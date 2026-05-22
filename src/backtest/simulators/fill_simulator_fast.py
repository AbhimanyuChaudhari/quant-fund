"""
Fast Vectorized Fill Simulator — Fixed Version
===============================================
Fixes from benchmark debug:
    1. Terminal bars: don't mark as NaN — handle inside fill loop
    2. bid_q1 in shares: use lot_size not hardcoded 75
    3. Two-pass mismatch: single pass with inventory tracked inline
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("[fast_backtest] Numba not available — using NumPy fallback")

SESSION_START_IST = 33300   # 09:15 IST
SESSION_END_IST   = 55800   # 15:30 IST
SESSION_SECONDS   = 22500.0
FLATTEN_IST       = 54720   # 15:12 IST
OPEN_END_IST      = 35100   # 09:45 IST
GST_RATE          = 0.18


@dataclass
class FastBacktestResult:
    net_pnl:         float
    gross_pnl:       float
    total_costs:     float
    total_fills:     int
    buy_fills:       int
    sell_fills:      int
    fill_rate:       float
    win_rate:        float
    sharpe_ratio:    float
    max_drawdown:    float
    max_inventory:   int
    final_inventory: int
    bars_processed:  int


# ─────────────────────────────────────────────────────────────────────────────
# Core Numba JIT loop — quotes + fills in single pass
# ─────────────────────────────────────────────────────────────────────────────

def _run_loop_numpy(
    ts_sec:           np.ndarray,
    mid:              np.ndarray,
    sigma:            np.ndarray,   # rolling vol (precomputed)
    lows:             np.ndarray,
    highs:            np.ndarray,
    bid_q1:           np.ndarray,
    ask_q1:           np.ndarray,
    tick_count:       np.ndarray,
    bad_bars:         np.ndarray,   # bool mask: skip these bars
    gamma:            float,
    kappa:            float,
    min_spread:       float,
    max_spread:       float,
    open_mult:        float,
    max_inventory:    int,
    queue_aggression: float,
    lot_size:         int,
) -> tuple:
    """
    Single-pass loop: compute quotes and simulate fills together.
    Inventory is tracked bar-by-bar so quotes adapt to current position.
    """
    n         = len(ts_sec)
    inventory = 0

    # Pre-compute constants
    log_term  = (2.0 / gamma) * np.log(1.0 + gamma / kappa)

    fill_prices = []
    fill_sides  = []
    fill_bars   = []
    inv_series  = np.zeros(n, dtype=np.int32)

    for i in range(n):
        ist = (ts_sec[i] + 19800) % 86400

        # Skip bad bars (imbalance spike filter)
        if bad_bars[i]:
            inv_series[i] = inventory
            continue

        # Time remaining
        elapsed   = max(0.0, ist - SESSION_START_IST)
        remaining = max(60.0, SESSION_SECONDS - elapsed)
        T         = remaining / SESSION_SECONDS

        # Urgency
        urgency = 1.0 + (abs(inventory) / max(max_inventory, 1)) * (1.0 - T)

        # Reservation price
        s   = sigma[i]
        r   = mid[i] - inventory * gamma * s * s * T * urgency

        # Spread
        spread = min(max(gamma * s * s * T + log_term, min_spread), max_spread)

        # Open period multiplier
        if SESSION_START_IST <= ist <= OPEN_END_IST:
            spread = min(spread * open_mult, max_spread)

        # Terminal flatten — send market order
        if ist >= FLATTEN_IST and abs(inventory) > 1:
            if inventory > 0:
                fill_prices.append(mid[i] - 0.05)
                fill_sides.append(-1)
                fill_bars.append(i)
                inventory -= 1
            elif inventory < 0:
                fill_prices.append(mid[i] + 0.05)
                fill_sides.append(1)
                fill_bars.append(i)
                inventory += 1
            inv_series[i] = inventory
            continue

        bid = round(r - spread / 2, 2)
        ask = round(r + spread / 2, 2)

        # Fill probability using lot_size (not hardcoded 75)
        # bid_q1 is in shares → convert to lots
        bid_q1_lots = max(bid_q1[i] / lot_size, 0.5)
        ask_q1_lots = max(ask_q1[i] / lot_size, 0.5)
        ticks       = max(tick_count[i], 1.0)

        # BUY fill
        if lows[i] <= bid and inventory < max_inventory:
            # ticks needed to clear queue vs ticks available at our level
            price_frac    = max(0.01, 1.0 - abs(bid - lows[i]) /
                                max(highs[i] - lows[i], 0.05))
            ticks_at_level = ticks * price_frac
            ticks_to_clear = bid_q1_lots  # lots ahead of us
            fill_prob      = min(1.0, (ticks_at_level / max(ticks_to_clear, 0.5))
                                 * queue_aggression)
            if fill_prob > 0.05:
                fill_prices.append(bid)
                fill_sides.append(1)
                fill_bars.append(i)
                inventory += 1

        # SELL fill
        if highs[i] >= ask and inventory > -max_inventory:
            price_frac    = max(0.01, 1.0 - abs(ask - highs[i]) /
                                max(highs[i] - lows[i], 0.05))
            ticks_at_level = ticks * price_frac
            ticks_to_clear = ask_q1_lots
            fill_prob      = min(1.0, (ticks_at_level / max(ticks_to_clear, 0.5))
                                 * queue_aggression)
            if fill_prob > 0.05:
                fill_prices.append(ask)
                fill_sides.append(-1)
                fill_bars.append(i)
                inventory -= 1

        inv_series[i] = inventory

    return (np.array(fill_prices, dtype=np.float64),
            np.array(fill_sides,  dtype=np.int32),
            np.array(fill_bars,   dtype=np.int64),
            inv_series)


if NUMBA_AVAILABLE:
    @numba.jit(nopython=True, cache=True)
    def _run_loop_numba(
        ts_sec, mid, sigma, lows, highs,
        bid_q1, ask_q1, tick_count, bad_bars,
        gamma, kappa, min_spread, max_spread, open_mult,
        max_inventory, queue_aggression, lot_size,
    ):
        n         = len(ts_sec)
        inventory = numba.int32(0)
        log_term  = (2.0 / gamma) * np.log(1.0 + gamma / kappa)

        fill_prices = numba.typed.List.empty_list(numba.float64)
        fill_sides  = numba.typed.List.empty_list(numba.int32)
        fill_bars   = numba.typed.List.empty_list(numba.int64)
        inv_series  = np.zeros(n, dtype=numba.int32)

        for i in range(n):
            ist = (ts_sec[i] + 19800.0) % 86400.0

            if bad_bars[i]:
                inv_series[i] = inventory
                continue

            elapsed   = max(0.0, ist - 33300.0)
            remaining = max(60.0, 22500.0 - elapsed)
            T         = remaining / 22500.0

            urgency = 1.0 + (abs(inventory) / max(max_inventory, 1)) * (1.0 - T)

            s   = sigma[i]
            r   = mid[i] - inventory * gamma * s * s * T * urgency

            spread = min(max(gamma * s * s * T + log_term, min_spread), max_spread)

            if 33300.0 <= ist <= 35100.0:
                spread = min(spread * open_mult, max_spread)

            # Terminal flatten
            if ist >= 54720.0 and abs(inventory) > 1:
                if inventory > 0:
                    fill_prices.append(mid[i] - 0.05)
                    fill_sides.append(numba.int32(-1))
                    fill_bars.append(numba.int64(i))
                    inventory -= numba.int32(1)
                elif inventory < 0:
                    fill_prices.append(mid[i] + 0.05)
                    fill_sides.append(numba.int32(1))
                    fill_bars.append(numba.int64(i))
                    inventory += numba.int32(1)
                inv_series[i] = inventory
                continue

            bid = round(r - spread / 2.0, 2)
            ask = round(r + spread / 2.0, 2)

            bid_q1_lots = max(bid_q1[i] / lot_size, 0.5)
            ask_q1_lots = max(ask_q1[i] / lot_size, 0.5)
            ticks       = max(tick_count[i], 1.0)

            # BUY
            if lows[i] <= bid and inventory < max_inventory:
                bar_range     = max(highs[i] - lows[i], 0.05)
                price_frac    = max(0.01, 1.0 - abs(bid - lows[i]) / bar_range)
                ticks_at_lvl  = ticks * price_frac
                fill_prob     = min(1.0, (ticks_at_lvl / max(bid_q1_lots, 0.5))
                                    * queue_aggression)
                if fill_prob > 0.05:
                    fill_prices.append(bid)
                    fill_sides.append(numba.int32(1))
                    fill_bars.append(numba.int64(i))
                    inventory += numba.int32(1)

            # SELL
            if highs[i] >= ask and inventory > -max_inventory:
                bar_range     = max(highs[i] - lows[i], 0.05)
                price_frac    = max(0.01, 1.0 - abs(ask - highs[i]) / bar_range)
                ticks_at_lvl  = ticks * price_frac
                fill_prob     = min(1.0, (ticks_at_lvl / max(ask_q1_lots, 0.5))
                                    * queue_aggression)
                if fill_prob > 0.05:
                    fill_prices.append(ask)
                    fill_sides.append(numba.int32(-1))
                    fill_bars.append(numba.int64(i))
                    inventory -= numba.int32(1)

            inv_series[i] = inventory

        return fill_prices, fill_sides, fill_bars, inv_series


def _run_loop(ts_sec, mid, sigma, lows, highs, bid_q1, ask_q1,
              tick_count, bad_bars, gamma, kappa, min_spread,
              max_spread, open_mult, max_inventory,
              queue_aggression, lot_size):
    if NUMBA_AVAILABLE:
        fp, fs, fb, inv = _run_loop_numba(
            ts_sec, mid, sigma, lows, highs, bid_q1, ask_q1,
            tick_count, bad_bars, gamma, kappa, min_spread,
            max_spread, open_mult, max_inventory,
            queue_aggression, float(lot_size)
        )
        # Numba typed.List must be converted via list() outside nopython
        fp_arr  = np.empty(len(fp),  dtype=np.float64)
        fs_arr  = np.empty(len(fs),  dtype=np.int32)
        fb_arr  = np.empty(len(fb),  dtype=np.int64)
        for _i in range(len(fp)): fp_arr[_i] = fp[_i]
        for _i in range(len(fs)): fs_arr[_i] = fs[_i]
        for _i in range(len(fb)): fb_arr[_i] = fb[_i]
        inv_arr = inv if isinstance(inv, np.ndarray) else np.array(list(inv), dtype=np.int32)
        return (fp_arr, fs_arr, fb_arr, inv_arr)
    else:
        return _run_loop_numpy(
            ts_sec, mid, sigma, lows, highs, bid_q1, ask_q1,
            tick_count, bad_bars, gamma, kappa, min_spread,
            max_spread, open_mult, max_inventory,
            queue_aggression, lot_size
        )


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics_fast(
    fill_prices:     np.ndarray,
    fill_sides:      np.ndarray,
    lot_size:        int,
    instrument_type: str = 'equity_futures',
) -> dict:
    if len(fill_prices) == 0:
        return {
            'gross_pnl': 0.0, 'total_costs': 0.0, 'net_pnl': 0.0,
            'total_fills': 0, 'buy_fills': 0, 'sell_fills': 0,
            'win_rate': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0,
        }

    trade_values = fill_prices * lot_size

    if instrument_type == 'equity_futures':
        stt_rate, exchange_rate = 0.0005, 0.0000173
        sebi_rate, stamp_rate   = 0.000001, 0.00002
    else:
        stt_rate, exchange_rate = 0.0, 0.000009
        sebi_rate, stamp_rate   = 0.000001, 0.00001

    brokerage = np.full(len(fill_prices), 20.0)
    exchange  = trade_values * exchange_rate
    sebi      = trade_values * sebi_rate
    stt       = np.where(fill_sides == -1, trade_values * stt_rate, 0.0)
    stamp     = np.where(fill_sides ==  1, trade_values * stamp_rate, 0.0)
    gst       = (brokerage + exchange + sebi) * GST_RATE
    costs     = brokerage + exchange + sebi + stt + stamp + gst
    total_costs = float(costs.sum())

    # PnL: mirrors original engine compute_metrics exactly.
    # Track inventory + cost basis. Realize on sells (long close) only.
    # This matches how compute_metrics in metrics.py works.
    inventory  = 0
    cost_basis = 0.0
    gross_pnl  = 0.0
    pnl_list   = []

    for fp, fs in zip(fill_prices, fill_sides):
        fp = float(fp)
        if fs == 1:   # BUY
            cost_basis += fp
            inventory  += 1
        else:         # SELL
            if inventory > 0:
                avg_cost   = cost_basis / inventory
                trade_pnl  = (fp - avg_cost) * lot_size
                gross_pnl += trade_pnl
                pnl_list.append(gross_pnl)
                cost_basis -= avg_cost
                inventory  -= 1
            # If inventory <= 0, this is a short sell — skip PnL realization
            # (matches original engine behavior which only realizes on long closes)

    pnl_series = np.array(pnl_list, dtype=np.float64)

    net_pnl = gross_pnl - total_costs

    # Sharpe
    sharpe = 0.0
    if len(pnl_series) > 1:
        diffs = np.diff(pnl_series)
        std   = diffs.std()
        if std > 1e-4 and len(diffs) >= 5:
            sharpe = float((diffs.mean() / std) * np.sqrt(252))

    # Max drawdown
    max_dd = 0.0
    if len(pnl_series) > 0:
        peak  = np.maximum.accumulate(pnl_series)
        max_dd = float((pnl_series - peak).min())

    # Win rate
    win_rate = 0.0
    if len(pnl_series) > 1:
        win_rate = float((np.diff(pnl_series) > 0).mean())

    return {
        'gross_pnl':   gross_pnl,
        'total_costs': total_costs,
        'net_pnl':     net_pnl,
        'total_fills': len(fill_prices),
        'buy_fills':   int((fill_sides == 1).sum()),
        'sell_fills':  int((fill_sides == -1).sum()),
        'win_rate':    win_rate,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_fast_backtest(
    df:               pd.DataFrame,
    gamma:            float = 0.001,
    kappa:            float = 1.5,
    min_spread:       float = 0.10,
    max_spread:       float = 10.0,
    open_mult:        float = 2.0,
    lot_size:         int   = 75,
    max_inventory:    int   = 5,
    queue_aggression: float = 0.3,
    instrument_type:  str   = 'equity_futures',
) -> FastBacktestResult:

    if df.empty:
        return FastBacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    df = df.sort_values('ts_sec').reset_index(drop=True)
    n  = len(df)

    # ── Extract arrays ─────────────────────────────────────────────────────────
    ts_sec     = df['ts_sec'].values.astype(np.float64)
    mid        = df['weighted_mid'].fillna(df['close']).values.astype(np.float64)
    vol_raw    = df['realized_vol_60s'].fillna(2.0).values.astype(np.float64)
    vol_raw    = np.where(vol_raw <= 0, 2.0, vol_raw)
    lows       = df['low'].values.astype(np.float64)
    highs      = df['high'].values.astype(np.float64)
    bid_q1     = df['bid_q1'].fillna(lot_size).values.astype(np.float64)
    ask_q1     = df['ask_q1'].fillna(lot_size).values.astype(np.float64)
    tick_count = df['tick_count'].fillna(1).values.astype(np.float64)

    # ── Rolling σ (300-bar mean) ───────────────────────────────────────────────
    kernel = np.ones(300) / 300.0
    sigma  = np.convolve(vol_raw, kernel, mode='full')[:n]
    sigma[:300] = vol_raw[:300]
    sigma  = np.where(sigma <= 0, 2.0, sigma)

    # ── Imbalance filter ───────────────────────────────────────────────────────
    imb_last  = df['imbalance_last'].fillna(0).values.astype(np.float64)
    imb_ma30  = df['imbalance_ma_30s'].fillna(0).values.astype(np.float64)
    vol_ratio = df['volume_ratio'].fillna(1.0).values.astype(np.float64)
    bad_bars  = ((np.abs(imb_last - imb_ma30) > 0.3) &
                 (vol_ratio < 0.5)).astype(np.bool_)

    # ── Single-pass fill simulation ────────────────────────────────────────────
    fill_prices, fill_sides, fill_bars, inv_arr = _run_loop(
        ts_sec, mid, sigma, lows, highs, bid_q1, ask_q1,
        tick_count, bad_bars,
        gamma, kappa, min_spread, max_spread, open_mult,
        max_inventory, queue_aggression, float(lot_size)
    )

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics = compute_metrics_fast(fill_prices, fill_sides,
                                    lot_size, instrument_type)

    # Fill rate = fills / (2 × active bars)
    active_bars = int((~bad_bars).sum())
    fill_rate   = len(fill_prices) / max(active_bars * 2, 1) * 100

    max_inv   = int(np.abs(inv_arr).max()) if len(inv_arr) > 0 else 0
    final_inv = int(inv_arr[-1]) if len(inv_arr) > 0 else 0

    return FastBacktestResult(
        net_pnl         = metrics['net_pnl'],
        gross_pnl       = metrics['gross_pnl'],
        total_costs      = metrics['total_costs'],
        total_fills      = metrics['total_fills'],
        buy_fills        = metrics['buy_fills'],
        sell_fills       = metrics['sell_fills'],
        fill_rate        = round(fill_rate, 2),
        win_rate         = round(metrics['win_rate'] * 100, 1),
        sharpe_ratio     = round(metrics['sharpe_ratio'], 2),
        max_drawdown     = round(metrics['max_drawdown'], 2),
        max_inventory    = max_inv,
        final_inventory  = final_inv,
        bars_processed   = n,
    )