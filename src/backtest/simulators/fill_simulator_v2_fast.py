"""
Fast Vectorized V2 Fill Simulator — Ricci Hawkes-Alpha
=======================================================
Numba JIT compiled version of the full V2 strategy loop.

All five V2 steps compiled into one Numba function:
    1. Market classifier (influential MO detection)
    2. Hawkes process (λ±t decay + jump)
    3. Short-term alpha (αt decay + jump)
    4. Fill rate (κ±t decay + jump)
    5. Optimal quotes (Corollary 3.5.3) + fill simulation

Speed: ~0.02s per symbol/day after first compile (~5s compile once per session).

Usage:
    from src.backtest.simulators.fill_simulator_v2_fast import run_fast_v2_backtest
    result = run_fast_v2_backtest(df, params, lot_size=625)
"""

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("[v2_fast] Numba not available — using NumPy fallback")

GST_RATE = 0.18


@dataclass
class FastV2Result:
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
# Numba JIT core loop — all V2 steps in one function
# ─────────────────────────────────────────────────────────────────────────────

def _v2_loop_numpy(
    ts_sec, mid, sigma, lows, highs,
    bid_q1, ask_q1, tick_count,
    price_impact, volume_delta, imbalance_last,
    volume_ratio, bad_bars,
    # Hawkes
    beta, theta, eta, nu, rho,
    # Alpha
    zeta, epsilon_plus, epsilon_minus,
    # Fill rate
    beta_kappa, theta_kappa, eta_kappa, nu_kappa,
    # Strategy
    phi, min_spread, max_spread, open_mult,
    max_inventory, classifier_threshold, lot_size,
):
    n = len(ts_sec)

    # State
    lambda_buy  = theta
    lambda_sell = theta
    alpha_t     = 0.0
    kappa_buy   = theta_kappa
    kappa_sell  = theta_kappa
    inventory   = 0

    fill_prices = []
    fill_sides  = []
    inv_series  = np.zeros(n, dtype=np.int32)

    for i in range(n):
        ist = (ts_sec[i] + 19800.0) % 86400.0

        if bad_bars[i]:
            inv_series[i] = inventory
            continue

        # Terminal flatten
        if ist >= 54720.0 and abs(inventory) > 0:
            offset = max(0.05, mid[i] * 0.0002)
            if inventory > 0:
                fill_prices.append(mid[i] - offset)
                fill_sides.append(-1)
                inventory -= 1
            elif inventory < 0:
                fill_prices.append(mid[i] + offset)
                fill_sides.append(1)
                inventory += 1
            inv_series[i] = inventory
            continue

        # ── Step 1: Classify ──────────────────────────────────────────────────
        score = (0.35 * min(abs(price_impact[i]) * 10.0, 1.0) +
                 0.25 * min(max(volume_ratio[i] - 1.0, 0.0), 1.0) +
                 0.10 * min(abs(imbalance_last[i]) / 0.5, 1.0))
        is_influential = score >= classifier_threshold
        is_buy_inf  = is_influential and volume_delta[i] > 0
        is_sell_inf = is_influential and volume_delta[i] < 0

        # ── Step 2: Hawkes ────────────────────────────────────────────────────
        lambda_buy  = theta + (lambda_buy  - theta) * math.exp(-beta)
        lambda_sell = theta + (lambda_sell - theta) * math.exp(-beta)
        if is_buy_inf:
            lambda_buy  += eta
            lambda_sell += nu
        if is_sell_inf:
            lambda_sell += eta
            lambda_buy  += nu
        lambda_buy  = max(lambda_buy,  1e-6)
        lambda_sell = max(lambda_sell, 1e-6)

        # Stability check β > ρ(η+ν)
        if beta <= rho * (eta + nu):
            inv_series[i] = inventory
            continue

        # ── Step 3: Alpha ─────────────────────────────────────────────────────
        alpha_t *= math.exp(-zeta)
        if is_buy_inf:
            alpha_t += epsilon_plus
        if is_sell_inf:
            alpha_t -= epsilon_minus

        # ── Step 4: Fill rate ─────────────────────────────────────────────────
        kappa_buy  = theta_kappa + (kappa_buy  - theta_kappa) * math.exp(-beta_kappa)
        kappa_sell = theta_kappa + (kappa_sell - theta_kappa) * math.exp(-beta_kappa)
        if is_buy_inf:
            kappa_sell += eta_kappa
            kappa_buy  += nu_kappa
        if is_sell_inf:
            kappa_buy  += eta_kappa
            kappa_sell += nu_kappa

        # Blend with LOB observation
        if bid_q1[i] > 0 and ask_q1[i] > 0:
            bid_lots = bid_q1[i] / max(lot_size, 1.0)
            ask_lots = ask_q1[i] / max(lot_size, 1.0)
            kappa_buy  = 0.7 * kappa_buy  + 0.3 * (theta_kappa * 2.0 / max(bid_lots, 0.1))
            kappa_sell = 0.7 * kappa_sell + 0.3 * (theta_kappa * 2.0 / max(ask_lots, 0.1))

        kappa_buy  = max(0.1, min(kappa_buy,  10.0))
        kappa_sell = max(0.1, min(kappa_sell, 10.0))

        # ── Step 5: Optimal quotes (Corollary 3.5.3) ─────────────────────────
        T = max((55800.0 - max(ist - 33300.0, 0.0)) / 22500.0, 60.0 / 22500.0)

        d0_bid = 1.0 / kappa_buy
        d0_ask = 1.0 / kappa_sell
        B_bid  = 1.0 / (2.0 + kappa_buy)
        B_ask  = 1.0 / (2.0 + kappa_sell)

        alpha_integral = alpha_t / max(zeta, 1e-6) * (1.0 - math.exp(-zeta * T))

        beta_hat   = max(beta - rho * (eta - nu), 1e-6)
        h          = math.exp(-1.0)
        lambda_imb = lambda_buy - lambda_sell
        lambda_adj = phi * 2.0 * h * lambda_imb * (
            T / beta_hat -
            (1.0 - math.exp(-beta_hat * T)) / (beta_hat ** 2)
        )

        urgency     = 1.0 + (abs(inventory) / max(max_inventory, 1)) * (1.0 - T)
        inv_adj_bid = phi * (1.0 + 2.0 * inventory) * T * urgency
        inv_adj_ask = phi * (1.0 - 2.0 * inventory) * T * urgency

        delta_ask = d0_ask + B_ask * (-alpha_integral + inv_adj_ask + lambda_adj)
        delta_bid = d0_bid + B_bid * (+alpha_integral + inv_adj_bid + lambda_adj)

        # Dynamic min_spread (5 bps of mid)
        dyn_min   = max(min_spread, mid[i] * 0.0005)
        half_min  = dyn_min / 2.0
        half_max  = max_spread / 2.0

        # Open period multiplier 09:15-09:45 IST
        if 33300.0 <= ist <= 35100.0:
            delta_ask = min(delta_ask * open_mult, half_max)
            delta_bid = min(delta_bid * open_mult, half_max)

        delta_ask = max(half_min, min(delta_ask, half_max))
        delta_bid = max(half_min, min(delta_bid, half_max))

        bid = round(mid[i] - delta_bid, 2)
        ask = round(mid[i] + delta_ask, 2)

        # ── Step 6: Fill simulation ───────────────────────────────────────────
        ticks    = max(tick_count[i], 1.0)
        bq_lots  = max(bid_q1[i] / max(lot_size, 1.0), 0.5)
        aq_lots  = max(ask_q1[i] / max(lot_size, 1.0), 0.5)
        bar_rng  = max(highs[i] - lows[i], 0.05)

        if lows[i] <= bid and inventory < max_inventory:
            pf        = max(0.01, 1.0 - abs(bid - lows[i]) / bar_rng)
            fill_prob = min(1.0, (ticks * pf / bq_lots) * 0.3)
            if fill_prob > 0.05:
                fill_prices.append(bid)
                fill_sides.append(1)
                inventory += 1

        if highs[i] >= ask and inventory > -max_inventory:
            pf        = max(0.01, 1.0 - abs(ask - highs[i]) / bar_rng)
            fill_prob = min(1.0, (ticks * pf / aq_lots) * 0.3)
            if fill_prob > 0.05:
                fill_prices.append(ask)
                fill_sides.append(-1)
                inventory -= 1

        inv_series[i] = inventory

    return (np.array(fill_prices, dtype=np.float64),
            np.array(fill_sides,  dtype=np.int32),
            inv_series)


if NUMBA_AVAILABLE:
    @numba.jit(nopython=True, cache=True)
    def _v2_loop_numba(
        ts_sec, mid, sigma, lows, highs,
        bid_q1, ask_q1, tick_count,
        price_impact, volume_delta, imbalance_last,
        volume_ratio, bad_bars,
        beta, theta, eta, nu, rho,
        zeta, epsilon_plus, epsilon_minus,
        beta_kappa, theta_kappa, eta_kappa, nu_kappa,
        phi, min_spread, max_spread, open_mult,
        max_inventory, classifier_threshold, lot_size,
    ):
        n = len(ts_sec)

        lambda_buy  = theta
        lambda_sell = theta
        alpha_t     = 0.0
        kappa_buy   = theta_kappa
        kappa_sell  = theta_kappa
        inventory   = numba.int32(0)

        fill_prices = numba.typed.List.empty_list(numba.float64)
        fill_sides  = numba.typed.List.empty_list(numba.int32)
        inv_series  = np.zeros(n, dtype=numba.int32)

        for i in range(n):
            ist = (ts_sec[i] + 19800.0) % 86400.0

            if bad_bars[i]:
                inv_series[i] = inventory
                continue

            if ist >= 54720.0 and abs(inventory) > 0:
                offset = max(0.05, mid[i] * 0.0002)
                if inventory > 0:
                    fill_prices.append(mid[i] - offset)
                    fill_sides.append(numba.int32(-1))
                    inventory -= numba.int32(1)
                elif inventory < 0:
                    fill_prices.append(mid[i] + offset)
                    fill_sides.append(numba.int32(1))
                    inventory += numba.int32(1)
                inv_series[i] = inventory
                continue

            # Classify
            score = (0.35 * min(abs(price_impact[i]) * 10.0, 1.0) +
                     0.25 * min(max(volume_ratio[i] - 1.0, 0.0), 1.0) +
                     0.10 * min(abs(imbalance_last[i]) / 0.5, 1.0))
            is_influential = score >= classifier_threshold
            is_buy_inf  = is_influential and volume_delta[i] > 0.0
            is_sell_inf = is_influential and volume_delta[i] < 0.0

            # Hawkes
            lambda_buy  = theta + (lambda_buy  - theta) * math.exp(-beta)
            lambda_sell = theta + (lambda_sell - theta) * math.exp(-beta)
            if is_buy_inf:
                lambda_buy  += eta
                lambda_sell += nu
            if is_sell_inf:
                lambda_sell += eta
                lambda_buy  += nu
            lambda_buy  = max(lambda_buy,  1e-6)
            lambda_sell = max(lambda_sell, 1e-6)

            # Stability
            if beta <= rho * (eta + nu):
                inv_series[i] = inventory
                continue

            # Alpha
            alpha_t *= math.exp(-zeta)
            if is_buy_inf:
                alpha_t += epsilon_plus
            if is_sell_inf:
                alpha_t -= epsilon_minus

            # Fill rate
            kappa_buy  = theta_kappa + (kappa_buy  - theta_kappa) * math.exp(-beta_kappa)
            kappa_sell = theta_kappa + (kappa_sell - theta_kappa) * math.exp(-beta_kappa)
            if is_buy_inf:
                kappa_sell += eta_kappa
                kappa_buy  += nu_kappa
            if is_sell_inf:
                kappa_buy  += eta_kappa
                kappa_sell += nu_kappa

            if bid_q1[i] > 0.0 and ask_q1[i] > 0.0:
                bid_lots   = bid_q1[i] / max(lot_size, 1.0)
                ask_lots   = ask_q1[i] / max(lot_size, 1.0)
                kappa_buy  = 0.7 * kappa_buy  + 0.3 * (theta_kappa * 2.0 / max(bid_lots, 0.1))
                kappa_sell = 0.7 * kappa_sell + 0.3 * (theta_kappa * 2.0 / max(ask_lots, 0.1))

            kappa_buy  = max(0.1, min(kappa_buy,  10.0))
            kappa_sell = max(0.1, min(kappa_sell, 10.0))

            # Optimal quotes
            T = max((55800.0 - max(ist - 33300.0, 0.0)) / 22500.0, 60.0 / 22500.0)

            d0_bid = 1.0 / kappa_buy
            d0_ask = 1.0 / kappa_sell
            B_bid  = 1.0 / (2.0 + kappa_buy)
            B_ask  = 1.0 / (2.0 + kappa_sell)

            alpha_integral = alpha_t / max(zeta, 1e-6) * (1.0 - math.exp(-zeta * T))

            beta_hat   = max(beta - rho * (eta - nu), 1e-6)
            h          = math.exp(-1.0)
            lambda_imb = lambda_buy - lambda_sell
            lambda_adj = phi * 2.0 * h * lambda_imb * (
                T / beta_hat -
                (1.0 - math.exp(-beta_hat * T)) / (beta_hat ** 2)
            )

            urgency     = 1.0 + (abs(inventory) / max(max_inventory, 1)) * (1.0 - T)
            inv_adj_bid = phi * (1.0 + 2.0 * inventory) * T * urgency
            inv_adj_ask = phi * (1.0 - 2.0 * inventory) * T * urgency

            delta_ask = d0_ask + B_ask * (-alpha_integral + inv_adj_ask + lambda_adj)
            delta_bid = d0_bid + B_bid * (+alpha_integral + inv_adj_bid + lambda_adj)

            dyn_min  = max(min_spread, mid[i] * 0.0005)
            half_min = dyn_min / 2.0
            half_max = max_spread / 2.0

            if 33300.0 <= ist <= 35100.0:
                delta_ask = min(delta_ask * open_mult, half_max)
                delta_bid = min(delta_bid * open_mult, half_max)

            delta_ask = max(half_min, min(delta_ask, half_max))
            delta_bid = max(half_min, min(delta_bid, half_max))

            bid = round(mid[i] - delta_bid, 2)
            ask = round(mid[i] + delta_ask, 2)

            # Fill simulation
            ticks   = max(tick_count[i], 1.0)
            bq_lots = max(bid_q1[i] / max(lot_size, 1.0), 0.5)
            aq_lots = max(ask_q1[i] / max(lot_size, 1.0), 0.5)
            bar_rng = max(highs[i] - lows[i], 0.05)

            if lows[i] <= bid and inventory < max_inventory:
                pf        = max(0.01, 1.0 - abs(bid - lows[i]) / bar_rng)
                fill_prob = min(1.0, (ticks * pf / bq_lots) * 0.3)
                if fill_prob > 0.05:
                    fill_prices.append(bid)
                    fill_sides.append(numba.int32(1))
                    inventory += numba.int32(1)

            if highs[i] >= ask and inventory > -max_inventory:
                pf        = max(0.01, 1.0 - abs(ask - highs[i]) / bar_rng)
                fill_prob = min(1.0, (ticks * pf / aq_lots) * 0.3)
                if fill_prob > 0.05:
                    fill_prices.append(ask)
                    fill_sides.append(numba.int32(-1))
                    inventory -= numba.int32(1)

            inv_series[i] = inventory

        return fill_prices, fill_sides, inv_series


def _run_v2_loop(ts_sec, mid, sigma, lows, highs, bid_q1, ask_q1,
                 tick_count, price_impact, volume_delta, imbalance_last,
                 volume_ratio, bad_bars, beta, theta, eta, nu, rho,
                 zeta, epsilon_plus, epsilon_minus, beta_kappa, theta_kappa,
                 eta_kappa, nu_kappa, phi, min_spread, max_spread, open_mult,
                 max_inventory, classifier_threshold, lot_size):
    if NUMBA_AVAILABLE:
        fp, fs, inv = _v2_loop_numba(
            ts_sec, mid, sigma, lows, highs, bid_q1, ask_q1,
            tick_count, price_impact, volume_delta, imbalance_last,
            volume_ratio, bad_bars, beta, theta, eta, nu, rho,
            zeta, epsilon_plus, epsilon_minus, beta_kappa, theta_kappa,
            eta_kappa, nu_kappa, phi, min_spread, max_spread, open_mult,
            float(max_inventory), classifier_threshold, float(lot_size)
        )
        fp_arr  = np.empty(len(fp),  dtype=np.float64)
        fs_arr  = np.empty(len(fs),  dtype=np.int32)
        for _i in range(len(fp)):
            fp_arr[_i] = fp[_i]
        for _i in range(len(fs)):
            fs_arr[_i] = fs[_i]
        inv_arr = np.array(inv, dtype=np.int32) if not isinstance(inv, np.ndarray) \
                  else inv
        return fp_arr, fs_arr, inv_arr
    else:
        return _v2_loop_numpy(
            ts_sec, mid, sigma, lows, highs, bid_q1, ask_q1,
            tick_count, price_impact, volume_delta, imbalance_last,
            volume_ratio, bad_bars, beta, theta, eta, nu, rho,
            zeta, epsilon_plus, epsilon_minus, beta_kappa, theta_kappa,
            eta_kappa, nu_kappa, phi, min_spread, max_spread, open_mult,
            max_inventory, classifier_threshold, lot_size
        )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (reuse from V1)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(fill_prices, fill_sides, lot_size,
                     instrument_type='equity_futures'):
    if len(fill_prices) == 0:
        return {'gross_pnl': 0.0, 'total_costs': 0.0, 'net_pnl': 0.0,
                'total_fills': 0, 'buy_fills': 0, 'sell_fills': 0,
                'win_rate': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}

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

    # FIFO PnL
    from collections import deque
    buy_queue = deque()
    gross_pnl = 0.0
    pnl_list  = []

    for price, side in zip(fill_prices, fill_sides):
        if side == 1:
            buy_queue.append(float(price))
        else:
            if buy_queue:
                buy_price  = buy_queue.popleft()
                trade_pnl  = (float(price) - buy_price) * lot_size
                gross_pnl += trade_pnl
                pnl_list.append(gross_pnl)

    pnl_series = np.array(pnl_list, dtype=np.float64)
    net_pnl    = gross_pnl - total_costs

    sharpe = 0.0
    if len(pnl_series) > 1:
        diffs = np.diff(pnl_series)
        std   = diffs.std()
        if std > 0:
            sharpe = float((diffs.mean() / std) * np.sqrt(252))

    max_dd = 0.0
    if len(pnl_series) > 0:
        peak   = np.maximum.accumulate(pnl_series)
        max_dd = float((pnl_series - peak).min())

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

def run_fast_v2_backtest(
    df:               pd.DataFrame,
    # Hawkes
    beta:             float = 1.0,
    theta:            float = 2.0,
    eta:              float = 0.5,
    nu:               float = 0.2,
    rho:              float = 0.30,
    # Alpha
    zeta:             float = 0.5,
    epsilon_plus:     float = 0.002,
    epsilon_minus:    float = 0.002,
    # Fill rate
    beta_kappa:       float = 0.5,
    theta_kappa:      float = 1.5,
    eta_kappa:        float = 0.3,
    nu_kappa:         float = 0.1,
    # Strategy
    phi:              float = 0.001,
    min_spread:       float = 0.10,
    max_spread:       float = 10.0,
    open_mult:        float = 2.0,
    lot_size:         int   = 75,
    max_inventory:    int   = 5,
    classifier_threshold: float = 0.50,
    instrument_type:  str   = 'equity_futures',
) -> FastV2Result:

    if df.empty:
        return FastV2Result(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    df = df.sort_values('ts_sec').reset_index(drop=True)
    n  = len(df)

    # Extract arrays
    ts_sec       = df['ts_sec'].values.astype(np.float64)
    mid          = df['weighted_mid'].fillna(df['close']).values.astype(np.float64)
    vol_raw      = df['realized_vol_60s'].fillna(2.0).values.astype(np.float64)
    vol_raw      = np.where(vol_raw <= 0, 2.0, vol_raw)
    lows         = df['low'].values.astype(np.float64)
    highs        = df['high'].values.astype(np.float64)
    bid_q1       = df['bid_q1'].fillna(lot_size).values.astype(np.float64)
    ask_q1       = df['ask_q1'].fillna(lot_size).values.astype(np.float64)
    tick_count   = df['tick_count'].fillna(1).values.astype(np.float64)
    price_impact = df['price_impact'].fillna(0).values.astype(np.float64) \
                   if 'price_impact' in df.columns else np.zeros(n)
    volume_delta = df['volume_delta'].fillna(0).values.astype(np.float64)
    imb_last     = df['imbalance_last'].fillna(0).values.astype(np.float64)
    vol_ratio    = df['volume_ratio'].fillna(1.0).values.astype(np.float64)

    # Rolling sigma (300-bar mean)
    kernel = np.ones(300) / 300.0
    sigma  = np.convolve(vol_raw, kernel, mode='full')[:n]
    sigma[:300] = vol_raw[:300]
    sigma  = np.where(sigma <= 0, 2.0, sigma)

    # Imbalance filter
    imb_ma30  = df['imbalance_ma_30s'].fillna(0).values.astype(np.float64)
    bad_bars  = ((np.abs(imb_last - imb_ma30) > 0.3) &
                 (vol_ratio < 0.5)).astype(np.bool_)

    # Run loop
    fill_prices, fill_sides, inv_arr = _run_v2_loop(
        ts_sec, mid, sigma, lows, highs, bid_q1, ask_q1,
        tick_count, price_impact, volume_delta, imb_last,
        vol_ratio, bad_bars,
        beta, theta, eta, nu, rho,
        zeta, epsilon_plus, epsilon_minus,
        beta_kappa, theta_kappa, eta_kappa, nu_kappa,
        phi, min_spread, max_spread, open_mult,
        max_inventory, classifier_threshold, float(lot_size)
    )

    # Metrics
    fp_arr = np.array(fill_prices, dtype=np.float64)
    fs_arr = np.array(fill_sides,  dtype=np.int32)
    m      = _compute_metrics(fp_arr, fs_arr, lot_size, instrument_type)

    active_bars = int((~bad_bars).sum())
    fill_rate   = len(fp_arr) / max(active_bars * 2, 1) * 100
    inv_series  = np.array(inv_arr, dtype=np.int32)
    max_inv     = int(np.abs(inv_series).max()) if len(inv_series) > 0 else 0
    final_inv   = int(inv_series[-1]) if len(inv_series) > 0 else 0

    return FastV2Result(
        net_pnl         = m['net_pnl'],
        gross_pnl       = m['gross_pnl'],
        total_costs      = m['total_costs'],
        total_fills      = m['total_fills'],
        buy_fills        = m['buy_fills'],
        sell_fills       = m['sell_fills'],
        fill_rate        = round(fill_rate, 2),
        win_rate         = round(m['win_rate'] * 100, 1),
        sharpe_ratio     = round(m['sharpe_ratio'], 2),
        max_drawdown     = round(m['max_drawdown'], 2),
        max_inventory    = max_inv,
        final_inventory  = final_inv,
        bars_processed   = n,
    )
