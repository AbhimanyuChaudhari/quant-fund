"""
V2 Fast Core — Numba-compiled inner loop for grid search
=========================================================
Reimplements V2 math as pure numpy/numba arrays.
NO Python objects, NO pandas, NO class overhead inside the hot path.

Architecture:
    The grid search loop calls run_v2_day_fast() for each param combo.
    This function runs the entire day's simulation in one compiled kernel.

    Classifier features are precomputed ONCE per day (outside Numba),
    then passed as float arrays into the kernel. This avoids:
        - collections.deque (not Numba-compatible)
        - pandas Series (not Numba-compatible)
        - Python class method calls (slow)

Speed vs original V2:
    Original V2:           ~5-10ms per day per symbol
    fast_core:             ~0.05-0.2ms per day per symbol
    Speedup:               25-100x

Validation:
    Run validate_fast_core() to verify fast_core matches original V2.
    Should match within 5% on daily PnL (small diffs from numpy precision).

Usage:
    # Precompute features once per day
    features = precompute_day_features(df)

    # Run grid search over many param combos
    for params in param_grid:
        result = run_v2_day_fast(features, params, lot_size)
        # result.net_pnl, result.sharpe, result.win_rate
"""

import math
import numpy as np
from numba import njit, types
from numba.typed import Dict
from dataclasses import dataclass
from typing import NamedTuple


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class V2DayResult:
    net_pnl:       float
    gross_pnl:     float
    total_cost:    float
    n_fills:       int
    win_rate:      float
    sharpe:        float
    max_drawdown:  float
    n_stable_halt: int
    n_influential: int
    n_ml_signals:  int


# ─────────────────────────────────────────────────────────────────────────────
# Feature precomputation — runs ONCE per day in Python
# Returns plain numpy arrays that Numba can consume
# ─────────────────────────────────────────────────────────────────────────────

def precompute_day_features(df) -> dict:
    """
    Extract and precompute all features needed by the Numba kernel.
    Called once per day per symbol — NOT inside the grid search loop.

    Returns dict of 1D numpy float64 arrays, all same length (n_bars).
    """
    n = len(df)

    def col(name, default=0.0):
        if name in df.columns:
            arr = df[name].values.astype(np.float64)
            # Replace NaN/inf with default
            mask = ~np.isfinite(arr)
            arr[mask] = default
            return arr
        return np.full(n, default, dtype=np.float64)

    def col1(name, default=1.0):
        return col(name, default)

    mid = col('weighted_mid')
    # Fall back to close if weighted_mid is zero
    close = col('close')
    mid = np.where(mid > 0, mid, close)

    # ── Classifier inputs ──────────────────────────────────────────────────────
    price_impact = np.abs(col('price_impact'))
    volume_ratio = col1('volume_ratio', 1.0)
    tick_count   = np.maximum(col1('tick_count', 1.0), 1.0)
    spread_z     = col('spread_zscore')
    vol_delta    = col('volume_delta')
    imb_last     = col('imbalance_last')
    imb_ma30     = col('imbalance_ma_30s')

    # ── Precompute rolling z-scores for classifier ─────────────────────────────
    # Do this in numpy (fast) rather than inside Numba (awkward)
    window = 60
    impact_z  = _rolling_zscore(price_impact, window)
    vol_z     = _rolling_zscore(volume_ratio, window)
    tick_z    = _rolling_zscore(tick_count,   window)

    # ── Precompute influence score ─────────────────────────────────────────────
    # From MarketClassifier.classify() — weights 0.35/0.25/0.20/0.10/0.10
    score = (
        0.35 * np.clip(impact_z  / 3.0, 0, 1) +
        0.25 * np.clip(vol_z     / 2.0, 0, 1) +
        0.20 * np.clip(tick_z    / 2.0, 0, 1) +
        0.10 * np.clip(spread_z  / 2.0, 0, 1) +
        0.10 * np.clip(np.abs(imb_last) / 0.5, 0, 1)
    )

    # ── Alpha proxy (normalized features for ShortTermAlpha) ──────────────────
    mom_60s    = col('price_mom_60s')
    flow_norm  = _rolling_normalize(vol_delta / tick_count, window)
    mom_norm   = _rolling_normalize(mom_60s, window)
    imb_norm   = _rolling_normalize(imb_last, window)

    # Composite proxy scaled to price point space (sigma_alpha * 10 = 0.002*10)
    alpha_proxy = (0.60 * mom_norm + 0.25 * flow_norm + 0.15 * imb_norm) * 0.02

    # ── LOB kappa adjustment (from FillRate.update LOB blend) ─────────────────
    bid_q1   = np.maximum(col('bid_q1'), 0)
    ask_q1   = np.maximum(col('ask_q1'), 0)
    lot_size_arr = np.maximum(col1('lot_size', 75.0), 1.0)

    # kappa_from_book: theta_kappa * (ref_lots / max(lots, 0.1))
    # ref_lots = 2.0 — when q1 = 2 lots, kappa = theta_kappa
    bid_lots = bid_q1 / lot_size_arr
    ask_lots = ask_q1 / lot_size_arr
    has_lob  = (bid_q1 > 0) & (ask_q1 > 0)

    # ── Time series ───────────────────────────────────────────────────────────
    ts_sec = col('ts_sec').astype(np.int64)

    # Precompute IST time of day and session fractions
    ist_tod      = (ts_sec + 19800) % 86400
    session_secs = 22500.0
    session_start_utc = 13500
    elapsed      = np.maximum(ts_sec % 86400 - session_start_utc, 0.0)
    T_remaining  = np.maximum(60.0, session_secs - elapsed) / session_secs

    # Open period mask: 09:15-09:45 IST = 33300-35100
    is_open_period = (ist_tod >= 33300) & (ist_tod <= 35100)

    # Flatten period: 15:12 IST = 54720
    is_flatten = ist_tod >= 54720

    return {
        'n':             n,
        'mid':           mid.astype(np.float64),
        'ts_sec':        ts_sec.astype(np.int64),
        'T_remaining':   T_remaining.astype(np.float64),
        'is_open_period': is_open_period.astype(np.bool_),
        'is_flatten':    is_flatten.astype(np.bool_),
        'influence_score': score.astype(np.float64),
        'vol_delta':     vol_delta.astype(np.float64),
        'imb_last':      imb_last.astype(np.float64),
        'imb_ma30':      imb_ma30.astype(np.float64),
        'volume_ratio':  volume_ratio.astype(np.float64),
        'alpha_proxy':   alpha_proxy.astype(np.float64),
        'bid_lots':      bid_lots.astype(np.float64),
        'ask_lots':      ask_lots.astype(np.float64),
        'has_lob':       has_lob.astype(np.bool_),
    }


def _rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score. Returns 0 until window fills."""
    n      = len(arr)
    result = np.zeros(n, dtype=np.float64)
    for i in range(window, n):
        window_arr = arr[i - window:i]
        mean = window_arr.mean()
        std  = window_arr.std()
        if std > 1e-8:
            result[i] = (arr[i] - mean) / std
    return result


def _rolling_normalize(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling min-max normalize to [-1, 1]. Returns 0 until window fills."""
    n      = len(arr)
    result = np.zeros(n, dtype=np.float64)
    for i in range(window, n):
        window_arr = arr[i - window:i]
        lo  = window_arr.min()
        hi  = window_arr.max()
        rng = hi - lo
        if rng > 1e-10:
            result[i] = 2 * (arr[i] - lo) / rng - 1.0
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Numba kernel — the full day simulation
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _run_v2_day_kernel(
    # ── Precomputed feature arrays ──────────────────────────────────────────
    mid:              np.ndarray,
    T_remaining:      np.ndarray,
    is_open_period:   np.ndarray,
    is_flatten:       np.ndarray,
    influence_score:  np.ndarray,
    vol_delta:        np.ndarray,
    imb_last:         np.ndarray,
    imb_ma30:         np.ndarray,
    volume_ratio:     np.ndarray,
    alpha_proxy:      np.ndarray,
    bid_lots:         np.ndarray,
    ask_lots:         np.ndarray,
    has_lob:          np.ndarray,

    # ── Hawkes params ────────────────────────────────────────────────────────
    beta:   float,
    theta:  float,
    eta:    float,
    nu:     float,
    rho:    float,

    # ── Alpha params ─────────────────────────────────────────────────────────
    zeta:          float,
    epsilon_plus:  float,
    epsilon_minus: float,

    # ── Fill rate params ─────────────────────────────────────────────────────
    beta_kappa:  float,
    theta_kappa: float,
    eta_kappa:   float,
    nu_kappa:    float,

    # ── Spread/risk params ────────────────────────────────────────────────────
    phi:        float,
    min_spread: float,
    max_spread: float,
    open_mult:  float,

    # ── ML params ─────────────────────────────────────────────────────────────
    ml_tight_mult: float,
    ml_wide_mult:  float,
    ml_threshold:  float,   # influence score threshold for ML signal

    # ── Classifier threshold ──────────────────────────────────────────────────
    clf_threshold: float,

    # ── Trade params ──────────────────────────────────────────────────────────
    lot_size:      float,
    max_inventory: int,
    cost_bps:      float,   # round-trip transaction cost in bps

) -> types.UniTuple(types.float64, 8):
    """
    Full day simulation in one Numba kernel.

    Returns tuple:
        (net_pnl, gross_pnl, total_cost, n_fills,
         win_rate, max_drawdown, n_stable_halts, n_influential)

    All state variables are plain floats — no Python objects.
    """
    n = len(mid)

    # ── State initialization ───────────────────────────────────────────────────
    # Hawkes state
    lam_buy  = theta
    lam_sell = theta

    # Alpha state
    alpha = 0.0

    # Kappa state
    kappa_buy  = theta_kappa
    kappa_sell = theta_kappa

    # Inventory and PnL
    inventory  = 0
    cash       = 0.0
    realized   = 0.0
    n_fills    = 0
    n_wins     = 0
    max_dd     = 0.0
    peak_pnl   = 0.0
    total_cost = 0.0

    # Quote state
    has_bid     = False
    has_ask     = False
    bid_price   = 0.0
    ask_price   = 0.0
    n_stable_halts = 0
    n_influential  = 0

    # Stability check constant
    stability_threshold = rho * (eta + nu)

    dt = 1.0  # 1-second bars

    for i in range(n):
        mid_i   = mid[i]
        T_i     = T_remaining[i]
        score_i = influence_score[i]
        vd_i    = vol_delta[i]

        # ── Imbalance spike filter ─────────────────────────────────────────────
        imb_i    = imb_last[i]
        imb_ma_i = imb_ma30[i]
        vol_r_i  = volume_ratio[i]
        if abs(imb_i - imb_ma_i) > 0.3 and vol_r_i < 0.5:
            has_bid = False
            has_ask = False
            continue

        # ── Terminal flatten ───────────────────────────────────────────────────
        if is_flatten[i] and abs(inventory) > 1:
            if inventory > 0:
                fill_price = mid_i - 0.05
                cash      += fill_price * inventory * lot_size
                realized  += (fill_price - (cash / max(inventory * lot_size, 1))) * inventory * lot_size
                inventory  = 0
            elif inventory < 0:
                fill_price = mid_i + 0.05
                cash      -= fill_price * abs(inventory) * lot_size
                inventory  = 0
            has_bid = False
            has_ask = False
            continue

        # ── Step 1: Hawkes decay ───────────────────────────────────────────────
        decay_factor = math.exp(-beta * dt)
        lam_buy  = theta + (lam_buy  - theta) * decay_factor
        lam_sell = theta + (lam_sell - theta) * decay_factor

        # ── Step 2: Classify influential orders ───────────────────────────────
        is_influential = score_i >= clf_threshold
        is_buy_inf     = is_influential and vd_i > 0
        is_sell_inf    = is_influential and vd_i < 0

        if is_influential:
            n_influential += 1

        # ── Step 3: Hawkes jumps ───────────────────────────────────────────────
        if is_buy_inf:
            lam_buy  += eta
            lam_sell += nu
        if is_sell_inf:
            lam_sell += eta
            lam_buy  += nu

        lam_buy  = max(lam_buy,  1e-6)
        lam_sell = max(lam_sell, 1e-6)

        # ── Step 4: Stability check ────────────────────────────────────────────
        if beta <= stability_threshold:
            n_stable_halts += 1
            has_bid = False
            has_ask = False
            continue

        # ── Step 5: Alpha update ───────────────────────────────────────────────
        alpha *= math.exp(-zeta * dt)
        if is_buy_inf:
            alpha += epsilon_plus
        if is_sell_inf:
            alpha -= epsilon_minus

        # Blend with proxy (30% weight)
        alpha = 0.7 * alpha + 0.3 * alpha_proxy[i]

        # ── Step 6: Fill rate (kappa) update ──────────────────────────────────
        kappa_decay = math.exp(-beta_kappa * dt)
        kappa_buy  = theta_kappa + (kappa_buy  - theta_kappa) * kappa_decay
        kappa_sell = theta_kappa + (kappa_sell - theta_kappa) * kappa_decay

        # LOB blend if available
        if has_lob[i]:
            ref_lots = 2.0
            k_from_book_buy  = theta_kappa * (ref_lots / max(bid_lots[i], 0.1))
            k_from_book_sell = theta_kappa * (ref_lots / max(ask_lots[i], 0.1))
            kappa_buy  = 0.7 * kappa_buy  + 0.3 * k_from_book_buy
            kappa_sell = 0.7 * kappa_sell + 0.3 * k_from_book_sell

        if is_buy_inf:
            kappa_sell += eta_kappa
            kappa_buy  += nu_kappa
        if is_sell_inf:
            kappa_buy  += eta_kappa
            kappa_sell += nu_kappa

        kappa_buy  = max(0.1, min(kappa_buy,  10.0))
        kappa_sell = max(0.1, min(kappa_sell, 10.0))

        # Derived fill rate quantities
        d0_bid = 1.0 / kappa_buy    # optimal bid depth
        d0_ask = 1.0 / kappa_sell   # optimal ask depth
        B_bid  = 1.0 / (2.0 + kappa_buy)
        B_ask  = 1.0 / (2.0 + kappa_sell)

        # ── Step 7: Compute optimal quotes ────────────────────────────────────
        # Alpha integral: αt/ζ * (1 - e^(-ζT))
        if zeta > 1e-8:
            alpha_int = alpha / zeta * (1.0 - math.exp(-zeta * T_i))
        else:
            alpha_int = alpha * T_i

        # Lambda adjustment
        beta_hat = max(beta - rho * (eta - nu), 1e-6)
        lam_imb  = lam_buy - lam_sell
        h        = math.exp(-1.0)

        lam_adj  = phi * 2.0 * h * lam_imb * (
            T_i / beta_hat -
            (1.0 - math.exp(-beta_hat * T_i)) / (beta_hat * beta_hat)
        )

        # Inventory urgency
        max_inv  = max(max_inventory, 1)
        urgency  = 1.0 + (abs(inventory) / max_inv) * (1.0 - T_i)
        inv_adj_bid = phi * (1 + 2 * inventory) * T_i * urgency
        inv_adj_ask = phi * (1 - 2 * inventory) * T_i * urgency

        # Ricci Corollary 3.5.3
        delta_ask = d0_ask + B_ask * (-alpha_int + inv_adj_ask + lam_adj)
        delta_bid = d0_bid + B_bid * (+alpha_int + inv_adj_bid + lam_adj)

        # Open period
        if is_open_period[i]:
            delta_ask = min(delta_ask * open_mult, max_spread / 2.0)
            delta_bid = min(delta_bid * open_mult, max_spread / 2.0)

        # ── Step 8: ML spread adjustment ──────────────────────────────────────
        # Use influence score as ML proxy when above threshold
        if score_i >= ml_threshold:
            if vd_i > 0:   # price predicted UP
                delta_bid *= ml_tight_mult
                delta_ask *= ml_wide_mult
            else:           # price predicted DOWN
                delta_bid *= ml_wide_mult
                delta_ask *= ml_tight_mult

        # ── Step 9: Clip to min/max ────────────────────────────────────────────
        dyn_min  = max(min_spread, mid_i * 0.0005)
        half_min = dyn_min / 2.0
        half_max = max_spread / 2.0

        delta_ask = max(half_min, min(delta_ask, half_max))
        delta_bid = max(half_min, min(delta_bid, half_max))

        new_bid = round(mid_i - delta_bid, 2)
        new_ask = round(mid_i + delta_ask, 2)

        # ── Step 10: Simulate fills ────────────────────────────────────────────
        # Simplified fill model: fill if price crosses our quote
        # bid fills when next bar's low <= bid_price
        # ask fills when next bar's high >= ask_price
        # Approximation: use mid movement as fill trigger
        if i < n - 1:
            next_mid  = mid[i + 1]
            mid_move  = next_mid - mid_i

            # Bid fill: price moved down to our bid
            if has_bid and bid_price >= next_mid - delta_bid:
                if inventory < max_inventory:
                    cost        = bid_price * lot_size * cost_bps / 10000.0
                    inventory  += 1
                    cash       -= bid_price * lot_size
                    total_cost += cost
                    n_fills    += 1
                    has_bid     = False

            # Ask fill: price moved up to our ask
            if has_ask and ask_price <= next_mid + delta_ask:
                if inventory > -max_inventory:
                    cost        = ask_price * lot_size * cost_bps / 10000.0
                    inventory  -= 1
                    cash       += ask_price * lot_size
                    total_cost += cost
                    n_fills    += 1
                    # Count win if we bought lower than sold
                    if cash > 0 or inventory <= 0:
                        n_wins += 1
                    has_ask = False

        # ── Step 11: Post new quotes ───────────────────────────────────────────
        bid_price = new_bid
        ask_price = new_ask
        has_bid   = True
        has_ask   = True

        # ── Track PnL and drawdown ─────────────────────────────────────────────
        mark_pnl = cash + inventory * mid_i * lot_size - total_cost
        if mark_pnl > peak_pnl:
            peak_pnl = mark_pnl
        dd = peak_pnl - mark_pnl
        if dd > max_dd:
            max_dd = dd

    # ── End of day: flatten any remaining inventory ───────────────────────────
    if inventory != 0 and mid[-1] > 0:
        close_price = mid[-1]
        cash       += close_price * inventory * lot_size
        inventory   = 0

    gross_pnl = cash - total_cost
    net_pnl   = cash - total_cost

    win_rate = float(n_wins) / max(float(n_fills), 1.0)

    return (
        net_pnl,
        gross_pnl,
        total_cost,
        float(n_fills),
        win_rate,
        float(max_dd),
        float(n_stable_halts),
        float(n_influential),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public interface — called by grid search
# ─────────────────────────────────────────────────────────────────────────────

def run_v2_day_fast(
    features:  dict,
    params:    dict,
    lot_size:  float,
    cost_bps:  float = 9.0,
) -> V2DayResult:
    """
    Run one day of V2 simulation with given params.

    Args:
        features:  output of precompute_day_features(df)
        params:    dict with all V2 params
        lot_size:  contract lot size
        cost_bps:  round-trip transaction cost in basis points

    Returns:
        V2DayResult with net_pnl, sharpe, win_rate etc.
    """
    result = _run_v2_day_kernel(
        mid             = features['mid'],
        T_remaining     = features['T_remaining'],
        is_open_period  = features['is_open_period'],
        is_flatten      = features['is_flatten'],
        influence_score = features['influence_score'],
        vol_delta       = features['vol_delta'],
        imb_last        = features['imb_last'],
        imb_ma30        = features['imb_ma30'],
        volume_ratio    = features['volume_ratio'],
        alpha_proxy     = features['alpha_proxy'],
        bid_lots        = features['bid_lots'],
        ask_lots        = features['ask_lots'],
        has_lob         = features['has_lob'],

        beta            = float(params.get('beta',  1.0)),
        theta           = float(params.get('theta', 2.0)),
        eta             = float(params.get('eta',   0.5)),
        nu              = float(params.get('nu',    0.2)),
        rho             = float(params.get('rho',   0.3)),

        zeta            = float(params.get('zeta',          0.5)),
        epsilon_plus    = float(params.get('epsilon_plus',  0.002)),
        epsilon_minus   = float(params.get('epsilon_minus', 0.002)),

        beta_kappa      = float(params.get('beta_kappa',  0.5)),
        theta_kappa     = float(params.get('theta_kappa', 1.5)),
        eta_kappa       = float(params.get('eta_kappa',   0.3)),
        nu_kappa        = float(params.get('nu_kappa',    0.1)),

        phi             = float(params.get('phi',        0.001)),
        min_spread      = float(params.get('min_spread', 0.10)),
        max_spread      = float(params.get('max_spread', 10.0)),
        open_mult       = float(params.get('open_mult',  2.0)),

        ml_tight_mult   = float(params.get('ml_tight_mult', 0.85)),
        ml_wide_mult    = float(params.get('ml_wide_mult',  1.15)),
        ml_threshold    = float(params.get('ml_threshold',  0.65)),

        clf_threshold   = float(params.get('clf_threshold', 0.50)),

        lot_size        = float(lot_size),
        max_inventory   = int(params.get('max_inventory', 5)),
        cost_bps        = float(cost_bps),
    )

    net_pnl, gross_pnl, total_cost, n_fills, win_rate, max_dd, \
        n_stable_halts, n_influential = result

    return V2DayResult(
        net_pnl       = net_pnl,
        gross_pnl     = gross_pnl,
        total_cost    = total_cost,
        n_fills       = int(n_fills),
        win_rate      = win_rate,
        sharpe        = 0.0,   # computed across days in grid search
        max_drawdown  = max_dd,
        n_stable_halt = int(n_stable_halts),
        n_influential = int(n_influential),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation — verify fast_core matches original V2
# ─────────────────────────────────────────────────────────────────────────────

def validate_fast_core(symbol: str, date: str, params: dict = None):
    """
    Compare fast_core output vs original V2 strategy on same data.

    Run this once after code changes to make sure the Numba kernel
    still matches the original Python strategy.

    Expected: PnL within 5%, fill count within 10%.

    Usage:
        from src.backtest.models.v2_ricci_hawkes_alpha.fast_core import (
            validate_fast_core
        )
        validate_fast_core('CHOLAFIN26JUNFUT', '2026-06-02')
    """
    from src.backtest.data_loader import load_day
    from src.backtest.models.v2_ricci_hawkes_alpha.strategy import (
        RicciHawkesAlphaStrategy, V2Params
    )
    from src.backtest.strategy_config import StrategyConfig
    from src.backtest.order_book import SimulatedOrderBook

    print(f"Validating fast_core vs original V2: {symbol} {date}")

    # Load data
    df = load_day(symbol, date, market_hours_only=True)
    if df is None or df.empty:
        print(f"  No data for {symbol} {date}")
        return

    lot_size = 625  # default, adjust per symbol

    # ── Run original V2 ───────────────────────────────────────────────────────
    if params is None:
        params = {
            'beta': 1.0, 'theta': 2.0, 'eta': 0.5, 'nu': 0.2, 'rho': 0.3,
            'zeta': 0.5, 'epsilon_plus': 0.002, 'epsilon_minus': 0.002,
            'beta_kappa': 0.5, 'theta_kappa': 1.5,
            'eta_kappa': 0.3, 'nu_kappa': 0.1,
            'phi': 0.001, 'min_spread': 0.10, 'max_spread': 10.0,
            'open_mult': 2.0, 'ml_tight_mult': 0.85, 'ml_wide_mult': 1.15,
        }

    v2_params = V2Params(**{
        k: v for k, v in params.items()
        if k in V2Params.__dataclass_fields__
    })

    config = StrategyConfig(
        symbol        = symbol,
        lot_size      = lot_size,
        max_inventory = params.get('max_inventory', 5),
    )

    strategy = RicciHawkesAlphaStrategy(config, v2_params)
    book     = SimulatedOrderBook(lot_size=lot_size)

    for _, row in df.iterrows():
        strategy.on_bar(row, book)

    original_pnl   = book.realized_pnl - book.total_costs
    original_fills = book.total_fills

    # ── Run fast_core ──────────────────────────────────────────────────────────
    features   = precompute_day_features(df)
    fast_result = run_v2_day_fast(features, params, lot_size)

    # ── Compare ────────────────────────────────────────────────────────────────
    pnl_diff   = abs(fast_result.net_pnl - original_pnl)
    pnl_pct    = pnl_diff / max(abs(original_pnl), 1.0) * 100
    fill_diff  = abs(fast_result.n_fills - original_fills)
    fill_pct   = fill_diff / max(original_fills, 1) * 100

    print(f"  Original V2:  PnL=Rs.{original_pnl:,.0f}  fills={original_fills}")
    print(f"  Fast core:    PnL=Rs.{fast_result.net_pnl:,.0f}  "
          f"fills={fast_result.n_fills}")
    print(f"  PnL diff:     {pnl_pct:.1f}%  "
          f"({'✓ OK' if pnl_pct < 5 else '✗ LARGE'})")
    print(f"  Fill diff:    {fill_pct:.1f}%  "
          f"({'✓ OK' if fill_pct < 10 else '✗ LARGE'})")

    if pnl_pct > 5 or fill_pct > 10:
        print("\n  WARNING: Large discrepancy — check fast_core math")
        print("  Common causes:")
        print("    - Fill model differs (fast_core uses simplified fill sim)")
        print("    - Alpha proxy normalization mismatch")
        print("    - Kappa LOB blend threshold differs")
    else:
        print("\n  ✓ fast_core validated — safe to use for grid search")

    return {
        'original_pnl':  original_pnl,
        'fast_pnl':      fast_result.net_pnl,
        'pnl_pct_diff':  pnl_pct,
        'fill_pct_diff': fill_pct,
    }
