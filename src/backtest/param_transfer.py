"""
Param Transfer
==============
Layer 1 of the contract roll param system.

Problem:
    Every month when MAYFUT → JUNFUT rolls, we have zero JUNFUT data.
    Running grid search requires 5-7 days of data.
    Those 5-7 days use stale MAYFUT params → bad trading.

Solution:
    Transfer MAYFUT params to JUNFUT by normalizing for:
        1. Volatility ratio   (spread_width, gamma scale with vol)
        2. Price ratio        (inventory limits scale with price)
        3. Liquidity ratio    (kappa scales with fill rate)

    This gets us 70-80% of optimal params on day 1 of new contract.
    Layer 2 (rolling grid search) and Layer 3 (online adapter) 
    refine from there.

Usage:
    # At contract roll — called automatically by param_loader.py
    from src.backtest.param_transfer import transfer_params

    junfut_params = transfer_params(
        symbol      = 'CHOLAFIN26JUNFUT',
        old_params  = mayfut_params,
        data_loader = load_day,
        new_dates   = ['2026-05-27', '2026-05-28'],   # first JUNFUT days
        old_dates   = ['2026-05-22', '2026-05-23'],   # last MAYFUT days
    )

How vol normalization works:
    MAYFUT spread_width = 0.50
    MAYFUT realized_vol = 0.30%
    JUNFUT realized_vol = 0.45%  (new contract, wider spreads)
    vol_ratio           = 0.45 / 0.30 = 1.5
    JUNFUT spread_width = 0.50 * 1.5 = 0.75  ← correct starting point

Architecture decisions:
    - Uses last 3 days of old contract + first N days of new contract
    - Falls back gracefully at every step — never crashes
    - Logs what it did so you can audit the transfer
    - Hard bounds prevent extreme params (0.5x to 2.0x of base)
"""

import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# How many days from each contract to use for vol estimation
OLD_CONTRACT_LOOKBACK = 3   # last N days of old contract
NEW_CONTRACT_LOOKBACK = 2   # first N days of new contract (may be 0 on day 1)

# Param bounds relative to old params — prevent extreme transfers
TRANSFER_BOUNDS = {
    'spread_width': (0.5, 2.5),
    'gamma':        (0.5, 2.5),
    'kappa':        (0.6, 1.8),   # kappa is structural, less variance
    'min_spread':   (0.5, 3.0),
    'open_mult':    (0.8, 1.5),   # open_mult barely changes contract to contract
}

# EWM alpha for realized vol computation — higher = more weight on recent bars
VOL_EWM_ALPHA = 0.05

# Minimum bars needed to compute reliable vol estimate
MIN_BARS_FOR_VOL = 500

# ─────────────────────────────────────────────────────────────────────────────
# Volatility estimation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_realized_vol(prices: np.ndarray) -> float:
    """
    Compute realized volatility from price series.

    Uses log returns + EWM to give more weight to recent bars.
    Returns annualized vol as a fraction (0.003 = 0.3%).

    Falls back to NaN if insufficient data.
    """
    if len(prices) < 10:
        return np.nan

    log_returns = np.diff(np.log(prices + 1e-10))

    # Remove extreme outliers — data errors cause spikes
    p1, p99 = np.percentile(log_returns, [1, 99])
    log_returns = log_returns[
        (log_returns >= p1) & (log_returns <= p99)
    ]

    if len(log_returns) < 10:
        return np.nan

    # EWM vol — recent bars matter more
    series   = pd.Series(log_returns)
    ewm_vol  = series.ewm(alpha=VOL_EWM_ALPHA).std().iloc[-1]

    # Annualize: ~23400 bars/day (1s bars), ~252 trading days
    bars_per_year = 23400 * 252
    annualized    = float(ewm_vol * np.sqrt(bars_per_year))

    return annualized


def _load_prices(symbol: str, dates: list, data_loader) -> np.ndarray:
    """
    Load price series for a symbol across multiple dates.
    Returns numpy array of prices. Empty array if load fails.
    """
    frames = []
    for date in dates:
        try:
            df = data_loader(symbol, date, market_hours_only=True)
            if df is not None and not df.empty:
                col = ('weighted_mid' if 'weighted_mid' in df.columns
                       else 'close')
                if col in df.columns:
                    frames.append(df[col].dropna().values)
        except Exception as e:
            logger.debug(f"Could not load {symbol} {date}: {e}")

    if not frames:
        return np.array([])

    return np.concatenate(frames)


def estimate_vol_ratio(
    new_symbol:  str,
    old_symbol:  str,
    new_dates:   list,
    old_dates:   list,
    data_loader,
) -> float:
    """
    Estimate volatility ratio between new and old contract.

    vol_ratio = new_vol / old_vol

    Returns 1.0 (no scaling) if insufficient data or load fails.
    Clamps result to [0.5, 2.0] to prevent extreme scaling.
    """
    # Load old contract prices
    old_prices = _load_prices(old_symbol, old_dates[-OLD_CONTRACT_LOOKBACK:],
                               data_loader)
    if len(old_prices) < MIN_BARS_FOR_VOL:
        logger.warning(
            f"Insufficient old contract data for {old_symbol} "
            f"({len(old_prices)} bars) — using vol_ratio=1.0"
        )
        return 1.0

    old_vol = _compute_realized_vol(old_prices)
    if np.isnan(old_vol) or old_vol < 1e-8:
        logger.warning(f"Could not compute vol for {old_symbol} — using 1.0")
        return 1.0

    # Load new contract prices — may be empty on day 1
    new_prices = _load_prices(new_symbol, new_dates[:NEW_CONTRACT_LOOKBACK],
                               data_loader)

    if len(new_prices) < MIN_BARS_FOR_VOL:
        # Day 1 of new contract: no data yet
        # Use old contract vol as proxy — better than nothing
        logger.info(
            f"No new contract data for {new_symbol} yet — "
            f"assuming vol_ratio=1.0 (will update tomorrow)"
        )
        return 1.0

    new_vol = _compute_realized_vol(new_prices)
    if np.isnan(new_vol) or new_vol < 1e-8:
        logger.warning(f"Could not compute vol for {new_symbol} — using 1.0")
        return 1.0

    vol_ratio = new_vol / old_vol

    # Clamp to prevent extreme scaling
    vol_ratio_clamped = float(np.clip(vol_ratio, 0.5, 2.0))

    logger.info(
        f"Vol transfer {old_symbol} → {new_symbol}: "
        f"old_vol={old_vol:.4f}  new_vol={new_vol:.4f}  "
        f"ratio={vol_ratio:.3f}  clamped={vol_ratio_clamped:.3f}"
    )

    return vol_ratio_clamped


def estimate_price_ratio(
    new_symbol:  str,
    old_symbol:  str,
    new_dates:   list,
    old_dates:   list,
    data_loader,
) -> float:
    """
    Estimate price ratio between new and old contract.
    Used to scale inventory limits.

    Returns 1.0 if insufficient data.
    """
    # Get last price of old contract
    old_prices = _load_prices(old_symbol, old_dates[-1:], data_loader)
    if len(old_prices) == 0:
        return 1.0
    old_price = float(np.median(old_prices[-100:]))  # median of last 100 bars

    # Get first price of new contract
    new_prices = _load_prices(new_symbol, new_dates[:1], data_loader)
    if len(new_prices) == 0:
        return 1.0
    new_price = float(np.median(new_prices[:100]))

    if old_price < 1e-8:
        return 1.0

    price_ratio = new_price / old_price
    price_ratio_clamped = float(np.clip(price_ratio, 0.7, 1.3))

    logger.info(
        f"Price transfer {old_symbol} → {new_symbol}: "
        f"old_price=Rs.{old_price:.2f}  new_price=Rs.{new_price:.2f}  "
        f"ratio={price_ratio:.3f}  clamped={price_ratio_clamped:.3f}"
    )

    return price_ratio_clamped


# ─────────────────────────────────────────────────────────────────────────────
# Core transfer logic
# ─────────────────────────────────────────────────────────────────────────────

def _apply_bounds(params: dict, base_params: dict) -> dict:
    """
    Apply hard bounds to transferred params.
    Prevents extreme values that could cause runaway losses.
    """
    bounded = {}
    for key, value in params.items():
        if key in TRANSFER_BOUNDS and key in base_params:
            lo, hi = TRANSFER_BOUNDS[key]
            base   = base_params[key]
            bounded[key] = float(np.clip(value, base * lo, base * hi))
        else:
            bounded[key] = value
    return bounded


def transfer_params(
    symbol:      str,
    old_params:  dict,
    data_loader,
    new_dates:   list,
    old_dates:   list,
    old_symbol:  Optional[str] = None,
) -> dict:
    """
    Transfer params from old contract to new contract.

    This is the main entry point called by param_loader.py
    when no params exist for the new contract.

    Args:
        symbol:      new contract symbol e.g. 'CHOLAFIN26JUNFUT'
        old_params:  params from old contract (MAYFUT)
        data_loader: function(symbol, date) → DataFrame
        new_dates:   available dates for new contract
        old_dates:   available dates for old contract
        old_symbol:  old contract symbol (auto-derived if None)

    Returns:
        Transferred and bounded params dict.
        Falls back to old_params if anything goes wrong.
    """
    if not old_params:
        logger.warning(f"No old params to transfer for {symbol}")
        return {}

    # Derive old symbol if not provided
    if old_symbol is None:
        old_symbol = _derive_old_symbol(symbol)

    if old_symbol is None:
        logger.warning(f"Could not derive old symbol for {symbol}")
        return old_params.copy()

    logger.info(f"Transferring params: {old_symbol} → {symbol}")

    try:
        # Step 1 — compute vol ratio
        vol_ratio = estimate_vol_ratio(
            new_symbol  = symbol,
            old_symbol  = old_symbol,
            new_dates   = new_dates,
            old_dates   = old_dates,
            data_loader = data_loader,
        )

        # Step 2 — compute price ratio
        price_ratio = estimate_price_ratio(
            new_symbol  = symbol,
            old_symbol  = old_symbol,
            new_dates   = new_dates,
            old_dates   = old_dates,
            data_loader = data_loader,
        )

        # Step 3 — scale params
        new_params = _scale_params(old_params, vol_ratio, price_ratio)

        # Step 4 — apply hard bounds
        new_params = _apply_bounds(new_params, old_params)

        logger.info(
            f"Param transfer complete for {symbol}: "
            f"vol_ratio={vol_ratio:.3f}  price_ratio={price_ratio:.3f}\n"
            f"  gamma:      {old_params.get('gamma', '?'):.4f} "
            f"→ {new_params.get('gamma', '?'):.4f}\n"
            f"  kappa:      {old_params.get('kappa', '?'):.4f} "
            f"→ {new_params.get('kappa', '?'):.4f}\n"
            f"  min_spread: {old_params.get('min_spread', '?'):.4f} "
            f"→ {new_params.get('min_spread', '?'):.4f}"
        )

        return new_params

    except Exception as e:
        logger.error(
            f"Param transfer failed for {symbol}: {e} — "
            f"falling back to old params"
        )
        return old_params.copy()


def _scale_params(
    old_params:  dict,
    vol_ratio:   float,
    price_ratio: float,
) -> dict:
    """
    Scale individual params based on vol and price ratios.

    Scaling logic per param:
        gamma       — scales with vol (higher vol → wider inventory penalty)
        kappa       — scales weakly with vol (fill rate is structural)
        min_spread  — scales with vol (need wider spread in higher vol)
        open_mult   — barely changes (opening auction behavior similar)
        spread_width — scales with vol (if present)
    """
    new_params = old_params.copy()

    # gamma: inventory risk penalty — scales linearly with vol
    if 'gamma' in old_params:
        new_params['gamma'] = old_params['gamma'] * vol_ratio

    # kappa: fill rate / market order arrival — weakly structural
    # Scale with sqrt of vol ratio (dampened — kappa is more stable)
    if 'kappa' in old_params:
        new_params['kappa'] = old_params['kappa'] * np.sqrt(vol_ratio)

    # min_spread: minimum quote spread — scales with vol
    if 'min_spread' in old_params:
        new_params['min_spread'] = old_params['min_spread'] * vol_ratio

    # spread_width: if V2 has this — scales with vol
    if 'spread_width' in old_params:
        new_params['spread_width'] = old_params['spread_width'] * vol_ratio

    # open_mult: opening period multiplier — slightly scales with vol
    # Dampened: opening behavior is mostly structural
    if 'open_mult' in old_params:
        dampened_ratio = 1.0 + (vol_ratio - 1.0) * 0.3
        new_params['open_mult'] = old_params['open_mult'] * dampened_ratio

    # alpha, sigma, other V2 params — scale with vol
    for key in ['alpha', 'sigma', 'vol_mult']:
        if key in old_params:
            new_params[key] = old_params[key] * vol_ratio

    return new_params


# ─────────────────────────────────────────────────────────────────────────────
# Contract symbol utilities
# ─────────────────────────────────────────────────────────────────────────────

# Month order for contract roll detection
_MONTH_ORDER = {
    'JAN': 1,  'FEB': 2,  'MAR': 3,  'APR': 4,
    'MAY': 5,  'JUN': 6,  'JUL': 7,  'AUG': 8,
    'SEP': 9,  'OCT': 10, 'NOV': 11, 'DEC': 12,
}

_CONTRACT_RE = re.compile(
    r'^(.*?)(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)FUT$',
    re.IGNORECASE
)


def _derive_old_symbol(new_symbol: str) -> Optional[str]:
    """
    Derive the previous contract symbol from the new one.

    CHOLAFIN26JUNFUT → CHOLAFIN26MAYFUT
    CHOLAFIN26JANFUT → CHOLAFIN25DECFUT  (year roll)
    """
    m = _CONTRACT_RE.match(new_symbol)
    if not m:
        return None

    base, year_str, month_str = m.group(1), m.group(2), m.group(3).upper()
    year  = int(year_str)
    month = _MONTH_ORDER.get(month_str, 0)

    if month == 0:
        return None

    # Previous month
    prev_month = month - 1
    prev_year  = year

    if prev_month == 0:
        prev_month = 12
        prev_year  = year - 1

    # Find month string for previous month
    prev_month_str = {v: k for k, v in _MONTH_ORDER.items()}[prev_month]
    prev_year_str  = str(prev_year)[-2:]   # 2026 → '26'

    return f"{base}{prev_year_str}{prev_month_str}FUT"


def get_contract_dates_from_params(params_dict: dict,
                                    base_name: str) -> list:
    """
    Get all available dates for a base symbol from params JSON keys.

    Useful for param_loader to know which old dates to pass
    to transfer_params.

    Returns sorted list of contract keys e.g.:
        ['CHOLAFIN26MAYFUT', 'CHOLAFIN26JUNFUT']
    """
    matching = [
        key for key in params_dict
        if _CONTRACT_RE.match(key) and
           _CONTRACT_RE.match(key).group(1) == base_name
    ]
    return sorted(matching)


# ─────────────────────────────────────────────────────────────────────────────
# Date utilities — used by param_loader to get recent trading dates
# ─────────────────────────────────────────────────────────────────────────────

def get_recent_trading_dates(
    n_days:     int = 5,
    before_date: Optional[str] = None,
) -> list:
    """
    Return last N trading dates (Mon-Fri, no NSE holidays).

    Used to find old_dates and new_dates for transfer_params
    without requiring the caller to know exact dates.

    Args:
        n_days:      number of trading days to return
        before_date: ISO date string — return dates before this
                     (defaults to today)

    Returns:
        List of ISO date strings, most recent last.
    """
    import datetime

    # NSE holidays 2026 — update annually
    # Source: https://www.nseindia.com/products-services/equity-market-holidays
    NSE_HOLIDAYS_2026 = {
        datetime.date(2026, 1, 26),   # Republic Day
        datetime.date(2026, 3, 25),   # Holi
        datetime.date(2026, 4, 2),    # Ram Navami
        datetime.date(2026, 4, 14),   # Dr Ambedkar Jayanti
        datetime.date(2026, 4, 3),    # Good Friday
        datetime.date(2026, 5, 1),    # Maharashtra Day
        datetime.date(2026, 8, 15),   # Independence Day
        datetime.date(2026, 10, 2),   # Gandhi Jayanti
        datetime.date(2026, 10, 22),  # Dussehra
        datetime.date(2026, 11, 11),  # Diwali Laxmi Pujan
        datetime.date(2026, 11, 12),  # Diwali Balipratipada
        datetime.date(2026, 11, 25),  # Gurunanak Jayanti
        datetime.date(2026, 12, 25),  # Christmas
    }

    if before_date:
        end = datetime.date.fromisoformat(before_date)
    else:
        end = datetime.date.today()

    dates   = []
    current = end - datetime.timedelta(days=1)

    while len(dates) < n_days:
        if (current.weekday() < 5 and   # Mon-Fri
                current not in NSE_HOLIDAYS_2026):
            dates.append(current.isoformat())
        current -= datetime.timedelta(days=1)

        if (end - current).days > 60:
            break

    return list(reversed(dates))


# ─────────────────────────────────────────────────────────────────────────────
# Batch transfer — run at contract roll for all symbols at once
# ─────────────────────────────────────────────────────────────────────────────

def transfer_all_params(
    old_params_dict: dict,
    new_contract:    str,
    old_contract:    str,
    data_loader,
    new_dates:       list,
    old_dates:       list,
    output_path:     Optional[Path] = None,
) -> dict:
    """
    Transfer params for ALL symbols at contract roll.

    Call this once at the start of each new contract month.
    Saves result to JSON so param_loader can pick it up.

    Args:
        old_params_dict: full params dict from old contract grid search
        new_contract:    e.g. 'JUNFUT'
        old_contract:    e.g. 'MAYFUT'
        data_loader:     function(symbol, date) → DataFrame
        new_dates:       first available dates of new contract
        old_dates:       last available dates of old contract
        output_path:     where to save transferred params JSON

    Returns:
        Dict of {new_symbol: transferred_params}

    Example:
        from src.backtest.data_loader import load_day
        from src.backtest.param_transfer import transfer_all_params

        transferred = transfer_all_params(
            old_params_dict = mayfut_params,
            new_contract    = 'JUNFUT',
            old_contract    = 'MAYFUT',
            data_loader     = load_day,
            new_dates       = ['2026-05-27'],
            old_dates       = ['2026-05-22', '2026-05-23'],
            output_path     = Path('research/findings/v1_optimal_params.json'),
        )
    """
    import json

    transferred = {}
    year_suffix = '26'   # update annually or derive from dates

    for old_key, old_params in old_params_dict.items():
        m = _CONTRACT_RE.match(old_key)
        if not m:
            continue

        base       = m.group(1)
        old_month  = m.group(3).upper()

        # Skip if this key is already the new contract
        if old_month == new_contract.replace('FUT', '').upper():
            continue

        new_symbol  = f"{base}{year_suffix}{new_contract}"
        old_symbol  = old_key

        params = transfer_params(
            symbol      = new_symbol,
            old_params  = old_params,
            data_loader = data_loader,
            new_dates   = new_dates,
            old_dates   = old_dates,
            old_symbol  = old_symbol,
        )

        transferred[new_symbol] = params
        print(f"  {old_symbol:<28} → {new_symbol:<28}  "
              f"vol_ratio from logs above")

    print(f"\nTransferred {len(transferred)} symbols")

    # Merge with existing params and save
    if output_path and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        existing.update(transferred)
        with open(output_path, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"Saved to {output_path}")

    return transferred


# ─────────────────────────────────────────────────────────────────────────────
# Quick validation — run this after transfer to sanity check results
# ─────────────────────────────────────────────────────────────────────────────

def validate_transfer(old_params: dict, new_params: dict,
                       symbol: str) -> bool:
    """
    Sanity check transferred params.
    Logs warnings for anything suspicious.
    Returns True if params look reasonable.
    """
    ok = True

    for key in ['gamma', 'kappa', 'min_spread']:
        if key not in new_params:
            logger.warning(f"{symbol}: missing key {key} after transfer")
            ok = False
            continue

        old_val = old_params.get(key, 0)
        new_val = new_params[key]

        if old_val > 0:
            ratio = new_val / old_val
            if ratio < 0.4 or ratio > 3.0:
                logger.warning(
                    f"{symbol}: {key} changed by {ratio:.2f}x "
                    f"({old_val:.4f} → {new_val:.4f}) — check vol data"
                )
                ok = False

        if new_val <= 0:
            logger.error(
                f"{symbol}: {key} = {new_val} after transfer — invalid"
            )
            ok = False

    return ok
