"""
Parameter Loader
================
Loads per-symbol optimized parameters.

Lookup order (4 layers):
    1. Exact match:     CHOLAFIN26JUNFUT in params JSON
    2. Base name match: any CHOLAFIN* key (old contract fallback)
    3. Param transfer:  scale old contract params by vol ratio
    4. Defaults:        last resort hardcoded fallback

Param source priority:
    A. GCS bucket   — written nightly by rolling_optimizer.py on VM
    B. Local JSON   — fallback if GCS unavailable (e.g. no internet)

This means:
    - GCP VM writes fresh params to GCS every night automatically
    - Local backtest reads from GCS → always has latest params
    - If GCS is down, falls back to local JSON seamlessly
"""

import re
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

GCS_BUCKET     = 'hedge-fund-494103-marketdata-mumbai'
GCS_PARAMS_DIR = 'params'

# Regex to strip contract suffix
_CONTRACT_RE = re.compile(
    r'\d{2}(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)FUT$',
    re.IGNORECASE
)

# Default fallback params — last resort only
# Conservative values: won't blow up but won't perform well
_DEFAULTS = {
    'gamma':      0.001,
    'kappa':      1.5,
    'min_spread': 0.10,
    'open_mult':  2.0,
}

# In-memory caches
_params_cache:   dict = {}   # {model: params_dict}  — full params per model
_transfer_cache: dict = {}   # {(symbol, old_key): transferred_params}


# ─────────────────────────────────────────────────────────────────────────────
# Param loading — GCS first, local fallback
# ─────────────────────────────────────────────────────────────────────────────

def load_optimal_params(model: str = 'v1', force_refresh: bool = False) -> dict:
    """
    Load all optimal params for a model.

    Source priority:
        1. In-memory cache (fastest — same process)
        2. GCS bucket      (written nightly by VM)
        3. Local JSON      (fallback if GCS unavailable)

    Args:
        model:         'v1' or 'v2'
        force_refresh: bypass cache and reload from GCS/local

    Returns:
        Dict of {symbol: params_dict}
    """
    global _params_cache

    # Check memory cache first
    if not force_refresh and model in _params_cache:
        return _params_cache[model]

    # Try GCS
    params = _load_from_gcs(model)

    # Fall back to local
    if params is None:
        params = _load_from_local(model)

    if params is None:
        logger.warning(f"No params found for model={model} in GCS or local")
        params = {}

    # Cache in memory
    _params_cache[model] = params
    return params


def _load_from_gcs(model: str) -> Optional[dict]:
    """
    Load params from GCS bucket.
    Returns None if unavailable (no credentials, network error, etc.)
    """
    try:
        from google.cloud import storage

        gcs_path = f"{GCS_PARAMS_DIR}/{model}_optimal_params.json"
        client   = storage.Client()
        bucket   = client.bucket(GCS_BUCKET)
        blob     = bucket.blob(gcs_path)

        if not blob.exists():
            logger.debug(f"GCS params not found: gs://{GCS_BUCKET}/{gcs_path}")
            return None

        params = json.loads(blob.download_as_text())
        logger.info(
            f"Loaded {len(params)} {model} params from "
            f"gs://{GCS_BUCKET}/{gcs_path}"
        )
        return params

    except ImportError:
        # google-cloud-storage not installed — local development
        logger.debug("google-cloud-storage not installed — using local params")
        return None

    except Exception as e:
        logger.warning(f"Could not load params from GCS: {e} — trying local")
        return None


def _load_from_local(model: str) -> Optional[dict]:
    """Load params from local JSON file."""
    path = Path(f'research/findings/{model}_optimal_params.json')
    if path.exists():
        with open(path) as f:
            params = json.load(f)
        logger.info(f"Loaded {len(params)} {model} params from {path}")
        return params

    logger.debug(f"Local params not found: {path}")
    return None


def clear_params_cache(model: Optional[str] = None):
    """
    Clear in-memory params cache.

    Call this at the start of each trading session to pick up
    fresh params written overnight by rolling_optimizer.py.

    Args:
        model: specific model to clear, or None to clear all
    """
    global _params_cache, _transfer_cache

    if model:
        _params_cache.pop(model, None)
        # Clear transfer cache for this model too
        keys_to_remove = [k for k in _transfer_cache if k[1] == model]
        for k in keys_to_remove:
            del _transfer_cache[k]
        logger.info(f"Cleared params cache for model={model}")
    else:
        _params_cache   = {}
        _transfer_cache = {}
        logger.info("Cleared all params cache")


# ─────────────────────────────────────────────────────────────────────────────
# Symbol param lookup
# ─────────────────────────────────────────────────────────────────────────────

def strip_contract_suffix(symbol: str) -> str:
    """CHOLAFIN26JUNFUT → CHOLAFIN"""
    return _CONTRACT_RE.sub('', symbol)


def get_symbol_params(
    symbol:       str,
    model:        str = 'v1',
    data_loader   = None,
    new_dates:    Optional[list] = None,
    old_dates:    Optional[list] = None,
    use_transfer: bool = True,
) -> dict:
    """
    Get optimized params for a symbol.

    Lookup order:
        1. Exact match    → CHOLAFIN26JUNFUT in params JSON
        2. Base name      → any CHOLAFIN* key (most recent contract)
        3. Param transfer → scale old params by vol ratio      ← Layer 1
        4. Defaults       → hardcoded fallback

    Args:
        symbol:       full symbol e.g. 'CHOLAFIN26JUNFUT'
        model:        'v1' or 'v2'
        data_loader:  function(symbol, date) → DataFrame
                      needed for Layer 3 transfer — if None, skipped
        new_dates:    dates for new contract (auto-derived if None)
        old_dates:    dates for old contract (auto-derived if None)
        use_transfer: set False to skip param transfer (pure backtest mode)

    Returns:
        dict with gamma, kappa, min_spread, open_mult
    """
    params = load_optimal_params(model)

    # ── Layer 1: Exact match ──────────────────────────────────────────────────
    if symbol in params:
        return _extract(params[symbol])

    # ── Layer 2: Base name match ──────────────────────────────────────────────
    base = strip_contract_suffix(symbol)
    if base:
        matching = {
            key: val for key, val in params.items()
            if strip_contract_suffix(key) == base
        }
        if matching:
            best_key   = sorted(matching.keys())[-1]
            old_params = _extract(params[best_key])

            # ── Layer 3: Param transfer ───────────────────────────────────────
            if use_transfer and data_loader is not None:
                transferred = _get_transferred_params(
                    symbol      = symbol,
                    old_params  = old_params,
                    old_key     = best_key,
                    data_loader = data_loader,
                    new_dates   = new_dates,
                    old_dates   = old_dates,
                )
                if transferred:
                    return transferred

            # Layer 2 fallback — old params unscaled
            logger.debug(
                f"{symbol}: using unscaled {best_key} params"
            )
            return old_params

    # ── Layer 4: Defaults ─────────────────────────────────────────────────────
    logger.warning(
        f"{symbol}: no params found anywhere — using defaults. "
        f"Run grid search or check GCS bucket."
    )
    return _DEFAULTS.copy()


def _get_transferred_params(
    symbol:      str,
    old_params:  dict,
    old_key:     str,
    data_loader,
    new_dates:   Optional[list],
    old_dates:   Optional[list],
) -> Optional[dict]:
    """
    Run Layer 3 param transfer with in-memory caching.
    Returns None if transfer fails.
    """
    from src.backtest.param_transfer import (
        transfer_params,
        validate_transfer,
        get_recent_trading_dates,
        _derive_old_symbol,
    )

    cache_key = (symbol, old_key)
    if cache_key in _transfer_cache:
        return _transfer_cache[cache_key]

    try:
        if new_dates is None or old_dates is None:
            all_recent = get_recent_trading_dates(n_days=10)
            if new_dates is None:
                new_dates = all_recent[-3:]
            if old_dates is None:
                old_dates = all_recent[:5]

        old_symbol   = _derive_old_symbol(symbol) or old_key
        transferred  = transfer_params(
            symbol      = symbol,
            old_params  = old_params,
            data_loader = data_loader,
            new_dates   = new_dates,
            old_dates   = old_dates,
            old_symbol  = old_symbol,
        )

        if not transferred:
            return None

        is_valid = validate_transfer(old_params, transferred, symbol)
        if not is_valid:
            logger.warning(
                f"{symbol}: transferred params failed validation — "
                f"using old params"
            )
            _transfer_cache[cache_key] = old_params
            return old_params

        logger.info(f"{symbol}: Layer 3 transfer applied")
        _transfer_cache[cache_key] = transferred
        return transferred

    except Exception as e:
        logger.error(f"Transfer failed for {symbol}: {e}")
        return None


def _extract(p: dict) -> dict:
    """Extract standard params from a JSON entry."""
    return {
        'gamma':      p.get('gamma',      _DEFAULTS['gamma']),
        'kappa':      p.get('kappa',       _DEFAULTS['kappa']),
        'min_spread': p.get('min_spread',  _DEFAULTS['min_spread']),
        'open_mult':  p.get('open_mult',   _DEFAULTS['open_mult']),
    }


def get_all_symbols_with_params(model: str = 'v1') -> list:
    """Return list of all symbols that have optimized params."""
    return list(load_optimal_params(model).keys())


# ─────────────────────────────────────────────────────────────────────────────
# Manual contract roll helper
# ─────────────────────────────────────────────────────────────────────────────

def run_contract_roll_transfer(
    model:        str = 'v1',
    new_contract: str = 'JUNFUT',
    old_contract: str = 'MAYFUT',
    new_dates:    Optional[list] = None,
    old_dates:    Optional[list] = None,
):
    """
    One-shot manual transfer at contract roll.

    Normally rolling_optimizer.py handles this automatically.
    Use this only if the VM missed the roll for some reason.

    Usage:
        from src.backtest.param_loader import run_contract_roll_transfer
        run_contract_roll_transfer(
            model        = 'v1',
            new_contract = 'JULFUT',
            old_contract = 'JUNFUT',
            new_dates    = ['2026-07-01'],
            old_dates    = ['2026-06-24', '2026-06-25'],
        )
    """
    from src.backtest.data_loader import load_day
    from src.backtest.param_transfer import (
        transfer_all_params,
        get_recent_trading_dates,
    )

    old_params = load_optimal_params(model, force_refresh=True)
    if not old_params:
        print(f"No params found for model={model}")
        return

    if new_dates is None:
        new_dates = get_recent_trading_dates(n_days=3)
    if old_dates is None:
        old_dates = get_recent_trading_dates(
            n_days=5, before_date=new_dates[0]
        )

    print(f"Transferring params: {old_contract} → {new_contract}")
    print(f"  Old dates: {old_dates}")
    print(f"  New dates: {new_dates}")

    transferred = transfer_all_params(
        old_params_dict = old_params,
        new_contract    = new_contract,
        old_contract    = old_contract,
        data_loader     = load_day,
        new_dates       = new_dates,
        old_dates       = old_dates,
        output_path     = None,
    )

    # Merge and push to GCS
    old_params.update(transferred)

    try:
        from google.cloud import storage
        gcs_path = f"{GCS_PARAMS_DIR}/{model}_optimal_params.json"
        client   = storage.Client()
        bucket   = client.bucket(GCS_BUCKET)
        blob     = bucket.blob(gcs_path)
        blob.upload_from_string(
            json.dumps(old_params, indent=2),
            content_type='application/json'
        )
        print(f"Saved to gs://{GCS_BUCKET}/{gcs_path}")
    except Exception as e:
        # Save locally if GCS unavailable
        local_path = Path(f'research/findings/{model}_optimal_params.json')
        with open(local_path, 'w') as f:
            json.dump(old_params, f, indent=2)
        print(f"GCS unavailable ({e}) — saved to {local_path}")

    clear_params_cache(model)
    print(f"Done. {len(transferred)} symbols transferred.")
    return transferred
