"""
Rolling Param Optimizer
=======================
Runs nightly on GCP VM after market close + pipeline finishes.

What it does:
    1. Detects if today was a contract roll day
       → if yes: runs param transfer (Layer 1) for all symbols
    2. Runs fast grid search on last N days of data (Layer 2)
    3. Blends new params with existing params (smooth transition)
    4. Saves result to GCS bucket
    5. Logs everything for audit

Schedule (set by scheduler_setup.sh):
    Runs at 8:30 PM IST daily (after pipeline.py finishes)

GCS output:
    gs://hedge-fund-494103-marketdata/params/v1_optimal_params.json
    gs://hedge-fund-494103-marketdata/params/v2_optimal_params.json
    gs://hedge-fund-494103-marketdata/params/transfer_log.json

Usage:
    python -m src.automation.rolling_optimizer
    python -m src.automation.rolling_optimizer --model v2
    python -m src.automation.rolling_optimizer --force-transfer
    python -m src.automation.rolling_optimizer --dry-run
"""

import json
import logging
import argparse
import datetime
import traceback
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Logging — writes to file + stdout so journalctl picks it up
# ─────────────────────────────────────────────────────────────────────────────

LOG_DIR = Path('/home/ubuntu/quant-fund/logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level    = logging.INFO,
    format   = '%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt  = '%Y-%m-%d %H:%M:%S',
    handlers = [
        logging.FileHandler(LOG_DIR / 'rolling_optimizer.log'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

GCS_BUCKET       = 'hedge-fund-494103-marketdata'
GCS_PARAMS_DIR   = 'params'

# Rolling grid search config
MIN_DAYS_TO_OPTIMIZE = 2      # start optimizing after 2 days of data
LOOKBACK_DAYS        = 7      # use last 7 days for grid search
BLEND_ALPHA          = 0.4    # 40% new params, 60% old params per day

# Models to optimize
MODELS = ['v1', 'v2']

# NSE expiry dates — update monthly
NSE_EXPIRY_DATES = {
    (2026, 5):  datetime.date(2026, 5, 26),
    (2026, 6):  datetime.date(2026, 6, 25),
    (2026, 7):  datetime.date(2026, 7, 30),
    (2026, 8):  datetime.date(2026, 8, 27),
    (2026, 9):  datetime.date(2026, 9, 24),
    (2026, 10): datetime.date(2026, 10, 29),
    (2026, 11): datetime.date(2026, 11, 26),
    (2026, 12): datetime.date(2026, 12, 31),
}

# ─────────────────────────────────────────────────────────────────────────────
# GCS helpers
# ─────────────────────────────────────────────────────────────────────────────

def gcs_upload_json(data: dict, gcs_path: str):
    """Upload dict as JSON to GCS."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob   = bucket.blob(gcs_path)
    blob.upload_from_string(
        json.dumps(data, indent=2),
        content_type='application/json'
    )
    logger.info(f"Uploaded to gs://{GCS_BUCKET}/{gcs_path}")


def gcs_download_json(gcs_path: str) -> Optional[dict]:
    """Download JSON from GCS. Returns None if not found."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob   = bucket.blob(gcs_path)
        if not blob.exists():
            return None
        return json.loads(blob.download_as_text())
    except Exception as e:
        logger.warning(f"Could not download {gcs_path}: {e}")
        return None


def get_gcs_params_path(model: str) -> str:
    return f"{GCS_PARAMS_DIR}/{model}_optimal_params.json"


# ─────────────────────────────────────────────────────────────────────────────
# Contract roll detection
# ─────────────────────────────────────────────────────────────────────────────

def get_current_contract(date: datetime.date) -> str:
    """Return contract month string for a given date e.g. 'JUNFUT'"""
    month_names = {
        1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR',
        5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG',
        9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC',
    }

    # If today is after expiry, we're in next month's contract
    expiry = NSE_EXPIRY_DATES.get((date.year, date.month))
    if expiry and date > expiry:
        # Next month
        next_month = date.month + 1
        next_year  = date.year
        if next_month > 12:
            next_month = 1
            next_year += 1
        return f"{month_names[next_month]}FUT"

    return f"{month_names[date.month]}FUT"


def is_roll_day(today: datetime.date) -> bool:
    """
    True if today is the first trading day of a new contract.
    i.e. yesterday was expiry or today is the day after expiry.
    """
    yesterday = today - datetime.timedelta(days=1)
    # Skip weekends
    while yesterday.weekday() >= 5:
        yesterday -= datetime.timedelta(days=1)

    today_contract     = get_current_contract(today)
    yesterday_contract = get_current_contract(yesterday)

    return today_contract != yesterday_contract


def get_old_contract(today: datetime.date) -> str:
    """Return the contract that just expired."""
    yesterday = today - datetime.timedelta(days=1)
    while yesterday.weekday() >= 5:
        yesterday -= datetime.timedelta(days=1)
    return get_current_contract(yesterday)


# ─────────────────────────────────────────────────────────────────────────────
# Trading date utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_trading_dates_before(date: datetime.date, n: int) -> list:
    """Return last N trading dates before given date."""
    from src.backtest.param_transfer import get_recent_trading_dates
    return get_recent_trading_dates(
        n_days      = n,
        before_date = date.isoformat(),
    )


def get_available_contract_dates(
    contract: str,
    today:    datetime.date,
    max_lookback: int = 30,
) -> list:
    """
    Return trading dates where data exists for the given contract.
    Checks GCS for actual data files.
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    dates    = []
    check    = today - datetime.timedelta(days=1)
    lookback = 0

    while lookback < max_lookback:
        if check.weekday() < 5:   # trading day
            date_str = check.isoformat()
            # Check if any processed file exists for this date
            prefix = f"processed/features/"
            blobs  = list(bucket.list_blobs(
                prefix     = prefix,
                max_results = 1,
            ))
            # Simplified check — just use trading days in contract window
            contract_month = contract.replace('FUT', '')
            month_map = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
                'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
                'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
            }
            contract_month_num = month_map.get(contract_month, 0)
            if check.month == contract_month_num:
                dates.append(date_str)

        check    -= datetime.timedelta(days=1)
        lookback += 1

    return list(reversed(dates))


# ─────────────────────────────────────────────────────────────────────────────
# Param blending — smooth transition, avoid abrupt changes
# ─────────────────────────────────────────────────────────────────────────────

def blend_params(old: dict, new: dict, alpha: float = BLEND_ALPHA) -> dict:
    """
    Blend old and new params: result = (1-alpha)*old + alpha*new

    alpha=0.4 means 40% new params, 60% old params.
    Prevents whipsaw from noisy single-day optimization.

    Only blends numeric values. Non-numeric keys kept from new.
    """
    blended = {}
    for key in new:
        new_val = new[key]
        old_val = old.get(key, new_val)

        if isinstance(new_val, (int, float)) and isinstance(old_val, (int, float)):
            blended[key] = round(
                (1 - alpha) * float(old_val) + alpha * float(new_val),
                6
            )
        else:
            blended[key] = new_val

    # Keep any keys in old not in new
    for key in old:
        if key not in blended:
            blended[key] = old[key]

    return blended


def blend_all_params(old_params: dict, new_params: dict,
                      alpha: float = BLEND_ALPHA) -> dict:
    """Blend full params dicts symbol by symbol."""
    blended = {}

    all_symbols = set(list(old_params.keys()) + list(new_params.keys()))

    for symbol in all_symbols:
        if symbol in new_params and symbol in old_params:
            blended[symbol] = blend_params(
                old_params[symbol], new_params[symbol], alpha
            )
        elif symbol in new_params:
            # New symbol — use new params directly (no old to blend with)
            blended[symbol] = new_params[symbol]
        else:
            # Symbol dropped from new optimization — keep old
            blended[symbol] = old_params[symbol]

    return blended


# ─────────────────────────────────────────────────────────────────────────────
# Fast grid search wrapper
# ─────────────────────────────────────────────────────────────────────────────

def run_fast_grid_search(
    model:      str,
    dates:      list,
    dry_run:    bool = False,
) -> Optional[dict]:
    """
    Run grid search on given dates. Returns new params dict or None.

    Calls existing grid_search_v1.py logic but as a library function
    rather than subprocess so we can capture the result dict directly.
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would run grid search on {len(dates)} days")
        return None

    if len(dates) < MIN_DAYS_TO_OPTIMIZE:
        logger.info(
            f"Only {len(dates)} days available — "
            f"need {MIN_DAYS_TO_OPTIMIZE} minimum. Skipping grid search."
        )
        return None

    logger.info(
        f"Running grid search: model={model}  "
        f"dates={dates[0]} to {dates[-1]}  ({len(dates)} days)"
    )

    try:
        # Import grid search run function
        # Assumes grid_search_v1.py exposes run_grid_search(dates) → dict
        if model == 'v1':
            from research.grid_search_v1 import run_grid_search
        else:
            # v2 grid search — same interface
            from research.grid_search_v2 import run_grid_search

        new_params = run_grid_search(dates=dates)
        logger.info(f"Grid search complete: {len(new_params)} symbols")
        return new_params

    except ImportError as e:
        logger.error(
            f"Could not import grid search for {model}: {e}\n"
            f"Make sure grid_search_{model}.py exposes run_grid_search(dates)"
        )
        return None
    except Exception as e:
        logger.error(f"Grid search failed: {e}\n{traceback.format_exc()}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Transfer log — audit trail of all param transfers
# ─────────────────────────────────────────────────────────────────────────────

def save_transfer_log(log_entry: dict):
    """Append to transfer log in GCS."""
    log_path = f"{GCS_PARAMS_DIR}/transfer_log.json"
    existing = gcs_download_json(log_path) or []
    if not isinstance(existing, list):
        existing = []
    existing.append(log_entry)
    # Keep last 90 entries
    existing = existing[-90:]
    gcs_upload_json(existing, log_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main optimization loop
# ─────────────────────────────────────────────────────────────────────────────

def run_optimization(
    model:          str  = 'v1',
    force_transfer: bool = False,
    dry_run:        bool = False,
):
    """
    Main nightly optimization for one model.

    Steps:
        1. Load current params from GCS
        2. Check if today is a roll day
        3. If roll: run param transfer for all symbols
        4. Run rolling grid search on last N days
        5. Blend new params with old
        6. Save back to GCS
        7. Log what happened
    """
    today     = datetime.date.today()
    today_str = today.isoformat()

    logger.info(f"{'='*60}")
    logger.info(f"Rolling optimizer — {today_str}  model={model}")
    logger.info(f"{'='*60}")

    # ── Step 1: Load current params ───────────────────────────────────────────
    gcs_path      = get_gcs_params_path(model)
    current_params = gcs_download_json(gcs_path)

    if current_params is None:
        # Fall back to local params
        local_path = Path(f'research/findings/{model}_optimal_params.json')
        if local_path.exists():
            with open(local_path) as f:
                current_params = json.load(f)
            logger.info(f"Loaded {len(current_params)} params from local file")
        else:
            logger.warning("No params found in GCS or local — starting fresh")
            current_params = {}
    else:
        logger.info(
            f"Loaded {len(current_params)} params from "
            f"gs://{GCS_BUCKET}/{gcs_path}"
        )

    # ── Step 2: Check for contract roll ───────────────────────────────────────
    roll_happened  = is_roll_day(today) or force_transfer
    new_contract   = get_current_contract(today)
    old_contract   = get_old_contract(today)

    if roll_happened:
        logger.info(
            f"CONTRACT ROLL DETECTED: {old_contract} → {new_contract}"
        )

    # ── Step 3: Param transfer if roll ────────────────────────────────────────
    if roll_happened and current_params:
        logger.info("Running param transfer for all symbols...")

        try:
            from src.backtest.data_loader import load_day
            from src.backtest.param_transfer import transfer_all_params

            old_dates = get_trading_dates_before(today, n=3)
            new_dates = [today_str]

            if not dry_run:
                transferred = transfer_all_params(
                    old_params_dict = current_params,
                    new_contract    = new_contract,
                    old_contract    = old_contract,
                    data_loader     = load_day,
                    new_dates       = new_dates,
                    old_dates       = old_dates,
                    output_path     = None,   # we handle saving ourselves
                )

                # Merge transferred into current params
                current_params.update(transferred)
                logger.info(
                    f"Param transfer complete: {len(transferred)} symbols"
                )

                # Log the transfer
                save_transfer_log({
                    'date':          today_str,
                    'event':         'contract_roll',
                    'old_contract':  old_contract,
                    'new_contract':  new_contract,
                    'model':         model,
                    'n_transferred': len(transferred),
                })
            else:
                logger.info(
                    f"[DRY RUN] Would transfer {len(current_params)} symbols"
                )

        except Exception as e:
            logger.error(
                f"Param transfer failed: {e}\n{traceback.format_exc()}"
            )

    # ── Step 4: Rolling grid search ───────────────────────────────────────────
    # Get last N trading days for current contract
    all_recent = get_trading_dates_before(today, n=LOOKBACK_DAYS + 2)

    # Filter to current contract dates only
    contract_month_map = {
        'JANFUT': 1, 'FEBFUT': 2, 'MARFUT': 3, 'APRFUT': 4,
        'MAYFUT': 5, 'JUNFUT': 6, 'JULFUT': 7, 'AUGFUT': 8,
        'SEPFUT': 9, 'OCTFUT': 10, 'NOVFUT': 11, 'DECFUT': 12,
    }
    contract_month_num = contract_month_map.get(new_contract, 0)

    search_dates = [
        d for d in all_recent
        if datetime.date.fromisoformat(d).month == contract_month_num
    ][-LOOKBACK_DAYS:]

    logger.info(
        f"Grid search dates: {search_dates[0] if search_dates else 'none'} "
        f"to {search_dates[-1] if search_dates else 'none'} "
        f"({len(search_dates)} days)"
    )

    new_params = run_fast_grid_search(
        model   = model,
        dates   = search_dates,
        dry_run = dry_run,
    )

    # ── Step 5: Blend params ──────────────────────────────────────────────────
    if new_params and current_params:
        logger.info(
            f"Blending params: alpha={BLEND_ALPHA} "
            f"({int(BLEND_ALPHA*100)}% new, "
            f"{int((1-BLEND_ALPHA)*100)}% old)"
        )
        final_params = blend_all_params(current_params, new_params, BLEND_ALPHA)
    elif new_params:
        final_params = new_params
    else:
        # No new optimization — keep current params unchanged
        logger.info("No new params from grid search — keeping current params")
        final_params = current_params

    # ── Step 6: Save to GCS ───────────────────────────────────────────────────
    if not dry_run and final_params:
        gcs_upload_json(final_params, gcs_path)

        # Also save a dated snapshot for audit
        snapshot_path = (
            f"{GCS_PARAMS_DIR}/snapshots/"
            f"{model}_params_{today_str}.json"
        )
        gcs_upload_json(final_params, snapshot_path)

        logger.info(
            f"Saved {len(final_params)} params to "
            f"gs://{GCS_BUCKET}/{gcs_path}"
        )

        # Log the optimization run
        save_transfer_log({
            'date':         today_str,
            'event':        'grid_search',
            'model':        model,
            'n_symbols':    len(final_params),
            'n_dates':      len(search_dates),
            'roll_day':     roll_happened,
            'blend_alpha':  BLEND_ALPHA,
        })

    elif dry_run:
        logger.info(
            f"[DRY RUN] Would save {len(final_params)} params to GCS"
        )

    logger.info(f"Optimization complete for model={model}")
    return final_params


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Nightly rolling param optimizer'
    )
    parser.add_argument(
        '--model', default='all',
        help="Model to optimize: v1, v2, or all (default: all)"
    )
    parser.add_argument(
        '--force-transfer', action='store_true',
        help="Force param transfer even if not a roll day"
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Run without saving — for testing"
    )
    args = parser.parse_args()

    models = MODELS if args.model == 'all' else [args.model]

    for model in models:
        try:
            run_optimization(
                model          = model,
                force_transfer = args.force_transfer,
                dry_run        = args.dry_run,
            )
        except Exception as e:
            logger.error(
                f"Optimization failed for model={model}: "
                f"{e}\n{traceback.format_exc()}"
            )


if __name__ == '__main__':
    main()