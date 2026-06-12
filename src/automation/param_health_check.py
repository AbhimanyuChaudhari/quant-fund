"""
Param Health Check
==================
Runs at 8:45 AM IST before market open.

What it does:
    1. Loads current params from GCS
    2. Validates every symbol has sane params
    3. Checks param freshness (not stale)
    4. Archives params if contract roll happened
    5. Cleans up old daily snapshots (>30 days)
    6. Logs a clear health report
    7. Exits with code 1 if critical issues found
       (cron can alert on non-zero exit)

Schedule:
    8:45 AM IST = 03:15 UTC (set by scheduler_setup.sh)

Exit codes:
    0 = all good, safe to trade
    1 = critical issues found, needs attention
    2 = warnings only, can trade but monitor

Usage:
    python -m src.automation.param_health_check
    python -m src.automation.param_health_check --model v1
    python -m src.automation.param_health_check --archive-only
"""

import json
import logging
import argparse
import datetime
import traceback
from pathlib import Path
from typing import Optional

LOG_DIR = Path('/home/ubuntu/quant-fund/logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level    = logging.INFO,
    format   = '%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt  = '%Y-%m-%d %H:%M:%S',
    handlers = [
        logging.FileHandler(LOG_DIR / 'param_health_check.log'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

GCS_BUCKET         = 'hedge-fund-494103-marketdata-mumbai'
GCS_PARAMS_DIR     = 'params'
GCS_SNAPSHOTS_DIR  = 'params/snapshots'
GCS_ARCHIVE_DIR    = 'params/archive'

LOCAL_ARCHIVE_DIR  = Path('research/findings/params_archive')

MODELS = ['v1', 'v2']

# Param sanity bounds — values outside these are suspicious
PARAM_BOUNDS = {
    'gamma':      (1e-6,  0.1),
    'kappa':      (0.1,   10.0),
    'min_spread': (0.01,  5.0),
    'open_mult':  (0.5,   5.0),
    'phi':        (1e-6,  0.1),
    'rho':        (0.05,  0.95),
    'zeta':       (0.05,  5.0),
    'beta':       (0.1,   10.0),
    'theta':      (0.1,   10.0),
    'eta':        (0.01,  2.0),
    'nu':         (0.01,  2.0),
    'theta_kappa':(0.1,   5.0),
}

# Max age of params before flagging as stale (days)
MAX_PARAM_AGE_DAYS = 3

# Min symbols expected to have params
MIN_SYMBOLS_EXPECTED = 40

# Daily snapshots older than this get deleted from GCS
SNAPSHOT_RETENTION_DAYS = 30

# NSE expiry dates
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
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob   = bucket.blob(gcs_path)
    blob.upload_from_string(
        json.dumps(data, indent=2),
        content_type='application/json'
    )


def gcs_download_json(gcs_path: str) -> Optional[dict]:
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob   = bucket.blob(gcs_path)
        if not blob.exists():
            return None
        return json.loads(blob.download_as_text())
    except Exception as e:
        logger.warning(f"GCS download failed for {gcs_path}: {e}")
        return None


def gcs_get_blob_age(gcs_path: str) -> Optional[float]:
    """Return age of a GCS blob in days. None if not found."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob   = bucket.blob(gcs_path)
        blob.reload()
        if blob.updated is None:
            return None
        age = datetime.datetime.now(datetime.timezone.utc) - blob.updated
        return age.total_seconds() / 86400.0
    except Exception:
        return None


def gcs_list_blobs(prefix: str) -> list:
    """List all blob names under a GCS prefix."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        return [b.name for b in bucket.list_blobs(prefix=prefix)]
    except Exception:
        return []


def gcs_delete_blob(gcs_path: str):
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob   = bucket.blob(gcs_path)
        blob.delete()
    except Exception as e:
        logger.warning(f"Could not delete {gcs_path}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Param validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_symbol_params(symbol: str, params: dict) -> list:
    """
    Validate params for one symbol.
    Returns list of issue strings (empty = all good).
    """
    issues = []

    # Check required keys exist
    required = ['min_spread']
    for key in required:
        if key not in params:
            issues.append(f"missing required key: {key}")

    # Check bounds
    for key, (lo, hi) in PARAM_BOUNDS.items():
        if key not in params:
            continue
        val = params[key]
        if not isinstance(val, (int, float)):
            continue
        if val < lo or val > hi:
            issues.append(
                f"{key}={val:.4f} outside bounds [{lo}, {hi}]"
            )

    # Check for NaN/None
    for key, val in params.items():
        if isinstance(val, float):
            import math
            if math.isnan(val) or math.isinf(val):
                issues.append(f"{key} is NaN/Inf")

    return issues


def check_params_coverage(
    params: dict,
    model:  str,
) -> tuple:
    """
    Check overall param coverage.
    Returns (n_symbols, n_issues, issue_list).
    """
    n_symbols  = len(params)
    all_issues = []

    for symbol, p in params.items():
        issues = validate_symbol_params(symbol, p)
        for issue in issues:
            all_issues.append(f"{symbol}: {issue}")

    return n_symbols, len(all_issues), all_issues


# ─────────────────────────────────────────────────────────────────────────────
# Contract roll detection
# ─────────────────────────────────────────────────────────────────────────────

def is_roll_day(today: datetime.date) -> bool:
    """True if today is first day of new contract."""
    yesterday = today - datetime.timedelta(days=1)
    while yesterday.weekday() >= 5:
        yesterday -= datetime.timedelta(days=1)

    def get_contract(d: datetime.date) -> str:
        expiry = NSE_EXPIRY_DATES.get((d.year, d.month))
        month_names = {
            1:'JAN',2:'FEB',3:'MAR',4:'APR',5:'MAY',6:'JUN',
            7:'JUL',8:'AUG',9:'SEP',10:'OCT',11:'NOV',12:'DEC'
        }
        if expiry and d > expiry:
            nm = d.month + 1
            ny = d.year
            if nm > 12:
                nm = 1
                ny += 1
            return f"{month_names[nm]}FUT"
        return f"{month_names[d.month]}FUT"

    return get_contract(today) != get_contract(yesterday)


# ─────────────────────────────────────────────────────────────────────────────
# Archiving
# ─────────────────────────────────────────────────────────────────────────────

def archive_params(
    params:    dict,
    model:     str,
    today:     datetime.date,
    reason:    str = 'daily',
):
    """
    Save a dated snapshot of current params.

    reason='daily'    → goes to snapshots/ (auto-deleted after 30 days)
    reason='roll'     → goes to archive/   (kept forever)
    reason='manual'   → goes to archive/   (kept forever)
    """
    today_str  = today.isoformat()
    filename   = f"{model}_params_{today_str}.json"

    if reason == 'daily':
        gcs_path   = f"{GCS_SNAPSHOTS_DIR}/{filename}"
        local_path = LOCAL_ARCHIVE_DIR / 'snapshots' / filename
    else:
        gcs_path   = f"{GCS_ARCHIVE_DIR}/{filename}"
        local_path = LOCAL_ARCHIVE_DIR / filename

    # Add metadata to snapshot
    snapshot = {
        '_meta': {
            'date':   today_str,
            'model':  model,
            'reason': reason,
            'n_symbols': len(params),
        },
        **params,
    }

    # Save to GCS
    try:
        gcs_upload_json(snapshot, gcs_path)
        logger.info(
            f"Archived {model} params → "
            f"gs://{GCS_BUCKET}/{gcs_path} ({reason})"
        )
    except Exception as e:
        logger.warning(f"GCS archive failed: {e}")

    # Save locally too
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, 'w') as f:
            json.dump(snapshot, f, indent=2)
        logger.info(f"Archived locally → {local_path}")
    except Exception as e:
        logger.warning(f"Local archive failed: {e}")


def cleanup_old_snapshots(model: str, today: datetime.date):
    """
    Delete daily snapshots older than SNAPSHOT_RETENTION_DAYS from GCS.
    Contract roll archives are never deleted.
    """
    cutoff = today - datetime.timedelta(days=SNAPSHOT_RETENTION_DAYS)
    prefix = f"{GCS_SNAPSHOTS_DIR}/{model}_params_"
    blobs  = gcs_list_blobs(prefix)

    deleted = 0
    for blob_name in blobs:
        # Extract date from filename: v1_params_2026-05-01.json
        try:
            date_str = blob_name.split('_params_')[1].replace('.json', '')
            blob_date = datetime.date.fromisoformat(date_str)
            if blob_date < cutoff:
                gcs_delete_blob(blob_name)
                deleted += 1
        except Exception:
            continue

    if deleted > 0:
        logger.info(
            f"Cleaned up {deleted} old {model} snapshots "
            f"(older than {cutoff})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main health check
# ─────────────────────────────────────────────────────────────────────────────

def run_health_check(
    model:        str,
    today:        datetime.date,
    archive_only: bool = False,
) -> tuple:
    """
    Run full health check for one model.

    Returns:
        (status, issues) where status is 'OK', 'WARNING', or 'CRITICAL'
    """
    today_str = today.isoformat()
    issues    = []
    warnings  = []

    logger.info(f"{'─'*50}")
    logger.info(f"Health check: model={model}  date={today_str}")

    # ── Load params ────────────────────────────────────────────────────────────
    gcs_path = f"{GCS_PARAMS_DIR}/{model}_optimal_params.json"
    params   = gcs_download_json(gcs_path)

    if params is None:
        # Try local fallback
        local_path = Path(f'research/findings/{model}_optimal_params.json')
        if local_path.exists():
            with open(local_path) as f:
                params = json.load(f)
            warnings.append(f"Loaded from local file — GCS unavailable")
            logger.warning(f"Using local params for {model}")
        else:
            issues.append(f"No params found in GCS or local for {model}")
            return 'CRITICAL', issues

    # Remove metadata key if present
    params = {k: v for k, v in params.items() if not k.startswith('_')}

    if archive_only:
        archive_params(params, model, today, reason='daily')
        cleanup_old_snapshots(model, today)
        return 'OK', []

    # ── Check coverage ─────────────────────────────────────────────────────────
    n_symbols, n_invalid, invalid_list = check_params_coverage(params, model)

    logger.info(f"  Symbols with params: {n_symbols}")

    if n_symbols < MIN_SYMBOLS_EXPECTED:
        issues.append(
            f"Only {n_symbols} symbols have params "
            f"(expected ≥{MIN_SYMBOLS_EXPECTED})"
        )

    if n_invalid > 0:
        logger.warning(f"  Invalid params: {n_invalid} symbols")
        for issue in invalid_list[:10]:   # show first 10
            logger.warning(f"    {issue}")
        if n_invalid > 10:
            logger.warning(f"    ... and {n_invalid - 10} more")
        warnings.append(f"{n_invalid} symbols have invalid param values")

    # ── Check freshness ────────────────────────────────────────────────────────
    age_days = gcs_get_blob_age(gcs_path)
    if age_days is not None:
        logger.info(f"  Param age: {age_days:.1f} days")
        if age_days > MAX_PARAM_AGE_DAYS:
            issues.append(
                f"Params are {age_days:.1f} days old "
                f"(max {MAX_PARAM_AGE_DAYS}) — optimizer may have failed"
            )
    else:
        warnings.append("Could not determine param age")

    # ── Check contract roll ────────────────────────────────────────────────────
    if is_roll_day(today):
        logger.info(f"  CONTRACT ROLL DAY detected")

        # Archive old params before roll
        archive_params(params, model, today, reason='roll')
        logger.info(f"  Archived pre-roll params")

        # Check if new contract params exist
        import re
        contract_re = re.compile(
            r'\d{2}(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)FUT$',
            re.IGNORECASE
        )
        month_names = {
            1:'JAN',2:'FEB',3:'MAR',4:'APR',5:'MAY',6:'JUN',
            7:'JUL',8:'AUG',9:'SEP',10:'OCT',11:'NOV',12:'DEC'
        }

        expiry = NSE_EXPIRY_DATES.get((today.year, today.month))
        if expiry and today > expiry:
            nm = today.month + 1
            ny = today.year
            if nm > 12:
                nm = 1
                ny += 1
            new_contract = f"{str(ny)[-2:]}{month_names[nm]}FUT"
        else:
            new_contract = f"{str(today.year)[-2:]}{month_names[today.month]}FUT"

        new_contract_symbols = [
            k for k in params
            if new_contract.upper() in k.upper()
        ]

        if len(new_contract_symbols) < MIN_SYMBOLS_EXPECTED // 2:
            issues.append(
                f"Roll day but only {len(new_contract_symbols)} "
                f"{new_contract} symbols found — "
                f"run param transfer manually"
            )
            logger.error(
                f"  CRITICAL: Missing {new_contract} params. Run:\n"
                f"  python -m src.backtest.param_loader "
                f"run_contract_roll_transfer"
            )
        else:
            logger.info(
                f"  {new_contract} params found: "
                f"{len(new_contract_symbols)} symbols ✓"
            )
    else:
        # Regular day — just archive daily snapshot
        archive_params(params, model, today, reason='daily')

    # ── Cleanup old snapshots ──────────────────────────────────────────────────
    cleanup_old_snapshots(model, today)

    # ── Summary ────────────────────────────────────────────────────────────────
    if issues:
        status = 'CRITICAL'
        logger.error(f"  Status: CRITICAL — {len(issues)} issue(s)")
        for issue in issues:
            logger.error(f"    ✗ {issue}")
    elif warnings:
        status = 'WARNING'
        logger.warning(f"  Status: WARNING — {len(warnings)} warning(s)")
        for w in warnings:
            logger.warning(f"    ~ {w}")
    else:
        status = 'OK'
        logger.info(f"  Status: OK ✓")

    return status, issues + warnings


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Pre-market param health check'
    )
    parser.add_argument(
        '--model', default='all',
        help='Model to check: v1, v2, or all (default: all)'
    )
    parser.add_argument(
        '--archive-only', action='store_true',
        help='Only archive params, skip validation'
    )
    args = parser.parse_args()

    today  = datetime.date.today()
    models = MODELS if args.model == 'all' else [args.model]

    logger.info(f"{'='*50}")
    logger.info(f"Param Health Check — {today.isoformat()}")
    logger.info(f"{'='*50}")

    all_statuses = []

    for model in models:
        try:
            status, issues = run_health_check(
                model        = model,
                today        = today,
                archive_only = args.archive_only,
            )
            all_statuses.append(status)
        except Exception as e:
            logger.error(
                f"Health check failed for {model}: "
                f"{e}\n{traceback.format_exc()}"
            )
            all_statuses.append('CRITICAL')

    logger.info(f"{'='*50}")

    # Exit code for cron alerting
    if 'CRITICAL' in all_statuses:
        logger.error("OVERALL: CRITICAL — manual intervention needed")
        exit(1)
    elif 'WARNING' in all_statuses:
        logger.warning("OVERALL: WARNING — monitor closely")
        exit(2)
    else:
        logger.info("OVERALL: OK — safe to trade")
        exit(0)


if __name__ == '__main__':
    main()
