"""
Data Gap Detector
=================
Checks GCS market data for gaps, missing instruments, and low tick counts.
Saves a JSON + human-readable report to GCS logs bucket.

Usage:
    python data_gap_detector.py                        # checks yesterday
    python data_gap_detector.py --date 2026-05-09      # specific date
    python data_gap_detector.py --date 2026-05-09 --verbose  # full detail

Cron (VM):
    # Daily at 6:15pm IST (12:45pm UTC) — 45 mins after market close
    45 12 * * 1-5 cd /home/ubuntu/quant-fund && venv/bin/python data_gap_detector.py >> logs/gap_detector.log 2>&1
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, time
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq
from google.cloud import storage

# ─── Config ───────────────────────────────────────────────────────────────────

GCS_BUCKET          = "hedge-fund-494103-marketdata-mumbai"
GCS_LOG_PREFIX      = "logs/gap_reports"
PROCESSED_PREFIX    = "processed/features"

# Expected instrument counts
EXPECTED_TOTAL      = 261
EXPECTED_EQUITY_FUT = 89    # 5 index + 84 stocks
EXPECTED_NIFTY_OPT  = 84
EXPECTED_BNF_OPT    = 84
EXPECTED_USDINR     = 2
EXPECTED_SPOTS      = 2

# Market hours in UTC (ts_ist column is misnamed — actually stored as UTC)
# IST 9:15  = UTC 3:45
# IST 15:30 = UTC 10:00
EQUITY_OPEN   = time(3, 45)
EQUITY_CLOSE  = time(10, 0)

# CDS hours in UTC
# IST 9:00  = UTC 3:30
# IST 17:00 = UTC 11:30
CDS_OPEN      = time(3, 30)
CDS_CLOSE     = time(11, 30)

# Gap thresholds
MIN_COVERAGE_PCT    = 0.50   # instrument has < 50% expected ticks = bad
CONSECUTIVE_GAP_MIN = 5      # 5+ consecutive missing minutes = gap alert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_yesterday_ist() -> str:
    """Return yesterday's date in IST as YYYY-MM-DD string."""
    # IST = UTC+5:30
    ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    yesterday = ist_now.date() - timedelta(days=1)
    return str(yesterday)


def is_usdinr(symbol: str) -> bool:
    return "USDINR" in symbol.upper()


def expected_minutes(symbol: str) -> int:
    """Expected number of 1-minute bars for a given symbol."""
    if is_usdinr(symbol):
        # CDS: 3:30 to 11:30 UTC = 480 mins
        return 480
    else:
        # Equity: 3:45 to 10:00 UTC = 375 mins
        return 375


def consecutive_gaps(missing_minutes: list) -> int:
    """Return the longest streak of consecutive missing minutes."""
    if not missing_minutes:
        return 0
    mins = sorted(missing_minutes)
    max_streak = streak = 1
    for i in range(1, len(mins)):
        if mins[i] - mins[i - 1] == 1:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 1
    return max_streak


# ─── GCS helpers ──────────────────────────────────────────────────────────────

def list_processed_symbols(client: storage.Client, date: str) -> list[str]:
    """List all symbols that have a processed parquet for this date."""
    prefix = f"{PROCESSED_PREFIX}/"
    blobs = client.list_blobs(GCS_BUCKET, prefix=prefix)
    symbols = []
    for blob in blobs:
        # path: processed/features/{SYMBOL}/{YYYY-MM-DD}.parquet
        parts = blob.name.split("/")
        if len(parts) == 4 and parts[3] == f"{date}.parquet":
            symbols.append(parts[2])
    return symbols


def load_processed(client: storage.Client, symbol: str, date: str) -> pd.DataFrame | None:
    """Load processed parquet from GCS. Returns None if missing."""
    path = f"{PROCESSED_PREFIX}/{symbol}/{date}.parquet"
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(path)
    if not blob.exists():
        return None
    data = blob.download_as_bytes()
    import io
    return pd.read_parquet(io.BytesIO(data))


def save_report_to_gcs(client: storage.Client, date: str, report: dict):
    """Save JSON report + human-readable summary to GCS."""
    bucket = client.bucket(GCS_BUCKET)

    # JSON report
    json_path = f"{GCS_LOG_PREFIX}/{date}/report.json"
    bucket.blob(json_path).upload_from_string(
        json.dumps(report, indent=2, default=str),
        content_type="application/json"
    )

    # Human-readable summary
    summary = build_summary(report)
    txt_path = f"{GCS_LOG_PREFIX}/{date}/summary.txt"
    bucket.blob(txt_path).upload_from_string(summary, content_type="text/plain")

    log.info(f"Report saved → gs://{GCS_BUCKET}/{json_path}")
    log.info(f"Summary saved → gs://{GCS_BUCKET}/{txt_path}")


# ─── Core analysis ────────────────────────────────────────────────────────────

def analyze_symbol(df: pd.DataFrame, symbol: str) -> dict:
    """
    Analyze a single symbol's processed data for gaps.
    Returns a dict with gap metrics.
    """
    result = {
        "symbol":            symbol,
        "rows":              len(df),
        "expected_minutes":  expected_minutes(symbol),
        "coverage_pct":      0.0,
        "missing_minutes":   [],
        "max_consecutive_gap": 0,
        "any_missing_minute":  False,
        "large_gap":           False,       # 5+ consecutive
        "low_coverage":        False,       # < 50% expected
        "status":            "OK",
    }

    if df is None or len(df) == 0:
        result["status"] = "NO_DATA"
        result["low_coverage"] = True
        return result

    # Normalise timestamp column
    ts_col = None
    for c in ["ts_ist", "ts_sec", "timestamp", "datetime"]:
        if c in df.columns:
            ts_col = c
            break

    if ts_col is None:
        result["status"] = "NO_TIMESTAMP_COL"
        return result

    df = df.copy()
    df["_dt"] = pd.to_datetime(df[ts_col], utc=False, errors="coerce")
    df = df.dropna(subset=["_dt"])

    # Truncate to market hours
    if is_usdinr(symbol):
        open_t, close_t = CDS_OPEN, CDS_CLOSE
    else:
        open_t, close_t = EQUITY_OPEN, EQUITY_CLOSE

    df = df[
        (df["_dt"].dt.time >= open_t) &
        (df["_dt"].dt.time <= close_t)
    ]

    if len(df) == 0:
        result["status"] = "NO_DATA_IN_MARKET_HOURS"
        result["low_coverage"] = True
        return result

    # Minute-level presence
    df["_min"] = df["_dt"].dt.hour * 60 + df["_dt"].dt.minute
    present_mins = set(df["_min"].unique())

    open_min  = open_t.hour * 60 + open_t.minute
    close_min = close_t.hour * 60 + close_t.minute
    all_mins  = set(range(open_min, close_min + 1))

    missing = sorted(all_mins - present_mins)
    max_streak = consecutive_gaps(missing)

    coverage = len(present_mins) / len(all_mins)

    result["rows"]                 = len(df)
    result["coverage_pct"]         = round(coverage * 100, 1)
    result["missing_minutes"]      = missing
    result["max_consecutive_gap"]  = max_streak
    result["any_missing_minute"]   = len(missing) > 0
    result["large_gap"]            = max_streak >= CONSECUTIVE_GAP_MIN
    result["low_coverage"]         = coverage < MIN_COVERAGE_PCT

    # Status
    issues = []
    if result["low_coverage"]:
        issues.append("LOW_COVERAGE")
    if result["large_gap"]:
        issues.append("LARGE_GAP")
    elif result["any_missing_minute"]:
        issues.append("MINOR_GAPS")

    result["status"] = ", ".join(issues) if issues else "OK"
    return result


def run_detector(date: str, verbose: bool = False) -> dict:
    """Main detection run. Returns full report dict."""
    log.info(f"Running gap detector for date: {date}")

    client = storage.Client(project="hedge-fund-494103")
    symbols = list_processed_symbols(client, date)
    log.info(f"Found {len(symbols)} processed symbols for {date}")

    # ── Per-symbol analysis ──
    results = []
    usdinr_symbols   = []
    equity_fut       = []
    options_symbols  = []

    for sym in symbols:
        if "USDINR" in sym.upper():
            usdinr_symbols.append(sym)
        elif any(x in sym.upper() for x in ["CE", "PE"]):
            options_symbols.append(sym)
        else:
            equity_fut.append(sym)

    all_results = []
    for sym in symbols:
        df = load_processed(client, sym, date)
        r  = analyze_symbol(df, sym)
        all_results.append(r)
        if verbose:
            flag = "✓" if r["status"] == "OK" else "✗"
            log.info(f"  {flag} {sym:40s} cov={r['coverage_pct']:5.1f}%  gap={r['max_consecutive_gap']}min  status={r['status']}")

    # ── Aggregate stats ──
    ok_count          = sum(1 for r in all_results if r["status"] == "OK")
    low_coverage_list = [r["symbol"] for r in all_results if r["low_coverage"]]
    large_gap_list    = [r["symbol"] for r in all_results if r["large_gap"]]
    no_data_list      = [r["symbol"] for r in all_results if "NO_DATA" in r["status"]]
    minor_gap_list    = [r["symbol"] for r in all_results if "MINOR_GAPS" in r["status"]]

    # Instrument count checks
    count_checks = {
        "total_found":          len(symbols),
        "total_expected":       EXPECTED_TOTAL,
        "total_ok":             ok_count,
        "equity_fut_found":     len(equity_fut),
        "equity_fut_expected":  EXPECTED_EQUITY_FUT,
        "options_found":        len(options_symbols),
        "options_expected":     EXPECTED_NIFTY_OPT + EXPECTED_BNF_OPT,
        "usdinr_found":         len(usdinr_symbols),
        "usdinr_expected":      EXPECTED_USDINR,
        "instrument_count_ok":  len(symbols) >= EXPECTED_TOTAL * 0.95,
    }

    overall_health = "HEALTHY"
    if len(symbols) < EXPECTED_TOTAL * 0.80:
        overall_health = "CRITICAL"
    elif len(symbols) < EXPECTED_TOTAL * 0.95 or large_gap_list or low_coverage_list:
        overall_health = "WARNING"

    report = {
        "date":           date,
        "generated_at":   datetime.utcnow().isoformat() + "Z",
        "overall_health": overall_health,
        "count_checks":   count_checks,
        "summary": {
            "ok":           ok_count,
            "no_data":      len(no_data_list),
            "low_coverage": len(low_coverage_list),
            "large_gaps":   len(large_gap_list),
            "minor_gaps":   len(minor_gap_list),
        },
        "problem_symbols": {
            "no_data":      no_data_list,
            "low_coverage": low_coverage_list,
            "large_gaps":   large_gap_list,
            "minor_gaps":   minor_gap_list,
        },
        "per_symbol": all_results,
    }

    return report


# ─── Human-readable summary ───────────────────────────────────────────────────

def build_summary(report: dict) -> str:
    health_icon = {"HEALTHY": "✅", "WARNING": "⚠️", "CRITICAL": "🔴"}.get(
        report["overall_health"], "❓"
    )
    cc = report["count_checks"]
    s  = report["summary"]
    ps = report["problem_symbols"]

    lines = [
        "=" * 60,
        f"  QUANT FUND — DATA GAP REPORT",
        f"  Date: {report['date']}",
        f"  Generated: {report['generated_at']}",
        "=" * 60,
        "",
        f"  Overall Health: {health_icon} {report['overall_health']}",
        "",
        "── Instrument Count ─────────────────────────────────────",
        f"  Total:       {cc['total_found']:>4} / {cc['total_expected']} expected",
        f"  Equity Fut:  {cc['equity_fut_found']:>4} / {cc['equity_fut_expected']} expected",
        f"  Options:     {cc['options_found']:>4} / {cc['options_expected']} expected",
        f"  USDINR:      {cc['usdinr_found']:>4} / {cc['usdinr_expected']} expected",
        "",
        "── Data Quality ─────────────────────────────────────────",
        f"  ✓ Clean:         {s['ok']}",
        f"  ✗ No data:       {s['no_data']}",
        f"  ✗ Low coverage:  {s['low_coverage']}   (< 50% of expected ticks)",
        f"  ✗ Large gaps:    {s['large_gaps']}   (5+ consecutive missing mins)",
        f"  ~ Minor gaps:    {s['minor_gaps']}   (any missing minute)",
        "",
    ]

    if ps["no_data"]:
        lines.append("── No Data ──────────────────────────────────────────────")
        for sym in ps["no_data"][:20]:
            lines.append(f"  {sym}")
        if len(ps["no_data"]) > 20:
            lines.append(f"  ... and {len(ps['no_data']) - 20} more")
        lines.append("")

    if ps["low_coverage"]:
        lines.append("── Low Coverage (<50%) ──────────────────────────────────")
        for sym in ps["low_coverage"][:20]:
            lines.append(f"  {sym}")
        if len(ps["low_coverage"]) > 20:
            lines.append(f"  ... and {len(ps['low_coverage']) - 20} more")
        lines.append("")

    if ps["large_gaps"]:
        lines.append("── Large Gaps (5+ consecutive mins) ─────────────────────")
        for sym in ps["large_gaps"][:20]:
            lines.append(f"  {sym}")
        lines.append("")

    lines += [
        "=" * 60,
        f"  GCS log: gs://hedge-fund-494103-marketdata/logs/gap_reports/{report['date']}/",
        "=" * 60,
    ]

    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quant Fund — Data Gap Detector")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to check (YYYY-MM-DD). Defaults to yesterday IST."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-symbol status while running"
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip GCS upload (useful for local testing)"
    )
    args = parser.parse_args()

    date = args.date or get_yesterday_ist()

    report = run_detector(date, verbose=args.verbose)

    # Always print summary to terminal
    summary = build_summary(report)
    print(summary)

    # Save to GCS unless --no-upload
    if not args.no_upload:
        client = storage.Client()
        save_report_to_gcs(client, date, report)

    # Exit code reflects health (useful for cron alerting)
    if report["overall_health"] == "CRITICAL":
        sys.exit(2)
    elif report["overall_health"] == "WARNING":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
