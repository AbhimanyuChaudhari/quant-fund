"""
Data Loader — Backtest
======================
Loads processed features for a symbol and date.

Supports two GCS folder structures:

  OLD (flat):
    processed/features/CHOLAFIN26MAYFUT/2026-05-13.parquet

  NEW (hierarchical — contract-agnostic):
    processed/features/CHOLAFIN/2026/05/2026-05-13.parquet

When loading by symbol+date, tries new structure first,
then falls back to old structure (any contract suffix).
This means:
  - Old MAYFUT data loads transparently
  - New JUNFUT data loads transparently
  - No code changes needed at contract roll

Contract suffix stripping:
  CHOLAFIN26JUNFUT → CHOLAFIN (for new structure lookup)
  But also checks old structure with exact symbol name.
"""

import os
import re
import duckdb
import gcsfs
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Iterator
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

PROJECT_ID  = os.getenv("GCP_PROJECT_ID",  "hedge-fund-494103")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "hedge-fund-494103-marketdata")

MARKET_OPEN_IST  = 33300   # 09:15 IST
MARKET_CLOSE_IST = 55800   # 15:30 IST

# Regex to strip contract suffix: 26MAYFUT, 26JUNFUT, 27JANFUT etc.
_CONTRACT_RE = re.compile(
    r'\d{2}(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)FUT$',
    re.IGNORECASE
)


def strip_contract_suffix(symbol: str) -> str:
    """
    CHOLAFIN26JUNFUT → CHOLAFIN
    NIFTY26JUNFUT    → NIFTY
    USDINR26529FUT   → USDINR26529FUT  (currency, no change)
    """
    return _CONTRACT_RE.sub('', symbol)


def _get_con() -> duckdb.DuckDBPyConnection:
    """DuckDB connection with GCS access."""
    fs  = gcsfs.GCSFileSystem(project=PROJECT_ID)
    con = duckdb.connect()
    con.register_filesystem(fs)
    return con


def _market_filter(symbol: str = "") -> str:
    is_currency = any(x in symbol.upper()
                      for x in ["USDINR", "EURINR", "GBPINR", "JPYINR"])
    if is_currency:
        open_ist  = 9 * 3600
        close_ist = 17 * 3600
    else:
        open_ist  = MARKET_OPEN_IST
        close_ist = MARKET_CLOSE_IST

    return (
        f"((ts_sec + 19800) % 86400) >= {open_ist} "
        f"AND ((ts_sec + 19800) % 86400) <= {close_ist}"
    )


def _gcs_paths_for(symbol: str, date: str) -> list[str]:
    """
    Return candidate GCS paths for a symbol+date, in priority order.

    Priority:
      1. New structure: processed/features/CHOLAFIN/2026/05/2026-05-13.parquet
      2. Old structure: processed/features/CHOLAFIN26JUNFUT/2026-05-13.parquet
      3. Old structure: processed/features/CHOLAFIN26MAYFUT/2026-05-13.parquet
         (scanned dynamically — handles any contract suffix)
    """
    base  = strip_contract_suffix(symbol)
    year  = date[:4]
    month = date[5:7]

    paths = []

    # 1. New hierarchical structure (base name, no suffix)
    paths.append(
        f"gs://{BUCKET_NAME}/processed/features/"
        f"{base}/{year}/{month}/{date}.parquet"
    )

    # 2. Old flat structure — exact symbol name (e.g. CHOLAFIN26JUNFUT)
    if symbol != base:   # only if symbol has a suffix
        paths.append(
            f"gs://{BUCKET_NAME}/processed/features/"
            f"{symbol}/{date}.parquet"
        )

    return paths


def _find_gcs_path(symbol: str, date: str,
                   fs: gcsfs.GCSFileSystem) -> str | None:
    """
    Find the actual GCS path for a symbol+date.
    Tries candidate paths in order, then scans for any contract suffix.
    Returns None if not found.
    """
    # Try explicit candidate paths first
    for path in _gcs_paths_for(symbol, date):
        gcs_key = path.replace("gs://", "")
        if fs.exists(gcs_key):
            return path

    # Fallback: scan old structure for any contract suffix
    # e.g. CHOLAFIN26MAYFUT, CHOLAFIN26JUNFUT etc.
    base = strip_contract_suffix(symbol)
    pattern = f"{BUCKET_NAME}/processed/features/{base}*FUT/{date}.parquet"
    matches = fs.glob(pattern)
    if matches:
        # Return the most recent contract (alphabetically last = latest)
        return f"gs://{sorted(matches)[-1]}"

    return None


def load_day(symbol: str, date: str,
             market_hours_only: bool = True) -> pd.DataFrame:
    """
    Load one day of processed features for a symbol.

    Lookup order:
      1. DuckDB local cache (fastest — microseconds)
      2. New GCS structure: processed/features/{BASE}/{YEAR}/{MM}/{date}.parquet
      3. Old GCS structure: processed/features/{SYMBOL}/{date}.parquet
      4. Old GCS structure: any contract suffix for base name

    Cache miss auto-populates for next call.
    """

    # ── 1. DuckDB cache ────────────────────────────────────────────────────────
    try:
        from src.backtest.duckdb_cache import LocalCache
        _cache = LocalCache()
        _df    = _cache.load(symbol, date, market_hours_only)
        if not _df.empty:
            print(f"[data_loader] Cache: {len(_df):,} bars | {symbol} | {date}")
            return _df
    except Exception:
        pass

    # ── 2-4. GCS lookup ────────────────────────────────────────────────────────
    fs       = gcsfs.GCSFileSystem(project=PROJECT_ID)
    gcs_path = _find_gcs_path(symbol, date, fs)

    if gcs_path is None:
        print(f"[data_loader] Loaded 0 bars | {symbol} | {date}")
        return pd.DataFrame()

    where = f"WHERE {_market_filter(symbol)}" if market_hours_only else ""
    con   = _get_con()

    try:
        df = con.execute(f"""
            SELECT *
            FROM read_parquet('{gcs_path}')
            {where}
            ORDER BY ts_sec
        """).df()
    except Exception as e:
        print(f"[data_loader] Read error {symbol} | {date}: {e}")
        return pd.DataFrame()

    print(f"[data_loader] Loaded {len(df):,} bars | {symbol} | {date}")

    # ── Populate cache ─────────────────────────────────────────────────────────
    if not df.empty:
        try:
            from src.backtest.duckdb_cache import LocalCache
            LocalCache().insert(symbol, date, df)
        except Exception:
            pass

    return df


def load_date_range(symbol: str, start: str, end: str,
                    market_hours_only: bool = True) -> pd.DataFrame:
    """
    Load multiple days of processed features for a symbol.
    Handles contract roll seamlessly — scans both old and new structures.
    Each day goes through load_day() so cache is used automatically.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    end_dt   = datetime.strptime(end,   "%Y-%m-%d").date()

    base = strip_contract_suffix(symbol)
    fs   = gcsfs.GCSFileSystem(project=PROJECT_ID)

    # Collect all candidate parquet files for this symbol
    candidate_files = set()

    # New structure: processed/features/CHOLAFIN/2026/*/
    new_files = fs.glob(
        f"{BUCKET_NAME}/processed/features/{base}/*/*/*.parquet"
    )
    candidate_files.update(new_files)

    # Old structure: processed/features/CHOLAFIN26*FUT/
    old_files = fs.glob(
        f"{BUCKET_NAME}/processed/features/{base}*FUT/*.parquet"
    )
    candidate_files.update(old_files)

    frames = []
    seen_dates = set()

    for f in sorted(candidate_files):
        fname = Path(f).stem
        try:
            fdate = datetime.strptime(fname, "%Y-%m-%d").date()
        except ValueError:
            continue

        if not (start_dt <= fdate <= end_dt):
            continue

        date_str = fname
        if date_str in seen_dates:
            continue   # already have data for this date (new structure wins)
        seen_dates.add(date_str)

        df = load_day(symbol, date_str, market_hours_only)
        if not df.empty:
            frames.append(df)

    if not frames:
        print(f"[data_loader] No data: {symbol} | {start} → {end}")
        return pd.DataFrame()

    combined = pd.concat(
        frames, ignore_index=True
    ).sort_values("ts_sec").reset_index(drop=True)

    print(
        f"[data_loader] Total: {len(combined):,} bars | "
        f"{symbol} | {start} → {end}"
    )
    return combined


def get_active_symbols(start: str, end: str,
                       exclude_index: bool = True) -> list[str]:
    """
    Return symbols that have processed data in the given date range.
    Returns base names (e.g. CHOLAFIN) — contract-agnostic.

    Use this instead of hardcoded symbol lists in backtest scripts.
    """
    INDEX_BASES = {
        "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
        "SENSEX", "NIFTYNXT50", "BANKEX",
    }

    fs = gcsfs.GCSFileSystem(project=PROJECT_ID)

    # Scan both structures
    all_files = fs.glob(
        f"{BUCKET_NAME}/processed/features/*"
    )

    active = set()
    for f in all_files:
        parts = f.split("/")
        if len(parts) < 4:
            continue

        entry = parts[3]   # CHOLAFIN or CHOLAFIN26JUNFUT

        # Strip contract suffix to get base name
        base = strip_contract_suffix(entry)

        if exclude_index and base in INDEX_BASES:
            continue

        # Check if any file exists in date range
        # (approximate — checks if the directory exists, not specific dates)
        active.add(base)

    return sorted(active)


def iter_bars(symbol: str, start: str, end: str,
              market_hours_only: bool = True) -> Iterator[pd.Series]:
    """Iterate over bars one at a time for the backtest engine."""
    df = load_date_range(symbol, start, end, market_hours_only)
    if df.empty:
        return
    for _, row in df.iterrows():
        yield row


if __name__ == "__main__":
    # Quick sanity check
    print("=== Test: load_day (JUNFUT) ===")
    df = load_day("CHOLAFIN26JUNFUT", "2026-05-27")
    if not df.empty:
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        print(df[["ts_sec", "open", "close",
                   "imbalance_last", "realized_vol_60s"]].head(3).to_string())

    print("\n=== Test: strip_contract_suffix ===")
    for sym in ["CHOLAFIN26JUNFUT", "CHOLAFIN26MAYFUT",
                "NIFTY26JUNFUT", "USDINR26529FUT"]:
        print(f"  {sym} → {strip_contract_suffix(sym)}")