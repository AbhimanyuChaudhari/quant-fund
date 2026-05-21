import os
import duckdb
import gcsfs
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Iterator
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

PROJECT_ID  = os.getenv("GCP_PROJECT_ID", "hedge-fund-494103")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "hedge-fund-494103-marketdata")

MARKET_OPEN_IST  = 33300   # 09:15 IST
MARKET_CLOSE_IST = 55800   # 15:30 IST


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

    return f"""
        ((ts_sec + 19800) % 86400) >= {open_ist}
        AND ((ts_sec + 19800) % 86400) <= {close_ist}
    """


def load_day(symbol: str, date: str,
             market_hours_only: bool = True) -> pd.DataFrame:
    """
    Load one day of processed features for a symbol.
    Checks local DuckDB cache first — falls back to GCS if not cached.
    Cache miss automatically populates the cache for next time.
    """
    # ── DuckDB cache check (5 lines) ──────────────────────────────────────────
    try:
        from src.backtest.duckdb_cache import LocalCache
        _cache = LocalCache()
        _df    = _cache.load(symbol, date, market_hours_only)
        if not _df.empty:
            print(f"[data_loader] Cache: {len(_df):,} bars | {symbol} | {date}")
            return _df
    except Exception:
        pass
    # ─────────────────────────────────────────────────────────────────────────

    con      = _get_con()
    gcs_path = f"gs://{BUCKET_NAME}/processed/features/{symbol}/{date}.parquet"

    fs = gcsfs.GCSFileSystem(project=PROJECT_ID)
    if not fs.exists(gcs_path.replace("gs://", "")):
        print(f"[data_loader] No file: {symbol} | {date}")
        return pd.DataFrame()

    where = f"WHERE {_market_filter(symbol)}" if market_hours_only else ""

    df = con.execute(f"""
        SELECT *
        FROM read_parquet('{gcs_path}')
        {where}
        ORDER BY ts_sec
    """).df()

    print(f"[data_loader] Loaded {len(df):,} bars | {symbol} | {date}")

    # ── Populate cache on GCS hit so next call is instant ─────────────────────
    try:
        from src.backtest.duckdb_cache import LocalCache
        LocalCache().insert(symbol, date, df)
    except Exception:
        pass
    # ─────────────────────────────────────────────────────────────────────────

    return df


def load_date_range(symbol: str, start: str, end: str,
                    market_hours_only: bool = True) -> pd.DataFrame:
    """
    Load multiple days of processed features for a symbol.
    Each day goes through load_day() so cache is used automatically.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    end_dt   = datetime.strptime(end,   "%Y-%m-%d").date()

    fs    = gcsfs.GCSFileSystem(project=PROJECT_ID)
    files = fs.glob(f"{BUCKET_NAME}/processed/features/{symbol}/*.parquet")

    frames = []
    for f in sorted(files):
        fname = Path(f).stem
        try:
            fdate = datetime.strptime(fname, "%Y-%m-%d").date()
        except ValueError:
            continue
        if start_dt <= fdate <= end_dt:
            df = load_day(symbol, fname, market_hours_only)
            if not df.empty:
                frames.append(df)

    if not frames:
        print(f"[data_loader] No data: {symbol} | {start} → {end}")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True).sort_values(
        "ts_sec"
    ).reset_index(drop=True)
    print(f"[data_loader] Total: {len(combined):,} bars | {symbol} | {start} → {end}")
    return combined


def iter_bars(symbol: str, start: str, end: str,
              market_hours_only: bool = True) -> Iterator[pd.Series]:
    """Iterate over bars one at a time for the backtest engine."""
    df = load_date_range(symbol, start, end, market_hours_only)
    if df.empty:
        return
    for _, row in df.iterrows():
        yield row


if __name__ == "__main__":
    print("=== Test 1: load_day ===")
    df = load_day("NIFTY26MAYFUT", "2026-04-30")
    if not df.empty:
        print(f"Rows: {len(df):,}")
        print(f"First bar: {df.iloc[0]['ts_ist']}")
        print(f"Last bar:  {df.iloc[-1]['ts_ist']}")
        print(df[["ts_ist", "open", "close", "spread_mean",
                   "imbalance_last", "realized_vol_60s"]].head(5).to_string())

    print("\n=== Test 2: iter_bars (first 5) ===")
    count = 0
    for bar in iter_bars("NIFTY26MAYFUT", "2026-04-30", "2026-04-30"):
        print(f"  {bar['ts_ist']} | close={bar['close']} | "
              f"imbalance={bar['imbalance_last']:.4f} | "
              f"spread={bar['spread_mean']}")
        count += 1
        if count >= 5:
            break