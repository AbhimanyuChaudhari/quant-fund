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

# NSE market hours in IST:  09:15 → 15:30
# IST = UTC + 5:30 = UTC + 19800 seconds
# So to filter by IST time-of-day using UTC epoch:
#   IST seconds-of-day = (ts_sec + 19800) % 86400
# 09:15 IST = 9*3600 + 15*60 = 33300
# 15:30 IST = 15*3600 + 30*60 = 55800
MARKET_OPEN_IST  = 33300   # 09:15 IST
MARKET_CLOSE_IST = 55800   # 15:30 IST


def _get_con() -> duckdb.DuckDBPyConnection:
    """DuckDB connection with GCS access."""
    fs  = gcsfs.GCSFileSystem(project=PROJECT_ID)
    con = duckdb.connect()
    con.register_filesystem(fs)
    return con


def _market_filter(symbol: str = "") -> str:
    """SQL fragment: filter to market hours IST.
    NSE equity: 09:15-15:30 IST
    CDS currency: 09:00-17:00 IST
    """
    is_currency = any(x in symbol.upper() 
                      for x in ["USDINR", "EURINR", "GBPINR", "JPYINR"])
    if is_currency:
        open_ist  = 9 * 3600           # 09:00 IST = 32400
        close_ist = 17 * 3600          # 17:00 IST = 61200
    else:
        open_ist  = MARKET_OPEN_IST    # 09:15 IST = 33300
        close_ist = MARKET_CLOSE_IST   # 15:30 IST = 55800

    return f"""
        ((ts_sec + 19800) % 86400) >= {open_ist}
        AND ((ts_sec + 19800) % 86400) <= {close_ist}
    """


def load_day(symbol: str, date: str,
             market_hours_only: bool = True) -> pd.DataFrame:
    """
    Load one day of processed features for a symbol from GCS.

    Args:
        symbol:            e.g. 'NIFTY26MAYFUT'
        date:              e.g. '2026-04-30'
        market_hours_only: filter to 09:15-15:30 IST (default True)

    Returns:
        DataFrame with 33 feature columns sorted by ts_sec.
        Empty DataFrame if file not found.
    """
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
    return df


def load_date_range(symbol: str, start: str, end: str,
                    market_hours_only: bool = True) -> pd.DataFrame:
    """
    Load multiple days of processed features for a symbol.
    Loads each day separately and concatenates — avoids DuckDB glob issues.

    Args:
        symbol:  e.g. 'NIFTY26MAYFUT'
        start:   e.g. '2026-04-29'
        end:     e.g. '2026-04-30'
        market_hours_only: filter to market hours (default True)

    Returns:
        Combined DataFrame across all dates, sorted by ts_sec.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    end_dt   = datetime.strptime(end,   "%Y-%m-%d").date()

    # Find all available parquet files in date range
    fs    = gcsfs.GCSFileSystem(project=PROJECT_ID)
    files = fs.glob(f"{BUCKET_NAME}/processed/features/{symbol}/*.parquet")

    frames = []
    for f in sorted(files):
        fname = Path(f).stem  # e.g. '2026-04-30'
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

    combined = pd.concat(frames, ignore_index=True).sort_values("ts_sec").reset_index(drop=True)
    print(f"[data_loader] Total: {len(combined):,} bars | {symbol} | {start} → {end}")
    return combined


def iter_bars(symbol: str, start: str, end: str,
              market_hours_only: bool = True) -> Iterator[pd.Series]:
    """
    Iterate over bars one at a time for the backtest engine.
    Yields one pd.Series per second bar.

    Usage:
        for bar in iter_bars('NIFTY26MAYFUT', '2026-04-30', '2026-04-30'):
            strategy.on_bar(bar)
    """
    df = load_date_range(symbol, start, end, market_hours_only)
    if df.empty:
        return
    for _, row in df.iterrows():
        yield row


if __name__ == "__main__":
    # Test 1 — load single day
    print("=== Test 1: load_day ===")
    df = load_day("NIFTY26MAYFUT", "2026-04-30")
    if not df.empty:
        print(f"Rows: {len(df):,}")
        print(f"First bar: {df.iloc[0]['ts_ist']}")
        print(f"Last bar:  {df.iloc[-1]['ts_ist']}")
        print(df[["ts_ist", "open", "close", "spread_mean",
                   "imbalance_last", "realized_vol_60s"]].head(5).to_string())

    # Test 2 — iterate bars
    print("\n=== Test 2: iter_bars (first 5) ===")
    count = 0
    for bar in iter_bars("NIFTY26MAYFUT", "2026-04-30", "2026-04-30"):
        print(f"  {bar['ts_ist']} | close={bar['close']} | "
              f"imbalance={bar['imbalance_last']:.4f} | "
              f"spread={bar['spread_mean']}")
        count += 1
        if count >= 5:
            break