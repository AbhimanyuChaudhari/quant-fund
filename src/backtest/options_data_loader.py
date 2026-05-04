"""
Options Data Loader

Loads processed options features from GCS.
Output of options_pipeline.py — includes Greeks, IV, microstructure.

Key difference from futures data_loader.py:
  - One file per underlying per day (all strikes combined)
  - Has Greek columns: iv, delta, gamma, vega, theta
  - Has metadata: strike, opt_type, expiry, moneyness, tte
  - Needs filtering by strike, expiry, opt_type
"""

import logging
import gcsfs
import duckdb
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Iterator

logger = logging.getLogger(__name__)

PROJECT_ID  = "hedge-fund-494103"
BUCKET_NAME = "hedge-fund-494103-marketdata"

# NSE market hours filter (IST)
MARKET_OPEN_IST  = 33300   # 09:15 IST
MARKET_CLOSE_IST = 55800   # 15:30 IST


def _get_con():
    fs  = gcsfs.GCSFileSystem(project=PROJECT_ID)
    con = duckdb.connect()
    con.register_filesystem(fs)
    return con


def load_options_day(underlying: str, date_str: str,
                     opt_type:         Optional[str]   = None,
                     strike_range:     Optional[tuple] = None,
                     market_hours_only: bool            = True,
                     min_premium:      float            = 1.0,
                     max_tte:          float            = 1.0) -> pd.DataFrame:
    """
    Load one day of processed options features.

    Args:
        underlying:        'NIFTY' or 'BANKNIFTY'
        date_str:          'YYYY-MM-DD'
        opt_type:          'CE', 'PE', or None (both)
        strike_range:      (min_strike, max_strike) or None (all)
        market_hours_only: filter to 09:15-15:30 IST
        min_premium:       minimum option price (filter illiquid)
        max_tte:           maximum time to expiry in years (filter far expiries)

    Returns:
        DataFrame with all options features
    """
    con      = _get_con()
    gcs_path = (f"gs://{BUCKET_NAME}/processed/options/"
                f"{underlying}/{date_str}.parquet")

    fs = gcsfs.GCSFileSystem(project=PROJECT_ID)
    if not fs.exists(gcs_path.replace("gs://", "")):
        logger.warning(f"No options data: {underlying} | {date_str}")
        return pd.DataFrame()

    # Build WHERE clause
    conditions = []

    if market_hours_only:
        conditions.append(
            f"((ts_sec + 19800) % 86400) BETWEEN "
            f"{MARKET_OPEN_IST} AND {MARKET_CLOSE_IST}"
        )

    if opt_type:
        conditions.append(f"opt_type = '{opt_type}'")

    if strike_range:
        conditions.append(
            f"strike BETWEEN {strike_range[0]} AND {strike_range[1]}"
        )

    if min_premium > 0:
        conditions.append(f"close >= {min_premium}")

    if max_tte < 1.0:
        conditions.append(f"tte <= {max_tte}")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    df = con.execute(f"""
        SELECT *
        FROM read_parquet('{gcs_path}')
        {where}
        ORDER BY ts_sec, symbol
    """).df()

    logger.info(
        f"Loaded {len(df):,} option bars | "
        f"{underlying} | {date_str} | "
        f"symbols={df['symbol'].nunique() if not df.empty else 0}"
    )
    return df


def load_options_chain(underlying: str, date_str: str,
                       ts_sec: int,
                       atm_price: float,
                       strikes_each_side: int = 5) -> pd.DataFrame:
    """
    Load the full options chain at a specific timestamp.
    Returns ATM ± N strikes for both CE and PE.

    Args:
        underlying:        'NIFTY' or 'BANKNIFTY'
        date_str:          'YYYY-MM-DD'
        ts_sec:            unix timestamp to get chain at
        atm_price:         current underlying price (to find ATM strike)
        strikes_each_side: number of strikes each side of ATM

    Returns:
        DataFrame with one row per strike/type at that timestamp
    """
    con      = _get_con()
    gcs_path = (f"gs://{BUCKET_NAME}/processed/options/"
                f"{underlying}/{date_str}.parquet")

    fs = gcsfs.GCSFileSystem(project=PROJECT_ID)
    if not fs.exists(gcs_path.replace("gs://", "")):
        return pd.DataFrame()

    # Find ATM strike (nearest 50pt multiple for NIFTY)
    interval  = 50 if underlying == "NIFTY" else 100
    atm_strike = round(atm_price / interval) * interval
    min_strike = atm_strike - strikes_each_side * interval
    max_strike = atm_strike + strikes_each_side * interval

    df = con.execute(f"""
        SELECT *
        FROM read_parquet('{gcs_path}')
        WHERE ts_sec = {ts_sec}
          AND strike BETWEEN {min_strike} AND {max_strike}
          AND close > 0.5
        ORDER BY strike, opt_type
    """).df()

    return df


def iter_option_bars(underlying: str, date_str: str,
                     opt_type: Optional[str] = None,
                     strike_range: Optional[tuple] = None,
                     market_hours_only: bool = True,
                     min_premium: float = 5.0) -> Iterator[tuple]:
    """
    Iterate over option bars grouped by timestamp.
    Yields (ts_sec, chain_df) for each second.

    Usage:
        for ts_sec, chain in iter_option_bars('NIFTY', '2026-05-06'):
            strategy.on_chain(ts_sec, chain)
    """
    df = load_options_day(
        underlying        = underlying,
        date_str          = date_str,
        opt_type          = opt_type,
        strike_range      = strike_range,
        market_hours_only = market_hours_only,
        min_premium       = min_premium,
    )

    if df.empty:
        return

    for ts_sec, group in df.groupby("ts_sec"):
        yield int(ts_sec), group


if __name__ == "__main__":
    print("=== Options Data Loader Tests ===\n")

    import gcsfs
    fs = gcsfs.GCSFileSystem(project=PROJECT_ID)

    # Find available options dates
    files = fs.glob(
        f"{BUCKET_NAME}/processed/options/NIFTY/*.parquet"
    )
    dates = sorted([Path(f).stem for f in files])
    print(f"Available NIFTY options dates: {dates}")

    if dates:
        date_str = dates[-1]
        print(f"\nLoading {date_str}...")

        df = load_options_day(
            underlying = "NIFTY",
            date_str   = date_str,
            opt_type   = "CE",
            min_premium= 1.0,
        )

        if not df.empty:
            print(f"Rows: {len(df):,}")
            print(f"Symbols: {df['symbol'].nunique()}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"\nSample:")
            cols = ["ts_ist", "symbol", "strike", "opt_type",
                    "close", "iv", "delta", "gamma", "theta"]
            available = [c for c in cols if c in df.columns]
            print(df[available].head(5).to_string())
        else:
            print("No data yet — waiting for Tuesday May 5")
    else:
        print("No options data yet — will be available after May 5")