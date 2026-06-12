"""
Options Processing Pipeline

Reads raw options ticks from GCS, builds 1-second bars,
computes Black-Scholes Greeks, joins with spot price,
and saves processed features to GCS.

Output path:
    processed/options/{underlying}/{date}.parquet
    e.g. processed/options/NIFTY/2026-05-02.parquet

Each file contains ALL strikes for that underlying on that date.
One row per (symbol, ts_sec) — e.g. NIFTY26MAY24000CE at 09:15:01
"""

import os
import io
import re
import duckdb
import gcsfs
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, date
from scipy.stats import norm
from scipy.optimize import brentq
from dotenv import load_dotenv
from src.storage.gcs import get_bucket

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

PROJECT_ID  = os.getenv("GCP_PROJECT_ID", "hedge-fund-494103")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "hedge-fund-494103-marketdata-mumbai")

# Risk-free rate — 10yr GSec yield
RISK_FREE_RATE = 0.07

# Market hours filter (IST)
MARKET_OPEN_IST  = 33300   # 09:15 IST
MARKET_CLOSE_IST = 55800   # 15:30 IST


# ─────────────────────────────────────
# DuckDB connection
# ─────────────────────────────────────
def get_con():
    fs  = gcsfs.GCSFileSystem(project=PROJECT_ID)
    con = duckdb.connect()
    con.register_filesystem(fs)
    return con


# ─────────────────────────────────────
# Symbol parsing
# ─────────────────────────────────────
def parse_option_symbol(symbol: str) -> dict:
    """
    Parse option symbol into components.
    e.g. NIFTY26MAY24000CE →
        underlying=NIFTY, expiry=2026-05-26, strike=24000, type=CE

    Zerodha format: {UNDERLYING}{YY}{MMM}{STRIKE}{TYPE}
    e.g. NIFTY26MAY24000CE
         BANKNIFTY26MAY54800PE
    """
    pattern = r"^([A-Z&-]+)(\d{2})([A-Z]{3})(\d+)(CE|PE)$"
    match   = re.match(pattern, symbol)
    if not match:
        return {}

    underlying = match.group(1)
    yy         = match.group(2)
    mmm        = match.group(3)
    strike     = float(match.group(4))
    opt_type   = match.group(5)

    # Parse expiry date
    month_map = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
                 "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
    year  = 2000 + int(yy)
    month = month_map.get(mmm, 0)

    # Last Thursday of the month = NSE expiry
    # Approximate: use last day of month, then find last Thursday
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    expiry_dt = date(year, month, last_day)
    while expiry_dt.weekday() != 3:  # 3 = Thursday
        expiry_dt -= pd.Timedelta(days=1)

    return {
        "underlying": underlying,
        "expiry":     expiry_dt,
        "strike":     strike,
        "opt_type":   opt_type,
    }


def time_to_expiry(expiry_date: date, as_of_date: date) -> float:
    """Time to expiry in years."""
    days = (expiry_date - as_of_date).days
    return max(days / 365.0, 1/365.0)  # minimum 1 day


# ─────────────────────────────────────
# Black-Scholes
# ─────────────────────────────────────
def bs_price(S, K, T, r, sigma, opt_type):
    """Black-Scholes option price."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt_type == "CE":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, opt_type):
    """Compute all Greeks."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return {"delta":0, "gamma":0, "vega":0, "theta":0}

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    delta = norm.cdf(d1)  if opt_type == "CE" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100   # per 1% vol change
    theta = (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) -
             r*K*np.exp(-r*T)*(norm.cdf(d2) if opt_type=="CE"
                                else norm.cdf(-d2))) / 365

    return {"delta": delta, "gamma": gamma,
            "vega": vega,   "theta": theta}


def implied_vol(market_price, S, K, T, r, opt_type,
                low=0.001, high=20.0) -> float:
    """
    Solve for implied volatility using Brent's method.
    Returns NaN if no solution found.
    """
    if market_price <= 0 or S <= 0 or T <= 0:
        return np.nan

    # Intrinsic value check
    intrinsic = max(0, S-K) if opt_type=="CE" else max(0, K-S)
    if market_price < intrinsic * 0.999:
        return np.nan

    try:
        iv = brentq(
            lambda sigma: bs_price(S, K, T, r, sigma, opt_type) - market_price,
            low, high, xtol=1e-6, maxiter=100
        )
        return iv if 0 < iv < high else np.nan
    except (ValueError, RuntimeError):
        return np.nan


# ─────────────────────────────────────
# Load spot price for a date
# ─────────────────────────────────────
def load_spot_bars(underlying: str, date_str: str) -> pd.DataFrame:
    """
    Load NIFTY_SPOT or BANKNIFTY_SPOT 1-second bars for a date.
    Returns DataFrame with ts_sec → spot_price.
    Falls back to futures price if spot not available.
    """
    con = get_con()

    spot_symbol = f"{underlying}_SPOT"
    futures_symbol_prefix = underlying

    # Try spot first
    fs    = gcsfs.GCSFileSystem(project=PROJECT_ID)
    files = fs.glob(f"{BUCKET_NAME}/raw/orderbook/{spot_symbol}/{date_str}/*.parquet")

    if files:
        paths = [f"gs://{f}" for f in files]
        spot_df = pd.DataFrame()
        for p in paths:
            try:
                chunk = con.execute(f"""
                    SELECT
                        (ts_local_ns // 1000000000)::BIGINT AS ts_sec,
                        LAST(last_price) AS spot_price
                    FROM read_parquet('{p}')
                    WHERE last_price > 0
                    GROUP BY ts_sec
                    ORDER BY ts_sec
                """).df()
                spot_df = pd.concat([spot_df, chunk])
            except:
                continue

        if not spot_df.empty:
            spot_df = spot_df.groupby("ts_sec")["spot_price"].last().reset_index()
            print(f"  Spot: {len(spot_df):,} bars from {spot_symbol}")
            return spot_df

    # Fallback to futures
    fut_files = fs.glob(
        f"{BUCKET_NAME}/raw/orderbook/{underlying}*FUT/{date_str}/*.parquet"
    )
    fut_files = [f for f in fut_files if "BANK" not in f or underlying == "BANKNIFTY"]

    if fut_files:
        paths = [f"gs://{f}" for f in sorted(fut_files)[:1]]  # nearest expiry
        try:
            spot_df = con.execute(f"""
                SELECT
                    (ts_local_ns // 1000000000)::BIGINT AS ts_sec,
                    LAST(last_price) AS spot_price
                FROM read_parquet('{paths[0]}')
                WHERE last_price > 0
                GROUP BY ts_sec
                ORDER BY ts_sec
            """).df()
            print(f"  Spot: {len(spot_df):,} bars from futures (fallback)")
            return spot_df
        except:
            pass

    print(f"  Spot: no data found for {underlying}")
    return pd.DataFrame()


# ─────────────────────────────────────
# Build options bars for one symbol
# ─────────────────────────────────────
def build_option_bars(symbol: str, date_str: str,
                      con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Build 1-second OHLCV + microstructure bars for one option symbol."""
    fs    = gcsfs.GCSFileSystem(project=PROJECT_ID)
    files = fs.glob(f"{BUCKET_NAME}/raw/orderbook/{symbol}/{date_str}/*.parquet")

    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            chunk = con.execute(f"""
                SELECT
                    (ts_local_ns // 1000000000)::BIGINT AS ts_sec,
                    symbol,
                    FIRST(last_price)       AS open,
                    MAX(last_price)         AS high,
                    MIN(last_price)         AS low,
                    LAST(last_price)        AS close,
                    LAST(volume)            AS volume,
                    COUNT(*)                AS tick_count,
                    LAST(oi)                AS oi,
                    AVG(spread)             AS spread_mean,
                    MAX(spread)             AS spread_max,
                    AVG(book_imbalance)     AS imbalance_mean,
                    LAST(book_imbalance)    AS imbalance_last,
                    LAST(total_bid_qty)     AS total_bid_qty,
                    LAST(total_ask_qty)     AS total_ask_qty,
                    LAST(bid_p1)            AS bid_p1,
                    LAST(ask_p1)            AS ask_p1
                FROM read_parquet('gs://{f}')
                WHERE last_price > 0
                  AND ((ts_local_ns // 1000000000) + 19800) % 86400 >= {MARKET_OPEN_IST}
                  AND ((ts_local_ns // 1000000000) + 19800) % 86400 <= {MARKET_CLOSE_IST}
                GROUP BY symbol, ts_sec
                ORDER BY ts_sec
            """).df()
            frames.append(chunk)
        except Exception as e:
            print(f"    Warning: {f}: {e}")
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("ts_sec").reset_index(drop=True)
    return df


# ─────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────
def run_options_pipeline(underlying: str, date_str: str) -> pd.DataFrame:
    """
    Full options pipeline for one underlying and date.

    Args:
        underlying: 'NIFTY' or 'BANKNIFTY'
        date_str:   'YYYY-MM-DD'

    Returns:
        Combined DataFrame with all strikes, Greeks, and features.
        Saved to GCS: processed/options/{underlying}/{date}.parquet
    """
    print(f"\n{'='*55}")
    print(f"Options pipeline: {underlying} | {date_str}")
    print(f"{'='*55}")

    con = get_con()
    fs  = gcsfs.GCSFileSystem(project=PROJECT_ID)

    # ── Find all option symbols for this underlying/date ──
    all_folders = [f.split('/')[-1] for f in
                   fs.ls(f"{BUCKET_NAME}/raw/orderbook/")]
    opt_symbols = [s for s in all_folders
                   if s.startswith(underlying) and
                   s.endswith(("CE", "PE")) and
                   "BANK" not in s or underlying == "BANKNIFTY"]

    if not opt_symbols:
        print(f"No options symbols found for {underlying}")
        return pd.DataFrame()

    print(f"Found {len(opt_symbols)} option symbols")

    # ── Load spot price ───────────────────────────────
    spot_df = load_spot_bars(underlying, date_str)
    if spot_df.empty:
        print("No spot data — cannot compute Greeks")
        return pd.DataFrame()

    # ── Process each symbol ───────────────────────────
    as_of = datetime.strptime(date_str, "%Y-%m-%d").date()
    all_frames = []

    for symbol in sorted(opt_symbols):
        parsed = parse_option_symbol(symbol)
        if not parsed:
            continue

        bars = build_option_bars(symbol, date_str, con)
        if bars.empty:
            continue

        # Join spot price
        bars = bars.merge(spot_df, on="ts_sec", how="left")
        bars["spot_price"] = bars["spot_price"].ffill().bfill()

        # Option metadata
        bars["strike"]     = parsed["strike"]
        bars["opt_type"]   = parsed["opt_type"]
        bars["expiry"]     = str(parsed["expiry"])
        bars["underlying"] = underlying
        bars["tte"]        = time_to_expiry(parsed["expiry"], as_of)

        # Moneyness
        bars["moneyness"]  = (bars["spot_price"] - bars["strike"]) / bars["spot_price"]

        # Compute Greeks row by row
        ivs, deltas, gammas, vegas, thetas = [], [], [], [], []

        for _, row in bars.iterrows():
            S  = row["spot_price"]
            K  = row["strike"]
            T  = row["tte"]
            mp = row["close"]
            ot = row["opt_type"]

            iv = implied_vol(mp, S, K, T, RISK_FREE_RATE, ot)
            ivs.append(iv)

            if not np.isnan(iv):
                g = bs_greeks(S, K, T, RISK_FREE_RATE, iv, ot)
                deltas.append(g["delta"])
                gammas.append(g["gamma"])
                vegas.append(g["vega"])
                thetas.append(g["theta"])
            else:
                deltas.append(np.nan)
                gammas.append(np.nan)
                vegas.append(np.nan)
                thetas.append(np.nan)

        bars["iv"]    = ivs
        bars["delta"] = deltas
        bars["gamma"] = gammas
        bars["vega"]  = vegas
        bars["theta"] = thetas

        # Rolling IV features
        bars["iv_ma_60s"]    = bars["iv"].rolling(60).mean()
        bars["iv_zscore"]    = (bars["iv"] - bars["iv"].rolling(300).mean()) / \
                                bars["iv"].rolling(300).std()

        # ts_ist for readability
        bars["ts_ist"] = (
            pd.to_datetime(bars["ts_sec"], unit="s", utc=True)
            .dt.tz_convert("Asia/Kolkata")
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )

        all_frames.append(bars)
        print(f"  {symbol}: {len(bars):,} bars | "
              f"IV range: {bars['iv'].min():.2f}-{bars['iv'].max():.2f}")

    if not all_frames:
        print("No data processed")
        return pd.DataFrame()

    # ── Combine all strikes ───────────────────────────
    final = pd.concat(all_frames, ignore_index=True)
    final = final.sort_values(["ts_sec", "symbol"]).reset_index(drop=True)

    print(f"\nTotal rows: {len(final):,} | Symbols: {final['symbol'].nunique()}")

    # ── Save to GCS ───────────────────────────────────
    buf       = io.BytesIO()
    final.to_parquet(buf, index=False, compression="zstd")
    buf.seek(0)

    bucket    = get_bucket()
    blob_name = f"processed/options/{underlying}/{date_str}.parquet"
    blob      = bucket.blob(blob_name)
    blob.upload_from_file(buf, content_type="application/octet-stream")
    print(f"Saved → gs://{BUCKET_NAME}/{blob_name}")

    return final


if __name__ == "__main__":
    import time

    # Run for both underlyings — use yesterday's date once data exists
    for underlying in ["NIFTY", "BANKNIFTY"]:
        start = time.time()
        df    = run_options_pipeline(underlying, "2026-05-02")
        elapsed = time.time() - start

        if df is not None and not df.empty:
            print(f"\nCompleted in {elapsed:.1f}s")
            print(f"Columns: {df.columns.tolist()}")
            print(df[["ts_ist", "symbol", "close", "spot_price",
                       "iv", "delta", "gamma"]].head(10).to_string())
