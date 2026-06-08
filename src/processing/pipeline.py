import io
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from src.storage.gcs import get_bucket

# ─────────────────────────────────────
# Step 1 — Load raw ticks from GCS
# ─────────────────────────────────────

def load_raw_ticks(symbol: str, date: str) -> pd.DataFrame:
    """
    Load all raw tick files for a symbol and date from GCS.
    Combines all hourly parquet files into one dataframe.
    """
    bucket  = get_bucket()
    prefix  = f"raw/orderbook/{symbol}/{date}/"
    blobs   = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        print(f"No data found for {symbol} on {date}")
        return pd.DataFrame()

    frames = []
    for blob in blobs:
        buf = io.BytesIO()
        blob.download_to_file(buf)
        buf.seek(0)
        frames.append(pd.read_parquet(buf))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("ts_local_ns").reset_index(drop=True)

    print(f"Loaded {len(df):,} ticks for {symbol} on {date}")
    return df


# ─────────────────────────────────────
# Step 2 — Build 1-second bars
# ─────────────────────────────────────

def build_1s_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw ticks into 1-second OHLCV bars.
    Also computes VWAP per bar.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["ts_sec"] = df["ts_local_ns"] // 1_000_000_000
    df["ts_dt"]  = pd.to_datetime(df["ts_sec"], unit="s", utc=True)\
                     .dt.tz_convert("Asia/Kolkata")

    bars = df.groupby(["symbol", "ts_sec"]).agg(
        open       = ("last_price", "first"),
        high       = ("last_price", "max"),
        low        = ("last_price", "min"),
        close      = ("last_price", "last"),
        volume     = ("volume",     "last"),
        tick_count = ("last_price", "count"),
        vwap       = ("avg_price",  "last"),
        oi         = ("oi",         "last"),
    ).reset_index()

    bars["ts_dt"] = pd.to_datetime(bars["ts_sec"], unit="s", utc=True)\
                      .dt.tz_convert("Asia/Kolkata")

    bars = bars.sort_values("ts_sec").reset_index(drop=True)
    print(f"Built {len(bars):,} 1-second bars")
    return bars


# ─────────────────────────────────────
# Step 3 — Microstructure features
# ─────────────────────────────────────

def build_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute microstructure features from raw ticks.
    These are the inputs to your ML models.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["ts_sec"] = df["ts_local_ns"] // 1_000_000_000

    # ── Per-tick features ──────────────────────────

    # Spread in basis points
    df["spread_bps"] = (df["spread"] / df["mid_price"] * 10000)\
                         .replace([np.inf, -np.inf], 0).fillna(0)

    # Weighted mid price (more accurate than simple mid)
    # Weights bids and asks by their quantity
    df["weighted_mid"] = (
        df["bid_p1"] * df["ask_q1"] + df["ask_p1"] * df["bid_q1"]
    ) / (df["bid_q1"] + df["ask_q1"]).replace(0, np.nan)
    df["weighted_mid"] = df["weighted_mid"].fillna(df["mid_price"])

    # Price impact proxy
    df["price_impact"] = (df["last_price"] - df["mid_price"]).abs()

    # ── Per-second aggregation ─────────────────────

    features = df.groupby(["symbol", "ts_sec"]).agg(
        # Spread features
        spread_mean    = ("spread",       "mean"),
        spread_max     = ("spread",       "max"),
        spread_bps     = ("spread_bps",   "mean"),

        # Order book imbalance
        imbalance_mean = ("book_imbalance", "mean"),
        imbalance_std  = ("book_imbalance", "std"),
        imbalance_last = ("book_imbalance", "last"),

        # Weighted mid
        weighted_mid   = ("weighted_mid", "last"),

        # Quantities
        total_bid_qty  = ("total_bid_qty", "last"),
        total_ask_qty  = ("total_ask_qty", "last"),

        # Price impact
        price_impact   = ("price_impact", "mean"),

        # Tick count (activity proxy)
        tick_count     = ("last_price", "count"),
    ).reset_index()

    features["ts_dt"] = pd.to_datetime(
        features["ts_sec"], unit="s", utc=True
    ).dt.tz_convert("Asia/Kolkata")

    print(f"Built microstructure features: {len(features):,} rows")
    return features


# ─────────────────────────────────────
# Step 4 — Rolling features
# ─────────────────────────────────────

def build_rolling_features(bars: pd.DataFrame,
                           features: pd.DataFrame) -> pd.DataFrame:
    """
    Merge bars and microstructure features.
    Add rolling window features used by models.
    Windows: 10s, 30s, 60s, 300s
    """
    if bars.empty or features.empty:
        return pd.DataFrame()

    df = bars.merge(features, on=["symbol", "ts_sec"], how="left")
    df = df.sort_values("ts_sec").reset_index(drop=True)

    # Rolling realized volatility
    df["returns"]  = df["close"].pct_change()
    for window in [10, 30, 60, 300]:
        df[f"realized_vol_{window}s"] = (
            df["returns"]
            .rolling(window)
            .std() * np.sqrt(window)
        )

    # Rolling imbalance momentum
    for window in [10, 30, 60]:
        df[f"imbalance_ma_{window}s"] = (
            df["imbalance_last"]
            .rolling(window)
            .mean()
        )

    # Rolling spread z-score
    spread_mean = df["spread_mean"].rolling(300).mean()
    spread_std  = df["spread_mean"].rolling(300).std()
    df["spread_zscore"] = (
        (df["spread_mean"] - spread_mean) / spread_std
    ).fillna(0)

    # Volume ratio (current vs average)
    df["volume_ratio"] = (
        df["tick_count_x"] /
        df["tick_count_x"].rolling(60).mean()
    ).fillna(1)

    # Price momentum
    for window in [10, 30, 60]:
        df[f"price_mom_{window}s"] = df["close"].pct_change(window)

    df = df.drop(columns=["returns"], errors="ignore")
    print(f"Built rolling features: {len(df):,} rows, "
          f"{len(df.columns)} columns")
    return df


# ─────────────────────────────────────
# Step 5 — Save processed data to GCS
# ─────────────────────────────────────

def save_processed(df: pd.DataFrame, symbol: str, date: str):
    """Save processed features back to GCS."""
    if df.empty:
        return

    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    buf.seek(0)

    bucket    = get_bucket()
    blob_name = f"processed/features/{symbol}/{date}.parquet"
    blob      = bucket.blob(blob_name)
    blob.upload_from_file(
        buf, content_type="application/octet-stream"
    )
    print(f"Saved → GCS: {blob_name}")

# ─────────────────────────────────────
# Cleanup — delete raw files after processing
# ADD THIS FUNCTION TO pipeline.py
# ─────────────────────────────────────

def delete_raw_ticks(symbol: str, date: str, keep_days: int = 1):
    """
    Delete raw tick files for a symbol/date after successful processing.
    Keeps last keep_days days as safety net in case reprocessing needed.
    """
    from datetime import datetime, timedelta

    # Don't delete if date is within keep_days of today
    try:
        file_date = datetime.strptime(date, '%Y-%m-%d').date()
        cutoff    = datetime.today().date() - timedelta(days=keep_days)
        if file_date >= cutoff:
            print(f"  Keeping raw/{symbol}/{date} (within {keep_days}-day window)")
            return
    except Exception:
        pass

    bucket = get_bucket()
    prefix = f"raw/orderbook/{symbol}/{date}/"
    blobs  = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        return

    for blob in blobs:
        blob.delete()

    print(f"  Deleted {len(blobs)} raw files: raw/orderbook/{symbol}/{date}/")


# ─────────────────────────────────────
# UPDATED run_pipeline
# Only change vs original: added delete_raw_ticks() after save_processed()
# ─────────────────────────────────────

def run_pipeline(symbol: str, date: str):
    """
    Full processing pipeline for one symbol and date.
    Raw ticks → 1s bars → features → GCS → delete raw
    """
    print(f"\n{'='*50}")
    print(f"Processing: {symbol} | {date}")
    print(f"{'='*50}")

    # Step 1 — Load
    ticks = load_raw_ticks(symbol, date)
    if ticks.empty:
        return

    # Step 2 — 1s bars
    bars = build_1s_bars(ticks)

    # Step 3 — Microstructure
    micro = build_microstructure_features(ticks)

    # Step 4 — Rolling features
    final = build_rolling_features(bars, micro)

    # Step 5 — Save processed
    save_processed(final, symbol, date)

    # Step 6 — Delete raw (only if save succeeded — got here without exception)
    # Keeps yesterday's raw as 1-day safety net
    delete_raw_ticks(symbol, date, keep_days=1)

    print(f"Done: {symbol} | {date}")
    return final
# ─────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────

if __name__ == "__main__":
    # Test on available data
    result = run_pipeline("BANKNIFTY26APRFUT", "2026-04-27")
    if result is not None:
        print(f"\nSample output:")
        print(result[["ts_dt_x", "open", "close", "spread_mean",
                       "imbalance_last", "realized_vol_60s"]].head(10).to_string())