import os
import io
import duckdb
import gcsfs
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from src.storage.gcs import get_bucket

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

PROJECT_ID  = os.getenv("GCP_PROJECT_ID")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")


def get_duckdb_con():
    """Create DuckDB connection with GCS access."""
    fs  = gcsfs.GCSFileSystem(project=PROJECT_ID)
    con = duckdb.connect()
    con.register_filesystem(fs)
    return con


def build_features_duckdb(symbol: str, date: str) -> pd.DataFrame:
    """
    Build 1-second bars + microstructure features in one DuckDB query.
    Reads raw ticks directly from GCS — no download needed.
    Clean output: no duplicate columns, no _x/_y suffixes, correct timestamps.
    """
    con      = get_duckdb_con()
    gcs_path = f"gs://{BUCKET_NAME}/raw/orderbook/{symbol}/{date}/*.parquet"

    fs    = gcsfs.GCSFileSystem(project=PROJECT_ID)
    files = fs.glob(f"{BUCKET_NAME}/raw/orderbook/{symbol}/{date}/*.parquet")
    if not files:
        print(f"No data: {symbol} | {date}")
        return pd.DataFrame()

    df = con.execute(f"""
        WITH raw AS (
            SELECT
                -- INTEGER division to avoid float precision loss
                (ts_local_ns // 1000000000)::BIGINT AS ts_sec,
                symbol,
                last_price,
                volume,
                avg_price,
                oi,
                spread,
                mid_price,
                book_imbalance,
                total_bid_qty,
                total_ask_qty,
                bid_p1, bid_q1,
                ask_p1, ask_q1,
                (bid_p1 * ask_q1 + ask_p1 * bid_q1) /
                    NULLIF(bid_q1 + ask_q1, 0)        AS weighted_mid,
                ABS(last_price - mid_price)            AS price_impact,
                spread / NULLIF(mid_price, 0) * 10000 AS spread_bps
            FROM read_parquet('{gcs_path}')
            WHERE last_price > 0
        ),

        bars AS (
            SELECT
                symbol,
                ts_sec,
                FIRST(last_price)        AS open,
                MAX(last_price)          AS high,
                MIN(last_price)          AS low,
                LAST(last_price)         AS close,
                LAST(volume)             AS volume,
                COUNT(*)                 AS tick_count,
                LAST(avg_price)          AS vwap,
                LAST(oi)                 AS oi,
                AVG(spread)              AS spread_mean,
                MAX(spread)              AS spread_max,
                AVG(spread_bps)          AS spread_bps,
                AVG(book_imbalance)      AS imbalance_mean,
                STDDEV(book_imbalance)   AS imbalance_std,
                LAST(book_imbalance)     AS imbalance_last,
                LAST(total_bid_qty)      AS total_bid_qty,
                LAST(total_ask_qty)      AS total_ask_qty,
                LAST(weighted_mid)       AS weighted_mid,
                AVG(price_impact)        AS price_impact
            FROM raw
            GROUP BY symbol, ts_sec
            ORDER BY ts_sec
        )

        SELECT
            symbol,
            ts_sec,

            -- Readable timestamp in IST
            strftime(
                (epoch_ms(ts_sec * 1000) AT TIME ZONE 'Asia/Kolkata'),
                '%Y-%m-%d %H:%M:%S'
            ) AS ts_ist,

            open, high, low, close,
            volume, tick_count, vwap, oi,

            -- Spread
            spread_mean, spread_max, spread_bps,

            -- Imbalance
            imbalance_mean, imbalance_std, imbalance_last,

            -- Depth
            total_bid_qty, total_ask_qty,
            weighted_mid, price_impact,

            -- Rolling volatility
            STDDEV(close) OVER (
                ORDER BY ts_sec ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
            ) AS realized_vol_10s,

            STDDEV(close) OVER (
                ORDER BY ts_sec ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
            ) AS realized_vol_30s,

            STDDEV(close) OVER (
                ORDER BY ts_sec ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
            ) AS realized_vol_60s,

            STDDEV(close) OVER (
                ORDER BY ts_sec ROWS BETWEEN 299 PRECEDING AND CURRENT ROW
            ) AS realized_vol_300s,

            -- Rolling imbalance MAs
            AVG(imbalance_last) OVER (
                ORDER BY ts_sec ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
            ) AS imbalance_ma_10s,

            AVG(imbalance_last) OVER (
                ORDER BY ts_sec ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
            ) AS imbalance_ma_30s,

            AVG(imbalance_last) OVER (
                ORDER BY ts_sec ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
            ) AS imbalance_ma_60s,

            -- Spread z-score (300s rolling)
            (spread_mean - AVG(spread_mean) OVER (
                ORDER BY ts_sec ROWS BETWEEN 299 PRECEDING AND CURRENT ROW
            )) / NULLIF(STDDEV(spread_mean) OVER (
                ORDER BY ts_sec ROWS BETWEEN 299 PRECEDING AND CURRENT ROW
            ), 0) AS spread_zscore,

            -- Volume ratio
            tick_count / NULLIF(
                AVG(tick_count) OVER (
                    ORDER BY ts_sec ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
                ), 0
            ) AS volume_ratio,

            -- Price momentum
            (close - LAG(close, 10) OVER (ORDER BY ts_sec)) /
                NULLIF(LAG(close, 10) OVER (ORDER BY ts_sec), 0) AS price_mom_10s,

            (close - LAG(close, 30) OVER (ORDER BY ts_sec)) /
                NULLIF(LAG(close, 30) OVER (ORDER BY ts_sec), 0) AS price_mom_30s,

            (close - LAG(close, 60) OVER (ORDER BY ts_sec)) /
                NULLIF(LAG(close, 60) OVER (ORDER BY ts_sec), 0) AS price_mom_60s

        FROM bars
        ORDER BY ts_sec

    """).df()

    print(f"Built {len(df):,} bars | {len(df.columns)} columns | {symbol} | {date}")
    return df


def save_processed(df: pd.DataFrame, symbol: str, date: str):
    """Save processed features to GCS, overwriting any existing file."""
    if df.empty:
        return

    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    buf.seek(0)

    bucket    = get_bucket()
    blob_name = f"processed/features/{symbol}/{date}.parquet"
    blob      = bucket.blob(blob_name)
    blob.upload_from_file(buf, content_type="application/octet-stream")
    print(f"Saved → gs://{bucket.name}/{blob_name}")


def run_pipeline(symbol: str, date: str):
    """Full pipeline — raw ticks → features → GCS."""
    print(f"\n{'='*50}")
    print(f"Processing: {symbol} | {date}")
    print(f"{'='*50}")

    df = build_features_duckdb(symbol, date)
    if df.empty:
        return None

    save_processed(df, symbol, date)
    return df


if __name__ == "__main__":
    import time

    for date in ["2026-04-29", "2026-04-30"]:
        start   = time.time()
        df      = run_pipeline("NIFTY26MAYFUT", date)
        elapsed = time.time() - start

        if df is not None:
            print(f"Completed in {elapsed:.1f}s")
            print(df[["ts_ist", "open", "close", "spread_mean",
                       "imbalance_last", "realized_vol_60s",
                       "spread_zscore"]].head(5).to_string())