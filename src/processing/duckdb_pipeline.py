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

PROJECT_ID  = os.getenv("GCP_PROJECT_ID", "hedge-fund-494103")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "hedge-fund-494103-marketdata")


def get_duckdb_con():
    """Create DuckDB connection with GCS access."""
    fs  = gcsfs.GCSFileSystem(project=PROJECT_ID)
    con = duckdb.connect()
    con.register_filesystem(fs)
    return con


def build_features_duckdb(symbol: str, date: str) -> pd.DataFrame:
    """
    Build 1-second bars + microstructure features + L5 depth in one DuckDB query.
    Reads raw ticks directly from GCS.

    Output columns (40 total):
        Base:         symbol, ts_sec, ts_ist
        OHLCV:        open, high, low, close, volume, tick_count, vwap, oi
        Spread:       spread_mean, spread_max, spread_bps
        Imbalance:    imbalance_mean, imbalance_std, imbalance_last
        Depth agg:    total_bid_qty, total_ask_qty, weighted_mid, price_impact
        L5 depth:     bid_p1..5, bid_q1..5, ask_p1..5, ask_q1..5  (NEW)
        Rolling:      realized_vol_10s/30s/60s/300s
        Imb MA:       imbalance_ma_10s/30s/60s
        Other:        spread_zscore, volume_ratio
        Momentum:     price_mom_10s/30s/60s
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
                bid_p2, bid_q2,
                bid_p3, bid_q3,
                bid_p4, bid_q4,
                bid_p5, bid_q5,
                ask_p1, ask_q1,
                ask_p2, ask_q2,
                ask_p3, ask_q3,
                ask_p4, ask_q4,
                ask_p5, ask_q5,
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
                LAST(volume) - FIRST(volume) AS volume_delta,
                LAST(volume) AS volume_cumulative,
                COUNT(*)                 AS tick_count,
                LAST(avg_price)          AS vwap,
                LAST(oi)                 AS oi,

                -- Spread
                AVG(spread)              AS spread_mean,
                MAX(spread)              AS spread_max,
                AVG(spread_bps)          AS spread_bps,

                -- Imbalance
                AVG(book_imbalance)      AS imbalance_mean,
                STDDEV(book_imbalance)   AS imbalance_std,
                LAST(book_imbalance)     AS imbalance_last,

                -- Depth aggregates
                LAST(total_bid_qty)      AS total_bid_qty,
                LAST(total_ask_qty)      AS total_ask_qty,
                LAST(weighted_mid)       AS weighted_mid,
                AVG(price_impact)        AS price_impact,

                -- L5 order book snapshot (last tick of each second)
                LAST(bid_p1) AS bid_p1, LAST(bid_q1) AS bid_q1,
                LAST(bid_p2) AS bid_p2, LAST(bid_q2) AS bid_q2,
                LAST(bid_p3) AS bid_p3, LAST(bid_q3) AS bid_q3,
                LAST(bid_p4) AS bid_p4, LAST(bid_q4) AS bid_q4,
                LAST(bid_p5) AS bid_p5, LAST(bid_q5) AS bid_q5,
                LAST(ask_p1) AS ask_p1, LAST(ask_q1) AS ask_q1,
                LAST(ask_p2) AS ask_p2, LAST(ask_q2) AS ask_q2,
                LAST(ask_p3) AS ask_p3, LAST(ask_q3) AS ask_q3,
                LAST(ask_p4) AS ask_p4, LAST(ask_q4) AS ask_q4,
                LAST(ask_p5) AS ask_p5, LAST(ask_q5) AS ask_q5

            FROM raw
            GROUP BY symbol, ts_sec
            ORDER BY ts_sec
        )

        SELECT
            symbol,
            ts_sec,
            open, high, low, close,
            volume, tick_count, vwap, oi,

            -- Spread
            spread_mean, spread_max, spread_bps,

            -- Imbalance
            imbalance_mean, imbalance_std, imbalance_last,

            -- Depth aggregates
            total_bid_qty, total_ask_qty,
            weighted_mid, price_impact,

            -- L5 order book (NEW)
            bid_p1, bid_q1, bid_p2, bid_q2, bid_p3, bid_q3,
            bid_p4, bid_q4, bid_p5, bid_q5,
            ask_p1, ask_q1, ask_p2, ask_q2, ask_p3, ask_q3,
            ask_p4, ask_q4, ask_p5, ask_q5,

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
                NULLIF(LAG(close, 10) OVER (ORDER BY ts_sec), 0)
            AS price_mom_10s,

            (close - LAG(close, 30) OVER (ORDER BY ts_sec)) /
                NULLIF(LAG(close, 30) OVER (ORDER BY ts_sec), 0)
            AS price_mom_30s,

            (close - LAG(close, 60) OVER (ORDER BY ts_sec)) /
                NULLIF(LAG(close, 60) OVER (ORDER BY ts_sec), 0)
            AS price_mom_60s

        FROM bars
        ORDER BY ts_sec

    """).df()

    # Add readable IST timestamp
    df["ts_ist"] = (
        pd.to_datetime(df["ts_sec"], unit="s", utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

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
    print(f"Saved -> gs://{bucket.name}/{blob_name}")


def run_pipeline(symbol: str, date: str):
    """Full pipeline — raw ticks -> features -> GCS."""
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

    for date in ["2026-04-30"]:
        start   = time.time()
        df      = run_pipeline("NIFTY26MAYFUT", date)
        elapsed = time.time() - start

        if df is not None:
            print(f"\nCompleted in {elapsed:.1f}s")
            print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
            print(df[["ts_ist", "open", "close",
                       "bid_p1", "bid_q1", "ask_p1", "ask_q1",
                       "bid_q2", "bid_q3"]].head(5).to_string())