import duckdb
import gcsfs

PROJECT_ID  = "hedge-fund-494103"
BUCKET_NAME = "hedge-fund-494103-marketdata-mumbai"

fs  = gcsfs.GCSFileSystem(project=PROJECT_ID)
con = duckdb.connect()
con.register_filesystem(fs)

# Check all available dates and row counts for NIFTY
for date in ["2026-04-29", "2026-04-30"]:
    gcs_path = f"gs://{BUCKET_NAME}/processed/features/NIFTY26MAYFUT/{date}.parquet"
    try:
        count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{gcs_path}')").fetchone()[0]
        first = con.execute(f"SELECT ts_sec FROM read_parquet('{gcs_path}') ORDER BY ts_sec LIMIT 1").fetchone()[0]
        last  = con.execute(f"SELECT ts_sec FROM read_parquet('{gcs_path}') ORDER BY ts_sec DESC LIMIT 1").fetchone()[0]
        print(f"{date}: {count:,} rows | first ts: {first} | last ts: {last}")
    except Exception as e:
        print(f"{date}: ERROR - {e}")

# Also show full column list and a sample from Apr 30
print("\n--- Sample from Apr 30 ---")
gcs_path = f"gs://{BUCKET_NAME}/processed/features/NIFTY26MAYFUT/2026-04-30.parquet"
df = con.execute(f"""
    SELECT symbol, ts_sec, open, high, low, close, volume, 
           spread_mean, imbalance_last, realized_vol_60s
    FROM read_parquet('{gcs_path}')
    ORDER BY ts_sec
    LIMIT 10
""").df()
print(df.to_string())
