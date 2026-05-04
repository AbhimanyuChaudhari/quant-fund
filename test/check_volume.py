import duckdb
import gcsfs

fs  = gcsfs.GCSFileSystem(project='hedge-fund-494103')
con = duckdb.connect()
con.register_filesystem(fs)

df = con.execute("""
    SELECT
        AVG(volume)     as avg_volume,
        MAX(volume)     as max_volume,
        MIN(volume)     as min_volume,
        AVG(tick_count) as avg_ticks,
        AVG(bid_q1)     as avg_bid_q1,
        AVG(ask_q1)     as avg_ask_q1,
        AVG(total_bid_qty) as avg_total_bid,
        COUNT(*)        as total_bars
    FROM read_parquet('gs://hedge-fund-494103-marketdata/processed/features/NIFTY26MAYFUT/2026-04-30.parquet')
""").df()

print("NIFTY Apr 30 volume stats:")
for col in df.columns:
    print(f"  {col}: {df[col].iloc[0]:,.2f}")