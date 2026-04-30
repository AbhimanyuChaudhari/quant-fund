import duckdb
import gcsfs
from fsspec import filesystem

# Use existing Google ADC credentials — no HMAC keys needed
fs = gcsfs.GCSFileSystem(project='hedge-fund-494103')

con = duckdb.connect()
con.register_filesystem(fs)

result = con.execute("""
    SELECT COUNT(*) as total_ticks
    FROM read_parquet(
        'gs://hedge-fund-494103-marketdata/raw/orderbook/NIFTY26MAYFUT/2026-04-30/*.parquet'
    )
""").df()

print("Total ticks:", result)