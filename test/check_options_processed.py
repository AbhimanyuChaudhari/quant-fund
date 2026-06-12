import gcsfs
from datetime import date, timedelta

PROJECT_ID  = "hedge-fund-494103"
BUCKET_NAME = "hedge-fund-494103-marketdata-mumbai"

fs = gcsfs.GCSFileSystem(project=PROJECT_ID)

print("=== Options Pipeline Status ===\n")

# Check raw options data available
print("Raw options data:")
for days_ago in range(6, -1, -1):
    d = (date.today() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    nifty = len(fs.glob(
        f"{BUCKET_NAME}/raw/orderbook/NIFTY265*/{d}/*.parquet"
    ))
    if nifty > 0:
        print(f"  {d}: {nifty} raw files")

# Check processed options
print("\nProcessed options data:")
proc = fs.glob(f"{BUCKET_NAME}/processed/options/*/*.parquet")
if proc:
    for f in sorted(proc):
        print(f"  {f.split('marketdata/')[-1]}")
else:
    print("  NONE — options pipeline has never run")
    print("  Run: python src/processing/options_pipeline.py")

# Check what dates have enough raw data to process
print("\nDates with enough raw data to process (>100 files):")
for days_ago in range(6, -1, -1):
    d = (date.today() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    nifty = len(fs.glob(
        f"{BUCKET_NAME}/raw/orderbook/NIFTY265*/{d}/*.parquet"
    ))
    if nifty > 100:
        print(f"  {d}: {nifty} files ← can process this")
