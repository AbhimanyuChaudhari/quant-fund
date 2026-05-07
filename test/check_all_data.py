import gcsfs
from datetime import date, timedelta

PROJECT_ID  = "hedge-fund-494103"
BUCKET_NAME = "hedge-fund-494103-marketdata"

fs = gcsfs.GCSFileSystem(project=PROJECT_ID)

print("=== Data Collection Summary ===\n")

# Check last 7 days
dates = [(date.today() - timedelta(days=i)).strftime("%Y-%m-%d")
         for i in range(6, -1, -1)]

print(f"{'Date':<12} {'Futures':>8} {'Options':>8} {'USDINR':>8} {'Status'}")
print("-" * 50)

for d in dates:
    futures = len(fs.glob(
        f"{BUCKET_NAME}/raw/orderbook/NIFTY26MAYFUT/{d}/*.parquet"
    ))
    options = len(fs.glob(
        f"{BUCKET_NAME}/raw/orderbook/NIFTY265*/{d}/*.parquet"
    ))
    usdinr = len(fs.glob(
        f"{BUCKET_NAME}/raw/orderbook/USDINR*/{d}/*.parquet"
    ))

    if futures > 10:
        status = "GOOD"
    elif futures > 0:
        status = "PARTIAL"
    else:
        status = "NO DATA"

    print(f"{d:<12} {futures:>8} {options:>8} {usdinr:>8}  {status}")

print("\n=== Processed Features ===\n")
processed = fs.glob(
    f"{BUCKET_NAME}/processed/features/NIFTY26MAYFUT/*.parquet"
)
print(f"NIFTY processed dates: {len(processed)}")
for f in sorted(processed):
    print(f"  {f.split('/')[-1]}")