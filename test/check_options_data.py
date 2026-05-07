import gcsfs
from datetime import date

PROJECT_ID  = "hedge-fund-494103"
BUCKET_NAME = "hedge-fund-494103-marketdata"

fs    = gcsfs.GCSFileSystem(project=PROJECT_ID)
today = date.today().strftime("%Y-%m-%d")

print(f"=== Checking options data for {today} ===\n")

# Check NIFTY options
nifty_files = fs.glob(
    f"{BUCKET_NAME}/raw/orderbook/NIFTY265*/{today}/*.parquet"
)
bnifty_files = fs.glob(
    f"{BUCKET_NAME}/raw/orderbook/BANKNIFTY265*/{today}/*.parquet"
)

print(f"NIFTY options files:     {len(nifty_files)}")
print(f"BANKNIFTY options files: {len(bnifty_files)}")

if nifty_files:
    # Show which symbols have data
    symbols = set(f.split("/")[3] for f in nifty_files)
    print(f"\nNIFTY symbols with data ({len(symbols)}):")
    for s in sorted(symbols)[:10]:
        count = sum(1 for f in nifty_files if f"/raw/orderbook/{s}/" in f)
        print(f"  {s}: {count} files")
    if len(symbols) > 10:
        print(f"  ... and {len(symbols)-10} more")
else:
    print("\nNo NIFTY options data yet")
    print("Possible reasons:")
    print("  1. Collector just restarted — wait 60s for first flush")
    print("  2. Token issue — check collector logs")
    print("  3. Options not subscribed — check collector config")

# Check USDINR
usdinr_files = fs.glob(
    f"{BUCKET_NAME}/raw/orderbook/USDINR*/{today}/*.parquet"
)
print(f"\nUSDINR files: {len(usdinr_files)}")

# Check futures for comparison
nifty_fut = fs.glob(
    f"{BUCKET_NAME}/raw/orderbook/NIFTY26MAYFUT/{today}/*.parquet"
)
print(f"NIFTY futures files: {len(nifty_fut)}  ← if this is 0, collector is down")