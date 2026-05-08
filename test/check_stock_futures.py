import gcsfs
from datetime import date, timedelta

PROJECT_ID  = "hedge-fund-494103"
BUCKET_NAME = "hedge-fund-494103-marketdata"

fs = gcsfs.GCSFileSystem(project=PROJECT_ID)

print("=== Stock Futures Processed Data ===\n")

# Get all processed futures files
all_processed = fs.glob(
    f"{BUCKET_NAME}/processed/features/*26MAYFUT/*.parquet"
)

# Group by symbol
from collections import defaultdict
symbol_dates = defaultdict(list)
for f in all_processed:
    parts  = f.split("/")
    symbol = parts[3]
    date_  = parts[4].replace(".parquet", "")
    symbol_dates[symbol].append(date_)

# Filter to stock futures only (not index)
index_futures = {
    "NIFTY26MAYFUT", "BANKNIFTY26MAYFUT",
    "FINNIFTY26MAYFUT", "MIDCPNIFTY26MAYFUT",
    "SENSEX26MAYFUT", "NIFTYNXT5026MAYFUT"
}

stock_futures = {
    sym: sorted(dates)
    for sym, dates in symbol_dates.items()
    if sym not in index_futures
    and len(dates) >= 1
}

print(f"Stock futures with processed data: {len(stock_futures)}\n")

# Show top ones by number of dates
sorted_stocks = sorted(
    stock_futures.items(),
    key=lambda x: len(x[1]),
    reverse=True
)

print(f"{'Symbol':<25} {'Dates':>6}  {'Available dates'}")
print("-" * 65)
for sym, dates in sorted_stocks[:20]:
    print(f"  {sym:<23} {len(dates):>6}  {', '.join(dates)}")

print(f"\nRun backtest on any of these:")
print(f"  python backtest.py --symbol RELIANCE26MAYFUT \\")
print(f"    --start 2026-05-05 --end 2026-05-05 --realistic")