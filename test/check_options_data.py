import duckdb
import gcsfs
import datetime

PROJECT_ID  = "hedge-fund-494103"
BUCKET_NAME = "hedge-fund-494103-marketdata"

fs  = gcsfs.GCSFileSystem(project=PROJECT_ID)
con = duckdb.connect()
con.register_filesystem(fs)

# First find ALL options folders that have data today
all_folders = fs.ls(f"{BUCKET_NAME}/raw/orderbook/")
options_folders = [f.split('/')[-1] for f in all_folders 
                   if f.split('/')[-1].endswith(('CE', 'PE'))]

print(f"Total options symbols in GCS: {len(options_folders)}")
print(f"\nChecking data for today (2026-05-01)...\n")

print(f"{'Symbol':<32} {'Files':>6} {'Ticks':>8} {'First':>12} {'Last':>12}")
print("-" * 78)

for sym in sorted(options_folders):
    files = fs.glob(f"{BUCKET_NAME}/raw/orderbook/{sym}/2026-05-01/*.parquet")
    if not files:
        continue

    # Read all files by loading each individually and summing
    total_ticks = 0
    first_ts    = None
    last_ts     = None

    for f in files:
        try:
            row = con.execute(f"""
                SELECT COUNT(*), MIN(ts_local_ns), MAX(ts_local_ns)
                FROM read_parquet('gs://{f}')
            """).fetchone()
            total_ticks += row[0]
            if row[1]:
                first_ts = min(first_ts, row[1]) if first_ts else row[1]
                last_ts  = max(last_ts,  row[2]) if last_ts  else row[2]
        except:
            continue

    if total_ticks == 0:
        continue

    def to_ist(ns):
        sec = ns // 1_000_000_000
        return datetime.datetime.utcfromtimestamp(sec + 19800).strftime('%H:%M:%S')

    first_str = to_ist(first_ts) if first_ts else "?"
    last_str  = to_ist(last_ts)  if last_ts  else "?"

    print(f"{sym:<32} {len(files):>6} {total_ticks:>8,} {first_str:>12} {last_str:>12}")