import gcsfs
from datetime import date

fs    = gcsfs.GCSFileSystem(project='hedge-fund-494103')
today = '2026-05-08'

symbols = [
    'NIFTY26MAYFUT',
    'BANKNIFTY26MAYFUT',
    'USDINR26508FUT',
    'USDINR26515FUT',
]

for sym in symbols:
    files = sorted(fs.glob(
        f'hedge-fund-494103-marketdata/raw/orderbook/{sym}/{today}/*.parquet'
    ))
    print(f'\n{sym}: {len(files)} files')
    for f in files:
        print(f'  {f.split("/")[-1]}')