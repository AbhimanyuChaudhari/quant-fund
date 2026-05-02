from src.utils.auth import get_kite_client
import pandas as pd
import datetime

kite = get_kite_client()
instruments = kite.instruments('NFO')
df = pd.DataFrame(instruments)
df['expiry'] = pd.to_datetime(df['expiry'])

today = pd.Timestamp(datetime.date.today())

# Find all NIFTY options expiries
nifty_opts = df[
    (df['name'] == 'NIFTY') &
    (df['instrument_type'].isin(['CE', 'PE'])) &
    (df['expiry'] >= today)
].copy()

expiries = sorted(nifty_opts['expiry'].unique())

print("=== All NIFTY Options Expiries ===")
for e in expiries[:10]:
    count    = len(nifty_opts[nifty_opts['expiry'] == e])
    day_name = e.strftime('%A')
    is_weekly = day_name == 'Thursday'
    tag = '← WEEKLY (0DTE candidate)' if is_weekly else ''
    print(f"  {e.strftime('%Y-%m-%d')} ({day_name:<9}) {count:>4} strikes  {tag}")

# Show ATM strikes for nearest Thursday expiry
thursdays = [e for e in expiries if e.weekday() == 3]  # 3 = Thursday
if thursdays:
    nearest_thu = thursdays[0]
    thu_opts = nifty_opts[nifty_opts['expiry'] == nearest_thu]

    # Get current NIFTY price
    try:
        ltp = kite.ltp('NSE:NIFTY 50')
        nifty_price = list(ltp.values())[0]['last_price']
    except:
        nifty_price = 24000

    atm = round(nifty_price / 50) * 50

    print(f"\n=== ATM±5 strikes for {nearest_thu.strftime('%Y-%m-%d')} ===")
    print(f"NIFTY price: {nifty_price}")
    atm_opts = thu_opts[
        (thu_opts['strike'] >= atm - 250) &
        (thu_opts['strike'] <= atm + 250)
    ].sort_values(['strike', 'instrument_type'])

    for _, row in atm_opts.iterrows():
        print(f"  {row['tradingsymbol']:<25} strike={row['strike']:>7.0f} "
              f"type={row['instrument_type']} token={row['instrument_token']}")

    print(f"\nTotal ATM±250 strikes: {len(atm_opts)}")
    print(f"Days to expiry: {(nearest_thu - today).days}")