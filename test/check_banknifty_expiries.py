from src.utils.auth import get_kite_client
import pandas as pd
import datetime

kite = get_kite_client()
df   = pd.DataFrame(kite.instruments('NFO'))
df['expiry'] = pd.to_datetime(df['expiry'])
today = pd.Timestamp(datetime.date.today())

bnf = df[
    (df['name'] == 'BANKNIFTY') &
    (df['instrument_type'].isin(['CE', 'PE'])) &
    (df['expiry'] >= today)
]

print("=== BANKNIFTY Options Expiries ===")
expiries = sorted(bnf['expiry'].unique())
for e in expiries[:8]:
    count = len(bnf[bnf['expiry'] == e])
    day   = e.strftime("%A")
    print(f"  {e.strftime('%Y-%m-%d')} ({day:<9}) {count} strikes")

print(f"\nNearest expiry: {expiries[0].strftime('%Y-%m-%d')}")

# Also check what BANKNIFTY ATM looks like
try:
    fut = df[
        (df['name'] == 'BANKNIFTY') &
        (df['instrument_type'] == 'FUT') &
        (df['expiry'] >= today)
    ].sort_values('expiry').iloc[0]

    ltp   = kite.ltp(f"NFO:{fut['tradingsymbol']}")
    price = list(ltp.values())[0]['last_price']
    print(f"BANKNIFTY current price: {price}")
    print(f"ATM strike (nearest 100): {round(price/100)*100}")
except Exception as e:
    print(f"Could not fetch LTP: {e}")