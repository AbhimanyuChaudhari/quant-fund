from src.utils.auth import get_kite_client
import pandas as pd
import datetime

kite = get_kite_client()
instruments = kite.instruments('CDS')
df = pd.DataFrame(instruments)
df['expiry'] = pd.to_datetime(df['expiry'])

today = pd.Timestamp(datetime.date.today())

# ── USDINR Options — find actual strikes available ────
usdinr_opts = df[
    (df['name'] == 'USDINR') &
    (df['instrument_type'].isin(['CE', 'PE'])) &
    (df['expiry'] >= today)
].copy()

nearest_expiry = usdinr_opts['expiry'].min()
nearest_opts   = usdinr_opts[
    usdinr_opts['expiry'] == nearest_expiry
].sort_values('strike')

print(f"Nearest expiry: {nearest_expiry.strftime('%Y-%m-%d')}")
print(f"Strike range: {nearest_opts['strike'].min()} → {nearest_opts['strike'].max()}")
print(f"Total options: {len(nearest_opts)}")
print(f"\nSample strikes (middle of chain):")

# Show middle 20 strikes
mid = len(nearest_opts) // 2
sample = nearest_opts.iloc[mid-10:mid+10]
for _, row in sample.iterrows():
    print(f"  {row['tradingsymbol']:<25} strike={row['strike']:>8.4f} "
          f"type={row['instrument_type']} token={row['instrument_token']}")

# ── Get current USDINR price via LTP ─────────────────
print("\n=== Current USDINR rate ===")
usdinr_fut = df[
    (df['name'] == 'USDINR') &
    (df['instrument_type'] == 'FUT') &
    (df['expiry'] >= today)
].sort_values('expiry').iloc[0]

try:
    ltp = kite.ltp(f"CDS:{usdinr_fut['tradingsymbol']}")
    price = list(ltp.values())[0]['last_price']
    print(f"  {usdinr_fut['tradingsymbol']}: ₹{price}")
    print(f"  ATM strike would be: {round(price * 4) / 4:.2f} (nearest 0.25)")
except Exception as e:
    print(f"  Could not fetch LTP: {e}")

# ── Lot size clarification ────────────────────────────
print("\n=== Contract specs (real) ===")
print(f"  Zerodha lot_size field: {usdinr_fut['lot_size']}")
print(f"  Actual contract size:   1000 USD (NSE standard)")
print(f"  Tick size:              0.0025 paise")
print(f"  Actual notional:        ~₹84,500 per contract at 84.50")
