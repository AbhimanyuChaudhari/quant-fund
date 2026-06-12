from src.utils.auth import get_kite_client
import pandas as pd
import datetime

kite = get_kite_client()
instruments = pd.DataFrame(kite.instruments('NFO'))
instruments['expiry'] = pd.to_datetime(instruments['expiry'])
today = pd.Timestamp(datetime.date.today())

# Top liquid stock options to add
targets = ['RELIANCE', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'INFY']

print("=== Stock Options Availability ===\n")

total = 0
for stock in targets:
    opts = instruments[
        (instruments['name'] == stock) &
        (instruments['instrument_type'].isin(['CE', 'PE'])) &
        (instruments['expiry'] >= today)
    ]

    if opts.empty:
        print(f"{stock}: No options found")
        continue

    # Nearest expiry
    nearest = opts['expiry'].min()
    nearest_opts = opts[opts['expiry'] == nearest]

    # All available strikes
    strikes = sorted(nearest_opts['strike'].unique())
    mid_strike = strikes[len(strikes)//2]

    # Get current price via LTP
    try:
        fut = instruments[
            (instruments['name'] == stock) &
            (instruments['instrument_type'] == 'FUT') &
            (instruments['expiry'] >= today)
        ].sort_values('expiry').iloc[0]

        ltp = kite.ltp(f"NFO:{fut['tradingsymbol']}")
        price = list(ltp.values())[0]['last_price']
        atm   = round(price / (strikes[1]-strikes[0])) * (strikes[1]-strikes[0])
    except:
        price = mid_strike
        atm   = mid_strike

    # ATM ± 5 strikes
    interval   = strikes[1] - strikes[0] if len(strikes) > 1 else 50
    atm_strikes = [s for s in strikes
                   if atm - 5*interval <= s <= atm + 5*interval]
    count = len(atm_strikes) * 2  # CE + PE

    lot_size = int(nearest_opts.iloc[0]['lot_size'])
    total   += count

    print(f"{stock}:")
    print(f"  Price:       {price:.2f}")
    print(f"  ATM strike:  {atm}")
    print(f"  Interval:    {interval}")
    print(f"  ATM±5 opts:  {count} contracts")
    print(f"  Lot size:    {lot_size}")
    print(f"  Expiry:      {nearest.strftime('%Y-%m-%d')}")
    print()

print(f"Total new contracts: {total}")
print(f"Current total:       261")
print(f"New total:           {261 + total}")
print(f"Zerodha limit:       3000")
print(f"Headroom:            {3000 - 261 - total}")
