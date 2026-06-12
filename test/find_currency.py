from src.utils.auth import get_kite_client

kite = get_kite_client()

# Get CDS (Currency Derivative Segment) instruments
instruments = kite.instruments('CDS')

import pandas as pd
df = pd.DataFrame(instruments)

print(f"Total CDS instruments: {len(df)}")
print(f"\nInstrument types: {df['instrument_type'].unique()}")
print(f"\nAll currency pairs available:")
print(df['name'].unique())

# Filter USDINR
usdinr = df[df['name'] == 'USDINR'].copy()
usdinr['expiry'] = pd.to_datetime(usdinr['expiry'])

import datetime
today = pd.Timestamp(datetime.date.today())
active = usdinr[usdinr['expiry'] >= today].sort_values('expiry')

print(f"\n--- Active USDINR contracts ---")
for _, row in active.head(10).iterrows():
    print(f"  {row['tradingsymbol']:<20} type={row['instrument_type']:<4} "
          f"expiry={row['expiry'].strftime('%Y-%m-%d')} "
          f"token={row['instrument_token']}")

# Also check EURINR, GBPINR
for pair in ['EURINR', 'GBPINR', 'JPYINR']:
    pair_df = df[df['name'] == pair]
    active_pair = pair_df[pd.to_datetime(pair_df['expiry']) >= today]
    print(f"\n{pair}: {len(active_pair)} active contracts")
    if len(active_pair) > 0:
        nearest = active_pair.sort_values('expiry').iloc[0]
        print(f"  Nearest: {nearest['tradingsymbol']} "
              f"token={nearest['instrument_token']}")
