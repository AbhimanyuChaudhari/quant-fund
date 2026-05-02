from src.utils.auth import get_kite_client

kite = get_kite_client()
instruments = kite.instruments('NFO')

# Filter NIFTY options expiring this month
nifty_opts = [i for i in instruments 
              if i['name'] == 'NIFTY' 
              and i['instrument_type'] in ('CE', 'PE')
              and '26MAY' in i['tradingsymbol']]

# Show strikes around ATM (NIFTY ~24000)
atm_opts = [i for i in nifty_opts 
            if 23000 <= i['strike'] <= 25000]

atm_opts.sort(key=lambda x: (x['strike'], x['instrument_type']))
for i in atm_opts:
    print(f"{i['tradingsymbol']:<30} strike={i['strike']} type={i['instrument_type']} token={i['instrument_token']}")

print(f"\nTotal: {len(atm_opts)} instruments")
print(f"\nAll available NIFTY May expiries:")
expiries = sorted(set(i['expiry'] for i in nifty_opts))
for e in expiries:
    count = sum(1 for i in nifty_opts if i['expiry'] == e)
    print(f"  {e}  ({count} strikes)")