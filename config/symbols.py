from datetime import date
import pandas as pd

# These never change — the underlying instruments
FUTURES_TO_TRACK = [
    "NIFTY",
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
    "NIFTYNXT50",
]


def get_active_symbols(kite) -> list[dict]:
    """
    Fetches current NFO instruments from Zerodha and returns
    the nearest expiry futures contract for each symbol.
    Automatically handles rollover — always picks front month.
    """
    instruments = pd.DataFrame(kite.instruments("NFO"))
    
    # Filter to futures only
    futures = instruments[instruments["instrument_type"] == "FUT"].copy()
    futures["expiry"] = pd.to_datetime(futures["expiry"])
    
    today  = pd.Timestamp(date.today())
    active = []

    for underlying in FUTURES_TO_TRACK:
        # Get all futures for this underlying
        contracts = futures[
            futures["tradingsymbol"].str.startswith(underlying + "2")
        ].copy()

        # Filter out expired contracts
        valid = contracts[contracts["expiry"] >= today]

        if valid.empty:
            print(f"[WARN] No valid contracts found for {underlying}")
            continue

        # Pick nearest expiry
        nearest = valid.sort_values("expiry").iloc[0]

        active.append({
            "tradingsymbol":    nearest["tradingsymbol"],
            "instrument_token": int(nearest["instrument_token"]),
            "exchange":         "NFO",
            "expiry":           nearest["expiry"].strftime("%Y-%m-%d"),
            "lot_size":         int(nearest["lot_size"]),
        })

        print(f"Active contract: {nearest['tradingsymbol']} "
              f"(expires {nearest['expiry'].strftime('%Y-%m-%d')})")

    return active


def get_token_map(active_symbols: list[dict]) -> dict:
    """Returns {instrument_token: tradingsymbol} lookup."""
    return {s["instrument_token"]: s["tradingsymbol"] for s in active_symbols}