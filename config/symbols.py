# Active symbols to collect
# Update instrument_token each expiry rollover

SYMBOLS = [
    {
        "tradingsymbol": "NIFTY26APRFUT",
        "instrument_token": 17072898,
        "exchange": "NFO",
        "expiry": "2026-04-28",
        "next_tradingsymbol": "NIFTY26MAYFUT",
        "next_instrument_token": 16914178,
        "next_expiry": "2026-05-26",
    },
    {
        "tradingsymbol": "BANKNIFTY26APRFUT",
        "instrument_token": 17072130,
        "exchange": "NFO",
        "expiry": "2026-04-28",
        "next_tradingsymbol": "BANKNIFTY26MAYFUT",
        "next_instrument_token": 16913410,
        "next_expiry": "2026-05-26",
    },
    {
        "tradingsymbol": "FINNIFTY26APRFUT",
        "instrument_token": 17072386,
        "exchange": "NFO",
        "expiry": "2026-04-28",
        "next_tradingsymbol": "FINNIFTY26MAYFUT",
        "next_instrument_token": 16913666,
        "next_expiry": "2026-05-26",
    },
]

# Quick lookup by token
TOKEN_TO_SYMBOL = {s["instrument_token"]: s["tradingsymbol"] for s in SYMBOLS}