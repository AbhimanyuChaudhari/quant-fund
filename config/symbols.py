from datetime import date
import pandas as pd

# ─────────────────────────────────────
# Index futures — always track
# ─────────────────────────────────────
INDEX_FUTURES = [
    "NIFTY",
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
    "NIFTYNXT50",
]

# ─────────────────────────────────────
# Top 50 most liquid stock futures
# Sorted by typical OI and volume
# ─────────────────────────────────────
STOCK_FUTURES_TIER1 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BAJFINANCE", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "WIPRO", "HCLTECH", "MARUTI",
    "SUNPHARMA", "TATAMOTORS", "TATASTEEL", "NTPC", "POWERGRID",
    "ULTRACEMCO", "TECHM", "BAJAJFINSV", "TITAN", "NESTLEIND",
    "ADANIENT", "ADANIPORTS", "COALINDIA", "ONGC", "JSWSTEEL",
    "GRASIM", "INDUSINDBK", "DIVISLAB", "DRREDDY", "CIPLA",
    "EICHERMOT", "HEROMOTOCO", "APOLLOHOSP", "TATACONSUM", "BRITANNIA",
    "BPCL", "IOC", "HINDALCO", "VEDL", "SAIL",
    "PNB", "BANKBARODA", "CANBK", "FEDERALBNK", "IDFCFIRSTB",
]

# ─────────────────────────────────────
# All F&O stocks — collect everything
# ─────────────────────────────────────
STOCK_FUTURES_TIER2 = [
    "AUROPHARMA", "BIOCON", "CADILAHC", "LUPIN", "TORNTPHARM",
    "ALKEM", "ABBOTINDIA", "PFIZER", "GLAXO", "SANOFI",
    "MOTHERSON", "BALKRISIND", "EXIDEIND", "AMARAJABAT", "MINDA",
    "ASHOKLEY", "TVSMOTOR", "BAJAJ-AUTO", "M&M", "ESCORTS",
    "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "PHOENIXLTD",
    "HAVELLS", "VOLTAS", "BLUESTARCO", "CROMPTON", "POLYCAB",
    "PIDILITIND", "ASIANPAINT", "BERGERPAINTS", "KANSAINER", "AKZOINDIA",
    "MUTHOOTFIN", "CHOLAFIN", "BAJAJHLDNG", "LICHSGFIN", "RECLTD",
    "PFC", "IRFC", "HUDCO", "NBCC", "BEL",
    "HAL", "BDL", "COCHINSHIP", "GRSE", "MAZAGON",
]

# Combined — everything
ALL_FUTURES = INDEX_FUTURES + STOCK_FUTURES_TIER1 + STOCK_FUTURES_TIER2


def get_active_symbols(kite, tier: str = "all") -> list[dict]:
    """
    Fetch nearest expiry futures for specified tier.
    tier: "index", "tier1", "tier2", "all"
    """
    if tier == "index":
        track = INDEX_FUTURES
    elif tier == "tier1":
        track = INDEX_FUTURES + STOCK_FUTURES_TIER1
    elif tier == "tier2":
        track = STOCK_FUTURES_TIER2
    else:
        track = ALL_FUTURES

    instruments = pd.DataFrame(kite.instruments("NFO"))
    futures     = instruments[
        instruments["instrument_type"] == "FUT"
    ].copy()
    futures["expiry"] = pd.to_datetime(futures["expiry"])

    today  = pd.Timestamp(date.today())
    active = []

    for underlying in track:
        contracts = futures[
            futures["tradingsymbol"].str.startswith(underlying)
        ].copy()
        valid = contracts[contracts["expiry"] >= today]

        if valid.empty:
            continue

        nearest = valid.sort_values("expiry").iloc[0]
        active.append({
            "tradingsymbol":    nearest["tradingsymbol"],
            "instrument_token": int(nearest["instrument_token"]),
            "exchange":         "NFO",
            "expiry":           nearest["expiry"].strftime("%Y-%m-%d"),
            "lot_size":         int(nearest["lot_size"]),
        })

    print(f"Active symbols: {len(active)} contracts")
    return active


def get_token_map(active_symbols: list[dict]) -> dict:
    return {s["instrument_token"]: s["tradingsymbol"]
            for s in active_symbols}