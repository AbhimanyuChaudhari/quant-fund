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
# All F&O stocks
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

ALL_FUTURES = INDEX_FUTURES + STOCK_FUTURES_TIER1 + STOCK_FUTURES_TIER2

# ─────────────────────────────────────
# Options config
# ─────────────────────────────────────
OPTIONS_CONFIG = {
    "NIFTY": {
        "strikes_each_side": 10,   # ATM ± 10 strikes
        "strike_interval":   50,   # 50pt intervals
        "lot_size":          75,
    },
    "BANKNIFTY": {
        "strikes_each_side": 10,
        "strike_interval":   100,  # 100pt intervals
        "lot_size":          35,
    },
}


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
            "instrument_type":  "FUT",
            "underlying":       underlying,
        })

    print(f"Active futures: {len(active)} contracts")
    return active


def get_active_options(kite,
                       underlyings: list[str] = ["NIFTY", "BANKNIFTY"],
                       num_expiries: int = 1) -> list[dict]:
    """
    Fetch liquid option strikes for specified underlyings.
    Selects ATM ± N strikes for the nearest N expiries.

    Args:
        kite:        authenticated KiteConnect client
        underlyings: list of underlying names e.g. ['NIFTY', 'BANKNIFTY']
        num_expiries: how many expiries to collect (1=nearest only)

    Returns:
        List of dicts same structure as get_active_symbols()
    """
    instruments = pd.DataFrame(kite.instruments("NFO"))
    options     = instruments[
        instruments["instrument_type"].isin(["CE", "PE"])
    ].copy()
    options["expiry"] = pd.to_datetime(options["expiry"])

    # Also get futures to find current ATM price
    futures = instruments[
        instruments["instrument_type"] == "FUT"
    ].copy()
    futures["expiry"] = pd.to_datetime(futures["expiry"])

    today  = pd.Timestamp(date.today())
    active = []

    for underlying in underlyings:
        cfg = OPTIONS_CONFIG.get(underlying)
        if not cfg:
            print(f"No options config for {underlying}, skipping")
            continue

        # Get current price from nearest futures contract
        fut_contracts = futures[
            futures["tradingsymbol"].str.startswith(underlying)
        ]
        valid_futs = fut_contracts[fut_contracts["expiry"] >= today]
        if valid_futs.empty:
            print(f"No active futures for {underlying}, skipping")
            continue

        nearest_fut = valid_futs.sort_values("expiry").iloc[0]
        # Use last price if available, else use a rough ATM estimate
        # We'll compute ATM from strike_interval rounding
        # Actual price will be fetched at runtime — for now pick middle of range

        # Get available expiries for this underlying's options
        und_options = options[
            options["tradingsymbol"].str.startswith(underlying)
        ]
        expiries = sorted(und_options[und_options["expiry"] >= today]["expiry"].unique())

        if not expiries:
            print(f"No active options for {underlying}")
            continue

        # Take nearest N expiries
        selected_expiries = expiries[:num_expiries]

        for expiry in selected_expiries:
            exp_options = und_options[und_options["expiry"] == expiry]

            # Get all available strikes
            strikes = sorted(exp_options["strike"].unique())
            if not strikes:
                continue

            # Find ATM strike (middle of available strikes as proxy)
            # Collector will dynamically pick liquid strikes at runtime
            # For now select full ATM ± N range
            mid_strike   = strikes[len(strikes) // 2]
            interval     = cfg["strike_interval"]
            n            = cfg["strikes_each_side"]

            # Round mid to nearest interval
            atm = round(mid_strike / interval) * interval

            target_strikes = [atm + i * interval
                              for i in range(-n, n + 1)]

            for strike in target_strikes:
                for opt_type in ["CE", "PE"]:
                    match = exp_options[
                        (exp_options["strike"] == strike) &
                        (exp_options["instrument_type"] == opt_type)
                    ]
                    if match.empty:
                        continue

                    row = match.iloc[0]
                    active.append({
                        "tradingsymbol":    row["tradingsymbol"],
                        "instrument_token": int(row["instrument_token"]),
                        "exchange":         "NFO",
                        "expiry":           expiry.strftime("%Y-%m-%d"),
                        "lot_size":         cfg["lot_size"],
                        "instrument_type":  opt_type,
                        "underlying":       underlying,
                        "strike":           strike,
                    })

    print(f"Active options: {len(active)} contracts "
          f"({underlyings}, {num_expiries} expir{'y' if num_expiries==1 else 'ies'})")
    return active


def get_token_map(symbols: list[dict]) -> dict:
    """Map instrument_token → tradingsymbol for all symbols."""
    return {s["instrument_token"]: s["tradingsymbol"] for s in symbols}


if __name__ == "__main__":
    from src.utils.auth import get_kite_client
    kite = get_kite_client()

    print("=== Futures ===")
    futures = get_active_symbols(kite, tier="index")
    for s in futures:
        print(f"  {s['tradingsymbol']:<30} token={s['instrument_token']}")

    print("\n=== Options (NIFTY ATM ± 10) ===")
    options = get_active_options(kite, underlyings=["NIFTY"], num_expiries=1)
    for s in options:
        print(f"  {s['tradingsymbol']:<30} strike={s['strike']} "
              f"type={s['instrument_type']} token={s['instrument_token']}")

    print(f"\nTotal instruments: {len(futures) + len(options)}")