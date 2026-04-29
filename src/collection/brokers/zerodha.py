import os
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from kiteconnect import KiteConnect, KiteTicker
from src.collection.brokers.base import BaseBroker
from src.utils.auth import get_secret

ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

FUTURES_TO_TRACK = [
    "NIFTY",
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
    "NIFTYNXT50",
]


class ZerodhaBroker(BaseBroker):

    def __init__(self):
        self.api_key      = None
        self.access_token = None
        self.kite         = None
        self.ticker       = None

    def login(self):
        try:
            self.api_key      = get_secret("KITE_API_KEY")
            self.access_token = get_secret("KITE_ACCESS_TOKEN")
            print("Zerodha credentials loaded from Secret Manager.")
        except Exception:
            self.api_key      = os.getenv("KITE_API_KEY")
            self.access_token = os.getenv("KITE_ACCESS_TOKEN")
            print("Zerodha credentials loaded from .env")

        self.kite = KiteConnect(api_key=self.api_key)
        self.kite.set_access_token(self.access_token)
        print("Zerodha login successful.")

    def get_active_symbols(self) -> list[dict]:
        instruments = pd.DataFrame(self.kite.instruments("NFO"))
        futures     = instruments[
            instruments["instrument_type"] == "FUT"
        ].copy()
        futures["expiry"] = pd.to_datetime(futures["expiry"])

        today  = pd.Timestamp(date.today())
        active = []

        for underlying in FUTURES_TO_TRACK:
            contracts = futures[
                futures["tradingsymbol"].str.startswith(underlying + "2")
            ].copy()
            valid = contracts[contracts["expiry"] >= today]

            if valid.empty:
                print(f"[WARN] No valid contracts for {underlying}")
                continue

            nearest = valid.sort_values("expiry").iloc[0]
            active.append({
                "tradingsymbol":    nearest["tradingsymbol"],
                "instrument_token": int(nearest["instrument_token"]),
                "exchange":         "NFO",
                "expiry":           nearest["expiry"].strftime("%Y-%m-%d"),
                "lot_size":         int(nearest["lot_size"]),
            })
            print(f"Active: {nearest['tradingsymbol']} "
                  f"(expires {nearest['expiry'].strftime('%Y-%m-%d')})")

        return active

    def start_websocket(self, on_tick, on_connect,
                        on_error, on_close,
                        on_reconnect, on_noreconnect):
        self.ticker = KiteTicker(self.api_key, self.access_token)
        self.ticker.on_ticks       = on_tick
        self.ticker.on_connect     = on_connect
        self.ticker.on_error       = on_error
        self.ticker.on_close       = on_close
        self.ticker.on_reconnect   = on_reconnect
        self.ticker.on_noreconnect = on_noreconnect
        self.ticker.connect(threaded=False)

    def subscribe(self, tokens: list[int]):
        if self.ticker:
            self.ticker.subscribe(tokens)
            self.ticker.set_mode(self.ticker.MODE_FULL, tokens)

    def stop(self):
        if self.ticker:
            self.ticker.stop()