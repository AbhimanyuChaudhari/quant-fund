import os
import time
import pyotp
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
# from NorenRestApiPy.NorenApi import NorenApi
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


class ShoonyaBroker(BaseBroker):

    def __init__(self):
        from src.collection.brokers.api_helper import ShoonyaApiPy
        self.api     = ShoonyaApiPy()
        self.user_id     = None
        self.instruments = None

    def login(self):
        try:
            user_id     = get_secret("SHOONYA_USER_ID")
            password    = get_secret("SHOONYA_PASSWORD")
            totp_secret = get_secret("SHOONYA_TOTP_SECRET")
            vendor_code = get_secret("SHOONYA_VENDOR_CODE")
            api_key     = get_secret("SHOONYA_API_KEY")
            imei        = get_secret("SHOONYA_IMEI")
            print("Shoonya credentials loaded from Secret Manager.")
        except Exception:
            user_id     = os.getenv("SHOONYA_USER_ID")
            password    = os.getenv("SHOONYA_PASSWORD")
            totp_secret = os.getenv("SHOONYA_TOTP_SECRET")
            vendor_code = os.getenv("SHOONYA_VENDOR_CODE")
            api_key     = os.getenv("SHOONYA_API_KEY")
            imei        = os.getenv("SHOONYA_IMEI")
            print("Shoonya credentials loaded from .env")

        # Generate TOTP automatically
        totp = pyotp.TOTP(totp_secret).now()

        response = self.api.login(
            userid=user_id,
            password=password,
            twoFA=totp,
            vendor_code=vendor_code,
            api_secret=api_key,
            imei=imei
        )

        if response is None or response.get("stat") != "Ok":
            raise ValueError(f"Shoonya login failed: {response}")

        self.user_id = user_id
        print(f"Shoonya login successful. User: {user_id}")

    def get_active_symbols(self) -> list[dict]:
        """
        Shoonya uses exchange|token format.
        NFO instruments need to be fetched differently.
        """
        # Load NFO instrument list
        # Shoonya provides instrument master files
        import requests
        import gzip
        import io

        url = "https://api.shoonya.com/NFO_symbols.txt.gz"
        response = requests.get(url)
        content  = gzip.decompress(response.content).decode("utf-8")

        instruments = pd.read_csv(
            io.StringIO(content),
            header=None,
            names=[
                "exchange", "token", "lot_size", "symbol",
                "tradingsymbol", "expiry", "strike",
                "option_type", "tick_size"
            ]
        )

        # Filter futures only
        futures = instruments[
            instruments["option_type"] == "XX"
        ].copy()

        futures["expiry"] = pd.to_datetime(
            futures["expiry"], format="%d-%b-%Y", errors="coerce"
        )

        today  = pd.Timestamp(date.today())
        active = []

        for underlying in FUTURES_TO_TRACK:
            contracts = futures[
                futures["symbol"] == underlying
            ].copy()
            valid = contracts[contracts["expiry"] >= today]

            if valid.empty:
                print(f"[WARN] No valid contracts for {underlying}")
                continue

            nearest = valid.sort_values("expiry").iloc[0]
            active.append({
                "tradingsymbol":    nearest["tradingsymbol"],
                "instrument_token": str(nearest["token"]),
                "exchange":         "NFO",
                "expiry":           nearest["expiry"].strftime("%Y-%m-%d"),
                "lot_size":         int(nearest["lot_size"]),
                "shoonya_key":      f"NFO|{nearest['token']}",
            })
            print(f"Active: {nearest['tradingsymbol']} "
                  f"(expires {nearest['expiry'].strftime('%Y-%m-%d')})")

        return active

    def start_websocket(self, on_tick, on_connect,
                        on_error, on_close,
                        on_reconnect, on_noreconnect):
        self.api.start_websocket(
            subscribe_callback=on_tick,
            socket_open_callback=on_connect,
            socket_close_callback=on_close,
            socket_error_callback=on_error
        )

    def subscribe(self, tokens: list):
        """
        Shoonya tokens are strings in format NFO|TOKEN
        """
        self.api.subscribe(tokens)

    def stop(self):
        self.api.close_websocket()
