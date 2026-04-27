import io
import os
import time
import threading
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from kiteconnect import KiteTicker
from src.utils.auth import get_kite_client
from src.storage.gcs import upload_dataframe
from config.symbols import get_active_symbols, get_token_map

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

FLUSH_INTERVAL = 60

# Fetch active symbols at startup — auto handles rollover
kite           = get_kite_client()
SYMBOLS        = get_active_symbols(kite)
TOKEN_TO_SYMBOL = get_token_map(SYMBOLS)
TOKENS         = list(TOKEN_TO_SYMBOL.keys())

# RAM buffer
buffer      = []
buffer_lock = threading.Lock()


# ─────────────────────────────────────
# Tick parser
# ─────────────────────────────────────
def parse_tick(tick: dict) -> dict:
    depth = tick.get("depth", {})
    bids  = depth.get("buy",  [])
    asks  = depth.get("sell", [])

    while len(bids) < 5:
        bids.append({"price": 0, "quantity": 0, "orders": 0})
    while len(asks) < 5:
        asks.append({"price": 0, "quantity": 0, "orders": 0})

    best_bid      = bids[0].get("price", 0)
    best_ask      = asks[0].get("price", 0)
    mid           = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
    spread        = best_ask - best_bid if best_bid and best_ask else 0
    total_bid_qty = tick.get("total_buy_quantity",  0)
    total_ask_qty = tick.get("total_sell_quantity", 0)
    total_qty     = total_bid_qty + total_ask_qty
    imbalance     = (total_bid_qty - total_ask_qty) / total_qty if total_qty else 0

    return {
        "ts_local_ns":      time.time_ns(),
        "ts_exchange":      str(tick.get("exchange_timestamp", "")),
        "ts_trade":         str(tick.get("last_trade_time", "")),
        "symbol":           TOKEN_TO_SYMBOL.get(tick["instrument_token"], ""),
        "instrument_token": tick["instrument_token"],
        "last_price":       tick.get("last_price", 0),
        "last_qty":         tick.get("last_traded_quantity", 0),
        "avg_price":        tick.get("average_traded_price", 0),
        "volume":           tick.get("volume_traded", 0),
        "open":             tick.get("ohlc", {}).get("open", 0),
        "high":             tick.get("ohlc", {}).get("high", 0),
        "low":              tick.get("ohlc", {}).get("low", 0),
        "close":            tick.get("ohlc", {}).get("close", 0),
        "oi":               tick.get("oi", 0),
        "oi_day_high":      tick.get("oi_day_high", 0),
        "oi_day_low":       tick.get("oi_day_low", 0),
        "total_bid_qty":    total_bid_qty,
        "total_ask_qty":    total_ask_qty,
        "mid_price":        round(mid, 2),
        "spread":           round(spread, 2),
        "book_imbalance":   round(imbalance, 6),
        "bid_p1": bids[0].get("price", 0), "bid_q1": bids[0].get("quantity", 0), "bid_o1": bids[0].get("orders", 0),
        "bid_p2": bids[1].get("price", 0), "bid_q2": bids[1].get("quantity", 0), "bid_o2": bids[1].get("orders", 0),
        "bid_p3": bids[2].get("price", 0), "bid_q3": bids[2].get("quantity", 0), "bid_o3": bids[2].get("orders", 0),
        "bid_p4": bids[3].get("price", 0), "bid_q4": bids[3].get("quantity", 0), "bid_o4": bids[3].get("orders", 0),
        "bid_p5": bids[4].get("price", 0), "bid_q5": bids[4].get("quantity", 0), "bid_o5": bids[4].get("orders", 0),
        "ask_p1": asks[0].get("price", 0), "ask_q1": asks[0].get("quantity", 0), "ask_o1": asks[0].get("orders", 0),
        "ask_p2": asks[1].get("price", 0), "ask_q2": asks[1].get("quantity", 0), "ask_o2": asks[1].get("orders", 0),
        "ask_p3": asks[2].get("price", 0), "ask_q3": asks[2].get("quantity", 0), "ask_o3": asks[2].get("orders", 0),
        "ask_p4": asks[3].get("price", 0), "ask_q4": asks[3].get("quantity", 0), "ask_o4": asks[3].get("orders", 0),
        "ask_p5": asks[4].get("price", 0), "ask_q5": asks[4].get("quantity", 0), "ask_o5": asks[4].get("orders", 0),
    }


# ─────────────────────────────────────
# Flush to GCS
# ─────────────────────────────────────
def flush_buffer():
    global buffer

    with buffer_lock:
        if not buffer:
            return
        rows   = buffer.copy()
        buffer = []

    df            = pd.DataFrame(rows)
    now           = datetime.now()
    date_str      = now.strftime("%Y-%m-%d")
    timestamp_str = now.strftime("%H-%M-%S")

    for symbol, group in df.groupby("symbol"):
        blob_name = f"raw/orderbook/{symbol}/{date_str}/{timestamp_str}.parquet"
        try:
            upload_dataframe(group, blob_name)
            print(f"[{timestamp_str}] GCS ← {blob_name} ({len(group)} ticks)")
        except Exception as e:
            print(f"[GCS ERROR] {symbol}: {e}")


def flush_loop():
    while True:
        time.sleep(FLUSH_INTERVAL)
        flush_buffer()


# ─────────────────────────────────────
# WebSocket handlers
# ─────────────────────────────────────
def on_ticks(ws, ticks):
    with buffer_lock:
        for tick in ticks:
            if tick["instrument_token"] in TOKENS:
                buffer.append(parse_tick(tick))


def on_connect(ws, response):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connected.")
    ws.subscribe(TOKENS)
    ws.set_mode(ws.MODE_FULL, TOKENS)
    print(f"Subscribed to {[s['tradingsymbol'] for s in SYMBOLS]}")


def on_error(ws, code, reason):
    print(f"[ERROR] {code}: {reason}")


def on_close(ws, code, reason):
    print(f"[CLOSE] {code}: {reason}")
    flush_buffer()


def on_reconnect(ws, attempts_count):
    print(f"[RECONNECT] Attempt {attempts_count}...")


def on_noreconnect(ws):
    print("[NORECONNECT] Max attempts reached.")
    flush_buffer()


# ─────────────────────────────────────
# Entry point
# ─────────────────────────────────────
def run_collector():
    api_key      = os.getenv("KITE_API_KEY")
    access_token = os.getenv("KITE_ACCESS_TOKEN")

    # Start flush thread
    threading.Thread(target=flush_loop, daemon=True).start()
    print(f"Flush thread started. Interval: {FLUSH_INTERVAL}s")

    # Start WebSocket
    ticker                = KiteTicker(api_key, access_token)
    ticker.on_ticks       = on_ticks
    ticker.on_connect     = on_connect
    ticker.on_error       = on_error
    ticker.on_close       = on_close
    ticker.on_reconnect   = on_reconnect
    ticker.on_noreconnect = on_noreconnect

    print(f"Starting collector → GCS only, no local storage")
    print("Press Ctrl+C to stop.\n")
    ticker.connect(threaded=False)


if __name__ == "__main__":
    run_collector()