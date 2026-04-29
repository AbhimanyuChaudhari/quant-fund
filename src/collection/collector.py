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
from src.storage.gcs import upload_dataframe
from src.collection.brokers import ZerodhaBroker, ShoonyaBroker

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# ─────────────────────────────────────
# Config — change this one line to
# switch brokers
# ─────────────────────────────────────
ACTIVE_BROKER = "zerodha"  # "zerodha" or "shoonya"
FLUSH_INTERVAL = 60

# ─────────────────────────────────────
# RAM buffer
# ─────────────────────────────────────
buffer      = []
buffer_lock = threading.Lock()
TOKENS      = []
TOKEN_TO_SYMBOL = {}


# ─────────────────────────────────────
# Tick parsers
# ─────────────────────────────────────
def parse_zerodha_tick(tick: dict) -> dict:
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
        "instrument_token": str(tick["instrument_token"]),
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


def parse_shoonya_tick(tick: dict) -> dict:
    """
    Shoonya tick format is different from Zerodha.
    Fields: tk (token), lp (last price), bp1-bp10, sp1-sp10,
            bq1-bq10, sq1-sq10, v (volume), oi, etc.
    """
    def safe_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    def safe_int(val):
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

    token     = tick.get("tk", "")
    symbol    = TOKEN_TO_SYMBOL.get(token, "")
    last_price = safe_float(tick.get("lp", 0))

    # Extract up to 10 levels each side
    bids = []
    asks = []
    for i in range(1, 11):
        bid_p = safe_float(tick.get(f"bp{i}", 0))
        bid_q = safe_int(tick.get(f"bq{i}", 0))
        ask_p = safe_float(tick.get(f"sp{i}", 0))
        ask_q = safe_int(tick.get(f"sq{i}", 0))
        bids.append({"price": bid_p, "quantity": bid_q})
        asks.append({"price": ask_p, "quantity": ask_q})

    best_bid      = bids[0]["price"]
    best_ask      = asks[0]["price"]
    mid           = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
    spread        = best_ask - best_bid if best_bid and best_ask else 0
    total_bid_qty = sum(b["quantity"] for b in bids)
    total_ask_qty = sum(a["quantity"] for a in asks)
    total_qty     = total_bid_qty + total_ask_qty
    imbalance     = (total_bid_qty - total_ask_qty) / total_qty if total_qty else 0

    row = {
        "ts_local_ns":      time.time_ns(),
        "ts_exchange":      str(tick.get("ft", "")),
        "ts_trade":         str(tick.get("ltt", "")),
        "symbol":           symbol,
        "instrument_token": token,
        "last_price":       last_price,
        "last_qty":         safe_int(tick.get("ls", 0)),
        "avg_price":        safe_float(tick.get("ap", 0)),
        "volume":           safe_int(tick.get("v", 0)),
        "open":             safe_float(tick.get("o", 0)),
        "high":             safe_float(tick.get("h", 0)),
        "low":              safe_float(tick.get("l", 0)),
        "close":            safe_float(tick.get("c", 0)),
        "oi":               safe_int(tick.get("oi", 0)),
        "oi_day_high":      safe_int(tick.get("poi", 0)),
        "oi_day_low":       0,
        "total_bid_qty":    total_bid_qty,
        "total_ask_qty":    total_ask_qty,
        "mid_price":        round(mid, 2),
        "spread":           round(spread, 2),
        "book_imbalance":   round(imbalance, 6),
    }

    # Add 10 levels for Shoonya (vs 5 for Zerodha)
    for i, (bid, ask) in enumerate(zip(bids, asks), 1):
        row[f"bid_p{i}"] = bid["price"]
        row[f"bid_q{i}"] = bid["quantity"]
        row[f"ask_p{i}"] = ask["price"]
        row[f"ask_q{i}"] = ask["quantity"]

    return row


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
def make_on_ticks(broker_name: str):
    def on_ticks(ws, ticks):
        with buffer_lock:
            for tick in ticks:
                if broker_name == "zerodha":
                    if tick["instrument_token"] in TOKENS:
                        buffer.append(parse_zerodha_tick(tick))
                else:
                    token = tick.get("tk", "")
                    if token in TOKENS:
                        buffer.append(parse_shoonya_tick(tick))
    return on_ticks


def make_on_connect(broker, broker_name: str):
    def on_connect(ws, response):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Connected.")
        broker.subscribe(TOKENS)
        symbols = [TOKEN_TO_SYMBOL.get(t, t) for t in TOKENS]
        print(f"Subscribed to {symbols}")
    return on_connect


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
    global TOKENS, TOKEN_TO_SYMBOL

    # Initialize broker
    if ACTIVE_BROKER == "zerodha":
        broker = ZerodhaBroker()
    else:
        broker = ShoonyaBroker()

    # Login
    broker.login()

    # Get active symbols
    symbols = broker.get_active_symbols(tier="all")
    TOKEN_TO_SYMBOL = {s["instrument_token"]: s["tradingsymbol"]
                       for s in symbols}
    TOKENS          = list(TOKEN_TO_SYMBOL.keys())

    # Start flush thread
    threading.Thread(target=flush_loop, daemon=True).start()
    print(f"Flush thread started. Interval: {FLUSH_INTERVAL}s")
    print(f"Broker: {ACTIVE_BROKER.upper()}")
    print(f"Collecting: {[s['tradingsymbol'] for s in symbols]}")
    print("Press Ctrl+C to stop.\n")

    # Start WebSocket
    broker.start_websocket(
        on_tick      = make_on_ticks(ACTIVE_BROKER),
        on_connect   = make_on_connect(broker, ACTIVE_BROKER),
        on_error     = on_error,
        on_close     = on_close,
        on_reconnect = on_reconnect,
        on_noreconnect = on_noreconnect
    )


if __name__ == "__main__":
    run_collector()