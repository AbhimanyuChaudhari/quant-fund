import os
import json
from pathlib import Path
from dotenv import load_dotenv
from kiteconnect import KiteTicker
from src.utils.auth import get_kite_client

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Get authenticated client
kite = get_kite_client()

# We will test with NIFTY 50 index
# Instrument token for NIFTY 50 is 256265
NIFTY_TOKEN = 17072898

def on_ticks(ws, ticks):
    """Called every time a tick arrives."""
    for tick in ticks:
        print("\n" + "="*60)
        print(f"Symbol:     {tick.get('tradingsymbol', 'NIFTY26APRFUT')}")
        print(f"Last Price: {tick.get('last_price')}")
        print(f"Mode:       {tick.get('mode')}")
        
        # This is what we care about — how many depth levels
        depth = tick.get('depth', {})
        buy_levels  = depth.get('buy', [])
        sell_levels = depth.get('sell', [])
        
        print(f"\nBuy levels received:  {len(buy_levels)}")
        print(f"Sell levels received: {len(sell_levels)}")
        
        print("\nBUY SIDE:")
        for i, level in enumerate(buy_levels):
            print(f"  Level {i+1}: Price={level.get('price')} Qty={level.get('quantity')} Orders={level.get('orders')}")
        
        print("\nSELL SIDE:")
        for i, level in enumerate(sell_levels):
            print(f"  Level {i+1}: Price={level.get('price')} Qty={level.get('quantity')} Orders={level.get('orders')}")
        
        print("\nFULL RAW TICK:")
        print(json.dumps(tick, indent=2, default=str))
        
        # Stop after first tick so we can read the output
        ws.stop()


def on_connect(ws, response):
    print("WebSocket connected.")
    print("Subscribing to NIFTY 50 in FULL mode...")
    ws.subscribe([NIFTY_TOKEN])
    ws.set_mode(ws.MODE_FULL, [NIFTY_TOKEN])


def on_error(ws, code, reason):
    print(f"Error: {code} - {reason}")


def on_close(ws, code, reason):
    print(f"Closed: {code} - {reason}")


if __name__ == "__main__":
    api_key = os.getenv("KITE_API_KEY")
    access_token = os.getenv("KITE_ACCESS_TOKEN")

    ticker = KiteTicker(api_key, access_token)
    ticker.on_ticks   = on_ticks
    ticker.on_connect = on_connect
    ticker.on_error   = on_error
    ticker.on_close   = on_close

    print("Starting WebSocket — waiting for first tick...")
    print("NOTE: Market must be open for ticks to arrive.")
    print("Indian market hours: 9:15am - 3:30pm IST")
    print("That is 11:45pm - 5:00am EST\n")

    ticker.connect(threaded=False)