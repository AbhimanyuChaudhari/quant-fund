"""
Live Trading Monitor — CLI Dashboard

Real-time view during market hours.

Usage:
    python monitor.py                # full dashboard, refresh every 10s
    python monitor.py --interval 5  # refresh every 5s
"""

import os
import sys
import time
import json
import argparse
import gcsfs
from datetime import datetime, date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

PROJECT_ID  = "hedge-fund-494103"
BUCKET_NAME = "hedge-fund-494103-marketdata-mumbai"

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def colored(text, color):
    return f"{color}{text}{RESET}"


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


# ─────────────────────────────────────
# GCS checks (with timeout)
# ─────────────────────────────────────
def _count_files(fs, today):
    """Count files by category — runs in thread with timeout."""
    result = {
        "futures":  0, "options": 0,
        "currency": 0, "spot":    0,
        "latest_age": None,
    }
    try:
        folders = [f.split("/")[-1]
                   for f in fs.ls(f"{BUCKET_NAME}/raw/orderbook/")]
        now = time.time()

        for folder in folders:
            files = fs.glob(
                f"{BUCKET_NAME}/raw/orderbook/{folder}/{today}/*.parquet"
            )
            if not files:
                continue

            if folder.endswith("FUT") and "USDINR" not in folder:
                result["futures"] += len(files)
            elif folder.endswith(("CE", "PE")):
                result["options"] += len(files)
            elif "USDINR" in folder:
                result["currency"] += len(files)
            elif folder.endswith("_SPOT"):
                result["spot"] += len(files)

            # Track latest file age
            try:
                latest = sorted(files)[-1]
                info   = fs.info(latest)
                mtime  = info.get("updated") or info.get("timeCreated")
                if mtime and hasattr(mtime, "timestamp"):
                    age = now - mtime.timestamp()
                    if result["latest_age"] is None or age < result["latest_age"]:
                        result["latest_age"] = age
            except Exception:
                pass

    except Exception as e:
        result["error"] = str(e)

    return result


def get_collector_status(today: str, timeout: int = 15) -> dict:
    """Get collector status with timeout to prevent hanging."""
    fs = gcsfs.GCSFileSystem(project=PROJECT_ID)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_count_files, fs, today)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            return {"error": "Timeout — GCS slow to respond"}
        except Exception as e:
            return {"error": str(e)}


def count_processed(today: str, timeout: int = 10) -> dict:
    """Count processed files with timeout."""
    def _count():
        fs = gcsfs.GCSFileSystem(project=PROJECT_ID)
        futures = fs.glob(
            f"{BUCKET_NAME}/processed/features/*/{today}.parquet"
        )
        options = fs.glob(
            f"{BUCKET_NAME}/processed/options/*/{today}.parquet"
        )
        return {"futures": len(futures), "options": len(options)}

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_count)
        try:
            return future.result(timeout=timeout)
        except Exception:
            return {"futures": 0, "options": 0}


# ─────────────────────────────────────
# Portfolio / trade logs
# ─────────────────────────────────────
def load_portfolio() -> dict:
    path = Path("logs/portfolio_state.json")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_recent_trades(n: int = 5) -> list:
    path = Path(f"logs/trades/{date.today()}.csv")
    if not path.exists():
        return []
    try:
        import csv
        with open(path) as f:
            rows = list(csv.DictReader(f))
        return rows[-n:]
    except Exception:
        return []


def load_recent_orders(n: int = 5) -> list:
    path = Path(f"logs/orders/{date.today()}.csv")
    if not path.exists():
        return []
    try:
        import csv
        with open(path) as f:
            rows = list(csv.DictReader(f))
        return rows[-n:]
    except Exception:
        return []


# ─────────────────────────────────────
# IST time helpers
# ─────────────────────────────────────
def get_ist_seconds() -> int:
    utc = (datetime.utcnow().hour * 3600 +
           datetime.utcnow().minute * 60 +
           datetime.utcnow().second)
    return (utc + 19800) % 86400


def format_ist(ist_sec: int) -> str:
    h = ist_sec // 3600
    m = (ist_sec % 3600) // 60
    s = ist_sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def market_status() -> tuple[str, str]:
    """Returns (NSE status, CDS status)."""
    ist = get_ist_seconds()
    nse_open  = 9 * 3600 + 15 * 60
    nse_close = 15 * 3600 + 30 * 60
    cds_open  = 9 * 3600
    cds_close = 17 * 3600

    if nse_open <= ist <= nse_close:
        rem = nse_close - ist
        nse = f"OPEN ({rem//3600}h {(rem%3600)//60}m left)"
    else:
        to_open = (nse_open - ist) % 86400
        nse = f"CLOSED (opens in {to_open//3600}h {(to_open%3600)//60}m)"

    if cds_open <= ist <= cds_close:
        rem = cds_close - ist
        cds = f"OPEN ({rem//3600}h {(rem%3600)//60}m left)"
    else:
        cds = "CLOSED"

    return nse, cds


# ─────────────────────────────────────
# Render
# ─────────────────────────────────────
def render(refresh: int = 10):
    today   = date.today().strftime("%Y-%m-%d")
    now_str = datetime.utcnow().strftime("%H:%M:%S UTC")
    ist_str = format_ist(get_ist_seconds())

    clear_screen()
    print(colored("=" * 60, CYAN))
    print(colored(f"  QUANT FUND MONITOR  |  {now_str}  |  IST {ist_str}", BOLD))
    print(colored(f"  {today}  |  refresh={refresh}s", CYAN))
    print(colored("=" * 60, CYAN))

    # ── Collector ─────────────────────────────────
    print(f"\n{colored('COLLECTOR', BOLD)}  (fetching...)", end="", flush=True)
    status = get_collector_status(today)
    print(f"\r{colored('COLLECTOR', BOLD)}              ")
    print("-" * 40)

    if "error" in status:
        print(f"  {colored(status['error'], RED)}")
    else:
        age = status.get("latest_age")
        age_str = f"{age:.0f}s ago" if age else "no files yet"

        for cat in ["futures", "options", "currency", "spot"]:
            count = status.get(cat, 0)
            if count > 0:
                col = colored("LIVE", GREEN)
            else:
                col = colored("NO DATA", RED)
            print(f"  {cat:<12} {col:<20} {count:>5} files")

        print(f"  Latest file:  {age_str}")

    # ── Processed ─────────────────────────────────
    print(f"\n{colored('PROCESSED TODAY', BOLD)}")
    print("-" * 40)
    proc = count_processed(today)
    print(f"  Futures:  {proc['futures']} symbols")
    print(f"  Options:  {proc['options']} underlyings")

    # ── Portfolio ─────────────────────────────────
    print(f"\n{colored('PORTFOLIO', BOLD)}")
    print("-" * 40)
    port = load_portfolio()

    if port and port.get("date") == today:
        pnl   = port.get("realized_pnl", 0)
        col   = GREEN if pnl >= 0 else RED
        print(f"  Realized PnL:  {colored(f'Rs.{pnl:+,.0f}', col)}")
        positions = {s: p for s, p in port.get("positions", {}).items()
                     if p.get("lots", 0) != 0}
        if positions:
            for sym, pos in positions.items():
                print(f"  Position:      {sym} {pos['lots']:+d}L "
                      f"@ {pos['avg_price']:.2f}")
        else:
            print(f"  Position:      {colored('FLAT', GREEN)}")
    else:
        print(f"  No active session today")
        print(f"  {colored('Run:', YELLOW)} python trade_runner.py "
              f"--symbol NIFTY26MAYFUT --paper-live")

    # ── Recent trades ─────────────────────────────
    trades = load_recent_trades(5)
    if trades:
        print(f"\n{colored('RECENT TRADES', BOLD)}")
        print("-" * 40)
        for t in trades:
            pnl   = float(t.get("pnl", 0))
            col   = GREEN if pnl >= 0 else RED
            mode  = "[P]" if t.get("paper") == "True" else "[L]"
            print(f"  {mode} {t.get('timestamp','')[-8:]}  "
                  f"{t.get('side',''):<5} {t.get('lots','')}L "
                  f"@ {float(t.get('price',0)):.2f}  "
                  f"{colored(f'Rs.{pnl:+,.0f}', col)}")

    # ── Recent orders ─────────────────────────────
    orders = load_recent_orders(5)
    if orders:
        print(f"\n{colored('RECENT ORDERS', BOLD)}")
        print("-" * 40)
        for o in orders:
            st  = o.get("status", "")
            col = GREEN if st == "FILLED" else \
                  YELLOW if st == "PENDING" else RED
            print(f"  {o.get('timestamp','')[-8:]}  "
                  f"{o.get('side',''):<5} {o.get('lots','')}L "
                  f"@ {float(o.get('price',0)):.2f}  "
                  f"{colored(st, col)}")

    # ── Market hours ──────────────────────────────
    print(f"\n{colored('MARKET STATUS', BOLD)}")
    print("-" * 40)
    nse, cds = market_status()
    nse_col = GREEN if "OPEN" in nse else RED
    cds_col = GREEN if "OPEN" in cds else RED
    print(f"  NSE (equity):   {colored(nse, nse_col)}")
    print(f"  CDS (currency): {colored(cds, cds_col)}")

    print(f"\n{colored('Ctrl+C to exit', YELLOW)}")
    print(colored("=" * 60, CYAN))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()

    print("Starting monitor...")
    try:
        while True:
            render(args.interval)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
