"""
Batch Backtest — All Stock Futures
Runs realistic backtest on all stock futures and ranks by profitability.

Usage:
    python backtest_all_futures.py
    python backtest_all_futures.py --start 2026-04-29 --end 2026-05-07
    python backtest_all_futures.py --min-bars 500
"""

import argparse
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
import gcsfs

PROJECT_ID  = "hedge-fund-494103"
BUCKET_NAME = "hedge-fund-494103-marketdata"

INDEX_FUTURES = {
    "NIFTY26MAYFUT", "BANKNIFTY26MAYFUT",
    "FINNIFTY26MAYFUT", "MIDCPNIFTY26MAYFUT",
    "SENSEX26MAYFUT", "NIFTYNXT5026MAYFUT",
    "BANKEX26MAYFUT",
}

# ─────────────────────────────────────
# Load credentials ONCE at startup
# ─────────────────────────────────────
_lot_sizes = {}

def init():
    """Load lot sizes once — called once before the loop."""
    global _lot_sizes
    try:
        from src.utils.auth import get_kite_client
        import pandas as pd
        print("Loading lot sizes from Zerodha...")
        kite = get_kite_client()
        df   = pd.DataFrame(kite.instruments("NFO"))
        df   = df[df["instrument_type"] == "FUT"]
        _lot_sizes = {
            row["name"]: int(row["lot_size"])
            for _, row in df.iterrows()
        }
        print(f"Loaded {len(_lot_sizes)} lot sizes\n")
    except Exception as e:
        print(f"Could not load lot sizes ({e}) — using default 75\n")


def get_symbols(start: str, end: str) -> list:
    """Get all stock futures with processed data in date range."""
    fs    = gcsfs.GCSFileSystem(project=PROJECT_ID)
    files = fs.glob(
        f"{BUCKET_NAME}/processed/features/*26MAYFUT/*.parquet"
    )
    symbols = set()
    for f in files:
        parts    = f.split("/")
        sym      = parts[3]
        date_str = parts[4].replace(".parquet", "")
        if sym not in INDEX_FUTURES and start <= date_str <= end:
            symbols.add(sym)
    return sorted(symbols)


# ─────────────────────────────────────
# Single symbol backtest
# ─────────────────────────────────────
def run_one(symbol: str, start: str, end: str) -> dict:
    """Run exactly like backtest.py --realistic does."""
    try:
        from src.backtest.strategy import StrategyConfig
        from src.backtest.mm_strategy import AvellanedaStoikovStrategy
        from src.backtest.engine import BacktestEngine, BacktestConfig
        from src.trading.fill_simulator import RealisticOrderBook

        # Lot size from cache (no API call)
        name     = symbol.replace("26MAYFUT", "")
        lot_size = _lot_sizes.get(name, 75)

        strat_config = StrategyConfig(
            symbol        = symbol,
            lot_size      = lot_size,
            max_inventory = 5,
        )
        strategy = AvellanedaStoikovStrategy(
            config     = strat_config,
            gamma      = 0.1,
            kappa      = 1.5,
            min_spread = 0.5,
            max_spread = 10.0,
        )
        bt_config = BacktestConfig(
            symbol        = symbol,
            start         = start,
            end           = end,
            lot_size      = lot_size,
            max_inventory = 5,
        )
        order_book = RealisticOrderBook(
            max_inventory    = 5,
            lot_size         = lot_size,
            queue_aggression = 0.3,
        )

        # Run silently
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            engine  = BacktestEngine(bt_config, strategy,
                                     order_book=order_book)
            metrics = engine.run()

        fill_rate = (
            order_book.fill_successes /
            max(order_book.fill_attempts, 1) * 100
        )

        # Get bars from engine or estimate
        bars = getattr(engine, 'bars_processed',
               getattr(metrics, 'bars_processed',
               metrics.total_fills * 50))

        return {
            "symbol":    symbol,
            "lot_size":  lot_size,
            "gross_pnl": round(metrics.gross_pnl, 2),
            "costs":     round(abs(metrics.total_costs), 2),
            "net_pnl":   round(metrics.net_pnl, 2),
            "fills":     metrics.total_fills,
            "fill_rate": round(fill_rate, 2),
            "win_rate":  round(metrics.win_rate * 100, 1),
            "sharpe":    round(metrics.sharpe_ratio, 2),
            "max_dd":    round(abs(metrics.max_drawdown), 2),
            "bars":      bars,
            "ok":        True,
        }

    except Exception as e:
        return {
            "symbol":  symbol,
            "net_pnl": 0,
            "bars":    0,
            "fills":   0,
            "ok":      False,
            "error":   str(e)[:80],
        }


# ─────────────────────────────────────
# Main
# ─────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Batch backtest all stock futures"
    )
    parser.add_argument("--start",    default="2026-04-29")
    parser.add_argument("--end",      default="2026-05-07")
    parser.add_argument("--min-bars", type=int, default=100,
                        help="Min bars to include symbol (default 100)")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  Batch Backtest — All Stock Futures")
    print(f"  Period: {args.start} → {args.end}")
    print(f"{'='*65}\n")

    # Load credentials once
    init()

    # Get symbols
    print("Finding available symbols...")
    symbols = get_symbols(args.start, args.end)
    print(f"Found {len(symbols)} stock futures\n")

    if not symbols:
        print("No data found.")
        return

    # ── Run backtests ──────────────────────────────────
    print(f"{'#':>3} {'Symbol':<28} {'NetPnL':>12} "
          f"{'Fills':>6} {'Rate':>6} {'Bars':>7}")
    print("-" * 65)

    results = []
    errors  = []

    for i, sym in enumerate(symbols, 1):
        print(f"{i:>3} {sym:<28}", end="", flush=True)
        r = run_one(sym, args.start, args.end)

        if not r["ok"]:
            errors.append(r)
            print(f"  ERROR: {r.get('error', '')[:40]}")
            continue

        if r["bars"] < args.min_bars:
            print(f"  SKIP  ({r['bars']} bars)")
            continue

        results.append(r)
        sign = "+" if r["net_pnl"] >= 0 else ""
        print(f"  Rs.{sign}{r['net_pnl']:>9,.0f}  "
              f"{r['fills']:>5}  "
              f"{r['fill_rate']:>5.1f}%  "
              f"{r['bars']:>6}")

    if not results:
        print("\nNo results after filtering.")
        return

    # Sort by net PnL
    results.sort(key=lambda x: x["net_pnl"], reverse=True)
    profitable = [r for r in results if r["net_pnl"] > 0]

    # ── Full ranking ───────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  FULL RANKING — {len(results)} symbols")
    print(f"{'='*80}")
    print(f"\n  {'#':>3} {'Symbol':<25} {'Lot':>4} "
          f"{'Gross':>11} {'Costs':>10} {'Net':>11} "
          f"{'Win%':>6} {'Sharpe':>7}")
    print("  " + "-" * 80)

    for i, r in enumerate(results, 1):
        sign   = "+" if r["net_pnl"] >= 0 else ""
        marker = " ✓" if r["net_pnl"] > 0 else ""
        print(f"  {i:>3} {r['symbol']:<25} {r['lot_size']:>4} "
              f"Rs.{r['gross_pnl']:>9,.0f} "
              f"Rs.{r['costs']:>8,.0f} "
              f"Rs.{sign}{r['net_pnl']:>9,.0f} "
              f"{r['win_rate']:>5.1f}% "
              f"{r['sharpe']:>7.2f}"
              f"{marker}")

    # ── Profitable summary ─────────────────────────────
    print(f"\n{'='*80}")
    print(f"  PROFITABLE: {len(profitable)} / {len(results)} symbols")
    print(f"{'='*80}\n")

    if profitable:
        total = sum(r["net_pnl"] for r in profitable)
        print(f"  Combined net PnL if trading all: Rs.+{total:,.0f}\n")
        print(f"  {'Symbol':<25} {'Lot':>4} {'Net PnL':>12} "
              f"{'Fill%':>7} {'Win%':>6} {'MaxDD':>10}")
        print("  " + "-" * 68)
        for r in profitable:
            print(f"  {r['symbol']:<25} {r['lot_size']:>4} "
                  f"Rs.+{r['net_pnl']:>10,.0f} "
                  f"{r['fill_rate']:>6.1f}% "
                  f"{r['win_rate']:>5.1f}% "
                  f"-Rs.{r['max_dd']:>7,.0f}")

    # ── Errors ─────────────────────────────────────────
    if errors:
        print(f"\n  {len(errors)} symbols had errors:")
        for e in errors[:5]:
            print(f"    {e['symbol']}: {e.get('error', 'unknown')}")

    print(f"\n  Tested: {len(results)} | "
          f"Profitable: {len(profitable)} | "
          f"Skipped: {len(symbols)-len(results)-len(errors)} | "
          f"Errors: {len(errors)}\n")


if __name__ == "__main__":
    main()