"""
Batch Backtest — All Stock Futures (Fast Parallel Version)
==========================================================
Runs realistic backtest on all stock futures using:
  - New optimized params (gamma=0.001, min_spread=0.10, kappa=1.5)
  - Local DuckDB cache (500x faster reads if synced)
  - Parallel execution (4 workers by default)

Usage:
    python backtest_all_futures.py --start 2026-04-30 --end 2026-04-30
    python backtest_all_futures.py --start 2026-05-13 --end 2026-05-13
    python backtest_all_futures.py --workers 8
    python backtest_all_futures.py --gamma 0.001 --min-spread 0.10 --kappa 1.5
"""

import argparse
import io
import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from multiprocessing import Pool, cpu_count
import gcsfs

PROJECT_ID  = "hedge-fund-494103"
BUCKET_NAME = "hedge-fund-494103-marketdata"

INDEX_FUTURES = {
    "NIFTY26MAYFUT", "BANKNIFTY26MAYFUT",
    "FINNIFTY26MAYFUT", "MIDCPNIFTY26MAYFUT",
    "SENSEX26MAYFUT", "NIFTYNXT5026MAYFUT",
    "BANKEX26MAYFUT",
}

# Global state shared across workers
_lot_sizes  = {}
_run_params = {}


def init():
    """Load lot sizes once before the loop."""
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
        print(f"Could not load lot sizes ({e}) — using defaults\n")


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


def worker_init(lot_sizes: dict, run_params: dict):
    """Initialize each worker process with shared state."""
    global _lot_sizes, _run_params
    _lot_sizes  = lot_sizes
    _run_params = run_params


def run_one(args_tuple: tuple) -> dict:
    """Run backtest for one symbol. Called by worker pool."""
    symbol, start, end = args_tuple

    try:
        from src.backtest.strategy import StrategyConfig
        from src.backtest.mm_strategy import AvellanedaStoikovStrategy
        from src.backtest.engine import BacktestEngine, BacktestConfig
        from src.trading.fill_simulator import RealisticOrderBook

        name     = symbol.replace("26MAYFUT", "")
        lot_size = _lot_sizes.get(name, 75)

        gamma      = _run_params.get("gamma",      0.001)
        kappa      = _run_params.get("kappa",      1.5)
        min_spread = _run_params.get("min_spread", 0.10)
        max_spread = _run_params.get("max_spread", 10.0)
        queue_agg  = _run_params.get("queue_agg",  0.3)
        max_inv    = _run_params.get("max_inv",    5)

        strat_config = StrategyConfig(
            symbol        = symbol,
            lot_size      = lot_size,
            max_inventory = max_inv,
        )
        strategy = AvellanedaStoikovStrategy(
            config     = strat_config,
            gamma      = gamma,
            kappa      = kappa,
            min_spread = min_spread,
            max_spread = max_spread,
        )
        bt_config = BacktestConfig(
            symbol        = symbol,
            start         = start,
            end           = end,
            lot_size      = lot_size,
            max_inventory = max_inv,
        )
        order_book = RealisticOrderBook(
            max_inventory    = max_inv,
            lot_size         = lot_size,
            queue_aggression = queue_agg,
        )

        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            engine  = BacktestEngine(bt_config, strategy,
                                     order_book=order_book)
            metrics = engine.run()

        fill_rate = (
            order_book.fill_successes /
            max(order_book.fill_attempts, 1) * 100
        )

        bars = getattr(engine, "bars_processed",
               getattr(metrics, "bars_processed",
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


def print_results(results: list, errors: list, total_symbols: int):
    """Print full ranking and profitable summary."""
    if not results:
        print("\nNo results after filtering.")
        return

    results.sort(key=lambda x: x["net_pnl"], reverse=True)
    profitable = [r for r in results if r["net_pnl"] > 0]

    print(f"\n{'='*80}")
    print(f"  FULL RANKING -- {len(results)} symbols")
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
              f"{r['sharpe']:>7.2f}{marker}")

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

    if errors:
        print(f"\n  {len(errors)} errors:")
        for e in errors[:5]:
            print(f"    {e['symbol']}: {e.get('error', 'unknown')}")

    skipped = total_symbols - len(results) - len(errors)
    print(f"\n  Tested: {len(results)} | "
          f"Profitable: {len(profitable)} | "
          f"Skipped: {skipped} | "
          f"Errors: {len(errors)}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fast parallel batch backtest — all stock futures"
    )
    parser.add_argument("--start",      default="2026-04-30")
    parser.add_argument("--end",        default="2026-04-30")
    parser.add_argument("--min-bars",   type=int,   default=100)
    parser.add_argument("--workers",    type=int,
                        default=min(4, cpu_count()))
    parser.add_argument("--gamma",      type=float, default=0.001)
    parser.add_argument("--kappa",      type=float, default=1.5)
    parser.add_argument("--min-spread", type=float, default=0.10)
    parser.add_argument("--max-spread", type=float, default=10.0)
    parser.add_argument("--queue-aggression", type=float, default=0.3)
    parser.add_argument("--max-inventory",    type=int,   default=5)
    args = parser.parse_args()

    run_params = {
        "gamma":      args.gamma,
        "kappa":      args.kappa,
        "min_spread": args.min_spread,
        "max_spread": args.max_spread,
        "queue_agg":  args.queue_aggression,
        "max_inv":    args.max_inventory,
    }

    print(f"\n{'='*70}")
    print(f"  Batch Backtest -- All Stock Futures")
    print(f"  Period:  {args.start} -> {args.end}")
    print(f"  Params:  gamma={args.gamma} min_spread={args.min_spread} "
          f"kappa={args.kappa}")
    print(f"  Workers: {args.workers}")
    print(f"{'='*70}\n")

    init()

    print("Finding available symbols...")
    symbols = get_symbols(args.start, args.end)
    print(f"Found {len(symbols)} stock futures\n")

    if not symbols:
        print("No data found.")
        return

    work_items = [(sym, args.start, args.end) for sym in symbols]

    print(f"{'#':>3} {'Symbol':<28} {'NetPnL':>12} "
          f"{'Fills':>6} {'Rate':>6} {'Bars':>7}")
    print("-" * 65)

    results = []
    errors  = []
    counter = [0]

    def handle_result(r):
        counter[0] += 1
        i = counter[0]
        if not r["ok"]:
            errors.append(r)
            print(f"{i:>3} {r['symbol']:<28}  ERROR: "
                  f"{r.get('error','')[:35]}")
            return
        if r["bars"] < args.min_bars:
            print(f"{i:>3} {r['symbol']:<28}  SKIP ({r['bars']} bars)")
            return
        results.append(r)
        sign = "+" if r["net_pnl"] >= 0 else ""
        print(f"{i:>3} {r['symbol']:<28}  "
              f"Rs.{sign}{r['net_pnl']:>9,.0f}  "
              f"{r['fills']:>5}  "
              f"{r['fill_rate']:>5.1f}%  "
              f"{r['bars']:>6}")

    if args.workers > 1:
        with Pool(
            processes=args.workers,
            initializer=worker_init,
            initargs=(_lot_sizes, run_params),
        ) as pool:
            for r in pool.imap_unordered(run_one, work_items):
                handle_result(r)
    else:
        worker_init(_lot_sizes, run_params)
        for item in work_items:
            handle_result(run_one(item))

    print_results(results, errors, len(symbols))


if __name__ == "__main__":
    main()