"""
Fast Batch Backtest — All Stock Futures
========================================
Supports both V1 (fast Numba engine) and V2 (Ricci Hawkes-Alpha).

Usage:
    # V1 — fast Numba engine (default)
    python backtest_all_futures_fast.py --start 2026-05-18 --end 2026-05-18

    # V2 — Ricci Hawkes-Alpha (slower, uses original engine)
    python backtest_all_futures_fast.py --start 2026-05-18 --end 2026-05-18 --model v2

    # Multi-day
    python backtest_all_futures_fast.py --start 2026-05-13 --end 2026-05-20 --model v2

Speed:
    V1 (Numba):  ~10-15 seconds per day across 84 symbols
    V2 (engine): ~2-3 minutes per day across 84 symbols
"""

import argparse
import time
import gcsfs
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.backtest.simulators.fill_simulator_fast import run_fast_backtest
from src.backtest.data_loader import load_day

PROJECT_ID  = "hedge-fund-494103"
BUCKET_NAME = "hedge-fund-494103-marketdata-mumbai"

INDEX_FUTURES = {
    "NIFTY26MAYFUT", "BANKNIFTY26MAYFUT",
    "FINNIFTY26MAYFUT", "MIDCPNIFTY26MAYFUT",
    "SENSEX26MAYFUT", "NIFTYNXT5026MAYFUT",
    "BANKEX26MAYFUT",
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_lot_sizes() -> dict:
    try:
        from src.utils.auth import get_secret
        from kiteconnect import KiteConnect
        api_key      = get_secret("KITE_API_KEY")
        access_token = get_secret("KITE_ACCESS_TOKEN")
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        df = pd.DataFrame(kite.instruments("NFO"))
        df = df[df["instrument_type"] == "FUT"]
        return {row["name"]: int(row["lot_size"]) for _, row in df.iterrows()}
    except Exception as e:
        print(f"Could not load lot sizes ({e}) — using defaults")
        return {}


def get_symbols(start: str, end: str) -> list:
    fs    = gcsfs.GCSFileSystem(project=PROJECT_ID)
    files = fs.glob(f"{BUCKET_NAME}/processed/features/*26MAYFUT/*.parquet")
    symbols = set()
    for f in files:
        parts    = f.split("/")
        sym      = parts[3]
        date_str = parts[4].replace(".parquet", "")
        if sym not in INDEX_FUTURES and start <= date_str <= end:
            symbols.add(sym)
    return sorted(symbols)


# ─────────────────────────────────────────────────────────────────────────────
# V1 — fast Numba runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one_v1(symbol: str, start: str, end: str,
               lot_sizes: dict, params: dict,
               min_bars: int = 100) -> dict:
    from datetime import datetime, timedelta

    name     = symbol.replace("26MAYFUT", "").replace("26JUNFUT", "")
    lot_size = lot_sizes.get(name, 75)
    inst     = 'currency_futures' if any(
        x in symbol.upper() for x in ['USDINR', 'EURINR']
    ) else 'equity_futures'

    try:
        start_dt = datetime.strptime(start, '%Y-%m-%d')
        end_dt   = datetime.strptime(end,   '%Y-%m-%d')
        frames   = []
        current  = start_dt

        while current <= end_dt:
            date_str = current.strftime('%Y-%m-%d')
            df = load_day(symbol, date_str, market_hours_only=True)
            if not df.empty:
                df['_date'] = date_str
                frames.append(df)
            current += timedelta(days=1)

        if not frames:
            return {'symbol': symbol, 'lot_size': lot_size,
                    'net_pnl': 0, 'bars': 0, 'ok': False, 'error': 'no data'}

        full_df = pd.concat(frames, ignore_index=True).sort_values('ts_sec')

        if len(full_df) < min_bars:
            return {'symbol': symbol, 'lot_size': lot_size,
                    'net_pnl': 0, 'bars': len(full_df), 'ok': False,
                    'error': f'only {len(full_df)} bars'}

        result = run_fast_backtest(
            df               = full_df,
            gamma            = params['gamma'],
            kappa            = params['kappa'],
            min_spread       = params['min_spread'],
            max_spread       = params['max_spread'],
            open_mult        = params.get('open_mult', 2.0),
            lot_size         = lot_size,
            max_inventory    = params.get('max_inv', 5),
            queue_aggression = params.get('queue_agg', 0.3),
            instrument_type  = inst,
        )

        return {
            'symbol':    symbol,
            'lot_size':  lot_size,
            'gross_pnl': round(result.gross_pnl, 2),
            'costs':     round(result.total_costs, 2),
            'net_pnl':   round(result.net_pnl, 2),
            'fills':     result.total_fills,
            'fill_rate': result.fill_rate,
            'win_rate':  result.win_rate,
            'sharpe':    result.sharpe_ratio,
            'max_dd':    result.max_drawdown,
            'bars':      result.bars_processed,
            'ok':        True,
        }

    except Exception as e:
        return {'symbol': symbol, 'lot_size': lot_size,
                'net_pnl': 0, 'bars': 0, 'ok': False, 'error': str(e)[:80]}


# ─────────────────────────────────────────────────────────────────────────────
# V2 — Ricci Hawkes-Alpha runner (uses original BacktestEngine)
# ─────────────────────────────────────────────────────────────────────────────

def run_one_v2(symbol: str, start: str, end: str,
               lot_sizes: dict, params: dict,
               min_bars: int = 100) -> dict:
    from datetime import datetime, timedelta
    from src.backtest.simulators.fill_simulator_v2_fast import run_fast_v2_backtest

    name     = symbol.replace("26MAYFUT", "").replace("26JUNFUT", "")
    lot_size = lot_sizes.get(name, 75)
    inst     = 'currency_futures' if any(
        x in symbol.upper() for x in ['USDINR', 'EURINR']
    ) else 'equity_futures'

    try:
        start_dt = datetime.strptime(start, '%Y-%m-%d')
        end_dt   = datetime.strptime(end,   '%Y-%m-%d')
        frames   = []
        current  = start_dt

        while current <= end_dt:
            date_str = current.strftime('%Y-%m-%d')
            df = load_day(symbol, date_str, market_hours_only=True)
            if not df.empty:
                df['_date'] = date_str
                frames.append(df)
            current += timedelta(days=1)

        if not frames:
            return {'symbol': symbol, 'lot_size': lot_size,
                    'net_pnl': 0, 'bars': 0, 'ok': False, 'error': 'no data'}

        full_df = pd.concat(frames, ignore_index=True).sort_values('ts_sec')

        if len(full_df) < min_bars:
            return {'symbol': symbol, 'lot_size': lot_size,
                    'net_pnl': 0, 'bars': len(full_df), 'ok': False,
                    'error': f'only {len(full_df)} bars'}

        result = run_fast_v2_backtest(
            df                   = full_df,
            beta                 = params.get('beta', 1.0),
            theta                = params.get('theta', 2.0),
            eta                  = params.get('eta', 0.5),
            nu                   = params.get('nu', 0.2),
            rho                  = params.get('rho', 0.30),
            zeta                 = params.get('zeta', 0.5),
            epsilon_plus         = params.get('epsilon_plus', 0.002),
            epsilon_minus        = params.get('epsilon_minus', 0.002),
            beta_kappa           = params.get('beta_kappa', 0.5),
            theta_kappa          = params.get('theta_kappa', 1.5),
            eta_kappa            = params.get('eta_kappa', 0.3),
            nu_kappa             = params.get('nu_kappa', 0.1),
            phi                  = params.get('phi', 0.001),
            min_spread           = params['min_spread'],
            max_spread           = params['max_spread'],
            open_mult            = params.get('open_mult', 2.0),
            lot_size             = lot_size,
            max_inventory        = params.get('max_inv', 5),
            classifier_threshold = params.get('classifier_threshold', 0.50),
            instrument_type      = inst,
        )

        return {
            'symbol':    symbol,
            'lot_size':  lot_size,
            'gross_pnl': round(result.gross_pnl, 2),
            'costs':     round(result.total_costs, 2),
            'net_pnl':   round(result.net_pnl, 2),
            'fills':     result.total_fills,
            'fill_rate': result.fill_rate,
            'win_rate':  result.win_rate,
            'sharpe':    result.sharpe_ratio,
            'max_dd':    result.max_drawdown,
            'bars':      result.bars_processed,
            'ok':        True,
        }

    except Exception as e:
        return {'symbol': symbol, 'lot_size': lot_size,
                'net_pnl': 0, 'bars': 0, 'ok': False, 'error': str(e)[:80]}


# ─────────────────────────────────────────────────────────────────────────────
# Results printer
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: list, errors: list,
                  total_symbols: int, model: str):
    if not results:
        print("\nNo results.")
        return

    results.sort(key=lambda x: x['net_pnl'], reverse=True)
    profitable = [r for r in results if r['net_pnl'] > 0]

    print(f"\n{'='*80}")
    print(f"  FULL RANKING -- {len(results)} symbols  [Model: {model.upper()}]")
    print(f"{'='*80}")
    print(f"\n  {'#':>3} {'Symbol':<25} {'Lot':>4} "
          f"{'Gross':>11} {'Costs':>10} {'Net':>11} "
          f"{'Win%':>6} {'Sharpe':>7}")
    print("  " + "-"*80)

    for i, r in enumerate(results, 1):
        sign   = "+" if r['net_pnl'] >= 0 else ""
        marker = " ✓" if r['net_pnl'] > 0 else ""
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
        total = sum(r['net_pnl'] for r in profitable)
        print(f"  Combined net PnL if trading all: Rs.+{total:,.0f}\n")
        print(f"  {'Symbol':<25} {'Lot':>4} {'Net PnL':>12} "
              f"{'Fills':>7} {'Win%':>6} {'MaxDD':>12}")
        print("  " + "-"*72)
        for r in profitable:
            print(f"  {r['symbol']:<25} {r['lot_size']:>4} "
                  f"Rs.+{r['net_pnl']:>10,.0f} "
                  f"{r['fills']:>7} "
                  f"{r['win_rate']:>5.1f}% "
                  f"-Rs.{abs(r['max_dd']):>9,.0f}")

    if errors:
        print(f"\n  {len(errors)} errors:")
        for e in errors[:5]:
            print(f"    {e['symbol']}: {e.get('error', 'unknown')}")

    skipped = total_symbols - len(results) - len(errors)
    print(f"\n  Tested: {len(results)} | Profitable: {len(profitable)} | "
          f"Skipped: {skipped} | Errors: {len(errors)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch backtest — all stock futures (V1 fast or V2 Ricci)"
    )
    parser.add_argument("--start",            default="2026-05-18")
    parser.add_argument("--end",              default="2026-05-18")
    parser.add_argument("--model",            default="v1",
                        choices=["v1", "v2"],
                        help="v1=fast Numba A-S, v2=Ricci Hawkes-Alpha")
    parser.add_argument("--min-bars",         type=int,   default=100)
    parser.add_argument("--workers",          type=int,   default=8)

    # V1 params
    parser.add_argument("--gamma",            type=float, default=0.001)
    parser.add_argument("--kappa",            type=float, default=1.5)
    parser.add_argument("--min-spread",       type=float, default=0.10)
    parser.add_argument("--max-spread",       type=float, default=10.0)
    parser.add_argument("--open-mult",        type=float, default=2.0)
    parser.add_argument("--queue-aggression", type=float, default=0.3)
    parser.add_argument("--max-inventory",    type=int,   default=5)

    # V2 params
    parser.add_argument("--phi",   type=float, default=0.001)
    parser.add_argument("--rho",   type=float, default=0.30)
    parser.add_argument("--beta",  type=float, default=1.0)
    parser.add_argument("--theta", type=float, default=2.0)
    parser.add_argument("--eta",   type=float, default=0.5)
    parser.add_argument("--nu",    type=float, default=0.2)
    parser.add_argument("--zeta",  type=float, default=0.5)

    args = parser.parse_args()

    params = {
        'gamma':      args.gamma,
        'kappa':      args.kappa,
        'min_spread': args.min_spread,
        'max_spread': args.max_spread,
        'open_mult':  args.open_mult,
        'queue_agg':  args.queue_aggression,
        'max_inv':    args.max_inventory,
        'phi':        args.phi,
        'rho':        args.rho,
        'beta':       args.beta,
        'theta':      args.theta,
        'eta':        args.eta,
        'nu':         args.nu,
        'zeta':       args.zeta,
    }

    t0 = time.perf_counter()

    print(f"\n{'='*70}")
    print(f"  Batch Backtest — All Stock Futures  [Model: {args.model.upper()}]")
    print(f"  Period:  {args.start} -> {args.end}")
    if args.model == 'v1':
        print(f"  Params:  gamma={args.gamma} min_spread={args.min_spread} "
              f"kappa={args.kappa} open_mult={args.open_mult}")
    else:
        print(f"  Params:  phi={args.phi} rho={args.rho} "
              f"beta={args.beta} eta={args.eta} nu={args.nu} "
              f"zeta={args.zeta}")
    print(f"  Workers: {args.workers}")
    print(f"{'='*70}\n")

    print("Loading lot sizes...")
    lot_sizes = get_lot_sizes()

    print("Finding available symbols...")
    symbols = get_symbols(args.start, args.end)
    print(f"Found {len(symbols)} stock futures\n")

    if not symbols:
        print("No data found.")
        return

    results = []
    errors  = []
    counter = [0]

    # Pick runner based on model
    run_one = run_one_v1 if args.model == 'v1' else run_one_v2

    print(f"{'#':>3} {'Symbol':<30} {'NetPnL':>12} "
          f"{'Fills':>6} {'Win%':>6} {'Sharpe':>7}")
    print("-"*65)

    def handle_result(r):
        counter[0] += 1
        i = counter[0]
        if not r['ok']:
            errors.append(r)
            print(f"{i:>3} {r['symbol']:<30}  "
                  f"SKIP ({r.get('error', '')[:40]})")
            return
        results.append(r)
        sign = "+" if r['net_pnl'] >= 0 else ""
        print(f"{i:>3} {r['symbol']:<30}  "
              f"Rs.{sign}{r['net_pnl']:>9,.0f}  "
              f"{r['fills']:>5}  "
              f"{r['win_rate']:>5.1f}%  "
              f"{r['sharpe']:>6.2f}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, sym, args.start, args.end,
                        lot_sizes, params, args.min_bars): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            handle_result(future.result())

    elapsed = time.perf_counter() - t0
    print_results(results, errors, len(symbols), args.model)
    print(f"  Total time: {elapsed:.1f}s  "
          f"({elapsed/max(len(symbols),1):.2f}s per symbol)\n")


if __name__ == "__main__":
    main()
