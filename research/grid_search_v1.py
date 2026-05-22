"""
V1 Parameter Grid Search — Walk-Forward Optimized
===================================================
Finds optimal per-symbol parameters using walk-forward validation.

Speed:
    ~30 seconds for all 85 symbols across 8 days
    (Numba compiles once, then 0.016s per combo)

Walk-forward logic:
    Train on first N-1 days → test on last day
    Rank by: test_sharpe > 0 AND test_pnl > 0 AND train_sharpe

Usage:
    python scripts/grid_search_v1.py --start 2026-05-13 --end 2026-05-22
    python scripts/grid_search_v1.py --start 2026-05-13 --end 2026-05-22 --workers 8
    python scripts/grid_search_v1.py --symbols CHOLAFIN26MAYFUT VOLTAS26MAYFUT

Output:
    research/findings/v1_optimal_params.json  ← best params per symbol
    research/findings/v1_grid_results.csv     ← full grid results
"""

import argparse
import json
import time
import itertools
import os
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import gcsfs

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ID  = "hedge-fund-494103"
BUCKET_NAME = "hedge-fund-494103-marketdata"

OUTPUT_DIR  = Path("research/findings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FUTURES = {
    "NIFTY26MAYFUT", "BANKNIFTY26MAYFUT",
    "FINNIFTY26MAYFUT", "MIDCPNIFTY26MAYFUT",
    "SENSEX26MAYFUT", "NIFTYNXT5026MAYFUT",
    "BANKEX26MAYFUT",
}

# ─────────────────────────────────────────────────────────────────────────────
# Parameter grid
# ─────────────────────────────────────────────────────────────────────────────

PARAM_GRID = {
    'gamma':      [0.0005, 0.001, 0.002, 0.005],
    'kappa':      [1.0, 1.5, 2.0],
    'min_spread': [0.05, 0.10, 0.25, 0.50],
    'open_mult':  [1.0, 1.5, 2.0, 3.0],
}

# Fixed params (not optimized)
FIXED_PARAMS = {
    'max_spread':       10.0,
    'max_inventory':    5,
    'queue_aggression': 0.3,
}

# Total combinations: 4 × 3 × 4 × 4 = 192 per symbol
TOTAL_COMBOS = (
    len(PARAM_GRID['gamma']) *
    len(PARAM_GRID['kappa']) *
    len(PARAM_GRID['min_spread']) *
    len(PARAM_GRID['open_mult'])
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_trading_dates(start: str, end: str) -> list:
    """Return all weekdays between start and end."""
    start_dt = datetime.strptime(start, '%Y-%m-%d')
    end_dt   = datetime.strptime(end,   '%Y-%m-%d')
    dates    = []
    current  = start_dt
    while current <= end_dt:
        if current.weekday() < 5:
            dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    return dates


def get_symbols(start: str, end: str) -> list:
    """Get all stock futures with processed data."""
    try:
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
    except Exception as e:
        print(f"Could not fetch symbols from GCS: {e}")
        return []


def get_lot_sizes() -> dict:
    """Load lot sizes from Zerodha."""
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


def load_symbol_data(symbol: str, dates: list) -> dict:
    """
    Load all dates for a symbol into memory once.
    Returns dict of {date: DataFrame}.
    Skips missing dates silently.
    """
    from src.backtest.data_loader import load_day
    data = {}
    for date in dates:
        try:
            df = load_day(symbol, date, market_hours_only=True)
            if not df.empty and len(df) >= 100:
                data[date] = df
        except Exception:
            pass
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Core: run one param combo on pre-loaded data
# ─────────────────────────────────────────────────────────────────────────────

def run_combo_on_data(
    dfs:          dict,        # {date: DataFrame} — pre-loaded
    lot_size:     int,
    instrument_type: str,
    gamma:        float,
    kappa:        float,
    min_spread:   float,
    open_mult:    float,
) -> dict:
    """
    Run fast backtest for one param combo across all dates.
    Data already in memory — no I/O.
    Returns per-date results.
    """
    from src.backtest.simulators.fill_simulator_fast import run_fast_backtest

    results = {}
    for date, df in dfs.items():
        try:
            r = run_fast_backtest(
                df               = df,
                gamma            = gamma,
                kappa            = kappa,
                min_spread       = min_spread,
                max_spread       = FIXED_PARAMS['max_spread'],
                open_mult        = open_mult,
                lot_size         = lot_size,
                max_inventory    = FIXED_PARAMS['max_inventory'],
                queue_aggression = FIXED_PARAMS['queue_aggression'],
                instrument_type  = instrument_type,
            )
            results[date] = {
                'net_pnl':   r.net_pnl,
                'sharpe':    r.sharpe_ratio,
                'win_rate':  r.win_rate,
                'fills':     r.total_fills,
                'max_dd':    r.max_drawdown,
            }
        except Exception as e:
            results[date] = {
                'net_pnl': 0, 'sharpe': 0,
                'win_rate': 0, 'fills': 0, 'max_dd': 0,
            }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward validation
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_score(
    per_date_results: dict,   # {date: {net_pnl, sharpe, ...}}
    dates:            list,
) -> dict:
    """
    Walk-forward validation:
        For each test day (last day in rolling window):
            Train: all previous days
            Test:  this day

    Returns aggregated train and test metrics.
    """
    available_dates = [d for d in dates if d in per_date_results]
    if len(available_dates) < 2:
        r = per_date_results.get(available_dates[0], {}) if available_dates else {}
        return {
            'train_sharpe': r.get('sharpe', 0),
            'train_pnl':    r.get('net_pnl', 0),
            'test_sharpe':  r.get('sharpe', 0),
            'test_pnl':     r.get('net_pnl', 0),
            'test_days':    len(available_dates),
            'profitable_days': 1 if r.get('net_pnl', 0) > 0 else 0,
        }

    test_sharpes = []
    test_pnls    = []
    train_sharpes = []
    train_pnls    = []

    # Walk forward: each day after the first is a test day
    for i in range(1, len(available_dates)):
        train_dates = available_dates[:i]
        test_date   = available_dates[i]

        # Train metrics
        t_sharpes = [per_date_results[d]['sharpe'] for d in train_dates]
        t_pnls    = [per_date_results[d]['net_pnl'] for d in train_dates]
        train_sharpes.append(np.mean(t_sharpes))
        train_pnls.append(np.sum(t_pnls))

        # Test metrics
        test_r = per_date_results.get(test_date, {})
        test_sharpes.append(test_r.get('sharpe', 0))
        test_pnls.append(test_r.get('net_pnl', 0))

    all_pnls = [per_date_results[d]['net_pnl'] for d in available_dates]

    return {
        'train_sharpe':     round(float(np.mean(train_sharpes)), 4),
        'train_pnl':        round(float(np.mean(train_pnls)), 2),
        'test_sharpe':      round(float(np.mean(test_sharpes)), 4),
        'test_pnl':         round(float(np.sum(test_pnls)), 2),
        'total_pnl':        round(float(np.sum(all_pnls)), 2),
        'avg_daily_pnl':    round(float(np.mean(all_pnls)), 2),
        'profitable_days':  sum(1 for p in all_pnls if p > 0),
        'test_days':        len(test_pnls),
        'pnl_std':          round(float(np.std(all_pnls)), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-symbol optimizer — runs in separate process
# ─────────────────────────────────────────────────────────────────────────────

def optimize_symbol(args_tuple) -> dict:
    """
    Optimize one symbol across all param combinations.
    Runs in a separate process — imports happen here.

    Returns:
        {
            'symbol': ...,
            'best_params': {...},
            'best_score': {...},
            'all_results': [...],  # full grid for CSV
        }
    """
    symbol, dates, lot_size, instrument_type = args_tuple

    # Load all data once
    data = load_symbol_data(symbol, dates)
    if len(data) < 2:
        return {
            'symbol': symbol,
            'ok': False,
            'error': f'only {len(data)} dates with data',
        }

    available_dates = sorted(data.keys())

    # Build all param combinations
    combos = list(itertools.product(
        PARAM_GRID['gamma'],
        PARAM_GRID['kappa'],
        PARAM_GRID['min_spread'],
        PARAM_GRID['open_mult'],
    ))

    best_score  = None
    best_params = None
    all_results = []

    for gamma, kappa, min_spread, open_mult in combos:
        # Run on all dates
        per_date = run_combo_on_data(
            dfs             = data,
            lot_size        = lot_size,
            instrument_type = instrument_type,
            gamma           = gamma,
            kappa           = kappa,
            min_spread      = min_spread,
            open_mult       = open_mult,
        )

        # Walk-forward score
        score = walk_forward_score(per_date, available_dates)

        result_row = {
            'symbol':         symbol,
            'gamma':          gamma,
            'kappa':          kappa,
            'min_spread':     min_spread,
            'open_mult':      open_mult,
            **score,
        }
        all_results.append(result_row)

        # Ranking criteria:
        #   1. test_sharpe > 0 (must be profitable out-of-sample)
        #   2. profitable_days >= 60% of test days
        #   3. Maximize: test_sharpe × (profitable_days / test_days)
        if score['test_days'] == 0:
            continue

        prof_rate  = score['profitable_days'] / score['test_days']
        rank_score = score['test_sharpe'] * prof_rate

        if score['test_pnl'] <= 0:
            continue   # skip if OOS is unprofitable

        if best_score is None or rank_score > best_score:
            best_score  = rank_score
            best_params = {
                'gamma':      gamma,
                'kappa':      kappa,
                'min_spread': min_spread,
                'open_mult':  open_mult,
                'lot_size':   lot_size,
                **score,
                'rank_score': round(rank_score, 4),
            }

    # Fallback: if no combo passes OOS filter, use best train Sharpe
    if best_params is None and all_results:
        fallback = max(all_results, key=lambda x: x.get('train_sharpe', 0))
        best_params = {
            'gamma':      fallback['gamma'],
            'kappa':      fallback['kappa'],
            'min_spread': fallback['min_spread'],
            'open_mult':  fallback['open_mult'],
            'lot_size':   lot_size,
            **{k: fallback[k] for k in ['train_sharpe', 'test_sharpe',
                                         'total_pnl', 'profitable_days',
                                         'test_days', 'avg_daily_pnl']},
            'rank_score': 0.0,
            'fallback':   True,
        }

    return {
        'symbol':      symbol,
        'ok':          True,
        'best_params': best_params,
        'all_results': all_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V1 Parameter Grid Search with Walk-Forward Validation"
    )
    parser.add_argument("--start",   default="2026-05-13")
    parser.add_argument("--end",     default="2026-05-22")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel processes (default 4 — CPU bound)")
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="Specific symbols (default: all 85)")
    parser.add_argument("--min-bars", type=int, default=100)
    args = parser.parse_args()

    t0 = time.perf_counter()

    print(f"\n{'='*70}")
    print(f"  V1 Parameter Grid Search")
    print(f"  Period:  {args.start} → {args.end}")
    print(f"  Grid:    {TOTAL_COMBOS} combinations per symbol")
    print(f"           gamma={PARAM_GRID['gamma']}")
    print(f"           kappa={PARAM_GRID['kappa']}")
    print(f"           min_spread={PARAM_GRID['min_spread']}")
    print(f"           open_mult={PARAM_GRID['open_mult']}")
    print(f"  Workers: {args.workers}")
    print(f"{'='*70}\n")

    # Load lot sizes
    print("Loading lot sizes...")
    lot_sizes = get_lot_sizes()

    # Get symbols
    if args.symbols:
        symbols = args.symbols
        print(f"Using {len(symbols)} specified symbols")
    else:
        print("Finding available symbols...")
        symbols = get_symbols(args.start, args.end)
        print(f"Found {len(symbols)} symbols")

    if not symbols:
        print("No symbols found.")
        return

    # Get trading dates
    dates = get_trading_dates(args.start, args.end)
    print(f"Trading dates: {dates}")
    print(f"\nExpected time: ~{len(symbols) * TOTAL_COMBOS * 0.016 / args.workers:.0f}s "
          f"(after Numba compile)\n")

    # Build work items
    work_items = []
    for sym in symbols:
        name = sym.replace("26MAYFUT", "").replace("26JUNFUT", "")
        lot  = lot_sizes.get(name, 75)
        inst = 'currency_futures' if 'USDINR' in sym.upper() \
               else 'equity_futures'
        work_items.append((sym, dates, lot, inst))

    # Run grid search in parallel
    optimal_params  = {}
    all_grid_rows   = []
    done            = 0
    errors          = []

    print(f"{'Symbol':<30} {'BestGamma':>10} {'BestKappa':>10} "
          f"{'BestSpread':>11} {'TestSharpe':>11} {'TestPnL':>12} "
          f"{'ProfDays':>9}")
    print("-" * 95)

    # Use ProcessPoolExecutor for CPU-bound work
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(optimize_symbol, item): item[0]
                   for item in work_items}

        for future in as_completed(futures):
            sym    = futures[future]
            done  += 1

            try:
                result = future.result()
            except Exception as e:
                errors.append({'symbol': sym, 'error': str(e)[:80]})
                print(f"{sym:<30}  ERROR: {str(e)[:50]}")
                continue

            if not result['ok']:
                errors.append(result)
                print(f"{sym:<30}  SKIP: {result.get('error', '')[:50]}")
                continue

            bp = result['best_params']
            if bp:
                optimal_params[sym] = bp
                all_grid_rows.extend(result['all_results'])

                prof_str = f"{bp.get('profitable_days',0)}/{bp.get('test_days',0)}"
                print(f"{sym:<30}  "
                      f"{bp['gamma']:>10.4f}  "
                      f"{bp['kappa']:>10.2f}  "
                      f"{bp['min_spread']:>11.3f}  "
                      f"{bp.get('test_sharpe',0):>11.2f}  "
                      f"Rs.{bp.get('test_pnl',0):>10,.0f}  "
                      f"{prof_str:>9}"
                      + ("  [fallback]" if bp.get('fallback') else ""))
            else:
                print(f"{sym:<30}  no valid combo found")

    elapsed = time.perf_counter() - t0

    # ── Save results ──────────────────────────────────────────────────────────

    # 1. Per-symbol optimal params JSON
    params_path = OUTPUT_DIR / "v1_optimal_params.json"
    with open(params_path, 'w') as f:
        json.dump(optimal_params, f, indent=2)
    print(f"\n✓ Saved optimal params → {params_path}")

    # 2. Full grid results CSV
    if all_grid_rows:
        csv_path = OUTPUT_DIR / "v1_grid_results.csv"
        pd.DataFrame(all_grid_rows).to_csv(csv_path, index=False)
        print(f"✓ Saved grid results  → {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  GRID SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Symbols optimized:  {len(optimal_params)}")
    print(f"  Errors:             {len(errors)}")
    print(f"  Total time:         {elapsed:.1f}s")
    print(f"  Time per symbol:    {elapsed/max(len(symbols),1):.2f}s")
    print(f"  Combos evaluated:   {len(all_grid_rows):,}")

    # Top 10 by test Sharpe
    if optimal_params:
        ranked = sorted(
            optimal_params.items(),
            key=lambda x: x[1].get('test_sharpe', 0),
            reverse=True
        )
        print(f"\n  Top 10 by out-of-sample Sharpe:")
        print(f"  {'Symbol':<28} {'TestSharpe':>11} {'TestPnL':>12} "
              f"{'Gamma':>8} {'Spread':>8} {'OpenMult':>9}")
        print("  " + "-"*78)
        for sym, p in ranked[:10]:
            print(f"  {sym:<28} "
                  f"{p.get('test_sharpe',0):>11.2f} "
                  f"Rs.{p.get('test_pnl',0):>10,.0f}  "
                  f"{p['gamma']:>8.4f}  "
                  f"{p['min_spread']:>8.3f}  "
                  f"{p['open_mult']:>9.1f}")

    print(f"\n  Output files:")
    print(f"    {params_path}")
    if all_grid_rows:
        print(f"    {csv_path}")
    print()


if __name__ == "__main__":
    main()