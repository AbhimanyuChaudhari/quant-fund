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

Contract roll:
    Automatically handles MAYFUT → JUNFUT transitions.
    Same expiry logic as backtest_all_futures_fast.py.

Usage:
    python research/grid_search_v1.py --start 2026-05-13 --end 2026-05-22
    python research/grid_search_v1.py --start 2026-05-13 --end 2026-05-22 --workers 8
    python research/grid_search_v1.py --symbols CHOLAFIN26MAYFUT VOLTAS26MAYFUT

Output:
    research/findings/v1_optimal_params.json  ← best params per symbol
    research/findings/v1_grid_results.csv     ← full grid results
"""

import re
import argparse
import calendar
import json
import time
import itertools
from datetime import date, datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import gcsfs

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ID  = "hedge-fund-494103"
BUCKET_NAME = "hedge-fund-494103-marketdata-mumbai"

OUTPUT_DIR  = Path("research/findings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Index futures base names — excluded from grid search
INDEX_BASES = {
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
    "SENSEX", "NIFTYNXT50", "BANKEX",
}

# Contract suffix regex — shared across all scripts
_CONTRACT_RE = re.compile(
    r'\d{2}(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)FUT$',
    re.IGNORECASE
)

_MONTH_MAP = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
    'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
    'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
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

FIXED_PARAMS = {
    'max_spread':       10.0,
    'max_inventory':    5,
    'queue_aggression': 0.3,
}

TOTAL_COMBOS = (
    len(PARAM_GRID['gamma']) *
    len(PARAM_GRID['kappa']) *
    len(PARAM_GRID['min_spread']) *
    len(PARAM_GRID['open_mult'])
)


# ─────────────────────────────────────────────────────────────────────────────
# Contract helpers — dynamic, no manual updates needed
# ─────────────────────────────────────────────────────────────────────────────

def strip_contract_suffix(symbol: str) -> str:
    """CHOLAFIN26JUNFUT → CHOLAFIN"""
    return _CONTRACT_RE.sub('', symbol)


def is_index_future(symbol: str) -> bool:
    return strip_contract_suffix(symbol) in INDEX_BASES


NSE_EXPIRY_DATES = {
    (2026, 1):  date(2026, 1, 29),
    (2026, 2):  date(2026, 2, 26),
    (2026, 3):  date(2026, 3, 26),
    (2026, 4):  date(2026, 4, 23),
    (2026, 5):  date(2026, 5, 26),   # adjusted (May 28 is holiday)
    (2026, 6):  date(2026, 6, 25),
    (2026, 7):  date(2026, 7, 30),
    (2026, 8):  date(2026, 8, 27),
    (2026, 9):  date(2026, 9, 24),
    (2026, 10): date(2026, 10, 29),
    (2026, 11): date(2026, 11, 26),
    (2026, 12): date(2026, 12, 31),
}


def get_contract_expiry(year: int, month: int) -> date:
    """
    NSE futures expiry date.
    Uses known dates (NSE adjusts for holidays),
    falls back to last Thursday for unknown months.
    """
    if (year, month) in NSE_EXPIRY_DATES:
        return NSE_EXPIRY_DATES[(year, month)]
    last_day  = calendar.monthrange(year, month)[1]
    last_date = date(year, month, last_day)
    days_back = (last_date.weekday() - 3) % 7
    return last_date.replace(day=last_day - days_back)


def is_contract_expired(symbol: str, start_date_str: str) -> bool:
    """True if contract expired before start_date."""
    m = _CONTRACT_RE.search(symbol)
    if not m:
        return False
    suffix = m.group(0)
    yy     = int(suffix[:2])
    mon    = suffix[2:5].upper()
    month  = _MONTH_MAP.get(mon)
    if not month:
        return False
    expiry = get_contract_expiry(2000 + yy, month)
    return date.fromisoformat(start_date_str) > expiry


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
    """
    Get all active stock futures with processed data in date range.
    Handles contract roll automatically — returns one symbol per underlying.
    """
    try:
        fs          = gcsfs.GCSFileSystem(project=PROJECT_ID)
        all_entries = fs.glob(f"{BUCKET_NAME}/processed/features/*")

        best: dict = {}

        for entry in all_entries:
            parts = entry.split("/")
            if len(parts) < 4:
                continue

            candidate = parts[3]

            if is_index_future(candidate):
                continue
            if not _CONTRACT_RE.search(candidate):
                continue
            if '.' in candidate or candidate.startswith('_'):
                continue
            if is_contract_expired(candidate, start):
                continue

            base  = strip_contract_suffix(candidate)
            files = fs.glob(
                f"{BUCKET_NAME}/processed/features/{candidate}/*.parquet"
            )
            has_data = any(
                start <= f.split("/")[-1].replace(".parquet", "") <= end
                for f in files
            )
            if not has_data:
                continue

            if base not in best or candidate > best[base]:
                best[base] = candidate

        return sorted(best.values())

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


def get_lot_size_for_symbol(symbol: str, lot_sizes: dict) -> int:
    base = strip_contract_suffix(symbol)
    if base in lot_sizes:
        return lot_sizes[base]
    if symbol in lot_sizes:
        return lot_sizes[symbol]
    return 75


def load_symbol_data(symbol: str, dates: list) -> dict:
    """
    Load all dates for a symbol into memory once.
    Returns dict of {date: DataFrame}.
    """
    from src.backtest.data_loader import load_day
    data = {}
    for date_str in dates:
        try:
            df = load_day(symbol, date_str, market_hours_only=True)
            if not df.empty and len(df) >= 100:
                data[date_str] = df
        except Exception:
            pass
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Core: run one param combo on pre-loaded data
# ─────────────────────────────────────────────────────────────────────────────

def run_combo_on_data(
    dfs:             dict,
    lot_size:        int,
    instrument_type: str,
    gamma:           float,
    kappa:           float,
    min_spread:      float,
    open_mult:       float,
) -> dict:
    """Run fast backtest for one param combo across all dates."""
    from src.backtest.simulators.fill_simulator_fast import run_fast_backtest

    results = {}
    for date_str, df in dfs.items():
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
            results[date_str] = {
                'net_pnl':  r.net_pnl,
                'sharpe':   r.sharpe_ratio,
                'win_rate': r.win_rate,
                'fills':    r.total_fills,
                'max_dd':   r.max_drawdown,
            }
        except Exception:
            results[date_str] = {
                'net_pnl': 0, 'sharpe': 0,
                'win_rate': 0, 'fills': 0, 'max_dd': 0,
            }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward validation
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_score(per_date_results: dict, dates: list) -> dict:
    """
    Walk-forward validation:
        Train on days 1..N-1, test on day N.
    """
    available_dates = [d for d in dates if d in per_date_results]
    if len(available_dates) < 2:
        r = per_date_results.get(available_dates[0], {}) \
            if available_dates else {}
        return {
            'train_sharpe':    r.get('sharpe', 0),
            'train_pnl':       r.get('net_pnl', 0),
            'test_sharpe':     r.get('sharpe', 0),
            'test_pnl':        r.get('net_pnl', 0),
            'test_days':       len(available_dates),
            'profitable_days': 1 if r.get('net_pnl', 0) > 0 else 0,
        }

    test_sharpes  = []
    test_pnls     = []
    train_sharpes = []
    train_pnls    = []

    for i in range(1, len(available_dates)):
        train_dates = available_dates[:i]
        test_date   = available_dates[i]

        train_sharpes.append(
            np.mean([per_date_results[d]['sharpe'] for d in train_dates])
        )
        train_pnls.append(
            np.sum([per_date_results[d]['net_pnl'] for d in train_dates])
        )
        test_r = per_date_results.get(test_date, {})
        test_sharpes.append(test_r.get('sharpe', 0))
        test_pnls.append(test_r.get('net_pnl', 0))

    all_pnls = [per_date_results[d]['net_pnl'] for d in available_dates]

    return {
        'train_sharpe':    round(float(np.mean(train_sharpes)), 4),
        'train_pnl':       round(float(np.mean(train_pnls)),    2),
        'test_sharpe':     round(float(np.mean(test_sharpes)),  4),
        'test_pnl':        round(float(np.sum(test_pnls)),      2),
        'total_pnl':       round(float(np.sum(all_pnls)),       2),
        'avg_daily_pnl':   round(float(np.mean(all_pnls)),      2),
        'profitable_days': sum(1 for p in all_pnls if p > 0),
        'test_days':       len(test_pnls),
        'pnl_std':         round(float(np.std(all_pnls)),       2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-symbol optimizer — runs in separate process
# ─────────────────────────────────────────────────────────────────────────────

def optimize_symbol(args_tuple) -> dict:
    """Optimize one symbol. Runs in a separate process."""
    symbol, dates, lot_size, instrument_type = args_tuple

    data = load_symbol_data(symbol, dates)
    if len(data) < 2:
        return {
            'symbol': symbol,
            'ok':     False,
            'error':  f'only {len(data)} dates with data',
        }

    available_dates = sorted(data.keys())

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
        per_date = run_combo_on_data(
            dfs             = data,
            lot_size        = lot_size,
            instrument_type = instrument_type,
            gamma           = gamma,
            kappa           = kappa,
            min_spread      = min_spread,
            open_mult       = open_mult,
        )

        score = walk_forward_score(per_date, available_dates)

        all_results.append({
            'symbol':     symbol,
            'gamma':      gamma,
            'kappa':      kappa,
            'min_spread': min_spread,
            'open_mult':  open_mult,
            **score,
        })

        if score['test_days'] == 0 or score['test_pnl'] <= 0:
            continue

        prof_rate  = score['profitable_days'] / score['test_days']
        rank_score = score['test_sharpe'] * prof_rate

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

    # Fallback: use best train Sharpe if no OOS-profitable combo found
    if best_params is None and all_results:
        fallback = max(all_results, key=lambda x: x.get('train_sharpe', 0))
        best_params = {
            'gamma':      fallback['gamma'],
            'kappa':      fallback['kappa'],
            'min_spread': fallback['min_spread'],
            'open_mult':  fallback['open_mult'],
            'lot_size':   lot_size,
            **{k: fallback.get(k, 0) for k in [
                'train_sharpe', 'test_sharpe', 'total_pnl',
                'profitable_days', 'test_days', 'avg_daily_pnl',
            ]},
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
    parser.add_argument("--start",    default="2026-05-13")
    parser.add_argument("--end",      default="2026-05-22")
    parser.add_argument("--workers",  type=int, default=4)
    parser.add_argument("--symbols",  nargs="*", default=None,
                        help="Specific symbols (default: all active)")
    parser.add_argument("--min-bars", type=int, default=100)
    args = parser.parse_args()

    t0 = time.perf_counter()

    print(f"\n{'='*70}")
    print(f"  V1 Parameter Grid Search")
    print(f"  Period:  {args.start} → {args.end}")
    print(f"  Grid:    {TOTAL_COMBOS} combinations per symbol")
    print(f"  Workers: {args.workers}")
    print(f"{'='*70}\n")

    # Show expiry info
    may_exp = get_contract_expiry(2026, 5)
    jun_exp = get_contract_expiry(2026, 6)
    print(f"  Contract expiries: MAY={may_exp}  JUN={jun_exp}\n")

    print("Loading lot sizes...")
    lot_sizes = get_lot_sizes()

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

    dates = get_trading_dates(args.start, args.end)
    print(f"Trading dates ({len(dates)}): {dates}")
    print(f"Expected time: ~{len(symbols) * TOTAL_COMBOS * 0.016 / args.workers:.0f}s\n")

    # Build work items
    work_items = [
        (sym, dates, get_lot_size_for_symbol(sym, lot_sizes),
         'currency_futures' if 'USDINR' in sym.upper() else 'equity_futures')
        for sym in symbols
    ]

    optimal_params = {}
    all_grid_rows  = []
    errors         = []

    print(f"{'Symbol':<30} {'BestGamma':>10} {'BestKappa':>10} "
          f"{'BestSpread':>11} {'TestSharpe':>11} {'TestPnL':>12} "
          f"{'ProfDays':>9}")
    print("-" * 95)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(optimize_symbol, item): item[0]
                   for item in work_items}

        for future in as_completed(futures):
            sym = futures[future]
            try:
                result = future.result()
            except Exception as e:
                errors.append({'symbol': sym, 'error': str(e)[:80]})
                print(f"{sym:<30}  ERROR: {str(e)[:50]}")
                continue

            if not result['ok']:
                errors.append(result)
                print(f"{sym:<30}  SKIP: {result.get('error','')[:50]}")
                continue

            bp = result['best_params']
            if bp:
                optimal_params[sym] = bp
                all_grid_rows.extend(result['all_results'])

                prof_str = (f"{bp.get('profitable_days',0)}/"
                            f"{bp.get('test_days',0)}")
                print(f"{sym:<30}  "
                      f"{bp['gamma']:>10.4f}  "
                      f"{bp['kappa']:>10.2f}  "
                      f"{bp['min_spread']:>11.3f}  "
                      f"{bp.get('test_sharpe',0):>11.2f}  "
                      f"Rs.{bp.get('test_pnl',0):>10,.0f}  "
                      f"{prof_str:>9}"
                      + ("  [fallback]" if bp.get('fallback') else ""))

    elapsed = time.perf_counter() - t0

    # ── Save results ──────────────────────────────────────────────────────────
    params_path = OUTPUT_DIR / "v1_optimal_params.json"
    with open(params_path, 'w') as f:
        json.dump(optimal_params, f, indent=2)
    print(f"\n✓ Saved optimal params → {params_path}")

    if all_grid_rows:
        csv_path = OUTPUT_DIR / "v1_grid_results.csv"
        pd.DataFrame(all_grid_rows).to_csv(csv_path, index=False)
        print(f"✓ Saved grid results  → {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  GRID SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Symbols optimized: {len(optimal_params)}")
    print(f"  Errors:            {len(errors)}")
    print(f"  Total time:        {elapsed:.1f}s")
    print(f"  Combos evaluated:  {len(all_grid_rows):,}")

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
    print()


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# Library entry point — called by rolling_optimizer.py
# ADD THIS TO THE BOTTOM OF grid_search_v1.pypython research/grid_search_v1.py --start 2026-05-27 --end 2026-06-03
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_search(
    dates:    list,
    symbols:  list = None,
    workers:  int  = 4,
) -> dict:
    """
    Library entry point for rolling_optimizer.py

    Same logic as main() but returns params dict instead of
    saving to JSON. rolling_optimizer handles saving + blending.

    Args:
        dates:   list of date strings e.g. ['2026-05-27', '2026-05-28']
        symbols: list of symbols (default: auto-detect from GCS)
        workers: parallel workers

    Returns:
        {symbol: best_params_dict}  — same format as v1_optimal_params.json
    """
    import time
    t0 = time.perf_counter()

    print(f"[V1 grid search] dates={dates[0]}..{dates[-1]}  "
          f"({len(dates)} days)")

    # Load lot sizes
    lot_sizes = get_lot_sizes()

    # Get symbols if not provided
    if symbols is None:
        symbols = get_symbols(dates[0], dates[-1])

    if not symbols:
        print("[V1 grid search] No symbols found")
        return {}

    # Build work items
    work_items = [
        (sym, dates,
         get_lot_size_for_symbol(sym, lot_sizes),
         'equity_futures')
        for sym in symbols
    ]

    optimal_params = {}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(optimize_symbol, item): item[0]
            for item in work_items
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                result = future.result()
                if result['ok'] and result.get('best_params'):
                    optimal_params[sym] = result['best_params']
            except Exception as e:
                print(f"[V1 grid search] {sym} failed: {e}")

    elapsed = time.perf_counter() - t0
    print(f"[V1 grid search] Done: {len(optimal_params)} symbols  "
          f"({elapsed:.1f}s)")

    return optimal_params
