"""
V2 Parameter Grid Search — Two-Stage Walk-Forward
==================================================
Uses existing fill_simulator_fast_v2.py (validated Numba kernel).

Two-stage search:

    Stage 1 — Spread/risk params (48 combos)
        Search: min_spread, phi, open_mult
        Fix:    all Hawkes/alpha/kappa at defaults
        ~30s for all symbols

    Stage 2 — Signal/ML params (81 combos)
        Search: rho, zeta, theta_kappa, classifier_threshold
        Fix:    best spread params from Stage 1
        ~45s for all symbols

Total: ~75s for all 84 symbols.

Usage:
    python research/grid_search_v2.py --start 2026-05-27 --end 2026-06-03
    python research/grid_search_v2.py --start 2026-05-27 --end 2026-06-03 --workers 8

Output:
    research/findings/v2_optimal_params.json
    research/findings/v2_grid_results.csv
"""

import json
import time
import argparse
import itertools
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('research/findings')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Reuse V1 helpers
# ─────────────────────────────────────────────────────────────────────────────

from research.grid_search_v1 import (
    get_symbols,
    get_lot_sizes,
    get_lot_size_for_symbol,
    get_trading_dates,
)

# ─────────────────────────────────────────────────────────────────────────────
# Two-stage parameter grids
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1: spread/risk — highest PnL impact, symbol-specific
STAGE1_GRID = {
    'min_spread': [0.05, 0.10, 0.25, 0.50],
    'phi':        [0.0005, 0.001, 0.002, 0.005],
    'open_mult':  [1.0, 1.5, 2.0, 3.0],
}

# Stage 2: signal sensitivity — refine after Stage 1
STAGE2_GRID = {
    'rho':                   [0.20, 0.30, 0.50],
    'zeta':                  [0.3,  0.5,  1.0 ],
    'theta_kappa':           [1.0,  1.5,  2.0 ],
    'classifier_threshold':  [0.40, 0.50, 0.60],
}

# Fixed params — structural, same across symbols
FIXED_PARAMS = {
    'beta':          1.0,
    'theta':         2.0,
    'eta':           0.5,
    'nu':            0.2,
    'epsilon_plus':  0.002,
    'epsilon_minus': 0.002,
    'beta_kappa':    0.5,
    'eta_kappa':     0.3,
    'nu_kappa':      0.1,
    'max_spread':    10.0,
    'max_inventory': 5,
}

STAGE1_COMBOS = (
    len(STAGE1_GRID['min_spread']) *
    len(STAGE1_GRID['phi']) *
    len(STAGE1_GRID['open_mult'])
)

STAGE2_COMBOS = (
    len(STAGE2_GRID['rho']) *
    len(STAGE2_GRID['zeta']) *
    len(STAGE2_GRID['theta_kappa']) *
    len(STAGE2_GRID['classifier_threshold'])
)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_symbol_data_v2(symbol: str, dates: list) -> dict:
    """
    Load dataframes for all dates once per symbol.
    Returns {date: df} — shared across all param combos.
    """
    from src.backtest.data_loader import load_day

    data = {}
    for date_str in dates:
        try:
            df = load_day(symbol, date_str, market_hours_only=True)
            if df is not None and not df.empty and len(df) >= 100:
                data[date_str] = df
        except Exception:
            pass
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Run one combo across all dates
# ─────────────────────────────────────────────────────────────────────────────

def run_combo_v2(
    dfs_by_date: dict,
    lot_size:    float,
    params:      dict,
) -> dict:
    """
    Run one param combo across all available dates.
    Returns {date: result_dict}.
    """
    from src.backtest.simulators.fill_simulator_v2_fast import run_fast_v2_backtest

    results = {}
    for date_str, df in dfs_by_date.items():
        try:
            r = run_fast_v2_backtest(
                df                    = df,
                beta                  = params.get('beta',          1.0),
                theta                 = params.get('theta',         2.0),
                eta                   = params.get('eta',           0.5),
                nu                    = params.get('nu',            0.2),
                rho                   = params.get('rho',           0.3),
                zeta                  = params.get('zeta',          0.5),
                epsilon_plus          = params.get('epsilon_plus',  0.002),
                epsilon_minus         = params.get('epsilon_minus', 0.002),
                beta_kappa            = params.get('beta_kappa',    0.5),
                theta_kappa           = params.get('theta_kappa',   1.5),
                eta_kappa             = params.get('eta_kappa',     0.3),
                nu_kappa              = params.get('nu_kappa',      0.1),
                phi                   = params.get('phi',           0.001),
                min_spread            = params.get('min_spread',    0.10),
                max_spread            = params.get('max_spread',    10.0),
                open_mult             = params.get('open_mult',     2.0),
                lot_size              = int(lot_size),
                max_inventory         = params.get('max_inventory', 5),
                classifier_threshold  = params.get('classifier_threshold', 0.50),
                instrument_type       = 'equity_futures',
            )
            results[date_str] = {
                'net_pnl':  r.net_pnl,
                'sharpe':   r.sharpe_ratio,
                'win_rate': r.win_rate / 100.0,   # normalize to [0,1]
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
# Walk-forward scoring (identical to V1)
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_score(per_date: dict, dates: list) -> dict:
    """Train on days 1..N-1, test on day N."""
    available = [d for d in dates if d in per_date]

    if len(available) < 2:
        r = per_date.get(available[0], {}) if available else {}
        return {
            'train_sharpe':    r.get('sharpe', 0),
            'test_sharpe':     r.get('sharpe', 0),
            'test_pnl':        r.get('net_pnl', 0),
            'total_pnl':       r.get('net_pnl', 0),
            'avg_daily_pnl':   r.get('net_pnl', 0),
            'profitable_days': 1 if r.get('net_pnl', 0) > 0 else 0,
            'test_days':       len(available),
            'pnl_std':         0,
        }

    test_sharpes  = []
    test_pnls     = []
    train_sharpes = []

    for i in range(1, len(available)):
        train_dates = available[:i]
        test_date   = available[i]

        train_pnls = [per_date[d]['net_pnl'] for d in train_dates]
        if len(train_pnls) > 1 and np.std(train_pnls) > 0:
            ts = float(np.mean(train_pnls) / np.std(train_pnls) * np.sqrt(252))
        else:
            ts = 0.0
        train_sharpes.append(ts)

        test_r = per_date.get(test_date, {})
        test_pnls.append(test_r.get('net_pnl', 0))
        test_sharpes.append(test_r.get('sharpe', 0))

    all_pnls = [per_date[d]['net_pnl'] for d in available]

    return {
        'train_sharpe':    round(float(np.mean(train_sharpes)), 4),
        'test_sharpe':     round(float(np.mean(test_sharpes)), 4),
        'test_pnl':        round(float(np.sum(test_pnls)), 2),
        'total_pnl':       round(float(np.sum(all_pnls)), 2),
        'avg_daily_pnl':   round(float(np.mean(all_pnls)), 2),
        'profitable_days': sum(1 for p in all_pnls if p > 0),
        'test_days':       len(test_pnls),
        'pnl_std':         round(float(np.std(all_pnls)), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-symbol optimizer — two-stage
# ─────────────────────────────────────────────────────────────────────────────

def optimize_symbol_v2(args_tuple) -> dict:
    """Two-stage optimization for one symbol. Runs in separate process."""
    symbol, dates, lot_size = args_tuple

    # Load data once — shared across all combos
    dfs = load_symbol_data_v2(symbol, dates)
    if len(dfs) < 2:
        return {
            'symbol': symbol,
            'ok':     False,
            'error':  f'only {len(dfs)} dates with data',
        }

    available_dates = sorted(dfs.keys())

    # ── Stage 1: Spread/risk params ───────────────────────────────────────────
    stage1_best_score  = None
    stage1_best_params = None

    for min_spread, phi, open_mult in itertools.product(
        STAGE1_GRID['min_spread'],
        STAGE1_GRID['phi'],
        STAGE1_GRID['open_mult'],
    ):
        params = {
            **FIXED_PARAMS,
            'min_spread': min_spread,
            'phi':        phi,
            'open_mult':  open_mult,
            'lot_size':   lot_size,
            # Stage 2 defaults
            'rho':                  0.30,
            'zeta':                 0.50,
            'theta_kappa':          1.50,
            'classifier_threshold': 0.50,
        }

        per_date = run_combo_v2(dfs, lot_size, params)
        score    = walk_forward_score(per_date, available_dates)

        if score['test_days'] == 0 or score['test_pnl'] <= 0:
            continue

        prof_rate  = score['profitable_days'] / max(score['test_days'], 1)
        rank_score = score['test_sharpe'] * prof_rate

        if stage1_best_score is None or rank_score > stage1_best_score:
            stage1_best_score  = rank_score
            stage1_best_params = {**params, **score,
                                   'rank_score': round(rank_score, 4)}

    # Fallback Stage 1
    if stage1_best_params is None:
        stage1_best_params = {
            **FIXED_PARAMS,
            'min_spread':           0.10,
            'phi':                  0.001,
            'open_mult':            2.0,
            'rho':                  0.30,
            'zeta':                 0.50,
            'theta_kappa':          1.50,
            'classifier_threshold': 0.50,
            'lot_size':             lot_size,
            'fallback':             True,
        }

    # ── Stage 2: Signal params ────────────────────────────────────────────────
    stage2_best_score  = None
    stage2_best_params = None
    all_results        = []

    for rho, zeta, theta_kappa, clf_thresh in itertools.product(
        STAGE2_GRID['rho'],
        STAGE2_GRID['zeta'],
        STAGE2_GRID['theta_kappa'],
        STAGE2_GRID['classifier_threshold'],
    ):
        params = {
            **stage1_best_params,
            'rho':                  rho,
            'zeta':                 zeta,
            'theta_kappa':          theta_kappa,
            'classifier_threshold': clf_thresh,
        }

        per_date = run_combo_v2(dfs, lot_size, params)
        score    = walk_forward_score(per_date, available_dates)

        all_results.append({
            'symbol':               symbol,
            'min_spread':           params['min_spread'],
            'phi':                  params['phi'],
            'open_mult':            params['open_mult'],
            'rho':                  rho,
            'zeta':                 zeta,
            'theta_kappa':          theta_kappa,
            'classifier_threshold': clf_thresh,
            **score,
        })

        if score['test_days'] == 0 or score['test_pnl'] <= 0:
            continue

        prof_rate  = score['profitable_days'] / max(score['test_days'], 1)
        rank_score = score['test_sharpe'] * prof_rate

        if stage2_best_score is None or rank_score > stage2_best_score:
            stage2_best_score  = rank_score
            stage2_best_params = {**params, **score,
                                   'rank_score': round(rank_score, 4)}

    final_params = stage2_best_params or stage1_best_params

    if 'rank_score' not in final_params:
        final_params['rank_score'] = 0.0
        final_params['fallback']   = True

    return {
        'symbol':      symbol,
        'ok':          True,
        'best_params': final_params,
        'all_results': all_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Library entry point — called by rolling_optimizer.py
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_search(
    dates:   list,
    symbols: list = None,
    workers: int  = 4,
) -> dict:
    """
    Library entry point for rolling_optimizer.py
    Returns {symbol: best_params_dict}
    """
    t0 = time.perf_counter()
    print(f"[V2 grid search] dates={dates[0]}..{dates[-1]} ({len(dates)} days)")

    lot_sizes = get_lot_sizes()

    if symbols is None:
        symbols = get_symbols(dates[0], dates[-1])

    if not symbols:
        print("[V2 grid search] No symbols found")
        return {}

    work_items = [
        (sym, dates, get_lot_size_for_symbol(sym, lot_sizes))
        for sym in symbols
    ]

    optimal_params = {}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(optimize_symbol_v2, item): item[0]
            for item in work_items
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                result = future.result()
                if result['ok'] and result.get('best_params'):
                    optimal_params[sym] = result['best_params']
            except Exception as e:
                print(f"[V2 grid search] {sym} failed: {e}")

    elapsed = time.perf_counter() - t0
    print(f"[V2 grid search] Done: {len(optimal_params)} symbols ({elapsed:.1f}s)")
    return optimal_params


# ─────────────────────────────────────────────────────────────────────────────
# Main — standalone run
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='V2 Parameter Grid Search — Two-Stage Walk-Forward'
    )
    parser.add_argument('--start',   default='2026-05-27')
    parser.add_argument('--end',     default='2026-06-03')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--symbols', nargs='*', default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()

    print(f"\n{'='*70}")
    print(f"  V2 Parameter Grid Search (Two-Stage)")
    print(f"  Period:  {args.start} → {args.end}")
    print(f"  Stage 1: {STAGE1_COMBOS} combos (min_spread × phi × open_mult)")
    print(f"  Stage 2: {STAGE2_COMBOS} combos (rho × zeta × theta_kappa × clf)")
    print(f"  Total:   {STAGE1_COMBOS + STAGE2_COMBOS} combos per symbol")
    print(f"  Workers: {args.workers}")
    print(f"{'='*70}\n")

    lot_sizes = get_lot_sizes()

    if args.symbols:
        symbols = args.symbols
    else:
        print("Finding available symbols...")
        symbols = get_symbols(args.start, args.end)
        print(f"Found {len(symbols)} symbols\n")

    if not symbols:
        print("No symbols found.")
        return

    dates = get_trading_dates(args.start, args.end)
    print(f"Trading dates ({len(dates)}): {dates}\n")

    work_items = [
        (sym, dates, get_lot_size_for_symbol(sym, lot_sizes))
        for sym in symbols
    ]

    optimal_params = {}
    all_results    = []
    errors         = []

    print(f"{'Symbol':<30} {'MinSpread':>10} {'Phi':>8} {'Rho':>6} "
          f"{'Zeta':>6} {'ThetaK':>7} {'Clf':>5} "
          f"{'TestShp':>9} {'TestPnL':>12} {'ProfDays':>9}")
    print("-" * 105)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(optimize_symbol_v2, item): item[0]
            for item in work_items
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                result = future.result()
            except Exception as e:
                errors.append({'symbol': sym, 'error': str(e)[:80]})
                print(f"{sym:<30}  ERROR: {str(e)[:60]}")
                continue

            if not result['ok']:
                errors.append(result)
                print(f"{sym:<30}  SKIP: {result.get('error','')[:60]}")
                continue

            bp = result['best_params']
            if bp:
                optimal_params[sym] = bp
                all_results.extend(result.get('all_results', []))

                prof_str = (f"{bp.get('profitable_days',0)}/"
                            f"{bp.get('test_days',0)}")
                print(f"{sym:<30}  "
                      f"{bp.get('min_spread',0):>10.3f}  "
                      f"{bp.get('phi',0):>8.4f}  "
                      f"{bp.get('rho',0):>6.2f}  "
                      f"{bp.get('zeta',0):>6.2f}  "
                      f"{bp.get('theta_kappa',0):>7.2f}  "
                      f"{bp.get('classifier_threshold',0):>5.2f}  "
                      f"{bp.get('test_sharpe',0):>9.2f}  "
                      f"Rs.{bp.get('test_pnl',0):>10,.0f}  "
                      f"{prof_str:>9}"
                      + ("  [fallback]" if bp.get('fallback') else ""))

    elapsed = time.perf_counter() - t0

    # Save
    params_path = OUTPUT_DIR / 'v2_optimal_params.json'
    with open(params_path, 'w') as f:
        json.dump(optimal_params, f, indent=2)
    print(f"\n✓ Saved optimal params → {params_path}")

    if all_results:
        csv_path = OUTPUT_DIR / 'v2_grid_results.csv'
        pd.DataFrame(all_results).to_csv(csv_path, index=False)
        print(f"✓ Saved grid results  → {csv_path}")

    print(f"\n{'='*70}")
    print(f"  V2 GRID SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Symbols optimized: {len(optimal_params)}")
    print(f"  Errors:            {len(errors)}")
    print(f"  Total time:        {elapsed:.1f}s")
    print(f"  Combos evaluated:  {len(all_results):,}")

    if optimal_params:
        ranked = sorted(
            optimal_params.items(),
            key=lambda x: x[1].get('test_sharpe', 0),
            reverse=True,
        )
        print(f"\n  Top 10 by OOS Sharpe:")
        print(f"  {'Symbol':<28} {'TestShp':>9} {'TestPnL':>12} "
              f"{'MinSpread':>10} {'Phi':>8} {'Rho':>6} {'Clf':>5}")
        print("  " + "-" * 80)
        for sym, p in ranked[:10]:
            print(f"  {sym:<28}  "
                  f"{p.get('test_sharpe',0):>9.2f}  "
                  f"Rs.{p.get('test_pnl',0):>10,.0f}  "
                  f"{p.get('min_spread',0):>10.3f}  "
                  f"{p.get('phi',0):>8.4f}  "
                  f"{p.get('rho',0):>6.2f}  "
                  f"{p.get('classifier_threshold',0):>5.2f}")
    print()


if __name__ == '__main__':
    main()