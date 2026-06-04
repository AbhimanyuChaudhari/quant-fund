"""
Fast Batch Backtest — All Stock Futures
========================================
Supports both V1 (fast Numba engine) and V2 (Ricci Hawkes-Alpha).

Usage:
    # V1 — default params
    python backtest_all_futures_fast.py --start 2026-05-27 --end 2026-05-27

    # V1 — with optimized per-symbol params from grid search
    python backtest_all_futures_fast.py --start 2026-05-27 --end 2026-05-27 --use-optimal-params

    # V2 — fast Numba version
    python backtest_all_futures_fast.py --start 2026-05-27 --end 2026-05-27 --model v2

    # Multi-day (works across contract rolls automatically)
    python backtest_all_futures_fast.py --start 2026-05-13 --end 2026-05-27 --use-optimal-params

Contract roll handling:
    Uses known NSE expiry dates (NSE adjusts for holidays).
    Add new entries to NSE_EXPIRY_DATES each month.
    Falls back to last-Thursday calculation for unknown months.

Speed:
    V1 (Numba):  ~10-15 seconds per day across 85 symbols
    V2 (Numba):  ~30 seconds per day across 85 symbols
"""

import re
import calendar
import argparse
import time
import gcsfs
import pandas as pd
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.backtest.simulators.fill_simulator_fast import run_fast_backtest
from src.backtest.data_loader import load_day
from src.backtest.param_loader import get_symbol_params

PROJECT_ID  = "hedge-fund-494103"
BUCKET_NAME = "hedge-fund-494103-marketdata"

# Index futures base names — excluded from stock futures backtest
INDEX_BASES = {
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
    "SENSEX", "NIFTYNXT50", "BANKEX",
}

# Regex to strip contract suffix: 26MAYFUT, 26JUNFUT, 27JANFUT etc.
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
# Known NSE expiry dates
# NSE sometimes adjusts for public holidays — always verify manually
# Add new entries each month before the contract expires
# ─────────────────────────────────────────────────────────────────────────────

NSE_EXPIRY_DATES = {
    (2026, 1):  date(2026, 1, 29),   # JANFUT
    (2026, 2):  date(2026, 2, 26),   # FEBFUT
    (2026, 3):  date(2026, 3, 26),   # MARFUT
    (2026, 4):  date(2026, 4, 23),   # APRFUT — adjusted (Apr 30 is holiday)
    (2026, 5):  date(2026, 5, 26),   # MAYFUT — adjusted (May 28 is holiday)
    (2026, 6):  date(2026, 6, 25),   # JUNFUT
    (2026, 7):  date(2026, 7, 30),   # JULFUT
    (2026, 8):  date(2026, 8, 27),   # AUGFUT
    (2026, 9):  date(2026, 9, 24),   # SEPFUT
    (2026, 10): date(2026, 10, 29),  # OCTFUT
    (2026, 11): date(2026, 11, 26),  # NOVFUT
    (2026, 12): date(2026, 12, 31),  # DECFUT
}


# ─────────────────────────────────────────────────────────────────────────────
# Contract helpers
# ─────────────────────────────────────────────────────────────────────────────

def strip_contract_suffix(symbol: str) -> str:
    """CHOLAFIN26JUNFUT → CHOLAFIN"""
    return _CONTRACT_RE.sub('', symbol)


def is_index_future(symbol: str) -> bool:
    """Return True if symbol is an index future."""
    return strip_contract_suffix(symbol) in INDEX_BASES


def get_contract_expiry(year: int, month: int) -> date:
    """
    Returns NSE futures expiry date for a given month/year.
    Uses known dates where available (NSE adjusts for holidays),
    falls back to last Thursday calculation for unknown months.
    """
    if (year, month) in NSE_EXPIRY_DATES:
        return NSE_EXPIRY_DATES[(year, month)]

    # Fallback: last Thursday of month
    last_day  = calendar.monthrange(year, month)[1]
    last_date = date(year, month, last_day)
    days_back = (last_date.weekday() - 3) % 7
    return last_date.replace(day=last_day - days_back)


def is_contract_expired(symbol: str, start_date_str: str) -> bool:
    """
    Returns True if the contract expired before start_date.
    Uses NSE_EXPIRY_DATES for accuracy — not just last-Thursday formula.
    """
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
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_lot_sizes() -> dict:
    """
    Load lot sizes from Zerodha API.
    Returns {base_name: lot_size} e.g. {'CHOLAFIN': 625}.
    Falls back to empty dict if API unavailable.
    """
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
    """
    Return active symbols for a date range.

    - Scans all contract directories in GCS
    - Skips index futures
    - Skips expired contracts using NSE_EXPIRY_DATES
    - Skips contracts with no data in date range
    - Returns one symbol per underlying (most recent active contract)
    """
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

        # Skip expired contracts
        if is_contract_expired(candidate, start):
            continue

        base = strip_contract_suffix(candidate)

        # Verify actual data files exist in date range
        files = fs.glob(
            f"{BUCKET_NAME}/processed/features/{candidate}/*.parquet"
        )
        has_data = any(
            start <= f.split("/")[-1].replace(".parquet", "") <= end
            for f in files
        )
        if not has_data:
            continue

        # Keep most recent contract per base name
        # JUNFUT > MAYFUT alphabetically = correct
        if base not in best or candidate > best[base]:
            best[base] = candidate

    return sorted(best.values())


def resolve_params(symbol: str, global_params: dict,
                   use_optimal: bool, model: str) -> dict:
    """
    Return params for a symbol.
    Contract roll handled via param_loader.get_symbol_params().
    """
    if not use_optimal:
        return global_params

    opt    = get_symbol_params(symbol, model=model)
    merged = global_params.copy()
    merged['gamma']      = opt.get('gamma',      global_params['gamma'])
    merged['kappa']      = opt.get('kappa',       global_params['kappa'])
    merged['min_spread'] = opt.get('min_spread',  global_params['min_spread'])
    merged['open_mult']  = opt.get('open_mult',   global_params['open_mult'])

    if model == 'v2':
        for k in ['phi', 'rho', 'beta', 'theta', 'eta', 'nu', 'zeta']:
            if k in opt:
                merged[k] = opt[k]

    return merged


def get_lot_size_for_symbol(symbol: str, lot_sizes: dict) -> int:
    """Get lot size handling contract suffix."""
    base = strip_contract_suffix(symbol)
    if base in lot_sizes:
        return lot_sizes[base]
    if symbol in lot_sizes:
        return lot_sizes[symbol]
    return 75


# ─────────────────────────────────────────────────────────────────────────────
# V1 — fast Numba runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one_v1(symbol: str, start: str, end: str,
               lot_sizes: dict, params: dict,
               min_bars: int = 100) -> dict:
    lot_size = get_lot_size_for_symbol(symbol, lot_sizes)
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
                    'net_pnl': 0, 'bars': 0, 'ok': False,
                    'error': 'no data'}

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
            'gamma':     params['gamma'],
            'open_mult': params.get('open_mult', 2.0),
        }

    except Exception as e:
        return {'symbol': symbol, 'lot_size': lot_size,
                'net_pnl': 0, 'bars': 0, 'ok': False,
                'error': str(e)[:80]}


# ─────────────────────────────────────────────────────────────────────────────
# V2 — fast Numba V2 runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one_v2(symbol: str, start: str, end: str,
               lot_sizes: dict, params: dict,
               min_bars: int = 100) -> dict:
    from src.backtest.simulators.fill_simulator_v2_fast import run_fast_v2_backtest

    lot_size = get_lot_size_for_symbol(symbol, lot_sizes)
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
                    'net_pnl': 0, 'bars': 0, 'ok': False,
                    'error': 'no data'}

        full_df = pd.concat(frames, ignore_index=True).sort_values('ts_sec')

        if len(full_df) < min_bars:
            return {'symbol': symbol, 'lot_size': lot_size,
                    'net_pnl': 0, 'bars': len(full_df), 'ok': False,
                    'error': f'only {len(full_df)} bars'}

        result = run_fast_v2_backtest(
            df                   = full_df,
            beta                 = params.get('beta',          1.0),
            theta                = params.get('theta',         2.0),
            eta                  = params.get('eta',           0.5),
            nu                   = params.get('nu',            0.2),
            rho                  = params.get('rho',           0.30),
            zeta                 = params.get('zeta',          0.5),
            epsilon_plus         = params.get('epsilon_plus',  0.002),
            epsilon_minus        = params.get('epsilon_minus', 0.002),
            beta_kappa           = params.get('beta_kappa',    0.5),
            theta_kappa          = params.get('theta_kappa',   1.5),
            eta_kappa            = params.get('eta_kappa',     0.3),
            nu_kappa             = params.get('nu_kappa',      0.1),
            phi                  = params.get('phi',           0.001),
            min_spread           = params['min_spread'],
            max_spread           = params['max_spread'],
            open_mult            = params.get('open_mult',     2.0),
            lot_size             = lot_size,
            max_inventory        = params.get('max_inv',       5),
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
                'net_pnl': 0, 'bars': 0, 'ok': False,
                'error': str(e)[:80]}


# ─────────────────────────────────────────────────────────────────────────────
# Results printer
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: list, errors: list,
                  total_symbols: int, model: str,
                  use_optimal: bool = False):
    if not results:
        print("\nNo results.")
        return

    results.sort(key=lambda x: x['net_pnl'], reverse=True)
    profitable = [r for r in results if r['net_pnl'] > 0]

    opt_tag = " [OPTIMIZED PARAMS]" if use_optimal else ""

    print(f"\n{'='*80}")
    print(f"  FULL RANKING -- {len(results)} symbols  "
          f"[Model: {model.upper()}{opt_tag}]")
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
    parser.add_argument("--start",              default="2026-05-27")
    parser.add_argument("--end",                default="2026-05-27")
    parser.add_argument("--model",              default="v1",
                        choices=["v1", "v2"])
    parser.add_argument("--use-optimal-params", action="store_true")
    parser.add_argument("--min-bars",           type=int,   default=100)
    parser.add_argument("--workers",            type=int,   default=8)
    parser.add_argument("--gamma",              type=float, default=0.001)
    parser.add_argument("--kappa",              type=float, default=1.5)
    parser.add_argument("--min-spread",         type=float, default=0.10)
    parser.add_argument("--max-spread",         type=float, default=10.0)
    parser.add_argument("--open-mult",          type=float, default=2.0)
    parser.add_argument("--queue-aggression",   type=float, default=0.3)
    parser.add_argument("--max-inventory",      type=int,   default=5)
    parser.add_argument("--phi",   type=float, default=0.001)
    parser.add_argument("--rho",   type=float, default=0.30)
    parser.add_argument("--beta",  type=float, default=1.0)
    parser.add_argument("--theta", type=float, default=2.0)
    parser.add_argument("--eta",   type=float, default=0.5)
    parser.add_argument("--nu",    type=float, default=0.2)
    parser.add_argument("--zeta",  type=float, default=0.5)

    args = parser.parse_args()

    global_params = {
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

    t0      = time.perf_counter()
    opt_tag = " [OPTIMIZED PARAMS]" if args.use_optimal_params else ""

    print(f"\n{'='*70}")
    print(f"  Batch Backtest — All Stock Futures  "
          f"[Model: {args.model.upper()}{opt_tag}]")
    print(f"  Period:  {args.start} -> {args.end}")
    if not args.use_optimal_params:
        if args.model == 'v1':
            print(f"  Params:  gamma={args.gamma} "
                  f"min_spread={args.min_spread} "
                  f"kappa={args.kappa} open_mult={args.open_mult}")
        else:
            print(f"  Params:  phi={args.phi} rho={args.rho} "
                  f"beta={args.beta} eta={args.eta} nu={args.nu}")
    else:
        print(f"  Params:  Per-symbol from "
              f"research/findings/{args.model}_optimal_params.json")
        print(f"           Fallback: gamma={args.gamma} "
              f"min_spread={args.min_spread}")
    print(f"  Workers: {args.workers}")

    # Show expiry dates for verification
    may_exp = get_contract_expiry(2026, 5)
    jun_exp = get_contract_expiry(2026, 6)
    print(f"  Expiries: MAY={may_exp}  JUN={jun_exp}")
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

    def run_with_resolved_params(sym):
        p = resolve_params(sym, global_params,
                           args.use_optimal_params, args.model)
        return run_one(sym, args.start, args.end,
                       lot_sizes, p, args.min_bars)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures_map = {
            pool.submit(run_with_resolved_params, sym): sym
            for sym in symbols
        }
        for future in as_completed(futures_map):
            handle_result(future.result())

    elapsed = time.perf_counter() - t0
    print_results(results, errors, len(symbols),
                  args.model, args.use_optimal_params)
    print(f"  Total time: {elapsed:.1f}s  "
          f"({elapsed/max(len(symbols),1):.2f}s per symbol)\n")


if __name__ == "__main__":
    main()