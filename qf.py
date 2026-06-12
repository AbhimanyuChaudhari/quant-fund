"""
Quant Fund CLI
==============
Single entry point for all backtest, grid search, pairs, and system commands.

Usage:
    python qf.py backtest --model v1 --start 2026-05-27 --end 2026-06-03
    python qf.py backtest --model both --start 2026-06-02 --end 2026-06-03
    python qf.py gridsearch --model v1 --start 2026-05-27 --end 2026-06-03
    python qf.py pairs --nb 3 --start 2026-05-13 --end 2026-06-03
    python qf.py health
    python qf.py optimize --model both
    python qf.py status
"""

import os
import sys
import json
import time
import argparse
import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def separator(char='=', width=68):
    print(char * width)

def header(title):
    separator()
    print(f"  {title}")
    separator()

def section(title):
    print(f"\n  {title}")
    print(f"  {'─' * (len(title) + 2)}")

def fmt_pnl(pnl):
    """Format PnL with color — green positive, red negative."""
    if pnl > 0:
        return f"\033[92m+Rs.{pnl:>10,.0f}\033[0m"
    elif pnl < 0:
        return f"\033[91m-Rs.{abs(pnl):>10,.0f}\033[0m"
    return f" Rs.{pnl:>10,.0f}"

def fmt_status(ok):
    return "\033[92m✓\033[0m" if ok else "\033[91m✗\033[0m"

def load_json(path):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}

def save_last_run(command, result):
    """Save last run metadata for status command."""
    runs_path = Path('research/findings/.last_runs.json')
    runs = load_json(runs_path)
    runs[command] = {
        'timestamp': datetime.datetime.now().isoformat(),
        'result':    result,
    }
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(runs_path, 'w') as f:
        json.dump(runs, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST command
# ─────────────────────────────────────────────────────────────────────────────

def cmd_backtest(args):
    from backtest_all_futures_fast import run_backtest_all

    models   = ['v1', 'v2'] if args.model == 'both' else [args.model]
    symbols  = args.symbols or None
    use_opt  = not args.no_optimal

    results = {}

    for model in models:
        header(
            f"BACKTEST — Model: {model.upper()}  "
            f"Period: {args.start} → {args.end}"
        )

        t0 = time.perf_counter()

        result = run_backtest_all(
            model            = model,
            start_date       = args.start,
            end_date         = args.end,
            symbols          = symbols,
            use_optimal_params = use_opt,
        )

        elapsed = time.perf_counter() - t0
        results[model] = result

        # Print results table
        print(f"\n  {'Symbol':<28} {'Net PnL':>12} {'Fills':>7} "
              f"{'Win%':>6} {'MaxDD':>12}")
        print(f"  {'─'*70}")

        profitable = [r for r in result if r['net_pnl'] > 0]
        losers     = [r for r in result if r['net_pnl'] <= 0]

        for r in sorted(profitable, key=lambda x: x['net_pnl'], reverse=True):
            dd_str = f"-Rs.{abs(r['max_dd']):>8,.0f}"
            print(f"  {r['symbol']:<28} "
                  f"{fmt_pnl(r['net_pnl'])}  "
                  f"{r['fills']:>7,}  "
                  f"{r['win_rate']:>5.1f}%  "
                  f"{dd_str:>12}")

        total_pnl = sum(r['net_pnl'] for r in result)
        print(f"\n  {'─'*70}")
        print(f"  Profitable: {len(profitable)}/{len(result)}  |  "
              f"Combined PnL: {fmt_pnl(total_pnl)}  |  "
              f"Time: {elapsed:.1f}s")

        save_last_run(f'backtest_{model}', {
            'date':        args.start + ' → ' + args.end,
            'profitable':  len(profitable),
            'total':       len(result),
            'combined_pnl': total_pnl,
        })

    # ── Comparison if both models run ─────────────────────────────────────────
    if args.model == 'both' and 'v1' in results and 'v2' in results:
        _print_comparison(results['v1'], results['v2'], args.start, args.end)


def _print_comparison(v1_results, v2_results, start, end):
    """Print side by side V1 vs V2 comparison with hybrid routing."""
    header(f"COMPARISON — V1 vs V2  |  {start} → {end}")

    # Build lookup dicts — strip contract suffix properly
    import re
    _re = re.compile(
        r'\d{2}(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)FUT$',
        re.IGNORECASE
    )
    def _base(sym): return _re.sub('', sym)

    v1 = {_base(r['symbol']): r for r in v1_results}
    v2 = {_base(r['symbol']): r for r in v2_results}

    all_syms = sorted(set(list(v1.keys()) + list(v2.keys())))

    print(f"\n  {'Symbol':<20} {'V1 PnL':>12} {'V2 PnL':>12} "
          f"{'Best':>6} {'Route':>6}")
    print(f"  {'─'*60}")

    hybrid_pnl  = 0
    v2_route    = []
    v1_route    = []

    for sym in all_syms:
        r1 = v1.get(sym)
        r2 = v2.get(sym)

        p1 = r1['net_pnl'] if r1 else None
        p2 = r2['net_pnl'] if r2 else None

        # Determine best
        if p1 is not None and p2 is not None:
            if p1 >= p2:
                best = p1
                route = 'V1'
                v1_route.append(sym)
            else:
                best = p2
                route = 'V2'
                v2_route.append(sym)
        elif p1 is not None:
            best  = p1
            route = 'V1'
            v1_route.append(sym)
        else:
            best  = p2
            route = 'V2'
            v2_route.append(sym)

        hybrid_pnl += best if best and best > 0 else 0

        p1_str = fmt_pnl(p1) if p1 is not None else f"{'—':>12}"
        p2_str = fmt_pnl(p2) if p2 is not None else f"{'—':>12}"
        best_c = '\033[92mV1\033[0m' if route == 'V1' else '\033[94mV2\033[0m'

        # Only print symbols where at least one model is profitable
        if (p1 and p1 > 0) or (p2 and p2 > 0):
            print(f"  {sym:<20} {p1_str} {p2_str} "
                  f"{'':>6} {best_c}")

    v1_total = sum(r['net_pnl'] for r in v1_results)
    v2_total = sum(r['net_pnl'] for r in v2_results)

    section("SUMMARY")
    print(f"    V1 total:   {fmt_pnl(v1_total)}")
    print(f"    V2 total:   {fmt_pnl(v2_total)}")
    print(f"    Hybrid:     {fmt_pnl(hybrid_pnl)}  "
          f"\033[92m(best model per symbol)\033[0m")

    section("ROUTING")
    print(f"    Route → V2: {', '.join(v2_route[:8])}"
          f"{'...' if len(v2_route) > 8 else ''}")
    print(f"    Route → V1: {', '.join(v1_route[:8])}"
          f"{'...' if len(v1_route) > 8 else ''}")

    save_last_run('backtest_comparison', {
        'date':       f'{start} → {end}',
        'v1_total':   v1_total,
        'v2_total':   v2_total,
        'hybrid':     hybrid_pnl,
        'v2_route':   v2_route,
        'v1_route':   v1_route,
    })


# ─────────────────────────────────────────────────────────────────────────────
# GRIDSEARCH command
# ─────────────────────────────────────────────────────────────────────────────

def cmd_gridsearch(args):
    models = ['v1', 'v2'] if args.model == 'both' else [args.model]

    for model in models:
        header(
            f"GRID SEARCH — Model: {model.upper()}  "
            f"Period: {args.start} → {args.end}"
        )

        t0 = time.perf_counter()

        if model == 'v1':
            from research.grid_search_v1 import run_grid_search
        else:
            from research.grid_search_v2 import run_grid_search

        # Build dates list
        from research.grid_search_v1 import get_trading_dates
        dates = get_trading_dates(args.start, args.end)

        symbols = args.symbols or None
        params  = run_grid_search(dates=dates, symbols=symbols,
                                   workers=args.workers)

        elapsed = time.perf_counter() - t0

        # Merge with existing params — never overwrite other symbols
        out_path = Path(f'research/findings/{model}_optimal_params.json')
        existing = load_json(out_path) if out_path.exists() else {}
        existing.update(params)   # new params override old for same symbols
        with open(out_path, 'w') as f:
            json.dump(existing, f, indent=2)

        print(f"\n  Optimized: {len(params)} symbols  |  "
              f"Time: {elapsed:.1f}s")
        print(f"  Saved → {out_path}")

        # Top 10
        if params:
            section("TOP 10 BY OOS SHARPE")
            ranked = sorted(
                params.items(),
                key=lambda x: x[1].get('test_sharpe', 0),
                reverse=True
            )[:10]
            print(f"  {'Symbol':<28} {'TestShp':>9} {'TestPnL':>12}")
            print(f"  {'─'*52}")
            for sym, p in ranked:
                print(f"  {sym:<28}  "
                      f"{p.get('test_sharpe',0):>9.2f}  "
                      f"Rs.{p.get('test_pnl',0):>10,.0f}")

        save_last_run(f'gridsearch_{model}', {
            'date':       f'{args.start} → {args.end}',
            'n_symbols':  len(params),
        })


# ─────────────────────────────────────────────────────────────────────────────
# PAIRS command
# ─────────────────────────────────────────────────────────────────────────────

def cmd_pairs(args):
    notebooks = {
        1: 'research/notebooks/pairs/NB_PAIRS_01_find_pairs.py',
        2: 'research/notebooks/pairs/NB_PAIRS_02_spread_analysis.py',
        3: 'research/notebooks/pairs/NB_PAIRS_03_backtest.py',
    }

    # Which notebooks to run
    if args.all:
        to_run = [1, 2, 3]
    else:
        to_run = [args.nb]

    for nb in to_run:
        header(f"PAIRS — Notebook {nb}  |  {args.start} → {args.end}")

        nb_path = notebooks.get(nb)
        if not nb_path or not Path(nb_path).exists():
            print(f"  Notebook {nb} not found at {nb_path}")
            continue

        # Inject dates into notebook via environment variables
        os.environ['PAIRS_START_DATE'] = args.start
        os.environ['PAIRS_END_DATE']   = args.end

        t0 = time.perf_counter()
        exec(open(nb_path).read(), {'__name__': '__main__'})
        elapsed = time.perf_counter() - t0

        print(f"\n  Notebook {nb} complete  |  Time: {elapsed:.1f}s")

        save_last_run(f'pairs_nb{nb}', {
            'date': f'{args.start} → {args.end}',
        })


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH command
# ─────────────────────────────────────────────────────────────────────────────

def cmd_health(args):
    header("PARAM HEALTH CHECK")

    try:
        from src.automation.param_health_check import run_health_check
        import datetime

        today  = datetime.date.today()
        models = ['v1', 'v2'] if args.model == 'all' else [args.model]

        for model in models:
            status, issues = run_health_check(
                model = model,
                today = today,
            )
            icon = (
                '\033[92m✓ OK\033[0m'       if status == 'OK'       else
                '\033[93m~ WARNING\033[0m'  if status == 'WARNING'  else
                '\033[91m✗ CRITICAL\033[0m'
            )
            print(f"  {model.upper()}: {icon}")
            for issue in issues:
                print(f"    → {issue}")

    except ImportError:
        print("  Health check not available — run from GCP VM")


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZE command
# ─────────────────────────────────────────────────────────────────────────────

def cmd_optimize(args):
    header("ROLLING OPTIMIZER")

    if args.transfer:
        # Manual param transfer at contract roll
        from src.backtest.param_loader import run_contract_roll_transfer
        models = ['v1', 'v2'] if args.model == 'both' else [args.model]

        for model in models:
            print(f"\n  Transferring {args.old} → {args.new} for {model}...")
            run_contract_roll_transfer(
                model        = model,
                new_contract = args.new,
                old_contract = args.old,
            )
    else:
        # Regular rolling optimization
        try:
            from src.automation.rolling_optimizer import run_optimization
            models = ['v1', 'v2'] if args.model == 'both' else [args.model]

            for model in models:
                print(f"\n  Optimizing {model}...")
                run_optimization(
                    model   = model,
                    dry_run = args.dry_run,
                )
        except ImportError:
            print("  Rolling optimizer not available — run from GCP VM")


# ─────────────────────────────────────────────────────────────────────────────
# STATUS command
# ─────────────────────────────────────────────────────────────────────────────

def cmd_status(args):
    header(f"QUANT FUND STATUS — {datetime.date.today()}")

    # ── Params ────────────────────────────────────────────────────────────────
    section("PARAMS")
    for model in ['v1', 'v2']:
        path   = Path(f'research/findings/{model}_optimal_params.json')
        if path.exists():
            params = load_json(path)
            params = {k: v for k, v in params.items()
                      if not k.startswith('_')}
            age    = datetime.datetime.now() - datetime.datetime.fromtimestamp(
                path.stat().st_mtime
            )
            age_h  = age.total_seconds() / 3600
            icon   = fmt_status(age_h < 24)
            print(f"    {model.upper()} params:  "
                  f"{len(params)} symbols  |  "
                  f"age: {age_h:.0f}h  |  "
                  f"source: local  {icon}")
        else:
            print(f"    {model.upper()} params:  \033[91mnot found\033[0m")

    # Contract expiry
    today = datetime.date.today()
    NSE_EXPIRY = {
        (2026, 6): datetime.date(2026, 6, 25),
        (2026, 7): datetime.date(2026, 7, 30),
    }
    expiry = NSE_EXPIRY.get((today.year, today.month))
    if expiry:
        days_left = (expiry - today).days
        print(f"\n    Contract: JUNFUT  |  "
              f"Expiry: {expiry}  |  "
              f"{days_left} days remaining")

    # ── Last runs ──────────────────────────────────────────────────────────────
    runs = load_json('research/findings/.last_runs.json')

    section("LAST BACKTEST")
    for model in ['v1', 'v2']:
        key = f'backtest_{model}'
        if key in runs:
            r = runs[key]
            ts = r['timestamp'][:10]
            d  = r['result']
            print(f"    {model.upper()}:  {ts}  |  "
                  f"{d.get('profitable','?')}/{d.get('total','?')} profitable  |  "
                  f"{fmt_pnl(d.get('combined_pnl', 0))}")
        else:
            print(f"    {model.upper()}:  never run")

    section("LAST GRID SEARCH")
    for model in ['v1', 'v2']:
        key = f'gridsearch_{model}'
        if key in runs:
            r = runs[key]
            ts = r['timestamp'][:10]
            d  = r['result']
            print(f"    {model.upper()}:  {ts}  |  "
                  f"{d.get('n_symbols','?')} symbols optimized")
        else:
            print(f"    {model.upper()}:  never run")

    section("LAST PAIRS RUN")
    for nb in [1, 2, 3]:
        key = f'pairs_nb{nb}'
        if key in runs:
            r  = runs[key]
            ts = r['timestamp'][:10]
            print(f"    NB{nb}:  {ts}  |  {r['result'].get('date','')}")
        else:
            print(f"    NB{nb}:  never run")

    # ── Comparison if available ────────────────────────────────────────────────
    if 'backtest_comparison' in runs:
        r = runs['backtest_comparison']
        d = r['result']
        section("LAST COMPARISON")
        print(f"    Date:    {d.get('date','')}")
        print(f"    V1:      {fmt_pnl(d.get('v1_total', 0))}")
        print(f"    V2:      {fmt_pnl(d.get('v2_total', 0))}")
        print(f"    Hybrid:  {fmt_pnl(d.get('hybrid', 0))}")
        v2_route = d.get('v2_route', [])
        if v2_route:
            print(f"    V2 symbols: {', '.join(v2_route[:6])}"
                  f"{'...' if len(v2_route) > 6 else ''}")

    print()
    separator()


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog        = 'qf',
        description = 'Quant Fund CLI',
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
Examples:
  python qf.py backtest --model v1 --start 2026-05-27 --end 2026-06-03
  python qf.py backtest --model both --start 2026-06-02 --end 2026-06-03
  python qf.py gridsearch --model v2 --start 2026-05-27 --end 2026-06-03
  python qf.py pairs --nb 3 --start 2026-05-13 --end 2026-06-03
  python qf.py pairs --all --start 2026-05-13 --end 2026-06-03
  python qf.py health
  python qf.py optimize --model both
  python qf.py optimize --transfer --old MAYFUT --new JUNFUT
  python qf.py status
        """
    )

    sub = parser.add_subparsers(dest='command', required=True)

    # ── backtest ──────────────────────────────────────────────────────────────
    bt = sub.add_parser('backtest', help='Run V1/V2/both backtest')
    bt.add_argument('--model',   required=True,
                    choices=['v1', 'v2', 'both'])
    bt.add_argument('--start',   required=True,
                    help='Start date YYYY-MM-DD')
    bt.add_argument('--end',     required=True,
                    help='End date YYYY-MM-DD')
    bt.add_argument('--symbols', nargs='*', default=None,
                    help='Specific symbols (default: all)')
    bt.add_argument('--no-optimal', action='store_true',
                    help='Use default params instead of optimized')

    # ── gridsearch ────────────────────────────────────────────────────────────
    gs = sub.add_parser('gridsearch', help='Run grid search')
    gs.add_argument('--model',   required=True,
                    choices=['v1', 'v2', 'both'])
    gs.add_argument('--start',   required=True)
    gs.add_argument('--end',     required=True)
    gs.add_argument('--symbols', nargs='*', default=None)
    gs.add_argument('--workers', type=int, default=4)

    # ── pairs ─────────────────────────────────────────────────────────────────
    pr = sub.add_parser('pairs', help='Run pairs notebooks')
    pr.add_argument('--nb',    type=int, choices=[1, 2, 3],
                    help='Notebook number to run')
    pr.add_argument('--all',   action='store_true',
                    help='Run all notebooks NB01 → NB02 → NB03')
    pr.add_argument('--start', required=True)
    pr.add_argument('--end',   required=True)

    # ── health ────────────────────────────────────────────────────────────────
    hc = sub.add_parser('health', help='Param health check')
    hc.add_argument('--model', default='all',
                    choices=['v1', 'v2', 'all'])

    # ── optimize ──────────────────────────────────────────────────────────────
    op = sub.add_parser('optimize', help='Run rolling optimizer')
    op.add_argument('--model',    default='both',
                    choices=['v1', 'v2', 'both'])
    op.add_argument('--transfer', action='store_true',
                    help='Run param transfer at contract roll')
    op.add_argument('--old',      default='MAYFUT',
                    help='Old contract e.g. MAYFUT')
    op.add_argument('--new',      default='JUNFUT',
                    help='New contract e.g. JUNFUT')
    op.add_argument('--dry-run',  action='store_true',
                    help='Run without saving')

    # ── status ────────────────────────────────────────────────────────────────
    sub.add_parser('status', help='Show system status')

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        'backtest':  cmd_backtest,
        'gridsearch': cmd_gridsearch,
        'pairs':     cmd_pairs,
        'health':    cmd_health,
        'optimize':  cmd_optimize,
        'status':    cmd_status,
    }

    fn = dispatch.get(args.command)
    if fn:
        try:
            fn(args)
        except KeyboardInterrupt:
            print("\n\n  Interrupted.")
            sys.exit(0)
        except Exception as e:
            print(f"\n  \033[91mError: {e}\033[0m")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
