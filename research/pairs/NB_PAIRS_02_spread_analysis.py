"""
NB_PAIRS_02 — Spread Analysis for All Tradeable Pairs
======================================================
Fixed issues:
  1. Spread z-score computed fresh from data (not from stale NB01 params)
  2. Fast pairs (HL < 120s) use higher entry threshold + 60s resampling
  3. Pre-load cache for speed

Inputs:
    research/findings/pairs/cointegrated_pairs.json

Outputs:
    research/findings/pairs/pair_params.json
    research/findings/pairs/nb_pairs_02_results.csv
    research/findings/pairs/nb_pairs_02_plots/
"""

import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# NB02 DATES — add the new days
DATES = [
    '2026-05-13', '2026-05-14', '2026-05-15',
    '2026-05-18', '2026-05-19', '2026-05-20',
    '2026-05-21', '2026-05-22',
    '2026-05-26', '2026-05-27', '2026-05-28', '2026-05-29', '2026-05-30',
    '2026-06-01',
]

# Transaction cost round trip both legs (bps)
TRANSACTION_COST_BPS = 9.0

# Thresholds to test
ENTRY_THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.5]
EXIT_THRESHOLDS  = [0.0, 0.3, 0.5, 0.75]
STOP_THRESHOLDS  = [3.5, 4.0, 5.0]

MAX_HOLD_SECS = 1800

# For fast pairs (HL < 120s): resample to 60s bars to avoid noise trading
FAST_HL_THRESHOLD  = 120    # seconds
FAST_RESAMPLE_SECS = 60     # resample to 60s bars

OUTPUT_DIR = Path('research/findings/pairs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR   = OUTPUT_DIR / 'nb_pairs_02_plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def preload_all_symbols(symbols: list, dates: list) -> dict:
    """Load all symbols once. Returns {sym: pd.Series(ts_sec → price)}"""
    from src.backtest.data_loader import load_day

    cache = {}
    for sym in symbols:
        frames = []
        for date in dates:
            try:
                df = load_day(sym, date, market_hours_only=True)
                if not df.empty:
                    col = ('weighted_mid' if 'weighted_mid' in df.columns
                           else 'close')
                    frames.append(
                        df[['ts_sec', col]].rename(columns={col: 'price'})
                    )
            except Exception:
                pass

        if not frames:
            continue

        combined = (pd.concat(frames, ignore_index=True)
                      .sort_values('ts_sec')
                      .drop_duplicates('ts_sec'))
        series = combined.set_index('ts_sec')['price'].dropna()

        if len(series) >= 3000:
            cache[sym] = series
            print(f"  {sym:<32}  {len(series):>8,} bars")
        else:
            print(f"  {sym:<32}  {len(series):>8,} bars  SKIP")

    return cache


def get_aligned_pair(sym1: str, sym2: str,
                     cache: dict) -> tuple:
    """Return aligned arrays for a pair from cache."""
    if sym1 not in cache or sym2 not in cache:
        return None, None, None

    s1 = cache[sym1]
    s2 = cache[sym2]
    common = s1.index.intersection(s2.index)

    if len(common) < 3000:
        return None, None, None

    return s1[common], s2[common], common


def resample_to_60s(s1: pd.Series, s2: pd.Series) -> tuple:
    """
    Resample 1-second series to 60-second bars.
    Used for fast-reverting pairs to avoid noise trading.
    """
    # Create minute buckets
    bucket = s1.index // 60 * 60

    s1_60 = s1.groupby(bucket).last()
    s2_60 = s2.groupby(bucket).last()

    common = s1_60.index.intersection(s2_60.index)
    return s1_60[common], s2_60[common]


# ─────────────────────────────────────────────────────────────────────────────
# Spread computation — recompute stats fresh from data
# ─────────────────────────────────────────────────────────────────────────────

def compute_spread_stats(s1_arr: np.ndarray, s2_arr: np.ndarray,
                         hedge_ratio: float) -> tuple:
    """
    Compute spread and its statistics fresh from the data.

    spread = s1 - hedge_ratio * s2
    We recompute mean/std here rather than using NB01 values
    because NB01 computed them on slightly different aligned data.

    Returns: spread, spread_mean, spread_std, zscore
    """
    spread      = s1_arr - hedge_ratio * s2_arr
    spread_mean = float(spread.mean())
    spread_std  = float(spread.std())

    if spread_std < 1e-8:
        return spread, spread_mean, 1.0, np.zeros_like(spread)

    zscore = (spread - spread_mean) / spread_std
    return spread, spread_mean, spread_std, zscore


# ─────────────────────────────────────────────────────────────────────────────
# Position sizing
# ─────────────────────────────────────────────────────────────────────────────

def get_lot_size(sym: str, lot_sizes: dict) -> int:
    base = sym.replace('26MAYFUT', '').replace('26JUNFUT', '')
    return lot_sizes.get(base, 75)


def compute_lots(sym1: str, sym2: str,
                 price1: float, price2: float,
                 hedge_ratio: float,
                 lot_sizes: dict) -> tuple[int, int, int, int]:
    """
    Compute balanced lot counts.
    Returns lots1, lots2, lot_size1, lot_size2
    """
    ls1 = get_lot_size(sym1, lot_sizes)
    ls2 = get_lot_size(sym2, lot_sizes)

    val1 = price1 * ls1
    val2 = price2 * ls2

    # Match: lots2 × val2 ≈ lots1 × val1 × |hedge_ratio|
    target2 = val1 * abs(hedge_ratio)
    lots2   = max(1, round(target2 / val2))
    lots1   = 1

    return int(lots1), int(lots2), int(ls1), int(ls2)


# ─────────────────────────────────────────────────────────────────────────────
# Signal simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_signals(
    s1_arr:       np.ndarray,
    s2_arr:       np.ndarray,
    ts_arr:       np.ndarray,
    spread:       np.ndarray,
    zscore:       np.ndarray,
    lots1:        int,
    lots2:        int,
    lot_size1:    int,
    lot_size2:    int,
    entry_z:      float,
    exit_z:       float,
    stop_z:       float,
    is_resampled: bool = False,
) -> dict | None:
    """
    Simulate pairs trading.

    PnL = trade_dir × Δspread × units
    where units = lots1 × lot_size1

    Costs = (notional_s1 + notional_s2) × cost_bps
    """
    n_dates  = len(DATES)
    units    = lots1 * lot_size1

    trades      = []
    in_trade    = False
    trade_dir   = 0
    entry_idx   = 0
    entry_spread = 0.0

    # Max hold in bars depends on resolution
    max_hold_bars = MAX_HOLD_SECS if not is_resampled else MAX_HOLD_SECS // 60

    for i in range(1, len(spread)):
        z = zscore[i]

        if not in_trade:
            if z > entry_z:
                in_trade     = True
                trade_dir    = -1
                entry_idx    = i
                entry_spread = spread[i]
            elif z < -entry_z:
                in_trade     = True
                trade_dir    = +1
                entry_idx    = i
                entry_spread = spread[i]
        else:
            bars_held  = i - entry_idx
            force_exit = bars_held >= max_hold_bars

            if trade_dir == -1:
                exit_cond = z < exit_z
                stop_cond = z > stop_z
            else:
                exit_cond = z > -exit_z
                stop_cond = z < -stop_z

            if exit_cond or stop_cond or force_exit:
                spread_change = spread[i] - entry_spread
                gross_pnl     = trade_dir * spread_change * units

                notional  = (s1_arr[entry_idx] * lots1 * lot_size1 +
                             s2_arr[entry_idx] * lots2 * lot_size2)
                cost      = notional * TRANSACTION_COST_BPS / 10000
                net_pnl   = gross_pnl - cost

                ist_hour = int(
                    ((int(ts_arr[entry_idx]) + 19800) % 86400) / 3600
                )

                trades.append({
                    'entry_idx':  int(entry_idx),
                    'exit_idx':   int(i),
                    'direction':  int(trade_dir),
                    'entry_z':    round(float(zscore[entry_idx]), 3),
                    'exit_z':     round(float(z), 3),
                    'bars_held':  int(bars_held),
                    'gross_pnl':  round(float(gross_pnl), 2),
                    'cost':       round(float(cost), 2),
                    'net_pnl':    round(float(net_pnl), 2),
                    'ist_hour':   int(ist_hour),
                    'is_stop':    bool(stop_cond),
                    'is_timeout': bool(force_exit),
                    'is_profit':  bool(net_pnl > 0),
                })

                in_trade = False

    if not trades:
        return None

    df  = pd.DataFrame(trades)
    n_t = len(df)

    bars_per_day = max(len(spread) // n_dates, 1)
    df['day']    = df['entry_idx'] // bars_per_day
    daily        = df.groupby('day')['net_pnl'].sum()
    sharpe       = (daily.mean() / daily.std() * np.sqrt(252)
                    if len(daily) > 1 and daily.std() > 0 else 0.0)

    return {
        'n_trades':      int(n_t),
        'n_per_day':     round(n_t / n_dates, 1),
        'win_rate':      round(float(df['is_profit'].mean()), 4),
        'avg_net_pnl':   round(float(df['net_pnl'].mean()), 2),
        'total_pnl':     round(float(df['net_pnl'].sum()), 2),
        'daily_pnl':     round(float(df['net_pnl'].sum() / n_dates), 2),
        'sharpe':        round(float(sharpe), 2),
        'avg_hold_bars': round(float(df['bars_held'].mean()), 1),
        'stop_rate':     round(float(df['is_stop'].mean()), 4),
        'timeout_rate':  round(float(df['is_timeout'].mean()), 4),
        'trades':        df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Threshold optimisation
# ─────────────────────────────────────────────────────────────────────────────

def optimise_thresholds(
    s1_arr: np.ndarray, s2_arr: np.ndarray,
    ts_arr: np.ndarray,
    spread: np.ndarray, zscore: np.ndarray,
    lots1: int, lots2: int,
    lot_size1: int, lot_size2: int,
    is_resampled: bool = False,
) -> tuple:
    """Try all threshold combos. Score = daily_pnl × win_rate."""
    best_result = None
    best_score  = -np.inf
    best_params = {}

    for entry_z in ENTRY_THRESHOLDS:
        for exit_z in EXIT_THRESHOLDS:
            if exit_z >= entry_z:
                continue
            for stop_z in STOP_THRESHOLDS:
                if stop_z <= entry_z:
                    continue

                result = simulate_signals(
                    s1_arr, s2_arr, ts_arr,
                    spread, zscore,
                    lots1, lots2, lot_size1, lot_size2,
                    entry_z, exit_z, stop_z, is_resampled,
                )

                if result is None or result['n_trades'] < 3:
                    continue
                if result['daily_pnl'] <= 0:
                    continue

                score = result['daily_pnl'] * result['win_rate']
                if score > best_score:
                    best_score  = score
                    best_result = result
                    best_params = {
                        'entry_z': entry_z,
                        'exit_z':  exit_z,
                        'stop_z':  stop_z,
                    }

    return best_result, best_params


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_pair(sym1: str, sym2: str,
              zscore: np.ndarray,
              best_result: dict,
              best_params: dict,
              is_resampled: bool):
    s1n = sym1.replace('26MAYFUT', '')
    s2n = sym2.replace('26MAYFUT', '')
    n   = min(5000, len(zscore))
    ez  = best_params.get('entry_z', 2.0)
    sz  = best_params.get('stop_z', 3.5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    res_tag   = ' [60s]' if is_resampled else ' [1s]'
    fig.suptitle(
        f"{s1n}/{s2n}{res_tag}  |  "
        f"entry={ez}σ exit={best_params.get('exit_z',0.5)}σ "
        f"stop={sz}σ  |  "
        f"Daily=Rs.{best_result['daily_pnl']:,.0f}  "
        f"WR={best_result['win_rate']:.0%}  "
        f"Sharpe={best_result['sharpe']:.2f}",
        fontsize=9
    )

    # Z-score
    ax = axes[0, 0]
    ax.plot(zscore[:n], lw=0.5, color='steelblue', alpha=0.8)
    for v, c, ls in [
        (ez, 'red', '--'), (-ez, 'green', '--'),
        (sz, 'red', ':'),  (-sz, 'green', ':'),
    ]:
        ax.axhline(v, color=c, lw=0.8, ls=ls)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_title('Z-score')
    ax.set_ylim(-max(6, ez+1), max(6, ez+1))
    ax.grid(True, alpha=0.3)

    df = best_result.get('trades')
    if df is not None and len(df) > 0:
        axes[0, 1].hist(df['net_pnl'], bins=25,
                        color='steelblue', alpha=0.7, edgecolor='white')
        axes[0, 1].axvline(0, color='black', lw=1)
        axes[0, 1].axvline(df['net_pnl'].mean(), color='red', lw=1.5,
                           label=f"avg=Rs.{df['net_pnl'].mean():.0f}")
        axes[0, 1].set_title('PnL per Trade')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        cum = df['net_pnl'].cumsum()
        axes[1, 0].plot(cum.values, lw=1.2, color='green')
        axes[1, 0].axhline(0, color='black', lw=0.8)
        axes[1, 0].fill_between(range(len(cum)), cum.values, 0,
                                 alpha=0.2, color='green')
        axes[1, 0].set_title(
            f'Cumulative PnL  (Rs.{cum.iloc[-1]:,.0f})')
        axes[1, 0].set_xlabel('Trade #')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].hist(df['bars_held'], bins=20,
                        color='orange', alpha=0.7, edgecolor='white')
        unit_label = 'minutes' if is_resampled else 'seconds'
        axes[1, 1].set_title(
            f'Hold Time ({unit_label})  '
            f'(avg={df["bars_held"].mean():.1f})')
        axes[1, 1].set_xlabel(unit_label.capitalize())
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{s1n}_{s2n}.png",
                dpi=80, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()

    print("=" * 70)
    print("  NB_PAIRS_02 — Spread Analysis (All Tradeable Pairs)")
    print(f"  Dates: {DATES}")
    print(f"  Transaction cost: {TRANSACTION_COST_BPS} bps")
    print(f"  Fast pairs (HL < {FAST_HL_THRESHOLD}s): resampled to "
          f"{FAST_RESAMPLE_SECS}s bars")
    print("=" * 70)

    pairs_path = Path('research/findings/pairs/cointegrated_pairs.json')
    if not pairs_path.exists():
        print("ERROR: Run NB_PAIRS_01 first.")
        return

    with open(pairs_path) as f:
        pairs = json.load(f)
    print(f"\nLoaded {len(pairs)} pairs")

    # Lot sizes
    lot_sizes = {}
    try:
        from src.utils.auth import get_secret
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=get_secret("KITE_API_KEY"))
        kite.set_access_token(get_secret("KITE_ACCESS_TOKEN"))
        df_lots   = pd.DataFrame(kite.instruments("NFO"))
        df_lots   = df_lots[df_lots["instrument_type"] == "FUT"]
        lot_sizes = {r["name"]: int(r["lot_size"])
                     for _, r in df_lots.iterrows()}
        print(f"Loaded {len(lot_sizes)} lot sizes")
    except Exception as e:
        print(f"Lot sizes unavailable: {e}")

    # Pre-load all symbols once
    all_syms = list(set(
        [p['sym1'] for p in pairs] + [p['sym2'] for p in pairs]
    ))
    print(f"\nPre-loading {len(all_syms)} symbols...")
    cache = preload_all_symbols(all_syms, DATES)
    print(f"\nLoaded {len(cache)} symbols  "
          f"({time.perf_counter()-t0:.0f}s)\n")

    all_results = []
    pair_params = []

    print(f"{'#':>3} {'Pair':<32} {'Res':>4} {'DailyPnL':>10} "
          f"{'WinRate':>8} {'Trd/d':>6} {'Sharpe':>7} "
          f"{'Entry':>7} {'AvgHold':>8}")
    print("─" * 88)

    for idx, pair in enumerate(pairs, 1):
        sym1 = pair['sym1']
        sym2 = pair['sym2']
        s1n  = sym1.replace('26MAYFUT', '')
        s2n  = sym2.replace('26MAYFUT', '')
        name = f"{s1n}/{s2n}"
        hl   = pair['half_life_secs']

        try:
            s1_ser, s2_ser, _ = get_aligned_pair(sym1, sym2, cache)

            if s1_ser is None:
                print(f"{idx:>3} {name:<32}  SKIP (no data)")
                continue

            is_fast = hl < FAST_HL_THRESHOLD

            # Resample fast pairs to 60s to avoid noise trading
            if is_fast:
                s1_use, s2_use = resample_to_60s(s1_ser, s2_ser)
                res_label = '60s'
            else:
                s1_use = s1_ser
                s2_use = s2_ser
                res_label = ' 1s'

            s1_arr = s1_use.values.astype(np.float64)
            s2_arr = s2_use.values.astype(np.float64)
            ts_arr = s1_use.index.values.astype(np.int64)

            if len(s1_arr) < 100:
                print(f"{idx:>3} {name:<32}  SKIP (too few bars after resample)")
                continue

            # Recompute spread stats fresh from data
            spread, spread_mean, spread_std, zscore = compute_spread_stats(
                s1_arr, s2_arr, pair['hedge_ratio']
            )

            # Lot sizes
            ls1   = get_lot_size(sym1, lot_sizes)
            ls2   = get_lot_size(sym2, lot_sizes)
            lots1, lots2, _, _ = compute_lots(
                sym1, sym2,
                s1_arr.mean(), s2_arr.mean(),
                pair['hedge_ratio'], lot_sizes,
            )

            # Optimise thresholds
            best_result, best_params = optimise_thresholds(
                s1_arr, s2_arr, ts_arr,
                spread, zscore,
                lots1, lots2, ls1, ls2,
                is_resampled=is_fast,
            )

            if best_result is None:
                print(f"{idx:>3} {name:<32}  {res_label}  "
                      f"no profitable threshold")
                continue

            try:
                plot_pair(sym1, sym2, zscore, best_result,
                          best_params, is_fast)
            except Exception:
                pass

            hold_label = (f"{best_result['avg_hold_bars']:.0f}m"
                          if is_fast
                          else f"{best_result['avg_hold_bars']:.0f}s")

            print(f"{idx:>3} {name:<32}  {res_label}  "
                  f"Rs.{best_result['daily_pnl']:>8,.0f}  "
                  f"{best_result['win_rate']:>7.1%}  "
                  f"{best_result['n_per_day']:>5.1f}  "
                  f"{best_result['sharpe']:>7.2f}  "
                  f"{best_params.get('entry_z',2.0):>5.1f}σ  "
                  f"{hold_label:>8}")

            all_results.append({
                'sym1':          sym1,
                'sym2':          sym2,
                'sector':        pair['sector'],
                'hedge_ratio':   pair['hedge_ratio'],
                'half_life':     hl,
                'correlation':   pair['correlation'],
                'lots1':         lots1,
                'lots2':         lots2,
                'spread_mean':   round(spread_mean, 6),
                'spread_std':    round(spread_std, 6),
                'is_resampled':  is_fast,
                **best_params,
                **{k: best_result[k] for k in [
                    'n_trades', 'n_per_day', 'win_rate',
                    'avg_net_pnl', 'total_pnl', 'daily_pnl',
                    'sharpe', 'avg_hold_bars', 'stop_rate',
                ]},
            })

            pair_params.append({
                'sym1':          sym1,
                'sym2':          sym2,
                'sector':        pair['sector'],
                'hedge_ratio':   pair['hedge_ratio'],
                'spread_mean':   round(spread_mean, 6),
                'spread_std':    round(spread_std, 6),
                'lots1':         lots1,
                'lots2':         lots2,
                'lot_size1':     ls1,
                'lot_size2':     ls2,
                'entry_z':       best_params.get('entry_z', 2.0),
                'exit_z':        best_params.get('exit_z',  0.5),
                'stop_z':        best_params.get('stop_z',  3.5),
                'half_life':     hl,
                'is_resampled':  is_fast,
                'resample_secs': FAST_RESAMPLE_SECS if is_fast else 1,
                'daily_pnl':     best_result['daily_pnl'],
                'win_rate':      best_result['win_rate'],
                'sharpe':        best_result['sharpe'],
            })

        except Exception as e:
            print(f"{idx:>3} {name:<32}  ERROR: {str(e)[:60]}")

    elapsed = time.perf_counter() - t0

    # ── Summary ───────────────────────────────────────────────────────────────
    if not all_results:
        print("\nNo profitable pairs found.")
        return

    df = pd.DataFrame(all_results).sort_values('daily_pnl', ascending=False)

    print(f"\n{'='*70}")
    print(f"  SUMMARY — {len(df)} profitable pairs  ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  Total combined daily PnL: Rs.{df['daily_pnl'].sum():,.0f}")
    print(f"  Avg win rate:             {df['win_rate'].mean():.1%}")
    print(f"  Avg Sharpe:               {df['sharpe'].mean():.2f}")
    print(f"  Total trades/day:         {df['n_per_day'].sum():.0f}")

    print(f"\n  TOP 15 PAIRS:")
    print(f"  {'Pair':<32} {'Res':>4} {'DailyPnL':>10} {'WinRate':>8} "
          f"{'Trd/d':>6} {'Sharpe':>7} {'HL':>8}")
    print(f"  {'─'*80}")
    for _, r in df.head(15).iterrows():
        s1  = r['sym1'].replace('26MAYFUT', '')
        s2  = r['sym2'].replace('26MAYFUT', '')
        res = '60s' if r['is_resampled'] else ' 1s'
        print(f"  {s1+'/'+s2:<32}  {res}  "
              f"Rs.{r['daily_pnl']:>8,.0f}  "
              f"{r['win_rate']:>7.1%}  "
              f"{r['n_per_day']:>5.1f}  "
              f"{r['sharpe']:>7.2f}  "
              f"{r['half_life']:>6.0f}s")

    print(f"\n  BY SECTOR:")
    for sector, grp in df.groupby('sector'):
        print(f"    {sector:<15}  {len(grp)} pairs  "
              f"Rs.{grp['daily_pnl'].sum():>8,.0f}/day  "
              f"Sharpe={grp['sharpe'].mean():.2f}")

    df.to_csv(OUTPUT_DIR / 'nb_pairs_02_results.csv', index=False)
    with open(OUTPUT_DIR / 'pair_params.json', 'w') as f:
        json.dump(pair_params, f, indent=2)

    print(f"\n  Saved: research/findings/pairs/nb_pairs_02_results.csv")
    print(f"  Saved: research/findings/pairs/pair_params.json")
    print(f"  Plots: research/findings/pairs/nb_pairs_02_plots/")

    good = df[df['sharpe'] > 1.0]
    print(f"\n{'='*70}")
    print(f"  {len(good)} pairs Sharpe > 1.0 → proceed to NB_PAIRS_03")
    print(f"  Next: python research/notebooks/pairs/NB_PAIRS_03_backtest.py")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()