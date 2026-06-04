"""
NB_PAIRS_03 — Walk-Forward Backtest for All Pairs
==================================================
Research Question:
    Do the pairs strategies hold out-of-sample?
    Which pairs are genuinely tradeable vs in-sample overfit?

Method:
    Rolling walk-forward validation:
        For each test day (last 2 days):
            Train: compute spread params on all prior days
            Test:  trade signals on that day using train params

    This prevents lookahead bias — we never use future data
    to compute spread mean/std.

Decision criteria:
    Green  → OOS Sharpe > 1.5, win rate > 55%, PnL > 0
    Yellow → OOS Sharpe > 0.8, win rate > 50%
    Red    → OOS Sharpe < 0.8 or PnL < 0  (don't trade)

Inputs:
    research/findings/pairs/cointegrated_pairs.json
    research/findings/pairs/pair_params.json

Outputs:
    research/findings/pairs/nb_pairs_03_oos_results.csv
    research/findings/pairs/tradeable_pairs_final.json  ← GREEN pairs only
    research/findings/pairs/nb_pairs_03_plots/
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

# All 8 trading days
# NEW — your full dataset
ALL_DATES = [
    '2026-05-13', '2026-05-14', '2026-05-15',
    '2026-05-18', '2026-05-19', '2026-05-20',
    '2026-05-21', '2026-05-22',
    '2026-05-26', '2026-05-27', '2026-05-28', '2026-05-29', '2026-05-30',
    '2026-06-01',
]
N_TEST_DAYS = 5

# Walk-forward: use last 2 days as OOS test days
# Train on all days before the test day
TRAIN_DATES   = ALL_DATES[:-N_TEST_DAYS]   # May 13-20
TEST_DATES    = ALL_DATES[-N_TEST_DAYS:]    # May 21-22

TRANSACTION_COST_BPS = 9.0
MAX_HOLD_SECS        = 1800
FAST_HL_THRESHOLD    = 120
FAST_RESAMPLE_SECS   = 60

# Decision thresholds
GREEN_SHARPE   = 1.5
GREEN_WINRATE  = 0.55
YELLOW_SHARPE  = 0.8
YELLOW_WINRATE = 0.50

OUTPUT_DIR = Path('research/findings/pairs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR   = OUTPUT_DIR / 'nb_pairs_03_plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def preload_all_symbols(symbols: list, dates: list) -> dict:
    """Load all symbols for given dates. Returns {sym: pd.Series}"""
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

        if len(series) >= 500:
            cache[sym] = series

    return cache


def get_date_mask(series: pd.Series, date: str) -> pd.Series:
    """Extract bars for a specific date from a series."""
    # ts_sec for a date in IST: date 09:15 to 15:30
    import datetime
    d   = datetime.date.fromisoformat(date)
    # Convert IST 09:15 and 15:30 to UTC unix timestamps
    t_open  = int(datetime.datetime(d.year, d.month, d.day,
                                     3, 45, 0).timestamp())   # 09:15 IST = 03:45 UTC
    t_close = int(datetime.datetime(d.year, d.month, d.day,
                                     10, 0, 0).timestamp())   # 15:30 IST = 10:00 UTC
    return series[(series.index >= t_open) & (series.index <= t_close)]


def resample_60s(s: pd.Series) -> pd.Series:
    bucket = s.index // 60 * 60
    return s.groupby(bucket).last()


# ─────────────────────────────────────────────────────────────────────────────
# Spread parameter estimation (train phase)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_spread_params(s1_train: np.ndarray,
                            s2_train: np.ndarray,
                            hedge_ratio: float) -> tuple[float, float]:
    """
    Estimate spread mean and std from training data.
    These params are then used to z-score the test data.
    """
    spread      = s1_train - hedge_ratio * s2_train
    spread_mean = float(spread.mean())
    spread_std  = float(spread.std())
    return spread_mean, max(spread_std, 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Signal simulation (test phase)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_one_day(
    s1_arr:       np.ndarray,
    s2_arr:       np.ndarray,
    ts_arr:       np.ndarray,
    hedge_ratio:  float,
    spread_mean:  float,   # from TRAIN data
    spread_std:   float,   # from TRAIN data
    lots1:        int,
    lots2:        int,
    lot_size1:    int,
    lot_size2:    int,
    entry_z:      float,
    exit_z:       float,
    stop_z:       float,
    is_resampled: bool = False,
) -> list:
    """
    Simulate trades for one day using train-estimated spread params.
    Returns list of trade dicts.
    """
    spread   = s1_arr - hedge_ratio * s2_arr
    zscore   = (spread - spread_mean) / spread_std
    units    = lots1 * lot_size1

    max_hold = MAX_HOLD_SECS if not is_resampled else MAX_HOLD_SECS // 60

    trades    = []
    in_trade  = False
    trade_dir = 0
    entry_idx = 0
    entry_spr = 0.0

    for i in range(1, len(spread)):
        z = zscore[i]

        if not in_trade:
            if z > entry_z:
                in_trade  = True
                trade_dir = -1
                entry_idx = i
                entry_spr = spread[i]
            elif z < -entry_z:
                in_trade  = True
                trade_dir = +1
                entry_idx = i
                entry_spr = spread[i]
        else:
            bars_held  = i - entry_idx
            force_exit = bars_held >= max_hold

            if trade_dir == -1:
                exit_cond = z < exit_z
                stop_cond = z > stop_z
            else:
                exit_cond = z > -exit_z
                stop_cond = z < -stop_z

            if exit_cond or stop_cond or force_exit:
                gross_pnl = trade_dir * (spread[i] - entry_spr) * units
                notional  = (s1_arr[entry_idx] * lots1 * lot_size1 +
                             s2_arr[entry_idx] * lots2 * lot_size2)
                cost      = notional * TRANSACTION_COST_BPS / 10000
                net_pnl   = gross_pnl - cost

                trades.append({
                    'gross_pnl':  round(float(gross_pnl), 2),
                    'cost':       round(float(cost), 2),
                    'net_pnl':    round(float(net_pnl), 2),
                    'bars_held':  int(bars_held),
                    'is_stop':    bool(stop_cond),
                    'is_profit':  bool(net_pnl > 0),
                    'entry_z':    round(float(zscore[entry_idx]), 3),
                    'exit_z':     round(float(z), 3),
                })
                in_trade = False

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward validation for one pair
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_pair(
    pair:       dict,
    params:     dict,
    full_cache: dict,
) -> dict:
    """
    Walk-forward validation for one pair.

    For each test day:
        1. Load all prior days (train set)
        2. Estimate spread mean/std from train
        3. Simulate trades on test day using train params
        4. Record PnL

    Returns OOS metrics.
    """
    sym1        = pair['sym1']
    sym2        = pair['sym2']
    hedge_ratio = pair['hedge_ratio']
    is_fast     = pair['half_life_secs'] < FAST_HL_THRESHOLD

    entry_z = params.get('entry_z', 2.0)
    exit_z  = params.get('exit_z',  0.5)
    stop_z  = params.get('stop_z',  3.5)
    lots1   = params.get('lots1',   1)
    lots2   = params.get('lots2',   1)
    ls1     = params.get('lot_size1', 75)
    ls2     = params.get('lot_size2', 75)

    if sym1 not in full_cache or sym2 not in full_cache:
        return {'ok': False, 'error': 'no data'}

    s1_full = full_cache[sym1]
    s2_full = full_cache[sym2]

    all_oos_trades  = []
    daily_oos_pnl   = {}

    # Rolling walk-forward: each test day uses all prior days as train
    for test_date in TEST_DATES:
        test_idx = ALL_DATES.index(test_date)
        train_dates_wf = ALL_DATES[:test_idx]

        if len(train_dates_wf) < 3:
            continue

        # ── Train: gather all prior data ──────────────────────────────────────
        train_frames_s1 = []
        train_frames_s2 = []
        for d in train_dates_wf:
            s1_d = get_date_mask(s1_full, d)
            s2_d = get_date_mask(s2_full, d)
            if len(s1_d) > 100 and len(s2_d) > 100:
                common = s1_d.index.intersection(s2_d.index)
                if len(common) > 100:
                    train_frames_s1.append(s1_d[common])
                    train_frames_s2.append(s2_d[common])

        if not train_frames_s1:
            continue

        s1_train = np.concatenate([f.values for f in train_frames_s1])
        s2_train = np.concatenate([f.values for f in train_frames_s2])

        if is_fast:
            # For fast pairs: resample train to 60s
            s1_tr_ser = pd.Series(s1_train)
            s2_tr_ser = pd.Series(s2_train)
            bucket    = s1_tr_ser.index // 60 * 60
            s1_train  = s1_tr_ser.groupby(bucket).last().values
            s2_train  = s2_tr_ser.groupby(bucket).last().values

        # Estimate spread params from train
        spread_mean, spread_std = estimate_spread_params(
            s1_train, s2_train, hedge_ratio
        )

        # ── Test: simulate on test day ────────────────────────────────────────
        s1_test_ser = get_date_mask(s1_full, test_date)
        s2_test_ser = get_date_mask(s2_full, test_date)
        common_test = s1_test_ser.index.intersection(s2_test_ser.index)

        if len(common_test) < 100:
            continue

        s1_test = s1_test_ser[common_test]
        s2_test = s2_test_ser[common_test]

        if is_fast:
            s1_test = resample_60s(s1_test)
            s2_test = resample_60s(s2_test)
            common_test = s1_test.index.intersection(s2_test.index)
            s1_test = s1_test[common_test]
            s2_test = s2_test[common_test]

        if len(s1_test) < 10:
            continue

        day_trades = simulate_one_day(
            s1_arr       = s1_test.values.astype(np.float64),
            s2_arr       = s2_test.values.astype(np.float64),
            ts_arr       = s1_test.index.values.astype(np.int64),
            hedge_ratio  = hedge_ratio,
            spread_mean  = spread_mean,
            spread_std   = spread_std,
            lots1        = lots1,
            lots2        = lots2,
            lot_size1    = ls1,
            lot_size2    = ls2,
            entry_z      = entry_z,
            exit_z       = exit_z,
            stop_z       = stop_z,
            is_resampled = is_fast,
        )

        day_pnl = sum(t['net_pnl'] for t in day_trades)
        daily_oos_pnl[test_date] = day_pnl
        for t in day_trades:
            t['date'] = test_date
        all_oos_trades.extend(day_trades)

    # ── Compute OOS metrics ───────────────────────────────────────────────────
    if not all_oos_trades:
        return {
            'ok':           True,
            'n_oos_trades': 0,
            'oos_pnl':      0.0,
            'oos_daily':    0.0,
            'oos_win_rate': 0.0,
            'oos_sharpe':   0.0,
            'daily_pnl':    daily_oos_pnl,
            'grade':        'RED',
        }

    df        = pd.DataFrame(all_oos_trades)
    n_trades  = len(df)
    total_pnl = float(df['net_pnl'].sum())
    daily_avg = total_pnl / max(len(TEST_DATES), 1)
    win_rate  = float(df['is_profit'].mean())

    daily_series = list(daily_oos_pnl.values())
    if len(daily_series) > 1:
        arr    = np.array(daily_series)
        sharpe = (arr.mean() / arr.std() * np.sqrt(252)
                  if arr.std() > 0 else 0.0)
    elif len(daily_series) == 1:
        sharpe = 10.0 if daily_series[0] > 0 else -10.0
    else:
        sharpe = 0.0

    # Grade
    if (sharpe >= GREEN_SHARPE and
            win_rate >= GREEN_WINRATE and
            total_pnl > 0):
        grade = 'GREEN'
    elif (sharpe >= YELLOW_SHARPE and
          win_rate >= YELLOW_WINRATE and
          total_pnl > 0):
        grade = 'YELLOW'
    else:
        grade = 'RED'

    return {
        'ok':           True,
        'n_oos_trades': int(n_trades),
        'n_oos_per_day': round(n_trades / max(len(TEST_DATES), 1), 1),
        'oos_pnl':      round(total_pnl, 2),
        'oos_daily':    round(daily_avg, 2),
        'oos_win_rate': round(win_rate, 4),
        'oos_sharpe':   round(float(sharpe), 2),
        'daily_pnl':    daily_oos_pnl,
        'trades':       df,
        'grade':        grade,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_oos_summary(results: list):
    """Plot OOS PnL distribution and grade breakdown."""
    df = pd.DataFrame([r for r in results if r.get('ok')])

    if df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('NB_PAIRS_03 — OOS Walk-Forward Results', fontsize=12)

    # Grade breakdown
    grade_counts = df['grade'].value_counts()
    colors = {'GREEN': 'green', 'YELLOW': 'gold', 'RED': 'red'}
    bars = axes[0].bar(
        grade_counts.index,
        grade_counts.values,
        color=[colors.get(g, 'grey') for g in grade_counts.index]
    )
    axes[0].set_title('Grade Distribution')
    axes[0].set_ylabel('Number of Pairs')
    for bar, val in zip(bars, grade_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     str(val), ha='center', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # OOS daily PnL distribution
    axes[1].hist(df['oos_daily'], bins=20,
                 color='steelblue', alpha=0.7, edgecolor='white')
    axes[1].axvline(0, color='black', lw=1.5)
    axes[1].axvline(df['oos_daily'].mean(), color='red', lw=1.5,
                    label=f"avg=Rs.{df['oos_daily'].mean():.0f}")
    axes[1].set_title('OOS Daily PnL Distribution')
    axes[1].set_xlabel('Rs./day')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # OOS Sharpe distribution
    sharpe_clipped = df['oos_sharpe'].clip(-20, 20)
    axes[2].hist(sharpe_clipped, bins=20,
                 color='orange', alpha=0.7, edgecolor='white')
    axes[2].axvline(0, color='black', lw=1.5)
    axes[2].axvline(GREEN_SHARPE, color='green', lw=1.5,
                    ls='--', label=f'Green={GREEN_SHARPE}')
    axes[2].axvline(YELLOW_SHARPE, color='gold', lw=1.5,
                    ls='--', label=f'Yellow={YELLOW_SHARPE}')
    axes[2].set_title('OOS Sharpe Distribution')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'nb_pairs_03_oos_summary.png',
                dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Plot: research/findings/pairs/nb_pairs_03_oos_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()

    print("=" * 70)
    print("  NB_PAIRS_03 — Walk-Forward OOS Backtest")
    print(f"  Train dates: {TRAIN_DATES}")
    print(f"  Test dates:  {TEST_DATES}")
    print(f"  Decision: GREEN=Sharpe>{GREEN_SHARPE} WR>{GREEN_WINRATE:.0%}")
    print("=" * 70)

    # Load pairs and params
    pairs_path  = Path('research/findings/pairs/cointegrated_pairs.json')
    params_path = Path('research/findings/pairs/pair_params.json')

    if not pairs_path.exists() or not params_path.exists():
        print("ERROR: Run NB_PAIRS_01 and NB_PAIRS_02 first.")
        return

    with open(pairs_path) as f:
        pairs = json.load(f)
    with open(params_path) as f:
        all_params = json.load(f)

    # Build params lookup
    params_lookup = {
        (p['sym1'], p['sym2']): p for p in all_params
    }

    print(f"\nLoaded {len(pairs)} pairs, {len(all_params)} param sets")

    # Pre-load ALL dates (train + test) for all symbols
    all_syms = list(set(
        [p['sym1'] for p in pairs] + [p['sym2'] for p in pairs]
    ))
    print(f"\nPre-loading {len(all_syms)} symbols for all {len(ALL_DATES)} dates...")
    full_cache = preload_all_symbols(all_syms, ALL_DATES)
    print(f"Loaded {len(full_cache)} symbols  "
          f"({time.perf_counter()-t0:.0f}s)\n")

    # Run walk-forward for each pair
    all_results = []

    grade_icon = {'GREEN': '✓', 'YELLOW': '~', 'RED': '✗'}

    print(f"{'#':>3} {'Pair':<32} {'Grade':>6} "
          f"{'OOS_PnL':>10} {'OOS_WR':>8} "
          f"{'OOS_Shp':>8} {'Trd/d':>6}")
    print("─" * 75)

    for idx, pair in enumerate(pairs, 1):
        sym1 = pair['sym1']
        sym2 = pair['sym2']
        s1n  = sym1.replace('26MAYFUT', '')
        s2n  = sym2.replace('26MAYFUT', '')
        name = f"{s1n}/{s2n}"

        key    = (sym1, sym2)
        params = params_lookup.get(key)

        if params is None:
            print(f"{idx:>3} {name:<32}  no params (run NB02 first)")
            continue

        result = walk_forward_pair(pair, params, full_cache)
        result['sym1']   = sym1
        result['sym2']   = sym2
        result['sector'] = pair['sector']
        result['half_life'] = pair['half_life_secs']
        result['correlation'] = pair['correlation']

        all_results.append(result)

        if not result['ok']:
            print(f"{idx:>3} {name:<32}  ERROR: {result.get('error','')}")
            continue

        grade = result['grade']
        icon  = grade_icon.get(grade, '?')

        if result['n_oos_trades'] == 0:
            print(f"{idx:>3} {name:<32}  {grade:>5}  "
                  f"no OOS trades")
        else:
            print(f"{idx:>3} {name:<32}  "
                  f"{icon} {grade:<5}  "
                  f"Rs.{result['oos_daily']:>8,.0f}  "
                  f"{result['oos_win_rate']:>7.1%}  "
                  f"{result['oos_sharpe']:>8.2f}  "
                  f"{result['n_oos_per_day']:>5.1f}")

    elapsed = time.perf_counter() - t0

    # ── Summary ───────────────────────────────────────────────────────────────
    valid   = [r for r in all_results if r.get('ok') and r['n_oos_trades'] > 0]
    green   = [r for r in valid if r['grade'] == 'GREEN']
    yellow  = [r for r in valid if r['grade'] == 'YELLOW']
    red     = [r for r in valid if r['grade'] == 'RED']

    print(f"\n{'='*70}")
    print(f"  OOS WALK-FORWARD RESULTS  ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  Total pairs tested:  {len(valid)}")
    print(f"  GREEN  (tradeable):  {len(green)}")
    print(f"  YELLOW (monitor):    {len(yellow)}")
    print(f"  RED    (skip):       {len(red)}")

    if green:
        green_daily = sum(r['oos_daily'] for r in green)
        print(f"\n  GREEN pairs combined OOS daily PnL: "
              f"Rs.{green_daily:,.0f}")

        print(f"\n  GREEN PAIRS — sorted by OOS Sharpe:")
        print(f"  {'Pair':<32} {'OOS_Daily':>10} {'OOS_WR':>8} "
              f"{'OOS_Shp':>8} {'HL':>8} {'Corr':>7}")
        print(f"  {'─'*72}")

        green_sorted = sorted(green,
                              key=lambda x: x['oos_sharpe'],
                              reverse=True)
        for r in green_sorted:
            s1 = r['sym1'].replace('26MAYFUT', '')
            s2 = r['sym2'].replace('26MAYFUT', '')
            print(f"  {s1+'/'+s2:<32}  "
                  f"Rs.{r['oos_daily']:>8,.0f}  "
                  f"{r['oos_win_rate']:>7.1%}  "
                  f"{r['oos_sharpe']:>8.2f}  "
                  f"{r['half_life']:>6.0f}s  "
                  f"{r['correlation']:>7.3f}")

    if yellow:
        print(f"\n  YELLOW PAIRS (monitor, paper trade first):")
        for r in sorted(yellow,
                        key=lambda x: x['oos_sharpe'], reverse=True):
            s1 = r['sym1'].replace('26MAYFUT', '')
            s2 = r['sym2'].replace('26MAYFUT', '')
            print(f"    {s1+'/'+s2:<32}  "
                  f"Rs.{r['oos_daily']:>7,.0f}/day  "
                  f"Sharpe={r['oos_sharpe']:.2f}  "
                  f"WR={r['oos_win_rate']:.0%}")

    # ── Save results ──────────────────────────────────────────────────────────

    # Full CSV
    rows = []
    for r in valid:
        rows.append({
            'sym1':          r['sym1'],
            'sym2':          r['sym2'],
            'sector':        r['sector'],
            'half_life':     r['half_life'],
            'correlation':   r['correlation'],
            'grade':         r['grade'],
            'oos_pnl':       r['oos_pnl'],
            'oos_daily':     r['oos_daily'],
            'oos_win_rate':  r['oos_win_rate'],
            'oos_sharpe':    r['oos_sharpe'],
            'n_oos_trades':  r['n_oos_trades'],
            'n_oos_per_day': r['n_oos_per_day'],
        })

    df_out = pd.DataFrame(rows).sort_values('oos_sharpe', ascending=False)
    df_out.to_csv(OUTPUT_DIR / 'nb_pairs_03_oos_results.csv', index=False)
    print(f"\n  Saved: research/findings/pairs/nb_pairs_03_oos_results.csv")

    # Final tradeable pairs JSON (GREEN only)
    if green:
        # Merge with pair_params for complete trading config
        params_lookup2 = {
            (p['sym1'], p['sym2']): p
            for p in all_params
        }
        final_pairs = []
        for r in green_sorted:
            key    = (r['sym1'], r['sym2'])
            params = params_lookup2.get(key, {})
            final_pairs.append({
                **params,
                'oos_daily':    r['oos_daily'],
                'oos_win_rate': r['oos_win_rate'],
                'oos_sharpe':   r['oos_sharpe'],
                'grade':        'GREEN',
            })

        with open(OUTPUT_DIR / 'tradeable_pairs_final.json', 'w') as f:
            json.dump(final_pairs, f, indent=2)
        print(f"  Saved: research/findings/pairs/tradeable_pairs_final.json")
        print(f"         ({len(final_pairs)} GREEN pairs ready for strategy)")

    # Plot summary
    try:
        plot_oos_summary(all_results)
    except Exception as e:
        print(f"  Plot error: {e}")

    # ── Next steps ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  NEXT STEPS")
    print(f"{'='*70}")

    if len(green) >= 5:
        print(f"""
  ✓ {len(green)} GREEN pairs confirmed OOS — build the strategy

  Next: build src/strategies/stat_arb/pairs_strategy.py
    - Uses tradeable_pairs_final.json for pair configs
    - Runs alongside MM strategy on different symbols
    - Paper trade first: 2 weeks before going live

  Capital needed per pair: Rs.10-20L (both legs combined)
  Start with top 5 GREEN pairs by OOS Sharpe
        """)
    elif len(green) > 0:
        print(f"""
  ~ {len(green)} GREEN pairs — limited but tradeable
  Consider also paper trading YELLOW pairs

  Recommendation: get more data (30 days) then re-run NB_PAIRS_01-03
  More data → better cointegration estimates → more GREEN pairs
        """)
    else:
        print(f"""
  ✗ No GREEN pairs OOS — strategy not ready yet

  Options:
    1. Wait for 30+ days of data and re-run
    2. Re-examine YELLOW pairs with looser thresholds
    3. Look at cross-sector pairs (currently only within-sector)
        """)

    print(f"  Total time: {elapsed:.0f}s\n")


if __name__ == '__main__':
    main()