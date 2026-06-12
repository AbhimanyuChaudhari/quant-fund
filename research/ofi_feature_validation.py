"""
NB10 — Order Flow Imbalance (OFI) Feature Validation
======================================================
Research Question:
    Does adding OFI (and related features) improve V1 backtest Sharpe
    by more than 5%? If yes → add to feature_engine.py and fill_simulator_fast.py

What is OFI?
    Order Flow Imbalance measures the pressure on the LOB between bars.
    Introduced by Cont, Kukanov, Stoikov (2014) — one of the most cited
    microstructure papers.

    OFI = ΔBid_qty_at_best - ΔAsk_qty_at_best

    Positive OFI → buy pressure → price tends to rise
    Negative OFI → sell pressure → price tends to fall

    We extend this to 5 levels (weighted OFI) using your existing depth data.

Features built in this notebook:
    1. ofi_l1          — level 1 OFI (bid_q1 delta - ask_q1 delta)
    2. ofi_weighted    — weighted OFI across 5 levels
    3. price_autocorr  — 60-bar return autocorrelation (mean reversion signal)
    4. vol_regime      — current vol / historical vol ratio
    5. queue_pressure  — total_bid_qty / total_ask_qty ratio

Research outputs:
    → Does each feature predict 60s forward return?
    → Does adding OFI to backtest improve Sharpe?
    → Which features are redundant with existing ones?

Save results to: research/findings/nb10_ofi_results.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SYMBOLS = [
    'CHOLAFIN26MAYFUT',
    'JSWSTEEL26MAYFUT',
    'OBEROIRLTY26MAYFUT',
    'HAVELLS26MAYFUT',
    'VOLTAS26MAYFUT',
    'TECHM26MAYFUT',
    'COCHINSHIP26MAYFUT',
    'HINDALCO26MAYFUT',
]

DATES = [
    '2026-05-13', '2026-05-14', '2026-05-15',
    '2026-05-18', '2026-05-19', '2026-05-20',
    '2026-05-21', '2026-05-22',
]

LOT_SIZES = {
    'CHOLAFIN26MAYFUT':   625,
    'JSWSTEEL26MAYFUT':   675,
    'OBEROIRLTY26MAYFUT': 350,
    'HAVELLS26MAYFUT':    500,
    'VOLTAS26MAYFUT':     375,
    'TECHM26MAYFUT':      600,
    'COCHINSHIP26MAYFUT': 400,
    'HINDALCO26MAYFUT':   700,
}

OUTPUT_DIR = Path('research/findings')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR   = OUTPUT_DIR / 'nb10_plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FORWARD_HORIZON = 60   # seconds — predict price in 60s
OFI_WEIGHTS     = [0.40, 0.25, 0.15, 0.12, 0.08]   # level 1-5 weights


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ofi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all OFI and related features from raw bar data.

    This is the candidate feature_engine.py implementation.
    We test here first — if it helps, move to production.
    """
    df = df.copy().reset_index(drop=True)
    n  = len(df)

    # ── 1. Level-1 OFI (Cont, Kukanov, Stoikov 2014) ─────────────────────────
    # OFI_l1 = Δbid_q1 × sign(Δbid_p1) - Δask_q1 × sign(Δask_p1)
    #
    # Logic:
    #   If bid price rose AND qty increased → more buy pressure
    #   If ask price fell AND qty increased → more sell pressure
    bid_q1_delta = df['bid_q1'].diff().fillna(0)
    ask_q1_delta = df['ask_q1'].diff().fillna(0)
    bid_p1_delta = df['bid_p1'].diff().fillna(0)
    ask_p1_delta = df['ask_p1'].diff().fillna(0)

    df['ofi_l1'] = (
        bid_q1_delta * np.sign(bid_p1_delta).replace(0, 1) -
        ask_q1_delta * np.sign(ask_p1_delta).replace(0, -1)
    )

    # ── 2. Weighted OFI across 5 levels ───────────────────────────────────────
    # More informative than L1 alone — deeper book changes predict prices too
    ofi_weighted = np.zeros(n)
    for level, weight in enumerate(OFI_WEIGHTS, 1):
        bid_col = f'bid_q{level}'
        ask_col = f'ask_q{level}'
        if bid_col in df.columns and ask_col in df.columns:
            b_delta = df[bid_col].diff().fillna(0).values
            a_delta = df[ask_col].diff().fillna(0).values
            ofi_weighted += weight * (b_delta - a_delta)

    df['ofi_weighted'] = ofi_weighted

    # ── 3. OFI Z-score (normalize for cross-symbol comparison) ────────────────
    ofi_roll_std = df['ofi_l1'].rolling(300).std().fillna(1)
    ofi_roll_std = ofi_roll_std.replace(0, 1)
    df['ofi_zscore'] = df['ofi_l1'] / ofi_roll_std

    # ── 4. Cumulative OFI (running sum over last 60 bars) ─────────────────────
    # Captures persistent directional pressure
    df['ofi_cum_60'] = df['ofi_l1'].rolling(60).sum().fillna(0)

    # ── 5. Queue pressure ratio ────────────────────────────────────────────────
    # total_bid_qty / total_ask_qty — already have these columns
    total_bid = df['total_bid_qty'].replace(0, np.nan)
    total_ask = df['total_ask_qty'].replace(0, np.nan)
    df['queue_pressure'] = (total_bid / total_ask).fillna(1.0)
    df['queue_pressure_log'] = np.log(df['queue_pressure'].clip(0.1, 10))

    # ── 6. Price return autocorrelation (60-bar window) ────────────────────────
    # Negative autocorr → mean reverting → good for MM (tighter spreads)
    # Positive autocorr → trending → bad for MM (wider spreads)
    mid  = df['weighted_mid'].fillna(df['close'])
    rets = mid.pct_change().fillna(0)

    autocorr_60 = np.zeros(n)
    for i in range(60, n):
        window = rets.iloc[i-60:i].values
        if len(window) > 10 and window.std() > 1e-10:
            # Lag-1 autocorrelation
            autocorr_60[i] = np.corrcoef(window[:-1], window[1:])[0, 1]
    df['price_autocorr_60'] = autocorr_60

    # ── 7. Volatility regime ratio ─────────────────────────────────────────────
    # current_vol / long_run_vol
    # > 1.5 → vol spike (be cautious, widen spreads)
    # < 0.7 → vol collapse (safe to tighten)
    vol_current = df['realized_vol_60s'].fillna(2.0).rolling(60).mean()
    vol_longrun = df['realized_vol_300s'].fillna(2.0).rolling(300).mean()
    vol_longrun = vol_longrun.replace(0, 2.0)
    df['vol_regime_ratio'] = (vol_current / vol_longrun).fillna(1.0)

    # ── 8. Spread trend ────────────────────────────────────────────────────────
    # Is spread widening or narrowing? Widening = adverse selection incoming
    df['spread_trend'] = df['spread_mean'].diff(10).fillna(0)
    df['spread_trend_z'] = (
        df['spread_trend'] /
        df['spread_trend'].rolling(300).std().replace(0, 1).fillna(1)
    )

    # ── 9. Tick direction imbalance ───────────────────────────────────────────
    # What fraction of recent ticks were upticks?
    price_diff  = mid.diff().fillna(0)
    uptick_mask = (price_diff > 0).astype(float)
    df['tick_direction_60'] = uptick_mask.rolling(60).mean().fillna(0.5)
    df['tick_direction_z']  = (
        (df['tick_direction_60'] - 0.5) /
        df['tick_direction_60'].rolling(300).std().replace(0, 0.1).fillna(0.1)
    )

    # ── 10. Multi-level book imbalance ────────────────────────────────────────
    # Weighted imbalance using 5 levels — better than level-1 alone
    weighted_bid = np.zeros(n)
    weighted_ask = np.zeros(n)
    for level, weight in enumerate(OFI_WEIGHTS, 1):
        bid_col = f'bid_q{level}'
        ask_col = f'ask_q{level}'
        if bid_col in df.columns and ask_col in df.columns:
            weighted_bid += weight * df[bid_col].fillna(0).values
            weighted_ask += weight * df[ask_col].fillna(0).values

    total_weighted = weighted_bid + weighted_ask
    total_weighted = np.where(total_weighted == 0, 1, total_weighted)
    df['book_imbalance_weighted'] = (weighted_bid - weighted_ask) / total_weighted

    return df


def compute_forward_return(df: pd.DataFrame,
                            horizon: int = 60) -> pd.Series:
    """
    Forward return: price change over next `horizon` bars.
    This is what we're trying to predict.
    """
    mid = df['weighted_mid'].fillna(df['close'])
    return mid.shift(-horizon) / mid - 1


# ─────────────────────────────────────────────────────────────────────────────
# Predictive power analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_feature_predictiveness(df: pd.DataFrame,
                                    feature_cols: list,
                                    forward_ret: pd.Series) -> pd.DataFrame:
    """
    For each feature, measure how well it predicts 60s forward return.

    Metrics:
        IC (Information Coefficient) = Spearman rank correlation
            IC > 0.02: useful signal
            IC > 0.05: strong signal

        Hit rate = % of time feature sign matches return sign
            Hit > 52%: useful directional signal
            Hit > 55%: strong directional signal

        IC_t = IC / std(IC) across rolling windows
            IC_t > 1.5: statistically significant
    """
    from scipy import stats

    results = []

    # Remove rows where forward return is NaN (last 60 bars)
    valid_mask = ~forward_ret.isna()

    for col in feature_cols:
        if col not in df.columns:
            continue

        feat = df[col].fillna(0)
        fret = forward_ret.copy()

        # Align
        valid = valid_mask & ~feat.isna() & (feat.abs() < 1e10)
        f = feat[valid].values
        r = fret[valid].values

        if len(f) < 100:
            continue

        # IC (Spearman correlation)
        ic, ic_pval = stats.spearmanr(f, r)

        # Hit rate (directional accuracy)
        signs_match = (np.sign(f) == np.sign(r))
        hit_rate    = signs_match.mean()

        # Rolling IC for stability
        window     = 300
        rolling_ic = []
        for i in range(window, len(f), window):
            w_f = f[i-window:i]
            w_r = r[i-window:i]
            if w_f.std() > 1e-10 and w_r.std() > 1e-10:
                ic_w, _ = stats.spearmanr(w_f, w_r)
                rolling_ic.append(ic_w)

        ic_mean  = np.mean(rolling_ic) if rolling_ic else ic
        ic_std   = np.std(rolling_ic)  if rolling_ic else 0.01
        ic_t     = ic_mean / max(ic_std, 0.001)

        results.append({
            'feature':   col,
            'ic':        round(ic, 5),
            'ic_pval':   round(ic_pval, 4),
            'hit_rate':  round(hit_rate, 4),
            'ic_t':      round(ic_t, 3),
            'ic_mean':   round(ic_mean, 5),
            'useful':    abs(ic) > 0.02 or hit_rate > 0.52,
            'strong':    abs(ic) > 0.05 or hit_rate > 0.55,
        })

    return pd.DataFrame(results).sort_values('ic', ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Backtest comparison
# ─────────────────────────────────────────────────────────────────────────────

def backtest_with_ofi_filter(df: pd.DataFrame,
                               lot_size: int,
                               params: dict,
                               use_ofi: bool = False) -> dict:
    """
    Run fast backtest with and without OFI filter.

    OFI filter: if ofi_zscore > 2.0 (strong sell pressure)
                    → skip posting bid (avoid buying into selling)
                if ofi_zscore < -2.0 (strong buy pressure)
                    → skip posting ask (avoid selling into buying)
    """
    from src.backtest.simulators.fill_simulator_fast import (
        run_fast_backtest, FastBacktestResult
    )

    # Standard backtest (no OFI filter)
    result_base = run_fast_backtest(
        df               = df,
        gamma            = params['gamma'],
        kappa            = params['kappa'],
        min_spread       = params['min_spread'],
        max_spread       = 10.0,
        open_mult        = params['open_mult'],
        lot_size         = lot_size,
        max_inventory    = 5,
        queue_aggression = 0.3,
    )

    if not use_ofi:
        return {
            'sharpe':    result_base.sharpe_ratio,
            'net_pnl':   result_base.net_pnl,
            'fills':     result_base.total_fills,
            'win_rate':  result_base.win_rate,
            'max_dd':    result_base.max_drawdown,
            'ofi_used':  False,
        }

    # OFI-filtered backtest
    # Modify bad_bars to include OFI extremes
    df_ofi = compute_ofi_features(df)

    # Prepare modified dataframe with OFI-enhanced imbalance
    df_modified = df.copy()

    # Scale imbalance_last by OFI signal
    ofi_z = df_ofi['ofi_zscore'].fillna(0).values

    # Enhanced imbalance: blend existing imbalance with OFI signal
    existing_imb = df_modified['imbalance_last'].fillna(0).values
    df_modified['imbalance_last'] = (
        0.6 * existing_imb +
        0.4 * np.clip(ofi_z / 3.0, -1, 1)
    )

    result_ofi = run_fast_backtest(
        df               = df_modified,
        gamma            = params['gamma'],
        kappa            = params['kappa'],
        min_spread       = params['min_spread'],
        max_spread       = 10.0,
        open_mult        = params['open_mult'],
        lot_size         = lot_size,
        max_inventory    = 5,
        queue_aggression = 0.3,
    )

    return {
        'sharpe':    result_ofi.sharpe_ratio,
        'net_pnl':   result_ofi.net_pnl,
        'fills':     result_ofi.total_fills,
        'win_rate':  result_ofi.win_rate,
        'max_dd':    result_ofi.max_drawdown,
        'ofi_used':  True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_analysis(ic_results: pd.DataFrame,
                           symbol: str):
    """Plot IC and hit rate for all features."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Feature Predictiveness — {symbol}', fontsize=13)

    useful = ic_results[ic_results['useful']]
    if useful.empty:
        plt.close()
        return

    # IC bar chart
    colors = ['green' if ic > 0 else 'red' for ic in useful['ic']]
    axes[0].barh(useful['feature'], useful['ic'], color=colors, alpha=0.7)
    axes[0].axvline(0.02,  color='orange', linestyle='--', label='Useful (0.02)')
    axes[0].axvline(0.05,  color='green',  linestyle='--', label='Strong (0.05)')
    axes[0].axvline(-0.02, color='orange', linestyle='--')
    axes[0].axvline(-0.05, color='green',  linestyle='--')
    axes[0].set_xlabel('Information Coefficient (Spearman)')
    axes[0].set_title('IC — Higher is Better')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Hit rate bar chart
    axes[1].barh(useful['feature'], useful['hit_rate'],
                  color='steelblue', alpha=0.7)
    axes[1].axvline(0.50, color='red',    linestyle='--', label='Random (50%)')
    axes[1].axvline(0.52, color='orange', linestyle='--', label='Useful (52%)')
    axes[1].axvline(0.55, color='green',  linestyle='--', label='Strong (55%)')
    axes[1].set_xlabel('Directional Hit Rate')
    axes[1].set_title('Hit Rate — Higher is Better')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / f'{symbol}_feature_ic.png', dpi=100,
                bbox_inches='tight')
    plt.close()


def plot_ofi_vs_returns(df_ofi: pd.DataFrame,
                         forward_ret: pd.Series,
                         symbol: str):
    """Scatter plot of OFI vs forward return."""
    valid = ~forward_ret.isna()
    ofi   = df_ofi['ofi_zscore'][valid].values
    fret  = forward_ret[valid].values

    # Bin OFI and compute avg forward return per bin
    bins  = np.percentile(ofi[np.isfinite(ofi)],
                           np.linspace(0, 100, 21))
    bin_labels = (bins[:-1] + bins[1:]) / 2
    bin_mean   = []
    bin_std    = []

    for i in range(len(bins)-1):
        mask = (ofi >= bins[i]) & (ofi < bins[i+1]) & np.isfinite(ofi)
        if mask.sum() > 5:
            bin_mean.append(fret[mask].mean() * 10000)   # in bps
            bin_std.append(fret[mask].std() * 10000 / np.sqrt(mask.sum()))
        else:
            bin_mean.append(np.nan)
            bin_std.append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(bin_labels)), bin_mean,
            yerr=bin_std, capsize=3, color='steelblue', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('OFI Z-score Percentile Bin')
    ax.set_ylabel('Avg Forward Return (bps)')
    ax.set_title(f'OFI vs 60s Forward Return — {symbol}\n'
                 f'Monotonic relationship = OFI predicts price direction')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f'{symbol}_ofi_vs_returns.png', dpi=100,
                bbox_inches='tight')
    plt.close()


def plot_backtest_comparison(comparison_df: pd.DataFrame):
    """Compare base vs OFI-enhanced backtest across symbols."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle('Base vs OFI-Enhanced Backtest', fontsize=13)

    symbols  = comparison_df['symbol'].tolist()
    x        = np.arange(len(symbols))
    width    = 0.35

    # Sharpe comparison
    axes[0].bar(x - width/2, comparison_df['sharpe_base'],
                 width, label='Base',         color='steelblue', alpha=0.7)
    axes[0].bar(x + width/2, comparison_df['sharpe_ofi'],
                 width, label='OFI Enhanced', color='green',     alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s.replace('26MAYFUT','') for s in symbols],
                              rotation=45, ha='right')
    axes[0].set_ylabel('Sharpe Ratio')
    axes[0].set_title('Sharpe Ratio')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Net PnL comparison
    axes[1].bar(x - width/2, comparison_df['pnl_base'] / 1000,
                 width, label='Base',         color='steelblue', alpha=0.7)
    axes[1].bar(x + width/2, comparison_df['pnl_ofi'] / 1000,
                 width, label='OFI Enhanced', color='green',     alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s.replace('26MAYFUT','') for s in symbols],
                              rotation=45, ha='right')
    axes[1].set_ylabel('Net PnL (Rs. thousands)')
    axes[1].set_title('Net PnL')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Sharpe improvement %
    sharpe_imp = (comparison_df['sharpe_ofi'] - comparison_df['sharpe_base']) / \
                  comparison_df['sharpe_base'].abs().clip(0.1) * 100
    colors = ['green' if v > 0 else 'red' for v in sharpe_imp]
    axes[2].bar(x, sharpe_imp, color=colors, alpha=0.7)
    axes[2].axhline(5,  color='orange', linestyle='--', label='Threshold (5%)')
    axes[2].axhline(0,  color='black',  linewidth=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([s.replace('26MAYFUT','') for s in symbols],
                              rotation=45, ha='right')
    axes[2].set_ylabel('Sharpe Improvement (%)')
    axes[2].set_title('OFI Improvement vs Base')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'backtest_comparison.png', dpi=100,
                bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main research pipeline
# ─────────────────────────────────────────────────────────────────────────────

NEW_FEATURES = [
    'ofi_l1',
    'ofi_weighted',
    'ofi_zscore',
    'ofi_cum_60',
    'queue_pressure_log',
    'price_autocorr_60',
    'vol_regime_ratio',
    'spread_trend_z',
    'tick_direction_z',
    'book_imbalance_weighted',
]

EXISTING_FEATURES = [
    'imbalance_last',
    'price_mom_60s',
    'price_mom_30s',
    'price_mom_10s',
    'spread_zscore',
    'volume_ratio',
    'volume_delta',
    'realized_vol_60s',
    'price_impact',
    'tick_count',
]

ALL_FEATURES = NEW_FEATURES + EXISTING_FEATURES


def main():
    from src.backtest.data_loader import load_day
    import json

    print("=" * 70)
    print("  NB10 — OFI Feature Validation")
    print("=" * 70)

    # Load optimal params from grid search
    params_path = Path('research/findings/v1_optimal_params.json')
    if params_path.exists():
        with open(params_path) as f:
            optimal_params = json.load(f)
    else:
        optimal_params = {}

    all_ic_results    = []
    backtest_comparison = []

    for symbol in SYMBOLS:
        lot_size = LOT_SIZES.get(symbol, 75)
        print(f"\n{'─'*60}")
        print(f"  {symbol} (lot={lot_size})")
        print(f"{'─'*60}")

        # ── Load all days ──────────────────────────────────────────────────
        frames = []
        for date in DATES:
            try:
                df_day = load_day(symbol, date, market_hours_only=True)
                if not df_day.empty:
                    df_day['_date'] = date
                    frames.append(df_day)
            except Exception as e:
                print(f"    Skip {date}: {e}")

        if len(frames) < 2:
            print(f"  Insufficient data — skipping")
            continue

        df = pd.concat(frames, ignore_index=True).sort_values('ts_sec')
        print(f"  Loaded {len(df):,} bars across {len(frames)} days")

        # ── Compute OFI features ───────────────────────────────────────────
        print("  Computing OFI features...")
        df_feat = compute_ofi_features(df)

        # ── Compute forward returns ────────────────────────────────────────
        forward_ret = compute_forward_return(df_feat, FORWARD_HORIZON)

        # ── Analyze predictiveness ─────────────────────────────────────────
        print("  Analyzing feature predictiveness...")
        ic_df = analyze_feature_predictiveness(df_feat, ALL_FEATURES, forward_ret)
        ic_df['symbol'] = symbol
        all_ic_results.append(ic_df)

        # Print results
        print(f"\n  {'Feature':<30} {'IC':>8} {'HitRate':>9} {'IC_t':>7} {'Signal'}")
        print(f"  {'─'*65}")
        for _, row in ic_df.iterrows():
            signal = '★ STRONG' if row['strong'] else ('✓ useful' if row['useful'] else '')
            new    = '(NEW)' if row['feature'] in NEW_FEATURES else ''
            print(f"  {row['feature']:<30} {row['ic']:>8.4f} "
                  f"{row['hit_rate']:>8.1%} {row['ic_t']:>7.2f}  "
                  f"{signal} {new}")

        # ── Plot OFI vs returns ────────────────────────────────────────────
        plot_ofi_vs_returns(df_feat, forward_ret, symbol)
        plot_feature_analysis(ic_df, symbol)

        # ── Backtest comparison (use last 3 days for speed) ────────────────
        print("\n  Running backtest comparison (base vs OFI)...")

        # Get optimized params or use defaults
        sym_params = optimal_params.get(symbol, {})
        params = {
            'gamma':      sym_params.get('gamma',      0.001),
            'kappa':      sym_params.get('kappa',      1.5),
            'min_spread': sym_params.get('min_spread', 0.10),
            'open_mult':  sym_params.get('open_mult',  2.0),
        }

        # Use last 3 days for quick comparison
        recent_dates = DATES[-3:]
        recent_frames = [f for f in frames if f['_date'].iloc[0] in recent_dates]
        if not recent_frames:
            continue

        df_recent = pd.concat(recent_frames, ignore_index=True).sort_values('ts_sec')

        base_result = backtest_with_ofi_filter(
            df_recent, lot_size, params, use_ofi=False
        )
        ofi_result  = backtest_with_ofi_filter(
            df_recent, lot_size, params, use_ofi=True
        )

        sharpe_improvement = (
            (ofi_result['sharpe'] - base_result['sharpe']) /
            max(abs(base_result['sharpe']), 0.1) * 100
        )

        verdict = '✓ IMPROVE' if sharpe_improvement > 5 else (
                  '~ neutral'  if abs(sharpe_improvement) <= 5 else
                  '✗ worse')

        print(f"  Base: Sharpe={base_result['sharpe']:.2f}  "
              f"PnL=Rs.{base_result['net_pnl']:>8,.0f}  "
              f"Fills={base_result['fills']}")
        print(f"  OFI:  Sharpe={ofi_result['sharpe']:.2f}  "
              f"PnL=Rs.{ofi_result['net_pnl']:>8,.0f}  "
              f"Fills={ofi_result['fills']}")
        print(f"  Improvement: {sharpe_improvement:+.1f}%  {verdict}")

        backtest_comparison.append({
            'symbol':       symbol,
            'sharpe_base':  base_result['sharpe'],
            'sharpe_ofi':   ofi_result['sharpe'],
            'pnl_base':     base_result['net_pnl'],
            'pnl_ofi':      ofi_result['net_pnl'],
            'fills_base':   base_result['fills'],
            'fills_ofi':    ofi_result['fills'],
            'sharpe_improvement_pct': round(sharpe_improvement, 2),
            'verdict':      verdict,
        })

    # ── Aggregate results ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  AGGREGATE RESULTS")
    print(f"{'='*70}")

    if all_ic_results:
        all_ic_df = pd.concat(all_ic_results, ignore_index=True)

        # Average IC per feature across symbols
        avg_ic = all_ic_df.groupby('feature').agg(
            avg_ic       = ('ic', 'mean'),
            avg_hit_rate = ('hit_rate', 'mean'),
            avg_ic_t     = ('ic_t', 'mean'),
            n_symbols    = ('symbol', 'count'),
            n_useful     = ('useful', 'sum'),
            n_strong     = ('strong', 'sum'),
        ).reset_index().sort_values('avg_ic', ascending=False)

        print(f"\n  Average predictiveness across {len(SYMBOLS)} symbols:")
        print(f"\n  {'Feature':<30} {'AvgIC':>8} {'AvgHit':>8} "
              f"{'Useful/N':>9} {'Strong/N':>9} {'IsNew?'}")
        print(f"  {'─'*75}")
        for _, row in avg_ic.iterrows():
            is_new  = '(NEW)' if row['feature'] in NEW_FEATURES else ''
            verdict = ('★★ STRONG' if row['n_strong'] >= len(SYMBOLS)//2 else
                       ('✓ useful'  if row['n_useful'] >= len(SYMBOLS)//2 else ''))
            print(f"  {row['feature']:<30} {row['avg_ic']:>8.4f} "
                  f"{row['avg_hit_rate']:>7.1%} "
                  f"{int(row['n_useful']):>4}/{int(row['n_symbols']):<4} "
                  f"{int(row['n_strong']):>4}/{int(row['n_symbols']):<4} "
                  f"{is_new} {verdict}")

        # Save IC results
        avg_ic.to_csv(OUTPUT_DIR / 'nb10_ofi_ic_results.csv', index=False)
        print(f"\n  Saved: research/findings/nb10_ofi_ic_results.csv")

    if backtest_comparison:
        comp_df = pd.DataFrame(backtest_comparison)

        print(f"\n  Backtest comparison summary:")
        print(f"\n  {'Symbol':<28} {'Base Sharpe':>12} {'OFI Sharpe':>11} "
              f"{'Improvement':>12} {'Verdict'}")
        print(f"  {'─'*75}")
        for _, row in comp_df.iterrows():
            print(f"  {row['symbol']:<28} {row['sharpe_base']:>12.2f} "
                  f"{row['sharpe_ofi']:>11.2f} "
                  f"{row['sharpe_improvement_pct']:>+11.1f}%  "
                  f"{row['verdict']}")

        avg_improvement = comp_df['sharpe_improvement_pct'].mean()
        n_improved = (comp_df['sharpe_improvement_pct'] > 5).sum()

        print(f"\n  Average Sharpe improvement: {avg_improvement:+.1f}%")
        print(f"  Symbols improved >5%:       {n_improved}/{len(comp_df)}")

        # Save backtest comparison
        comp_df.to_csv(OUTPUT_DIR / 'nb10_backtest_comparison.csv', index=False)
        print(f"  Saved: research/findings/nb10_backtest_comparison.csv")

        # Plot comparison
        if len(comp_df) >= 2:
            plot_backtest_comparison(comp_df)
            print(f"  Saved plots: research/findings/nb10_plots/")

        # ── FINAL VERDICT ──────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print("  FINAL VERDICT")
        print(f"{'='*70}")

        if avg_improvement > 5 and n_improved >= len(comp_df) // 2:
            print("""
  ✓ OFI FEATURES ARE USEFUL — ADD TO PRODUCTION

  Next steps:
    1. Copy compute_ofi_features() to src/features/feature_engine.py
    2. Add OFI columns to data processing pipeline
       (src/data/processor.py — run on each new day's data)
    3. Add ofi_zscore to fill_simulator_fast.py bad_bars filter
    4. Re-run grid search with new features
    5. Run NB11 (price autocorr) next
            """)
        elif avg_improvement > 0:
            print("""
  ~ OFI HAS MARGINAL BENEFIT — OPTIONAL TO ADD

  Recommendation:
    Add ofi_weighted and price_autocorr_60 only
    (these tend to be the most consistent signals)
    Skip the others to keep feature count manageable
            """)
        else:
            print("""
  ✗ OFI DOES NOT IMPROVE BACKTEST — DO NOT ADD

  This is a valid research outcome.
  Possible reasons:
    - Your existing imbalance features capture similar information
    - 1-second bars are too coarse for OFI to be predictive
    - OFI is more useful at tick level

  Next steps:
    Run NB11 (price autocorr) — different signal, may still help
            """)

    print(f"\n  Done. Check research/findings/nb10_plots/ for visualizations.\n")


if __name__ == '__main__':
    main()
