"""
NB_IMB_01 — Order Flow Imbalance: Exploratory Analysis
=======================================================
Research Question:
    Which imbalance measures best predict short-term price moves
    on NSE stock futures, at what timescale, and under what conditions?

Methodology:
    1. Load processed data for top liquid symbols
    2. Compute multiple imbalance measures
    3. Measure predictive power at multiple horizons
    4. Analyze signal decay, time-of-day effects, regime dependence
    5. Rank measures and select best for transformer input

References:
    Cont, Kukanov, Stoikov (2014) — OFI linear price impact
    Stoikov (2018) — micro-price
    Gould et al (2013) — LOB dynamics

Output:
    research/microstructure/imbalance/findings/imb_eda_results.json
    research/microstructure/imbalance/findings/imb_eda_plots/
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
from typing import Optional

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# Symbols to analyze — use liquid symbols with good data
SYMBOLS = [
    'CHOLAFIN26JUNFUT',
    'JSWSTEEL26JUNFUT',
    'HCLTECH26JUNFUT',
    'TECHM26JUNFUT',
    'VOLTAS26JUNFUT',
    'AXISBANK26JUNFUT',
    'SBIN26JUNFUT',
    'RELIANCE26JUNFUT',
]

DATES = [
    '2026-05-27', '2026-05-28', '2026-05-29', '2026-05-30',
    '2026-06-01', '2026-06-02', '2026-06-03', '2026-06-04',
    '2026-06-05', '2026-06-06', '2026-06-08', '2026-06-09',
]

# Prediction horizons to test (in bars = seconds)
HORIZONS = [1, 5, 10, 30, 60]

# Output
OUTPUT_DIR = Path('research/microstructure/imbalance/findings')
PLOT_DIR   = OUTPUT_DIR / 'imb_eda_plots'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# IST time filters
OPEN_START  = 33300   # 09:15
OPEN_END    = 35100   # 09:45
CLOSE_START = 52200   # 14:30
MARKET_END  = 55800   # 15:30


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_symbol_data(symbol: str, dates: list) -> pd.DataFrame:
    """Load all dates for a symbol into one DataFrame."""
    from src.backtest.data_loader import load_day

    frames = []
    for date in dates:
        try:
            df = load_day(symbol, date, market_hours_only=True)
            if df is not None and not df.empty:
                df['date'] = date
                frames.append(df)
        except Exception as e:
            pass

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True).sort_values('ts_sec')
    print(f"  {symbol}: {len(combined):,} bars across {len(frames)} days")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Imbalance measure computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_imbalance_measures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute multiple imbalance measures from available features.

    Measures:
        imb_simple    : (bid_q - ask_q) / (bid_q + ask_q)  — basic LOB
        imb_last      : instantaneous imbalance at last tick
        imb_delta     : change in imbalance from prev bar
        imb_momentum  : imbalance - rolling mean (deviation from baseline)
        imb_extreme   : |imbalance| > 2 std (extreme imbalance flag)
        vol_delta_norm: normalized signed volume (buy-sell proxy)
        micro_price   : Stoikov weighted mid price
        ofi_proxy     : order flow imbalance proxy from volume_delta
        imb_persistence: autocorrelation of imbalance (how sticky)
    """
    df = df.copy()

    # ── Basic imbalance ────────────────────────────────────────────────────────
    # From pipeline: imbalance_last is (bid_qty - ask_qty) / (bid_qty + ask_qty)
    df['imb_simple'] = df['imbalance_last'].fillna(0)
    df['imb_mean']   = df['imbalance_mean'].fillna(0) \
                       if 'imbalance_mean' in df.columns \
                       else df['imb_simple']

    # ── Imbalance momentum (deviation from rolling baseline) ──────────────────
    for window in [10, 30, 60]:
        col = f'imbalance_ma_{window}s'
        if col in df.columns:
            df[f'imb_dev_{window}s'] = df['imb_simple'] - df[col].fillna(0)
        else:
            df[f'imb_dev_{window}s'] = df['imb_simple'].rolling(window).mean()

    # ── Imbalance change (1st difference) ─────────────────────────────────────
    df['imb_delta'] = df['imb_simple'].diff().fillna(0)

    # ── Extreme imbalance flag ─────────────────────────────────────────────────
    imb_std = df['imb_simple'].rolling(300).std().fillna(0.1)
    df['imb_extreme'] = (df['imb_simple'].abs() > 2 * imb_std).astype(float)

    # ── Volume-based OFI proxy ─────────────────────────────────────────────────
    # volume_delta = buyer-initiated - seller-initiated volume proxy
    if 'volume_delta' in df.columns:
        vd = df['volume_delta'].fillna(0)
        tick = df['tick_count'].fillna(1).replace(0, 1) \
               if 'tick_count' in df.columns else 1
        df['ofi_proxy'] = vd / tick   # per-tick flow direction

        # Normalize OFI
        ofi_std = df['ofi_proxy'].rolling(300).std().fillna(1).replace(0, 1)
        df['ofi_norm'] = (df['ofi_proxy'] / ofi_std).clip(-3, 3)
    else:
        df['ofi_proxy'] = 0.0
        df['ofi_norm']  = 0.0

    # ── Micro-price (Stoikov 2018) ─────────────────────────────────────────────
    # micro_price = mid + imbalance * spread / 2
    # Better estimate of true price than simple mid
    if 'weighted_mid' in df.columns and 'spread_mean' in df.columns:
        spread = df['spread_mean'].fillna(0.1)
        df['micro_price'] = df['weighted_mid'].fillna(df['close']) + \
                            df['imb_simple'] * spread / 2
        df['micro_vs_mid'] = df['micro_price'] - \
                             df['weighted_mid'].fillna(df['close'])
    else:
        df['micro_vs_mid'] = df['imb_simple'] * 0.05

    # ── Queue depth ratio ──────────────────────────────────────────────────────
    if 'total_bid_qty' in df.columns and 'total_ask_qty' in df.columns:
        total_q = df['total_bid_qty'].fillna(0) + df['total_ask_qty'].fillna(0)
        df['depth_ratio'] = df['total_bid_qty'].fillna(0) / \
                            total_q.replace(0, np.nan)
        df['depth_ratio'] = df['depth_ratio'].fillna(0.5)
    else:
        df['depth_ratio'] = 0.5

    # ── Combined signal (equal weight baseline) ───────────────────────────────
    df['imb_combined'] = (
        0.40 * df['imb_simple'] +
        0.30 * df['ofi_norm'].clip(-1, 1) +
        0.20 * df['imb_dev_30s'].clip(-1, 1) +
        0.10 * df['imb_delta'].clip(-1, 1)
    )

    # ── IST time of day ────────────────────────────────────────────────────────
    df['ist_tod'] = (df['ts_sec'] + 19800) % 86400
    df['ist_hour'] = df['ist_tod'] // 3600

    # Session labels
    df['session'] = 'midday'
    df.loc[df['ist_tod'] < OPEN_END, 'session'] = 'open'
    df.loc[df['ist_tod'] >= CLOSE_START, 'session'] = 'close'

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Predictive power analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_forward_returns(df: pd.DataFrame,
                            horizons: list) -> pd.DataFrame:
    """
    Compute forward log returns at multiple horizons.
    These are what we're trying to predict.
    """
    price = df['weighted_mid'].fillna(df['close'])

    for h in horizons:
        df[f'fwd_ret_{h}s'] = np.log(price.shift(-h) / price)

    return df


def measure_predictive_power(
    df:       pd.DataFrame,
    signal:   str,
    horizons: list,
) -> dict:
    """
    Measure how well a signal predicts forward returns.

    Metrics:
        IC (Information Coefficient): Spearman rank correlation
        Hit rate: % of times signal direction = return direction
        Sharpe of signal: PnL if trading purely on signal
        Decay: how IC drops across horizons
    """
    results = {}

    for h in horizons:
        ret_col = f'fwd_ret_{h}s'
        if ret_col not in df.columns:
            continue

        mask = df[signal].notna() & df[ret_col].notna() & \
               np.isfinite(df[signal]) & np.isfinite(df[ret_col])
        sig  = df.loc[mask, signal]
        ret  = df.loc[mask, ret_col]

        if len(sig) < 100:
            continue

        # IC — Spearman rank correlation
        ic, p_val = stats.spearmanr(sig, ret)

        # Hit rate — directional accuracy
        sig_dir = np.sign(sig)
        ret_dir = np.sign(ret)
        hit_rate = (sig_dir == ret_dir).mean()

        # Signal Sharpe — if we trade direction of signal
        signal_pnl = sig_dir * ret
        sharpe = (signal_pnl.mean() / signal_pnl.std() * np.sqrt(252 * 23400)
                  if signal_pnl.std() > 0 else 0)

        results[h] = {
            'ic':       round(float(ic), 4),
            'p_value':  round(float(p_val), 4),
            'hit_rate': round(float(hit_rate), 4),
            'sharpe':   round(float(sharpe), 4),
            'n_obs':    int(len(sig)),
        }

    return results


def analyze_signal_decay(results_by_horizon: dict) -> dict:
    """
    Analyze how signal IC decays across horizons.
    Fit exponential decay: IC(h) = IC_0 * exp(-lambda * h)
    """
    horizons = sorted(results_by_horizon.keys())
    ics      = [results_by_horizon[h]['ic'] for h in horizons]

    if len(horizons) < 3:
        return {'half_life_s': None, 'peak_ic': max(ics) if ics else 0}

    # Fit exponential decay
    try:
        from scipy.optimize import curve_fit
        def exp_decay(x, ic0, lam):
            return ic0 * np.exp(-lam * x)

        popt, _ = curve_fit(exp_decay, horizons, ics,
                            p0=[ics[0], 0.1], maxfev=1000)
        ic0, lam  = popt
        half_life = np.log(2) / max(lam, 1e-8)

        return {
            'half_life_s': round(float(half_life), 1),
            'ic0':         round(float(ic0), 4),
            'decay_rate':  round(float(lam), 4),
            'peak_ic':     round(float(max(ics)), 4),
        }
    except Exception:
        return {
            'half_life_s': None,
            'peak_ic':     round(float(max(ics)) if ics else 0, 4),
        }


def analyze_by_session(
    df:      pd.DataFrame,
    signal:  str,
    horizon: int = 5,
) -> dict:
    """
    Measure signal IC by time-of-day session.
    Open / midday / close often have different dynamics.
    """
    ret_col = f'fwd_ret_{horizon}s'
    if ret_col not in df.columns:
        return {}

    results = {}
    for session in ['open', 'midday', 'close']:
        mask = (df['session'] == session) & \
               df[signal].notna() & df[ret_col].notna() & \
               np.isfinite(df[signal]) & np.isfinite(df[ret_col])
        sub = df[mask]
        if len(sub) < 50:
            continue

        ic, _ = stats.spearmanr(sub[signal], sub[ret_col])
        results[session] = {
            'ic':   round(float(ic), 4),
            'n':    len(sub),
        }

    return results


def analyze_by_vol_regime(
    df:      pd.DataFrame,
    signal:  str,
    horizon: int = 5,
) -> dict:
    """
    Measure signal IC by volatility regime.
    Signal may work better in mean-reverting vs trending regimes.
    """
    ret_col = f'fwd_ret_{horizon}s'
    if ret_col not in df.columns or 'realized_vol_60s' not in df.columns:
        return {}

    vol = df['realized_vol_60s'].fillna(0)
    vol_median = vol.median()
    vol_75     = vol.quantile(0.75)

    results = {}
    for regime, mask_fn in [
        ('low_vol',  lambda: vol <= vol_median),
        ('mid_vol',  lambda: (vol > vol_median) & (vol <= vol_75)),
        ('high_vol', lambda: vol > vol_75),
    ]:
        mask = mask_fn() & df[signal].notna() & df[ret_col].notna() & \
               np.isfinite(df[signal]) & np.isfinite(df[ret_col])
        sub = df[mask]
        if len(sub) < 50:
            continue

        ic, _ = stats.spearmanr(sub[signal], sub[ret_col])
        results[regime] = {
            'ic':       round(float(ic), 4),
            'n':        len(sub),
            'vol_mean': round(float(vol[mask_fn()].mean()), 4),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_signal_overview(df: pd.DataFrame, symbol: str,
                          signals: list):
    """Plot signal distributions and autocorrelations."""
    n_signals = len(signals)
    fig, axes = plt.subplots(2, n_signals, figsize=(4 * n_signals, 8))
    fig.suptitle(f'{symbol} — Imbalance Signal Overview', fontsize=11)

    for j, sig in enumerate(signals):
        if sig not in df.columns:
            continue

        # Distribution
        ax = axes[0, j] if n_signals > 1 else axes[0]
        vals = df[sig].dropna()
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=50, color='steelblue', alpha=0.7,
                edgecolor='white', density=True)
        ax.set_title(f'{sig}\nμ={vals.mean():.3f} σ={vals.std():.3f}',
                     fontsize=8)
        ax.axvline(0, color='black', lw=1)
        ax.grid(True, alpha=0.3)

        # Autocorrelation
        ax2 = axes[1, j] if n_signals > 1 else axes[1]
        lags  = range(1, 61)
        acfs  = [vals.autocorr(lag=l) for l in lags]
        ax2.bar(lags, acfs, color='orange', alpha=0.7)
        ax2.axhline(0, color='black', lw=0.8)
        ax2.axhline(0.1, color='red', lw=0.8, ls='--', label='0.1')
        ax2.axhline(-0.1, color='red', lw=0.8, ls='--')
        ax2.set_title('Autocorrelation (60 lags)', fontsize=8)
        ax2.set_xlabel('Lag (seconds)')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = PLOT_DIR / f'{symbol}_signal_overview.png'
    plt.savefig(fname, dpi=80, bbox_inches='tight')
    plt.close()


def plot_ic_decay(ic_results: dict, symbol: str):
    """Plot IC decay across horizons for all signals."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{symbol} — IC Decay Across Horizons', fontsize=11)

    colors = plt.cm.tab10(np.linspace(0, 1, len(ic_results)))

    for (signal, by_horizon), color in zip(ic_results.items(), colors):
        horizons = sorted(by_horizon.keys())
        ics      = [by_horizon[h]['ic'] for h in horizons]
        hit_rates = [by_horizon[h]['hit_rate'] for h in horizons]

        axes[0].plot(horizons, ics, 'o-', label=signal,
                     color=color, lw=1.5, ms=4)
        axes[1].plot(horizons, hit_rates, 'o-', label=signal,
                     color=color, lw=1.5, ms=4)

    axes[0].axhline(0, color='black', lw=0.8)
    axes[0].set_title('IC (Spearman) vs Horizon')
    axes[0].set_xlabel('Horizon (seconds)')
    axes[0].set_ylabel('IC')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0.5, color='black', lw=0.8, ls='--')
    axes[1].set_title('Hit Rate vs Horizon')
    axes[1].set_xlabel('Horizon (seconds)')
    axes[1].set_ylabel('Hit Rate')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = PLOT_DIR / f'{symbol}_ic_decay.png'
    plt.savefig(fname, dpi=80, bbox_inches='tight')
    plt.close()


def plot_session_ic(session_results: dict, symbol: str):
    """Plot IC by session and vol regime."""
    signals = list(session_results.keys())
    sessions = ['open', 'midday', 'close']

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f'{symbol} — IC by Time-of-Day Session', fontsize=11)

    x      = np.arange(len(signals))
    width  = 0.25
    colors = {'open': 'steelblue', 'midday': 'orange', 'close': 'green'}

    for i, session in enumerate(sessions):
        ics = [session_results[sig].get(session, {}).get('ic', 0)
               for sig in signals]
        ax.bar(x + i * width, ics, width, label=session,
               color=colors[session], alpha=0.7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(signals, rotation=30, ha='right', fontsize=8)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('IC (Spearman)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fname = PLOT_DIR / f'{symbol}_session_ic.png'
    plt.savefig(fname, dpi=80, bbox_inches='tight')
    plt.close()


def plot_intraday_ic(df: pd.DataFrame, signal: str,
                     symbol: str, horizon: int = 5):
    """Plot rolling IC by hour of day."""
    ret_col = f'fwd_ret_{horizon}s'
    if ret_col not in df.columns:
        return

    df = df.copy()
    mask = df[signal].notna() & df[ret_col].notna() & \
           np.isfinite(df[signal]) & np.isfinite(df[ret_col])
    df   = df[mask]

    # IC by 30-min bucket
    df['bucket'] = df['ist_tod'] // 1800 * 1800
    buckets = sorted(df['bucket'].unique())

    ics = []
    times = []
    for b in buckets:
        sub = df[df['bucket'] == b]
        if len(sub) < 30:
            continue
        ic, _ = stats.spearmanr(sub[signal], sub[ret_col])
        ics.append(ic)
        times.append(b / 3600)

    if not ics:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(times, ics, width=0.4,
           color=['green' if ic > 0 else 'red' for ic in ics],
           alpha=0.7)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel('Hour (IST)')
    ax.set_ylabel('IC')
    ax.set_title(
        f'{symbol} — {signal} IC by 30-min bucket '
        f'(horizon={horizon}s)', fontsize=10
    )
    ax.set_xticks(range(9, 16))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = PLOT_DIR / f'{symbol}_{signal}_intraday_ic.png'
    plt.savefig(fname, dpi=80, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_symbol(symbol: str) -> dict:
    """Full imbalance analysis for one symbol."""
    print(f"\n{'─'*60}")
    print(f"  Analyzing: {symbol}")
    print(f"{'─'*60}")

    # Load data
    df = load_symbol_data(symbol, DATES)
    if df.empty:
        print(f"  No data for {symbol}")
        return {}

    # Compute features
    df = compute_imbalance_measures(df)
    df = compute_forward_returns(df, HORIZONS)

    # Signals to test
    signals = [
        'imb_simple',
        'imb_dev_10s',
        'imb_dev_30s',
        'ofi_norm',
        'imb_delta',
        'depth_ratio',
        'micro_vs_mid',
        'imb_combined',
    ]
    signals = [s for s in signals if s in df.columns]

    print(f"  Testing {len(signals)} signals across "
          f"{len(HORIZONS)} horizons...")

    # IC analysis per signal
    ic_results      = {}
    decay_results   = {}
    session_results = {}
    regime_results  = {}

    for sig in signals:
        by_horizon  = measure_predictive_power(df, sig, HORIZONS)
        decay       = analyze_signal_decay(by_horizon)
        by_session  = analyze_by_session(df, sig, horizon=5)
        by_regime   = analyze_by_vol_regime(df, sig, horizon=5)

        ic_results[sig]      = by_horizon
        decay_results[sig]   = decay
        session_results[sig] = by_session
        regime_results[sig]  = by_regime

        # Print summary
        h5 = by_horizon.get(5, {})
        print(f"  {sig:<20}  "
              f"IC@5s={h5.get('ic', 0):>7.4f}  "
              f"hit={h5.get('hit_rate', 0):>5.1%}  "
              f"Sharpe={h5.get('sharpe', 0):>7.2f}  "
              f"HL={decay.get('half_life_s', '?')}s")

    # Find best signal
    best_signal = max(
        signals,
        key=lambda s: abs(ic_results[s].get(5, {}).get('ic', 0))
    )
    best_ic = ic_results[best_signal].get(5, {}).get('ic', 0)
    print(f"\n  Best signal: {best_signal} (IC@5s={best_ic:.4f})")

    # Plots
    try:
        plot_signal_overview(df, symbol, signals[:4])
        plot_ic_decay(ic_results, symbol)
        plot_session_ic(session_results, symbol)
        plot_intraday_ic(df, best_signal, symbol, horizon=5)
    except Exception as e:
        print(f"  Plot error: {e}")

    return {
        'symbol':          symbol,
        'n_bars':          len(df),
        'n_days':          len(df['date'].unique()) if 'date' in df.columns else 0,
        'signals_tested':  signals,
        'ic_results':      ic_results,
        'decay':           decay_results,
        'session_ic':      session_results,
        'regime_ic':       regime_results,
        'best_signal':     best_signal,
        'best_ic_5s':      best_ic,
    }


def rank_signals(all_results: dict) -> pd.DataFrame:
    """
    Rank signals across all symbols by average IC.
    Identifies which signals are universally predictive vs symbol-specific.
    """
    rows = []
    for symbol, r in all_results.items():
        for sig, by_horizon in r.get('ic_results', {}).items():
            for h, metrics in by_horizon.items():
                rows.append({
                    'symbol':    symbol,
                    'signal':    sig,
                    'horizon':   h,
                    'ic':        metrics['ic'],
                    'hit_rate':  metrics['hit_rate'],
                    'sharpe':    metrics['sharpe'],
                    'n_obs':     metrics['n_obs'],
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Average IC per signal per horizon
    summary = df.groupby(['signal', 'horizon']).agg(
        avg_ic     = ('ic',       'mean'),
        std_ic     = ('ic',       'std'),
        avg_hit    = ('hit_rate', 'mean'),
        avg_sharpe = ('sharpe',   'mean'),
        n_symbols  = ('symbol',   'nunique'),
    ).reset_index()

    return summary.sort_values(['horizon', 'avg_ic'],
                                ascending=[True, False])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import time
    t0 = time.perf_counter()

    print("=" * 70)
    print("  NB_IMB_01 — Order Flow Imbalance EDA")
    print(f"  Symbols: {len(SYMBOLS)}")
    print(f"  Dates:   {DATES[0]} → {DATES[-1]}")
    print(f"  Horizons: {HORIZONS}s")
    print("=" * 70)

    all_results = {}

    for symbol in SYMBOLS:
        try:
            result = analyze_symbol(symbol)
            if result:
                all_results[symbol] = result
        except Exception as e:
            print(f"  ERROR {symbol}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.perf_counter() - t0

    # ── Cross-symbol ranking ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CROSS-SYMBOL SIGNAL RANKING")
    print(f"{'='*70}")

    ranking = rank_signals(all_results)

    if not ranking.empty:
        for h in HORIZONS:
            h_df = ranking[ranking['horizon'] == h].head(5)
            if h_df.empty:
                continue
            print(f"\n  Horizon = {h}s:")
            print(f"  {'Signal':<22} {'Avg IC':>8} {'Std IC':>8} "
                  f"{'Hit Rate':>9} {'Sharpe':>8} {'N Sym':>6}")
            print(f"  {'─'*65}")
            for _, row in h_df.iterrows():
                print(f"  {row['signal']:<22} "
                      f"{row['avg_ic']:>8.4f} "
                      f"{row['std_ic']:>8.4f} "
                      f"{row['avg_hit']:>9.1%} "
                      f"{row['avg_sharpe']:>8.2f} "
                      f"{int(row['n_symbols']):>6}")

    # ── Best signals per symbol ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  BEST SIGNAL PER SYMBOL (horizon=5s)")
    print(f"{'='*70}")
    print(f"  {'Symbol':<28} {'Best Signal':<22} {'IC':>8} {'HL':>8}")
    print(f"  {'─'*70}")

    for symbol, r in sorted(all_results.items(),
                             key=lambda x: abs(x[1].get('best_ic_5s', 0)),
                             reverse=True):
        hl = r.get('decay', {}).get(
            r.get('best_signal', ''), {}
        ).get('half_life_s', '?')
        print(f"  {symbol:<28} "
              f"{r.get('best_signal', '?'):<22} "
              f"{r.get('best_ic_5s', 0):>8.4f} "
              f"{str(hl):>8}s")

    # ── Save results ───────────────────────────────────────────────────────────
    # Convert to JSON-serializable format
    json_results = {}
    for symbol, r in all_results.items():
        json_results[symbol] = {
            k: v for k, v in r.items()
            if k != 'df'   # don't save raw dataframe
        }

    out_path = OUTPUT_DIR / 'imb_eda_results.json'
    with open(out_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    if not ranking.empty:
        ranking.to_csv(OUTPUT_DIR / 'imb_signal_ranking.csv', index=False)

    print(f"\n  Saved: {out_path}")
    print(f"  Plots: {PLOT_DIR}")
    print(f"  Total time: {elapsed:.1f}s")

    # ── Recommendations for NB02 ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RECOMMENDATIONS FOR NB_IMB_02")
    print(f"{'='*70}")

    if not ranking.empty:
        h5 = ranking[ranking['horizon'] == 5]
        top_signals = h5[h5['avg_ic'].abs() > 0.02]['signal'].tolist()[:4]

        print(f"\n  Top signals to use in transformer (IC > 0.02 at 5s):")
        for sig in top_signals:
            row = h5[h5['signal'] == sig].iloc[0]
            print(f"    {sig:<22} avg IC={row['avg_ic']:.4f}")

        print(f"\n  Transformer input features:")
        print(f"    Sequence length: 60 bars (1s bars, 60s lookback)")
        print(f"    Features per bar: {len(top_signals)} imbalance signals")
        print(f"    Target: price direction at 5s horizon")
        print(f"\n  Next: python research/microstructure/imbalance/nb_imb_02_signals.py")

    print()


if __name__ == '__main__':
    main()
