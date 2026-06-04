"""
NB_PAIRS_01 — Find Cointegrated Pairs
======================================
Research Question:
    Which pairs of stocks from our universe are cointegrated?
    What are the hedge ratios and mean reversion speeds?
    Which pairs are tradeable after transaction costs?

Method:
    1. Load price series for all 85 symbols (MAYFUT, 8 days)
    2. Test all pairs for cointegration (Engle-Granger test)
    3. For cointegrated pairs: compute hedge ratio, half-life, spread stats
    4. Filter by tradability (spread > 2x transaction cost)
    5. Save results to research/findings/pairs/cointegrated_pairs.json

Usage:
    python research/notebooks/pairs/NB_PAIRS_01_find_pairs.py

Output:
    research/findings/pairs/cointegrated_pairs.json
    research/findings/pairs/nb_pairs_01_results.csv
    research/findings/pairs/nb_pairs_01_plots/
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
from datetime import datetime, timedelta
from itertools import combinations

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# Use MAYFUT — 8 full days of data
DATES = [
    '2026-05-13', '2026-05-14', '2026-05-15',
    '2026-05-18', '2026-05-19', '2026-05-20',
    '2026-05-21', '2026-05-22',
]

# All 85 symbols — grouped by sector for better pair discovery
SECTOR_GROUPS = {
    'metals':      ['JSWSTEEL26MAYFUT', 'HINDALCO26MAYFUT',
                    'TATASTEEL26MAYFUT', 'SAIL26MAYFUT', 'VEDL26MAYFUT'],
    'realty':      ['GODREJPROP26MAYFUT', 'OBEROIRLTY26MAYFUT',
                    'PRESTIGE26MAYFUT', 'PHOENIXLTD26MAYFUT',
                    'DLF26MAYFUT'],
    'it':          ['TECHM26MAYFUT', 'HCLTECH26MAYFUT', 'INFY26MAYFUT',
                    'TCS26MAYFUT', 'WIPRO26MAYFUT'],
    'nbfc':        ['CHOLAFIN26MAYFUT', 'BAJFINANCE26MAYFUT',
                    'INDUSINDBK26MAYFUT', 'LICHSGFIN26MAYFUT',
                    'FEDERALBNK26MAYFUT'],
    'banking':     ['ICICIBANK26MAYFUT', 'AXISBANK26MAYFUT',
                    'HDFCBANK26MAYFUT', 'SBIN26MAYFUT',
                    'KOTAKBANK26MAYFUT', 'BANKBARODA26MAYFUT',
                    'IDFCFIRSTB26MAYFUT'],
    'pharma':      ['SUNPHARMA26MAYFUT', 'CIPLA26MAYFUT',
                    'DRREDDY26MAYFUT', 'LUPIN26MAYFUT',
                    'DIVISLAB26MAYFUT', 'AUROPHARMA26MAYFUT',
                    'ALKEM26MAYFUT', 'TORNTPHARM26MAYFUT'],
    'fmcg':        ['HINDUNILVR26MAYFUT', 'NESTLEIND26MAYFUT',
                    'BRITANNIA26MAYFUT', 'TATACONSUM26MAYFUT'],
    'auto':        ['BAJAJ-AUTO26MAYFUT', 'HEROMOTOCO26MAYFUT',
                    'EICHERMOT26MAYFUT', 'MARUTI26MAYFUT',
                    'TVSMOTOR26MAYFUT', 'M&M26MAYFUT'],
    'industrials': ['HAVELLS26MAYFUT', 'VOLTAS26MAYFUT',
                    'BLUESTARCO26MAYFUT', 'CROMPTON26MAYFUT'],
    'energy':      ['BPCL26MAYFUT', 'ONGC26MAYFUT',
                    'IOC26MAYFUT', 'COALINDIA26MAYFUT'],
    'infra':       ['LT26MAYFUT', 'BEL26MAYFUT', 'BDL26MAYFUT',
                    'HAL26MAYFUT', 'POWERGRID26MAYFUT', 'NTPC26MAYFUT'],
    'mixed':       ['ADANIPORTS26MAYFUT', 'ADANIENT26MAYFUT',
                    'RELIANCE26MAYFUT', 'BHARTIARTL26MAYFUT',
                    'GRASIM26MAYFUT', 'ASIANPAINT26MAYFUT',
                    'PIDILITIND26MAYFUT', 'TITAN26MAYFUT',
                    'BAJAJFINSV26MAYFUT', 'BAJAJHLDNG26MAYFUT',
                    'MUTHOOTFIN26MAYFUT', 'COCHINSHIP26MAYFUT',
                    'OBEROIRLTY26MAYFUT'],
}

# Cointegration significance threshold
PVALUE_THRESHOLD  = 0.05    # p < 0.05 = cointegrated
STRONG_THRESHOLD  = 0.01    # p < 0.01 = strongly cointegrated

# Min half-life for mean reversion (too fast = noise, too slow = untradeable)
MIN_HALF_LIFE_SECS = 30     # at least 30 seconds
MAX_HALF_LIFE_SECS = 1800   # at most 30 minutes

# Min spread volatility relative to transaction cost
# Transaction cost ≈ 5-8 bps round trip
# Need spread std > 2x transaction cost to be profitable
MIN_SPREAD_STD_BPS = 15     # minimum 15 bps spread std

OUTPUT_DIR = Path('research/findings/pairs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR   = OUTPUT_DIR / 'nb_pairs_01_plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_price_series(symbols: list, dates: list) -> dict:
    """
    Load 1-second close prices for all symbols across all dates.
    Returns {symbol: pd.Series(index=ts_sec, values=close)}
    """
    from src.backtest.data_loader import load_day

    prices = {}
    for sym in symbols:
        frames = []
        for date in dates:
            try:
                df = load_day(sym, date, market_hours_only=True)
                if not df.empty:
                    frames.append(df[['ts_sec', 'close', 'weighted_mid']])
            except Exception:
                pass

        if not frames:
            continue

        combined = pd.concat(frames, ignore_index=True)\
                     .sort_values('ts_sec')\
                     .drop_duplicates('ts_sec')

        # Use weighted_mid if available, else close
        price_col = 'weighted_mid' if 'weighted_mid' in combined.columns \
                    else 'close'
        series = combined.set_index('ts_sec')[price_col].dropna()

        if len(series) >= 5000:
            prices[sym] = series
            print(f"  {sym:<30} {len(series):>8,} bars")
        else:
            print(f"  {sym:<30} {len(series):>8,} bars  SKIP (too few)")

    return prices


# ─────────────────────────────────────────────────────────────────────────────
# Cointegration testing
# ─────────────────────────────────────────────────────────────────────────────

def test_cointegration(s1: pd.Series, s2: pd.Series) -> dict:
    """
    Engle-Granger cointegration test.

    Steps:
      1. Align series on common timestamps
      2. OLS regression: s1 = β × s2 + α + ε
      3. Test residuals for stationarity (ADF test)
      4. Compute spread statistics

    Returns dict with pvalue, hedge_ratio, spread stats.
    """
    from statsmodels.tsa.stattools import coint, adfuller
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    # Align on common timestamps
    common = s1.index.intersection(s2.index)
    if len(common) < 5000:
        return {'ok': False, 'error': 'insufficient common bars'}

    y = s1[common].values
    x = s2[common].values

    # Remove any NaN/inf
    valid = np.isfinite(y) & np.isfinite(x)
    y, x  = y[valid], x[valid]

    if len(y) < 5000:
        return {'ok': False, 'error': 'too few valid bars'}

    # OLS: y = hedge_ratio * x + intercept
    X         = add_constant(x)
    model     = OLS(y, X).fit()
    hedge_ratio = model.params[1]
    intercept   = model.params[0]

    # Spread = y - hedge_ratio * x
    spread = y - hedge_ratio * x

    # ADF test on spread
    adf_result = adfuller(spread, maxlag=1, autolag=None)
    adf_stat   = adf_result[0]
    pvalue     = adf_result[1]

    # Half-life of mean reversion
    # Fit AR(1): Δspread = φ × spread_lag + ε
    # Half-life = -ln(2) / ln(|φ|)
    spread_lag    = spread[:-1]
    spread_diff   = np.diff(spread)
    valid_ar      = np.isfinite(spread_lag) & np.isfinite(spread_diff)
    if valid_ar.sum() > 100:
        phi_num = np.cov(spread_lag[valid_ar], spread_diff[valid_ar])[0, 1]
        phi_den = np.var(spread_lag[valid_ar])
        phi     = phi_num / phi_den if phi_den > 0 else 0
        if -1 < phi < 0:
            half_life = -np.log(2) / np.log(1 + phi)
        else:
            half_life = np.inf
    else:
        half_life = np.inf

    # Spread statistics
    spread_mean = float(spread.mean())
    spread_std  = float(spread.std())

    # Z-score stats (how many times per day does spread hit ±2σ?)
    zscore      = (spread - spread_mean) / max(spread_std, 1e-8)
    crossings   = int((np.diff(np.sign(zscore - 2)) != 0).sum() +
                      (np.diff(np.sign(zscore + 2)) != 0).sum())
    daily_signals = crossings / len(DATES)

    # Spread in bps
    avg_price      = (y.mean() + x.mean()) / 2
    spread_std_bps = (spread_std / avg_price) * 10000

    # Correlation
    correlation = float(np.corrcoef(y, x)[0, 1])

    return {
        'ok':              True,
        'pvalue':          round(float(pvalue), 6),
        'adf_stat':        round(float(adf_stat), 4),
        'hedge_ratio':     round(float(hedge_ratio), 6),
        'intercept':       round(float(intercept), 4),
        'half_life_secs':  round(float(half_life), 1),
        'spread_mean':     round(float(spread_mean), 4),
        'spread_std':      round(float(spread_std), 4),
        'spread_std_bps':  round(float(spread_std_bps), 2),
        'correlation':     round(float(correlation), 4),
        'n_bars':          int(len(y)),
        'daily_signals':   round(float(daily_signals), 1),
        'r_squared':       round(float(model.rsquared), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tradability check
# ─────────────────────────────────────────────────────────────────────────────

def is_tradeable(result: dict, lot_sizes: dict,
                 sym1: str, sym2: str) -> tuple[bool, str]:
    """
    Check if a cointegrated pair is actually tradeable.

    Filters:
      1. Half-life: 30s < half_life < 1800s
      2. Spread std > MIN_SPREAD_STD_BPS bps
      3. At least 5 signals per day
      4. Hedge ratio > 0 (same direction pair)
    """
    hl = result['half_life_secs']

    if hl < MIN_HALF_LIFE_SECS:
        return False, f'half_life={hl:.0f}s too fast (min {MIN_HALF_LIFE_SECS}s)'

    if hl > MAX_HALF_LIFE_SECS:
        return False, f'half_life={hl:.0f}s too slow (max {MAX_HALF_LIFE_SECS}s)'

    if result['spread_std_bps'] < MIN_SPREAD_STD_BPS:
        return False, (f'spread_std={result["spread_std_bps"]:.1f}bps '
                       f'too small (min {MIN_SPREAD_STD_BPS}bps)')

    if result['daily_signals'] < 5:
        return False, f'only {result["daily_signals"]:.1f} signals/day'

    if result['hedge_ratio'] <= 0:
        return False, f'hedge_ratio={result["hedge_ratio"]:.4f} negative'

    return True, 'ok'


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_pair(sym1: str, sym2: str, s1: pd.Series, s2: pd.Series,
              result: dict):
    """Plot price series, spread, and z-score for a pair."""
    common     = s1.index.intersection(s2.index)
    y          = s1[common].values
    x          = s2[common].values
    spread     = y - result['hedge_ratio'] * x
    spread_mean = result['spread_mean']
    spread_std  = result['spread_std']
    zscore      = (spread - spread_mean) / max(spread_std, 1e-8)

    # Use first 5000 bars for clarity
    n = min(5000, len(y))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    s1_name = sym1.replace('26MAYFUT', '')
    s2_name = sym2.replace('26MAYFUT', '')
    fig.suptitle(f'Pair: {s1_name} / {s2_name}  '
                 f'[p={result["pvalue"]:.4f}  '
                 f'HL={result["half_life_secs"]:.0f}s  '
                 f'corr={result["correlation"]:.3f}]',
                 fontsize=12)

    # Normalised price series
    axes[0].plot(y[:n] / y[0], label=s1_name, linewidth=0.8)
    axes[0].plot(x[:n] / x[0], label=s2_name, linewidth=0.8, alpha=0.8)
    axes[0].set_title('Normalised Price Series')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Spread
    axes[1].plot(spread[:n], linewidth=0.6, color='purple')
    axes[1].axhline(spread_mean,              color='black',  linewidth=1)
    axes[1].axhline(spread_mean + 2*spread_std, color='red',  linewidth=0.8,
                    linestyle='--', label='+2σ')
    axes[1].axhline(spread_mean - 2*spread_std, color='green', linewidth=0.8,
                    linestyle='--', label='-2σ')
    axes[1].set_title(f'Spread  (std={result["spread_std_bps"]:.1f}bps)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Z-score
    axes[2].plot(zscore[:n], linewidth=0.6, color='steelblue')
    axes[2].axhline( 2, color='red',   linewidth=0.8, linestyle='--')
    axes[2].axhline(-2, color='green', linewidth=0.8, linestyle='--')
    axes[2].axhline( 0, color='black', linewidth=0.8)
    axes[2].set_title('Z-score (trade at ±2)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{s1_name}_{s2_name}.png"
    plt.savefig(PLOT_DIR / fname, dpi=80, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()

    print("=" * 70)
    print("  NB_PAIRS_01 — Find Cointegrated Pairs")
    print(f"  Dates: {DATES}")
    print("=" * 70)

    # Load lot sizes for tradability check
    lot_sizes = {}
    try:
        from src.utils.auth import get_secret
        from kiteconnect import KiteConnect
        api_key      = get_secret("KITE_API_KEY")
        access_token = get_secret("KITE_ACCESS_TOKEN")
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        df_lots  = pd.DataFrame(kite.instruments("NFO"))
        df_lots  = df_lots[df_lots["instrument_type"] == "FUT"]
        lot_sizes = {r["name"]: int(r["lot_size"])
                     for _, r in df_lots.iterrows()}
        print(f"Loaded lot sizes for {len(lot_sizes)} instruments")
    except Exception as e:
        print(f"Could not load lot sizes: {e}")

    # Flatten all symbols
    all_symbols = []
    for sector, syms in SECTOR_GROUPS.items():
        all_symbols.extend(syms)
    all_symbols = list(dict.fromkeys(all_symbols))  # deduplicate

    # Load all price series
    print(f"\nLoading price series for {len(all_symbols)} symbols...")
    prices = load_price_series(all_symbols, DATES)
    print(f"\nLoaded {len(prices)} symbols with sufficient data")

    loaded_symbols = list(prices.keys())

    # Build pairs to test
    # Test within-sector pairs first (more likely to be cointegrated)
    # Then cross-sector pairs
    within_sector_pairs = []
    for sector, syms in SECTOR_GROUPS.items():
        valid = [s for s in syms if s in prices]
        for s1, s2 in combinations(valid, 2):
            within_sector_pairs.append((s1, s2, sector))

    # Total pairs
    total_pairs = len(within_sector_pairs)
    print(f"\nTesting {total_pairs} within-sector pairs...")
    print(f"{'─'*70}")

    all_results   = []
    tradeable     = []
    strong_coint  = []

    for i, (sym1, sym2, sector) in enumerate(within_sector_pairs, 1):
        s1 = prices[sym1]
        s2 = prices[sym2]

        result = test_cointegration(s1, s2)

        if not result['ok']:
            continue

        row = {
            'sym1':   sym1,
            'sym2':   sym2,
            'sector': sector,
            **result,
        }
        all_results.append(row)

        # Check tradability
        tradeable_flag, reason = is_tradeable(
            result, lot_sizes, sym1, sym2
        )

        s1_name = sym1.replace('26MAYFUT', '')
        s2_name = sym2.replace('26MAYFUT', '')

        if result['pvalue'] < PVALUE_THRESHOLD:
            tag = '★★ STRONG' if result['pvalue'] < STRONG_THRESHOLD \
                  else '✓ COINT'
            trade_tag = '  TRADEABLE' if tradeable_flag else f'  [{reason}]'

            print(f"  {s1_name:<14} / {s2_name:<14}  "
                  f"p={result['pvalue']:.4f}  "
                  f"HL={result['half_life_secs']:>7.1f}s  "
                  f"std={result['spread_std_bps']:>6.1f}bps  "
                  f"sig={result['daily_signals']:>5.1f}/day  "
                  f"{tag}{trade_tag}")

            if tradeable_flag:
                tradeable.append(row)
                # Generate plot for tradeable pairs
                try:
                    plot_pair(sym1, sym2, s1, s2, result)
                except Exception:
                    pass

            if result['pvalue'] < STRONG_THRESHOLD:
                strong_coint.append(row)

    elapsed = time.perf_counter() - t0

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Pairs tested:          {len(all_results)}")
    print(f"  Cointegrated (p<0.05): "
          f"{sum(1 for r in all_results if r['pvalue'] < 0.05)}")
    print(f"  Strong (p<0.01):       "
          f"{sum(1 for r in all_results if r['pvalue'] < 0.01)}")
    print(f"  Tradeable:             {len(tradeable)}")
    print(f"  Time: {elapsed:.1f}s")

    if tradeable:
        print(f"\n  {'─'*65}")
        print(f"  TRADEABLE PAIRS — sorted by half-life")
        print(f"  {'─'*65}")
        tradeable_sorted = sorted(tradeable,
                                   key=lambda x: x['half_life_secs'])
        print(f"  {'Pair':<32} {'Sector':<12} {'p-val':>7} "
              f"{'HalfLife':>9} {'StdBps':>8} {'Sigs/d':>8} {'Corr':>7}")
        print(f"  {'─'*85}")
        for r in tradeable_sorted:
            s1 = r['sym1'].replace('26MAYFUT', '')
            s2 = r['sym2'].replace('26MAYFUT', '')
            print(f"  {s1+'/'+s2:<32} {r['sector']:<12} "
                  f"{r['pvalue']:>7.4f} "
                  f"{r['half_life_secs']:>8.1f}s "
                  f"{r['spread_std_bps']:>8.1f}  "
                  f"{r['daily_signals']:>7.1f}  "
                  f"{r['correlation']:>7.3f}")

    # ── Save results ──────────────────────────────────────────────────────────

    # Full CSV
    if all_results:
        df_all = pd.DataFrame(all_results).sort_values('pvalue')
        df_all.to_csv(OUTPUT_DIR / 'nb_pairs_01_all_results.csv', index=False)
        print(f"\n  Saved: research/findings/pairs/nb_pairs_01_all_results.csv")

    # Tradeable pairs JSON — used by pairs strategy
    if tradeable:
        tradeable_clean = []
        for r in tradeable:
            tradeable_clean.append({
                'sym1':            r['sym1'],
                'sym2':            r['sym2'],
                'sector':          r['sector'],
                'hedge_ratio':     r['hedge_ratio'],
                'intercept':       r['intercept'],
                'spread_mean':     r['spread_mean'],
                'spread_std':      r['spread_std'],
                'spread_std_bps':  r['spread_std_bps'],
                'half_life_secs':  r['half_life_secs'],
                'pvalue':          r['pvalue'],
                'correlation':     r['correlation'],
                'daily_signals':   r['daily_signals'],
                'entry_zscore':    2.0,   # enter when spread > 2σ
                'exit_zscore':     0.5,   # exit when spread < 0.5σ
                'stop_zscore':     3.5,   # stop loss at 3.5σ
            })

        json_path = OUTPUT_DIR / 'cointegrated_pairs.json'
        with open(json_path, 'w') as f:
            json.dump(tradeable_clean, f, indent=2)
        print(f"  Saved: research/findings/pairs/cointegrated_pairs.json")

        print(f"\n  Plots saved to: research/findings/pairs/nb_pairs_01_plots/")

    # ── Decision ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  NEXT STEPS")
    print(f"{'='*70}")

    if len(tradeable) >= 3:
        print(f"""
  ✓ {len(tradeable)} tradeable pairs found — proceed to NB_PAIRS_02

  Next: python research/notebooks/pairs/NB_PAIRS_02_spread_analysis.py
  This will:
    - Analyse spread behaviour in detail per pair
    - Find optimal entry/exit thresholds
    - Estimate expected PnL per signal
    - Check for intraday patterns
        """)
    elif len(tradeable) > 0:
        print(f"""
  ~ {len(tradeable)} tradeable pairs found — marginal
  Consider lowering MIN_SPREAD_STD_BPS from {MIN_SPREAD_STD_BPS} to 10
  or MAX_HALF_LIFE_SECS from {MAX_HALF_LIFE_SECS} to 3600
        """)
    else:
        print(f"""
  ✗ No tradeable pairs found in within-sector pairs
  Try:
    1. Cross-sector pairs (e.g. CHOLAFIN vs HDFCBANK)
    2. More data — re-run after 30 days of data
    3. Relax filters — lower MIN_SPREAD_STD_BPS to 10
        """)

    print(f"\n  Total time: {elapsed:.1f}s\n")


if __name__ == '__main__':
    main()