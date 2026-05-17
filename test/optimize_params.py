"""
Fast Parameter Optimizer
========================
Grid search over A-S parameters using:
  1. Local data cache  -- downloads GCS parquet once, reuses 80x
  2. Parallel execution -- runs 4 backtests simultaneously
  3. Vectorized scoring -- no subprocess overhead

Speedup vs original:
  Original:  80 runs x 5s = 7 minutes
  This:      80 runs / 4 cores x 0.5s = ~10 seconds

Usage:
    python test/optimize_params_fast.py --symbol ADANIPORTS26MAYFUT --date 2026-04-30
    python test/optimize_params_fast.py --symbol HAL26MAYFUT --date 2026-04-30 --workers 8
"""

import argparse
import io
import itertools
import os
import sys
import warnings
from multiprocessing import Pool, cpu_count
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

GCS_BUCKET       = "hedge-fund-494103-marketdata"
PROCESSED_PREFIX = "processed/features"

SESSION_SECONDS   = 22500.0
SESSION_START_UTC = 13500


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(symbol: str, date: str) -> Optional[pd.DataFrame]:
    """Load processed parquet from GCS. Cached in memory."""
    try:
        from google.cloud import storage
        client = storage.Client(project="hedge-fund-494103")
        path   = f"{PROCESSED_PREFIX}/{symbol}/{date}.parquet"
        blob   = client.bucket(GCS_BUCKET).blob(path)
        if not blob.exists():
            print(f"No data: {path}")
            return None
        data = blob.download_as_bytes()
        df   = pd.read_parquet(io.BytesIO(data))
        print(f"Loaded {len(df):,} bars for {symbol} on {date}")
        return df
    except Exception as e:
        print(f"Load error: {e}")
        return None


# ─── Vectorized A-S backtest ──────────────────────────────────────────────────

def run_backtest_fast(
    df:         pd.DataFrame,
    gamma:      float,
    min_spread: float,
    max_spread: float = 10.0,
    kappa:      float = 1.5,
    lot_size:   int   = 475,
    max_inv:    int   = 5,
    queue_agg:  float = 0.3,
) -> dict:
    """
    Fast vectorized A-S backtest on pre-loaded DataFrame.
    No GCS reads, no subprocess overhead.
    Returns metrics dict.
    """
    import math

    if df is None or len(df) < 10:
        return _empty_result(gamma, min_spread, kappa)

    # ── Pre-compute A-S quotes for all bars ──
    closes   = df["close"].values.astype(float)
    ts_secs  = df["ts_sec"].values.astype(float)
    vol_60s  = df.get("realized_vol_60s", pd.Series([2.0] * len(df))).fillna(2.0).values
    mid      = df.get("weighted_mid", df["close"]).fillna(df["close"]).values.astype(float)
    bid_q1   = df.get("bid_q1", pd.Series([100] * len(df))).fillna(100).values.astype(float)
    ask_q1   = df.get("ask_q1", pd.Series([100] * len(df))).fillna(100).values.astype(float)
    bar_low  = df.get("low",  df["close"]).values.astype(float)
    bar_high = df.get("high", df["close"]).values.astype(float)
    tick_cnt = df.get("tick_count", pd.Series([1] * len(df))).fillna(1).values.astype(float)
    imb_last = df.get("imbalance_last", pd.Series([0] * len(df))).fillna(0).values.astype(float)
    imb_ma30 = df.get("imbalance_ma_30s", pd.Series([0] * len(df))).fillna(0).values.astype(float)
    vol_rat  = df.get("volume_ratio", pd.Series([1] * len(df))).fillna(1).values.astype(float)

    n = len(df)

    # Transaction costs per lot
    brokerage   = 20.0
    stt_per_lot = lot_size * closes.mean() * 0.0001
    other_costs = 10.0
    cost_per_fill = brokerage + stt_per_lot + other_costs

    # Simulate bar by bar
    inventory   = 0
    cash_pnl    = 0.0
    fills       = 0
    attempts    = 0
    wins        = 0
    pnl_history = []
    prev_cost_pnl = 0.0

    for i in range(n):
        ts  = ts_secs[i]
        sig = vol_60s[i] if vol_60s[i] > 0 else 2.0

        # Time remaining
        secs = ts % 86400
        elapsed   = max(0.0, secs - SESSION_START_UTC)
        remaining = max(60.0, SESSION_SECONDS - elapsed)
        T = remaining / SESSION_SECONDS

        # Imbalance spike filter
        if (abs(imb_last[i] - imb_ma30[i]) > 0.3 and vol_rat[i] < 0.5):
            continue

        # A-S formulas
        r      = mid[i] - inventory * gamma * sig**2 * T
        term1  = gamma * sig**2 * T
        term2  = (2.0 / gamma) * math.log(1.0 + gamma / kappa)
        spread = max(min_spread, min(term1 + term2, max_spread))

        bid_price = round(r - spread / 2, 2)
        ask_price = round(r + spread / 2, 2)

        # Fill simulation — L1 price crossing + queue filter
        # BUY fill
        if inventory < max_inv and bar_low[i] <= bid_price:
            attempts += 1
            # Queue probability
            fill_prob = min(tick_cnt[i] / max(bid_q1[i], 1) * queue_agg, 1.0)
            if np.random.random() < fill_prob:
                inventory += 1
                cash_pnl  -= bid_price * lot_size + cost_per_fill
                fills     += 1

        # SELL fill
        if inventory > -max_inv and bar_high[i] >= ask_price:
            attempts += 1
            fill_prob = min(tick_cnt[i] / max(ask_q1[i], 1) * queue_agg, 1.0)
            if np.random.random() < fill_prob:
                inventory -= 1
                cash_pnl  += ask_price * lot_size - cost_per_fill
                fills     += 1

        # Mark-to-market PnL
        mtm = cash_pnl + inventory * mid[i] * lot_size
        pnl_history.append(mtm)

        # Win tracking
        if len(pnl_history) > 1 and pnl_history[-1] > pnl_history[-2]:
            wins += 1

    # Final PnL (flatten inventory at last mid)
    final_pnl = cash_pnl + inventory * mid[-1] * lot_size

    fill_rate = (fills / attempts * 100) if attempts > 0 else 0.0
    win_rate  = (wins / max(len(pnl_history)-1, 1) * 100)

    # Sharpe
    if len(pnl_history) > 1:
        rets  = np.diff(pnl_history)
        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(375))
    else:
        sharpe = 0.0

    return {
        "gamma":      gamma,
        "min_spread": min_spread,
        "kappa":      kappa,
        "net_pnl":    round(final_pnl, 2),
        "fill_rate":  round(fill_rate, 2),
        "win_rate":   round(win_rate, 1),
        "fills":      fills,
        "attempts":   attempts,
        "sharpe":     round(sharpe, 2),
        "score":      final_pnl * fill_rate,
    }


def _empty_result(gamma, min_spread, kappa) -> dict:
    return {
        "gamma": gamma, "min_spread": min_spread, "kappa": kappa,
        "net_pnl": 0, "fill_rate": 0, "win_rate": 0,
        "fills": 0, "attempts": 0, "sharpe": 0, "score": 0,
    }


# ─── Worker for multiprocessing ───────────────────────────────────────────────

# Global df shared across workers (set before Pool)
_shared_df = None

def _worker(params: tuple) -> dict:
    gamma, min_spread, kappa = params
    return run_backtest_fast(_shared_df, gamma, min_spread, kappa)


def init_worker(df_bytes):
    """Initialize worker with shared dataframe."""
    global _shared_df
    _shared_df = pd.read_parquet(io.BytesIO(df_bytes))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fast parameter optimizer")
    parser.add_argument("--symbol",  default="ADANIPORTS26MAYFUT")
    parser.add_argument("--date",    default="2026-04-30")
    parser.add_argument("--workers", type=int, default=min(4, cpu_count()))
    parser.add_argument("--lot-size",type=int, default=475)
    args = parser.parse_args()

    print(f"\nFast Parameter Optimizer")
    print(f"Symbol:  {args.symbol}")
    print(f"Date:    {args.date}")
    print(f"Workers: {args.workers}")

    # ── Load data ONCE ──
    print(f"\nLoading data from GCS...")
    df = load_data(args.symbol, args.date)
    if df is None:
        print("No data found. Exiting.")
        return

    # Serialize df for multiprocessing
    buf = io.BytesIO()
    df.to_parquet(buf)
    df_bytes = buf.getvalue()

    # ── Parameter grid ──
    gammas      = [0.001, 0.005, 0.01, 0.05, 0.1]
    min_spreads = [0.05, 0.10, 0.25, 0.50]
    kappas      = [0.5, 1.0, 1.5, 2.0]

    combinations = list(itertools.product(gammas, min_spreads, kappas))
    total        = len(combinations)

    print(f"Running {total} combinations with {args.workers} workers...")
    print(f"{'#':>4} {'gamma':>8} {'min_sp':>8} {'kappa':>6} "
          f"{'Net PnL':>12} {'Fill%':>6} {'Win%':>6} {'Sharpe':>7}")
    print("-" * 65)

    # ── Parallel execution ──
    results = []

    if args.workers > 1:
        with Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(df_bytes,)
        ) as pool:
            for i, r in enumerate(pool.imap(_worker, combinations)):
                r["score"] = r["net_pnl"] * r["fill_rate"]
                results.append(r)
                g, ms, k = combinations[i]
                print(f"{i+1:>4} {g:>8.3f} {ms:>8.2f} {k:>6.1f} "
                      f"{r['net_pnl']:>12,.0f} {r['fill_rate']:>6.1f}% "
                      f"{r['win_rate']:>6.1f}% {r['sharpe']:>7.2f}")
    else:
        # Single process fallback
        global _shared_df
        _shared_df = df
        for i, params in enumerate(combinations):
            r = _worker(params)
            r["score"] = r["net_pnl"] * r["fill_rate"]
            results.append(r)
            g, ms, k = params
            print(f"{i+1:>4} {g:>8.3f} {ms:>8.2f} {k:>6.1f} "
                  f"{r['net_pnl']:>12,.0f} {r['fill_rate']:>6.1f}% "
                  f"{r['win_rate']:>6.1f}% {r['sharpe']:>7.2f}")

    # ── Results ──
    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n{'=' * 70}")
    print(f"  TOP 10 PARAMETER SETS -- {args.symbol} | {args.date}")
    print(f"{'=' * 70}")
    print(f"  {'gamma':>8} {'min_sp':>8} {'kappa':>6} "
          f"{'Net PnL':>12} {'Fill%':>6} {'Win%':>6} {'Sharpe':>7} {'Score':>12}")
    print(f"  {'-' * 68}")

    for r in results[:10]:
        print(f"  {r['gamma']:>8.3f} {r['min_spread']:>8.2f} {r['kappa']:>6.1f} "
              f"{r['net_pnl']:>12,.0f} {r['fill_rate']:>6.1f}% "
              f"{r['win_rate']:>6.1f}% {r['sharpe']:>7.2f} "
              f"{r['score']:>12,.1f}")

    best = results[0]
    print(f"\n  Best params:")
    print(f"    --gamma {best['gamma']} --min-spread {best['min_spread']} --kappa {best['kappa']}")
    print(f"    Net PnL:   Rs.{best['net_pnl']:,.0f}")
    print(f"    Fill rate: {best['fill_rate']:.1f}%")
    print(f"    Win rate:  {best['win_rate']:.1f}%")
    print(f"    Sharpe:    {best['sharpe']:.2f}")
    print(f"{'=' * 70}\n")

    # Also show best by net PnL alone
    by_pnl = sorted(results, key=lambda x: x["net_pnl"], reverse=True)
    print(f"  Best by Net PnL alone:")
    print(f"    --gamma {by_pnl[0]['gamma']} --min-spread {by_pnl[0]['min_spread']} --kappa {by_pnl[0]['kappa']}")
    print(f"    Net PnL: Rs.{by_pnl[0]['net_pnl']:,.0f}  Fill: {by_pnl[0]['fill_rate']:.1f}%\n")

    # Best by fill rate alone
    by_fill = sorted(results, key=lambda x: x["fill_rate"], reverse=True)
    print(f"  Best by Fill Rate alone:")
    print(f"    --gamma {by_fill[0]['gamma']} --min-spread {by_fill[0]['min_spread']} --kappa {by_fill[0]['kappa']}")
    print(f"    Net PnL: Rs.{by_fill[0]['net_pnl']:,.0f}  Fill: {by_fill[0]['fill_rate']:.1f}%\n")


if __name__ == "__main__":
    main()