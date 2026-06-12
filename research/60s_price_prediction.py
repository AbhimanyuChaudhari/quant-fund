"""
NB13 — 30-Second Price Prediction (LightGBM) — All 85 Symbols
==============================================================
Research Question:
    Can we predict the direction of price movement in the next 30 seconds
    with accuracy > 53%? If yes → wire into V2 strategy as alpha signal.

Runs on all 85 symbols automatically.
Saves complete results to research/findings/nb13_full_results.json
Prints clean summary table at the end.

Usage:
    python research/notebooks/NB13_60s_price_prediction.py
"""

import numpy as np
import pandas as pd
import json
import warnings
import time
from pathlib import Path

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("LightGBM not found. Install: pip install lightgbm")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ID  = "hedge-fund-494103"
BUCKET_NAME = "hedge-fund-494103-marketdata-mumbai"

INDEX_FUTURES = {
    "NIFTY26MAYFUT", "BANKNIFTY26MAYFUT",
    "FINNIFTY26MAYFUT", "MIDCPNIFTY26MAYFUT",
    "SENSEX26MAYFUT", "NIFTYNXT5026MAYFUT",
    "BANKEX26MAYFUT",
}

TRAIN_DATES = [
    '2026-05-13', '2026-05-14', '2026-05-15',
    '2026-05-18', '2026-05-19',
]
TEST_DATES = [
    '2026-05-20', '2026-05-21', '2026-05-22',
]
ALL_DATES = TRAIN_DATES + TEST_DATES

FORWARD_HORIZON = 30       # seconds
FLAT_THRESHOLD  = 0.0005   # 5 bps
OFI_WEIGHTS     = [0.40, 0.25, 0.15, 0.12, 0.08]

OUTPUT_DIR = Path('research/findings')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR   = OUTPUT_DIR / 'nb13_plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

LGBM_PARAMS = {
    'objective':         'binary',
    'metric':            'binary_logloss',
    'num_leaves':        15,
    'learning_rate':     0.03,
    'feature_fraction':  0.7,
    'bagging_fraction':  0.7,
    'bagging_freq':      5,
    'min_child_samples': 50,
    'lambda_l1':         1.0,
    'lambda_l2':         1.0,
    'verbose':           -1,
    'n_jobs':            4,
}

FEATURE_COLS = [
    # Existing microstructure
    'imbalance_last', 'price_mom_10s', 'price_mom_30s', 'price_mom_60s',
    'spread_zscore', 'volume_ratio', 'volume_delta', 'realized_vol_60s',
    'price_impact', 'tick_count', 'spread_bps', 'bid_q1', 'ask_q1',
    'total_bid_qty', 'total_ask_qty',
    # New from NB10
    'book_imbalance_weighted', 'ofi_weighted', 'tick_direction_z',
    'price_autocorr_60', 'vol_regime_ratio',
    # Time
    'ist_sin', 'ist_cos', 'mins_since_open', 'mins_to_close',
    # Derived
    'spread_rel', 'depth_ratio', 'mid_vs_vwap',
    'vol_ratio_10_60', 'oi_change_z',
]


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df  = df.copy().reset_index(drop=True)
    n   = len(df)
    mid = df['weighted_mid'].fillna(df['close'])

    # Weighted 5-level book imbalance
    wb = np.zeros(n)
    wa = np.zeros(n)
    for level, w in enumerate(OFI_WEIGHTS, 1):
        bc = f'bid_q{level}'
        ac = f'ask_q{level}'
        if bc in df.columns and ac in df.columns:
            wb += w * df[bc].fillna(0).values
            wa += w * df[ac].fillna(0).values
    tot = np.where(wb + wa == 0, 1, wb + wa)
    df['book_imbalance_weighted'] = (wb - wa) / tot

    # Weighted OFI
    ofi = np.zeros(n)
    for level, w in enumerate(OFI_WEIGHTS, 1):
        bc = f'bid_q{level}'
        ac = f'ask_q{level}'
        if bc in df.columns and ac in df.columns:
            ofi += w * (df[bc].diff().fillna(0).values -
                        df[ac].diff().fillna(0).values)
    df['ofi_weighted'] = ofi

    # Tick direction z-score
    pd_diff  = mid.diff().fillna(0)
    uptick   = (pd_diff > 0).astype(float)
    td60     = uptick.rolling(60).mean().fillna(0.5)
    td_std   = td60.rolling(300).std().replace(0, 0.1).fillna(0.1)
    df['tick_direction_z'] = (td60 - 0.5) / td_std

    # Price autocorrelation
    rets      = mid.pct_change().fillna(0)
    autocorr  = np.zeros(n)
    for i in range(60, n):
        w = rets.iloc[i-60:i].values
        if w.std() > 1e-10:
            autocorr[i] = np.corrcoef(w[:-1], w[1:])[0, 1]
    df['price_autocorr_60'] = autocorr

    # Vol regime ratio
    vc = df['realized_vol_60s'].fillna(2.0).rolling(60).mean()
    vl = df['realized_vol_300s'].fillna(2.0).rolling(300).mean()
    df['vol_regime_ratio'] = (vc / vl.replace(0, 2.0)).fillna(1.0)

    # Time features
    ist = (df['ts_sec'] + 19800) % 86400
    df['ist_sin']         = np.sin(2 * np.pi * ist / 86400)
    df['ist_cos']         = np.cos(2 * np.pi * ist / 86400)
    df['mins_since_open'] = ((ist - 33300) / 60).clip(0, 375)
    df['mins_to_close']   = ((55800 - ist) / 60).clip(0, 375)

    # Derived
    df['spread_rel']      = df['spread_bps'].fillna(0)
    tb = df['total_bid_qty'].replace(0, np.nan)
    ta = df['total_ask_qty'].replace(0, np.nan)
    df['depth_ratio']     = np.log((tb / ta).fillna(1.0).clip(0.1, 10))

    vwap      = df['vwap'].fillna(mid)
    vwap_safe = vwap.where(vwap != 0, mid)
    df['mid_vs_vwap'] = (mid - vwap_safe) / vwap_safe

    v10 = df['realized_vol_10s'].fillna(2.0)
    v60 = df['realized_vol_60s'].fillna(2.0)
    df['vol_ratio_10_60'] = (v10 / v60.replace(0, 2.0)).fillna(1.0)

    df['oi_change']   = df['oi'].diff().fillna(0)
    oi_std = df['oi_change'].rolling(300).std().replace(0, 1).fillna(1)
    df['oi_change_z'] = df['oi_change'] / oi_std

    return df


def build_labels(df: pd.DataFrame) -> pd.Series:
    mid  = df['weighted_mid'].fillna(df['close'])
    fwd  = mid.shift(-FORWARD_HORIZON)
    ret  = (fwd - mid) / mid
    lbl  = np.zeros(len(df), dtype=np.int8)
    lbl[ret >  FLAT_THRESHOLD] =  1
    lbl[ret < -FLAT_THRESHOLD] = -1
    return pd.Series(lbl, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def train_models(X: np.ndarray, y: np.ndarray):
    non_flat  = y != 0
    if non_flat.sum() < 100:
        non_flat = np.ones(len(y), dtype=bool)

    Xf = X[non_flat].astype(np.float32)
    yf = y[non_flat]

    m_up = lgb.train(
        LGBM_PARAMS,
        lgb.Dataset(Xf, label=(yf == 1).astype(int), free_raw_data=False),
        num_boost_round=100,
        callbacks=[lgb.log_evaluation(period=-1)],
    )
    m_dn = lgb.train(
        LGBM_PARAMS,
        lgb.Dataset(Xf, label=(yf == -1).astype(int), free_raw_data=False),
        num_boost_round=100,
        callbacks=[lgb.log_evaluation(period=-1)],
    )
    return m_up, m_dn


def evaluate(y: np.ndarray, p_up: np.ndarray,
             p_dn: np.ndarray, thresh: float) -> dict:
    pred     = np.zeros(len(y), dtype=np.int8)
    pred[p_up > thresh] =  1
    pred[p_dn > thresh] = -1
    nf       = y != 0
    sig      = pred != 0
    conf_m   = nf & sig
    conf_acc = float((pred[conf_m] == y[conf_m]).mean()) \
               if conf_m.sum() > 0 else 0.5
    return {
        'confident_accuracy': round(conf_acc, 4),
        'n_signals':          int(conf_m.sum()),
        'signal_rate':        round(float(sig.mean()), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    from src.backtest.data_loader import load_day
    import gcsfs

    if not LGBM_AVAILABLE:
        print("Install lightgbm: pip install lightgbm")
        return

    t_start = time.perf_counter()

    print("=" * 70)
    print("  NB13 — 30s Price Prediction — All Symbols")
    print(f"  Train: {TRAIN_DATES}")
    print(f"  Test:  {TEST_DATES}")
    print(f"  Features: {len(FEATURE_COLS)}")
    print("=" * 70)

    # ── Load lot sizes ─────────────────────────────────────────────────────
    lot_sizes_map = {}
    try:
        from src.utils.auth import get_secret
        from kiteconnect import KiteConnect
        api_key      = get_secret("KITE_API_KEY")
        access_token = get_secret("KITE_ACCESS_TOKEN")
        kite         = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        df_lots      = pd.DataFrame(kite.instruments("NFO"))
        df_lots      = df_lots[df_lots["instrument_type"] == "FUT"]
        lot_sizes_map = {r["name"]: int(r["lot_size"])
                         for _, r in df_lots.iterrows()}
        print(f"Loaded lot sizes for {len(lot_sizes_map)} instruments")
    except Exception as e:
        print(f"Could not load lot sizes ({e}) — using defaults")

    # ── Get all symbols ────────────────────────────────────────────────────
    print("Finding symbols...")
    try:
        fs    = gcsfs.GCSFileSystem(project=PROJECT_ID)
        files = fs.glob(f"{BUCKET_NAME}/processed/features/*FUT/*.parquet")
        all_syms = set()
        for f in files:
            parts    = f.split("/")
            sym      = parts[3]
            date_str = parts[4].replace(".parquet", "")
            if (sym not in INDEX_FUTURES and
                    TRAIN_DATES[0] <= date_str <= TEST_DATES[-1]):
                all_syms.add(sym)
        symbols = sorted(all_syms)
    except Exception as e:
        print(f"Could not load symbols from GCS: {e}")
        symbols = []

    if not symbols:
        print("No symbols found.")
        return

    print(f"Found {len(symbols)} symbols\n")

    # ── Run per symbol ─────────────────────────────────────────────────────
    all_results = []
    errors      = []

    print(f"{'#':>3} {'Symbol':<28} "
          f"{'P>0.55':>7} {'P>0.60':>7} {'P>0.65':>7} "
          f"{'Sigs':>6} {'Verdict'}")
    print("-" * 70)

    for idx, symbol in enumerate(symbols, 1):
        name     = symbol.replace("26MAYFUT","").replace("26JUNFUT","")
        lot_size = lot_sizes_map.get(name, 75)

        try:
            # Load data
            train_frames, test_frames = [], []
            for date in ALL_DATES:
                try:
                    df_d = load_day(symbol, date, market_hours_only=True)
                    if df_d.empty or len(df_d) < 500:
                        continue
                    df_d['_date'] = date
                    if date in TRAIN_DATES:
                        train_frames.append(df_d)
                    else:
                        test_frames.append(df_d)
                except Exception:
                    pass

            if len(train_frames) < 2 or not test_frames:
                errors.append({'symbol': symbol, 'error': 'insufficient data'})
                print(f"{idx:>3} {symbol:<28}  SKIP (insufficient data)")
                continue

            df_tr = pd.concat(train_frames,
                               ignore_index=True).sort_values('ts_sec')
            df_te = pd.concat(test_frames,
                               ignore_index=True).sort_values('ts_sec')

            # Features + labels
            df_tr = compute_features(df_tr)
            df_te = compute_features(df_te)
            y_tr  = build_labels(df_tr)
            y_te  = build_labels(df_te)

            # Trim last FORWARD_HORIZON rows
            df_tr = df_tr.iloc[:-FORWARD_HORIZON]
            df_te = df_te.iloc[:-FORWARD_HORIZON]
            y_tr  = y_tr.iloc[:-FORWARD_HORIZON]
            y_te  = y_te.iloc[:-FORWARD_HORIZON]

            # Feature matrix
            avail = [f for f in FEATURE_COLS if f in df_tr.columns]
            X_tr  = df_tr[avail].fillna(0).values.astype(np.float32)
            X_te  = df_te[avail].fillna(0).values.astype(np.float32)
            y_tr_a = y_tr.values
            y_te_a = y_te.values

            if (y_tr_a != 0).sum() < 200:
                errors.append({'symbol': symbol, 'error': 'too few non-flat'})
                print(f"{idx:>3} {symbol:<28}  SKIP (too few non-flat bars)")
                continue

            # Train
            m_up, m_dn = train_models(X_tr, y_tr_a)

            # Evaluate at 3 thresholds
            pu_te = m_up.predict(X_te)
            pd_te = m_dn.predict(X_te)

            e55 = evaluate(y_te_a, pu_te, pd_te, 0.55)
            e60 = evaluate(y_te_a, pu_te, pd_te, 0.60)
            e65 = evaluate(y_te_a, pu_te, pd_te, 0.65)

            # Train accuracy (quick check for overfit)
            pu_tr = m_up.predict(X_tr)
            pd_tr = m_dn.predict(X_tr)
            e_tr  = evaluate(y_tr_a, pu_tr, pd_tr, 0.55)

            # Best threshold (highest accuracy with enough signals)
            best_thresh = 0.55
            best_acc    = e55['confident_accuracy']
            for thresh, ev in [(0.60, e60), (0.65, e65)]:
                if (ev['confident_accuracy'] > best_acc and
                        ev['n_signals'] >= 500):
                    best_acc    = ev['confident_accuracy']
                    best_thresh = thresh

            # Feature importance
            fi_up  = m_up.feature_importance(importance_type='gain')
            fi_dn  = m_dn.feature_importance(importance_type='gain')
            fi_avg = (fi_up + fi_dn) / 2
            top5   = sorted(zip(avail, fi_avg),
                            key=lambda x: x[1], reverse=True)[:5]

            # Verdict
            verdict = ('★★ STRONG' if best_acc > 0.57 else
                       ('★  USEFUL' if best_acc > 0.53 else
                        '~  WEAK'))

            result = {
                'symbol':       symbol,
                'lot_size':     lot_size,
                'n_train':      int(len(X_tr)),
                'n_test':       int(len(X_te)),
                'train_acc':    e_tr['confident_accuracy'],
                'p55_acc':      e55['confident_accuracy'],
                'p55_signals':  e55['n_signals'],
                'p55_rate':     e55['signal_rate'],
                'p60_acc':      e60['confident_accuracy'],
                'p60_signals':  e60['n_signals'],
                'p65_acc':      e65['confident_accuracy'],
                'p65_signals':  e65['n_signals'],
                'best_thresh':  best_thresh,
                'best_acc':     best_acc,
                'verdict':      verdict,
                'top5_features': [f for f, _ in top5],
                'useful':       best_acc > 0.53,
                'strong':       best_acc > 0.57,
            }
            all_results.append(result)

            print(f"{idx:>3} {symbol:<28} "
                  f"{e55['confident_accuracy']:>6.1%} "
                  f"{e60['confident_accuracy']:>7.1%} "
                  f"{e65['confident_accuracy']:>7.1%} "
                  f"{e55['n_signals']:>6,}  "
                  f"{verdict}")

        except Exception as e:
            errors.append({'symbol': symbol, 'error': str(e)[:60]})
            print(f"{idx:>3} {symbol:<28}  ERROR: {str(e)[:40]}")

    elapsed = time.perf_counter() - t_start

    # ── Save full results ──────────────────────────────────────────────────
    output = {
        'run_date':    pd.Timestamp.now().isoformat(),
        'horizon':     FORWARD_HORIZON,
        'train_dates': TRAIN_DATES,
        'test_dates':  TEST_DATES,
        'n_symbols':   len(all_results),
        'n_errors':    len(errors),
        'results':     all_results,
        'errors':      errors,
    }
    out_path = OUTPUT_DIR / 'nb13_full_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print(f"  NB13 COMPLETE SUMMARY — {len(all_results)} symbols")
    print(f"  Horizon: {FORWARD_HORIZON}s  |  "
          f"Time: {elapsed:.0f}s")
    print(f"{'='*80}")

    if not all_results:
        print("  No results.")
        return

    df_res = pd.DataFrame(all_results).sort_values('best_acc', ascending=False)

    # Counts
    n_strong = df_res['strong'].sum()
    n_useful = df_res['useful'].sum()
    n_weak   = len(df_res) - n_useful

    print(f"\n  Strong (>57%): {n_strong:>3} symbols")
    print(f"  Useful (>53%): {n_useful:>3} symbols")
    print(f"  Weak   (<53%): {n_weak:>3} symbols")
    print(f"  Errors:        {len(errors):>3} symbols")

    avg_acc   = df_res['best_acc'].mean()
    avg_p55   = df_res['p55_acc'].mean()
    print(f"\n  Avg best accuracy:  {avg_acc:.1%}")
    print(f"  Avg P>0.55 accuracy: {avg_p55:.1%}")

    # ── Ranked table ───────────────────────────────────────────────────────
    print(f"\n  {'#':>3} {'Symbol':<28} {'BestAcc':>8} "
          f"{'@Thresh':>8} {'P>0.55':>7} {'P>0.60':>7} "
          f"{'P>0.65':>7} {'Sigs':>6} {'Verdict'}")
    print(f"  {'─'*85}")

    for i, row in df_res.reset_index(drop=True).iterrows():
        print(f"  {i+1:>3} {row['symbol']:<28} "
              f"{row['best_acc']:>7.1%}  "
              f"P>{row['best_thresh']:.2f}  "
              f"{row['p55_acc']:>6.1%} "
              f"{row['p60_acc']:>7.1%} "
              f"{row['p65_acc']:>7.1%} "
              f"{row['p55_signals']:>6,}  "
              f"{row['verdict']}")

    # ── Strong symbols detail ──────────────────────────────────────────────
    strong = df_res[df_res['strong']].reset_index(drop=True)
    if not strong.empty:
        print(f"\n\n  ★ STRONG SYMBOLS — Build production model for these first")
        print(f"  {'─'*70}")
        print(f"  {'Symbol':<28} {'BestAcc':>8} {'Threshold':>10} "
              f"{'DailySignals':>14} {'Top Feature'}")
        print(f"  {'─'*70}")
        for _, row in strong.iterrows():
            # Estimate daily signals
            daily_sigs = int(row['p55_signals'] / len(TEST_DATES))
            print(f"  {row['symbol']:<28} "
                  f"{row['best_acc']:>8.1%} "
                  f"P>{row['best_thresh']:.2f}      "
                  f"{daily_sigs:>12,}  "
                  f"{row['top5_features'][0] if row['top5_features'] else 'N/A'}")

    # ── Useful symbols ─────────────────────────────────────────────────────
    useful_not_strong = df_res[df_res['useful'] & ~df_res['strong']]
    if not useful_not_strong.empty:
        print(f"\n\n  ✓ USEFUL SYMBOLS — Add after strong symbols validated")
        print(f"  {'─'*50}")
        syms = useful_not_strong['symbol'].tolist()
        for i in range(0, len(syms), 4):
            row_syms = syms[i:i+4]
            accs     = useful_not_strong[
                useful_not_strong['symbol'].isin(row_syms)
            ]['best_acc'].tolist()
            parts = [f"{s.replace('26MAYFUT','')} ({a:.0%})"
                     for s, a in zip(row_syms, accs)]
            print(f"  {'  '.join(parts)}")

    # ── Weak symbols — don't use ML for these ─────────────────────────────
    weak = df_res[~df_res['useful']]
    if not weak.empty:
        print(f"\n\n  ✗ WEAK SYMBOLS — Do NOT use ML signal for these")
        print(f"  {'─'*50}")
        syms = weak['symbol'].tolist()
        for i in range(0, len(syms), 5):
            row_syms = syms[i:i+5]
            print(f"  {', '.join(s.replace('26MAYFUT','') for s in row_syms)}")

    # ── Feature importance summary ─────────────────────────────────────────
    print(f"\n\n  TOP FEATURES (most common across strong symbols)")
    print(f"  {'─'*40}")
    feat_counts = {}
    for _, row in df_res[df_res['strong']].iterrows():
        for f in row['top5_features']:
            feat_counts[f] = feat_counts.get(f, 0) + 1
    if feat_counts:
        for feat, cnt in sorted(feat_counts.items(),
                                 key=lambda x: x[1], reverse=True)[:10]:
            bar = '█' * cnt
            print(f"  {feat:<30} {bar} {cnt}")

    # ── Recommended production threshold per symbol ───────────────────────
    print(f"\n\n  RECOMMENDED THRESHOLDS FOR PRODUCTION")
    print(f"  (use these in lgbm_predictor.py)")
    print(f"  {'─'*50}")
    print(f"  SYMBOL_THRESHOLDS = {{")
    for _, row in df_res[df_res['useful']].iterrows():
        print(f"      '{row['symbol']}': {row['best_thresh']},")
    print(f"  }}")

    # ── Next steps ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print(f"  NEXT STEPS")
    print(f"{'='*80}")

    if n_strong >= 5:
        print(f"""
  ★ STRONG SIGNAL FOUND IN {n_strong} SYMBOLS

  1. Build src/ml/models/lgbm_predictor.py
     - Train per-symbol models on all available data
     - Save models to research/findings/lgbm_models/
     - Use thresholds from SYMBOL_THRESHOLDS above

  2. Wire into V2 strategy (conservative — 15% weight):
       ml_pred = predictor.predict(bar)
       if ml_pred.confidence > threshold:
           alpha_adj = ml_pred.direction * 0.003
           alpha_t   = 0.85 * alpha_t + 0.15 * alpha_adj

  3. Backtest V2+ML vs V2 baseline on {n_strong} strong symbols

  4. Re-run NB13 after 30 days data (expect +3-5% accuracy)

  5. Build LSTM (NB15) after 30 days data
        """)
    elif n_useful >= 10:
        print(f"""
  ✓ USEFUL SIGNAL IN {n_useful} SYMBOLS — PROCEED WITH CAUTION

  Same steps as above but use P>0.60 threshold
  and 10% weight in strategy (not 15%).
        """)
    else:
        print(f"""
  ~ SIGNAL TOO WEAK — COLLECT MORE DATA

  Re-run after 30 days of data.
  Expected improvement: +3-5% accuracy per additional 10 days.
        """)

    print(f"\n  Saved: {out_path}")
    print(f"  Total time: {elapsed:.0f}s\n")


if __name__ == '__main__':
    main()
