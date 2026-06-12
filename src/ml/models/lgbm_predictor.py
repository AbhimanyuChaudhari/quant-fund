"""
LightGBM Price Direction Predictor — Production Version
========================================================
Trained per-symbol on all available data.
Predicts 30s price direction: up (+1), down (-1), flat (0).

Contract roll handling:
    Models trained on MAYFUT work for JUNFUT automatically.
    get_threshold() and predict() strip contract suffix when
    looking up thresholds and models.

Usage:
    # Training
    python src/ml/models/lgbm_predictor.py --train

    # Inference
    predictor = LGBMPredictor()
    predictor.load_models()
    pred = predictor.predict('CHOLAFIN26JUNFUT', bar)
"""

import re
import json
import math
import time
import logging
import collections
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Contract suffix handling
# ─────────────────────────────────────────────────────────────────────────────

_CONTRACT_RE = re.compile(
    r'\d{2}(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)FUT$',
    re.IGNORECASE
)


def strip_contract_suffix(symbol: str) -> str:
    """CHOLAFIN26JUNFUT → CHOLAFIN"""
    return _CONTRACT_RE.sub('', symbol)


# ─────────────────────────────────────────────────────────────────────────────
# Constants — from NB13 findings
# ─────────────────────────────────────────────────────────────────────────────

FORWARD_HORIZON = 30
FLAT_THRESHOLD  = 0.0005
OFI_WEIGHTS     = [0.40, 0.25, 0.15, 0.12, 0.08]

# Per-symbol optimal thresholds from NB13
# Keys use MAYFUT — get_threshold() handles contract roll automatically
SYMBOL_THRESHOLDS = {
    'DLF26MAYFUT':          0.65,
    'PHOENIXLTD26MAYFUT':   0.65,
    'ASIANPAINT26MAYFUT':   0.65,
    'LUPIN26MAYFUT':        0.65,
    'BEL26MAYFUT':          0.65,
    'PRESTIGE26MAYFUT':     0.65,
    'BRITANNIA26MAYFUT':    0.65,
    'ALKEM26MAYFUT':        0.65,
    'GODREJPROP26MAYFUT':   0.65,
    'BLUESTARCO26MAYFUT':   0.65,
    'HDFCBANK26MAYFUT':     0.65,
    'POLYCAB26MAYFUT':      0.65,
    'TVSMOTOR26MAYFUT':     0.65,
    'COCHINSHIP26MAYFUT':   0.65,
    'LT26MAYFUT':           0.65,
    'TORNTPHARM26MAYFUT':   0.65,
    'CROMPTON26MAYFUT':     0.65,
    'HAL26MAYFUT':          0.65,
    'ULTRACEMCO26MAYFUT':   0.65,
    'BAJAJHLDNG26MAYFUT':   0.65,
    'BAJAJFINSV26MAYFUT':   0.65,
    'VEDL26MAYFUT':         0.65,
    'HEROMOTOCO26MAYFUT':   0.65,
    'LICHSGFIN26MAYFUT':    0.65,
    'CANBK26MAYFUT':        0.65,
    'HAVELLS26MAYFUT':      0.65,
    'MUTHOOTFIN26MAYFUT':   0.65,
    'OBEROIRLTY26MAYFUT':   0.65,
    'DIVISLAB26MAYFUT':     0.65,
    'EXIDEIND26MAYFUT':     0.60,
    'IOC26MAYFUT':          0.65,
    'INDUSINDBK26MAYFUT':   0.65,
    'DRREDDY26MAYFUT':      0.60,
    'WIPRO26MAYFUT':        0.65,
    'ASHOKLEY26MAYFUT':     0.60,
    'BHARTIARTL26MAYFUT':   0.65,
    'TITAN26MAYFUT':        0.65,
    'TATASTEEL26MAYFUT':    0.60,
    'PIDILITIND26MAYFUT':   0.65,
    'GRASIM26MAYFUT':       0.65,
    'FEDERALBNK26MAYFUT':   0.55,
    'BDL26MAYFUT':          0.65,
    'VOLTAS26MAYFUT':       0.65,
    'HCLTECH26MAYFUT':      0.65,
    'APOLLOHOSP26MAYFUT':   0.60,
    'SUNPHARMA26MAYFUT':    0.60,
    'BAJAJ-AUTO26MAYFUT':   0.65,
    'JSWSTEEL26MAYFUT':     0.60,
    'M&M26MAYFUT':          0.60,
    'COALINDIA26MAYFUT':    0.60,
    'AXISBANK26MAYFUT':     0.65,
    'TCS26MAYFUT':          0.60,
    'TATACONSUM26MAYFUT':   0.65,
    'MARUTI26MAYFUT':       0.65,
    'ADANIPORTS26MAYFUT':   0.65,
    'SAIL26MAYFUT':         0.65,
    'RECLTD26MAYFUT':       0.55,
    'IDFCFIRSTB26MAYFUT':   0.55,
    'PFC26MAYFUT':          0.65,
    'ONGC26MAYFUT':         0.60,
    'CIPLA26MAYFUT':        0.60,
    'CHOLAFIN26MAYFUT':     0.60,
    'POWERGRID26MAYFUT':    0.60,
    'KOTAKBANK26MAYFUT':    0.60,
    'NESTLEIND26MAYFUT':    0.60,
    'ADANIENT26MAYFUT':     0.60,
    'IRFC26MAYFUT':         0.60,
    'HINDALCO26MAYFUT':     0.55,
    'TECHM26MAYFUT':        0.60,
    'INFY26MAYFUT':         0.65,
    'ICICIBANK26MAYFUT':    0.65,
    'NBCC26MAYFUT':         0.55,
    'SBIN26MAYFUT':         0.60,
    'BPCL26MAYFUT':         0.60,
    'MOTHERSON26MAYFUT':    0.65,
    'BANKBARODA26MAYFUT':   0.60,
}

# Build base-name lookup once at import time
# e.g. 'CHOLAFIN' → 0.60
_BASE_THRESHOLDS: dict = {
    strip_contract_suffix(k): v
    for k, v in SYMBOL_THRESHOLDS.items()
}


def get_threshold(symbol: str) -> float:
    """
    Get ML confidence threshold for a symbol.
    Handles contract roll automatically.

    Lookup order:
      1. Exact match:      CHOLAFIN26MAYFUT → 0.60
      2. Base name match:  CHOLAFIN26JUNFUT → strips to CHOLAFIN → 0.60
      3. Default:          0.60
    """
    if symbol in SYMBOL_THRESHOLDS:
        return SYMBOL_THRESHOLDS[symbol]

    base = strip_contract_suffix(symbol)
    if base in _BASE_THRESHOLDS:
        return _BASE_THRESHOLDS[base]

    return 0.60   # conservative default


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
    'imbalance_last', 'price_mom_10s', 'price_mom_30s', 'price_mom_60s',
    'spread_zscore', 'volume_ratio', 'volume_delta', 'realized_vol_60s',
    'price_impact', 'tick_count', 'spread_bps', 'bid_q1', 'ask_q1',
    'total_bid_qty', 'total_ask_qty',
    'book_imbalance_weighted', 'ofi_weighted', 'tick_direction_z',
    'price_autocorr_60', 'vol_regime_ratio',
    'ist_sin', 'ist_cos', 'mins_since_open', 'mins_to_close',
    'spread_rel', 'depth_ratio', 'mid_vs_vwap',
    'vol_ratio_10_60', 'oi_change_z',
]

MODEL_DIR = Path('research/findings/lgbm_models')


# ─────────────────────────────────────────────────────────────────────────────
# Prediction result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MLPrediction:
    direction:  int
    confidence: float
    p_up:       float
    p_down:     float
    signal:     bool


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation — single bar (live trading)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureState:
    """
    Maintains rolling state for feature computation.
    Updated bar-by-bar in O(1). One instance per symbol.
    """

    def __init__(self, window: int = 300):
        self.window      = window
        self._mid_buf    = collections.deque(maxlen=window)
        self._ret_buf    = collections.deque(maxlen=window)
        self._td_buf     = collections.deque(maxlen=window)
        self._vc_buf     = collections.deque(maxlen=60)
        self._vl_buf     = collections.deque(maxlen=window)
        self._oi_buf     = collections.deque(maxlen=window)
        self._last_mid   = None
        self._last_bid   = [0.0] * 5
        self._last_ask   = [0.0] * 5
        self._autocorr   = 0.0
        self._autocorr_i = 0

    def _safe(self, val, default: float = 0.0) -> float:
        if val is None:
            return default
        try:
            f = float(val)
            return default if math.isnan(f) or math.isinf(f) else f
        except (TypeError, ValueError):
            return default

    def _buf_std(self, buf) -> float:
        if len(buf) < 5:
            return 1.0
        return max(float(np.array(buf).std()), 1e-8)

    def _buf_mean(self, buf) -> float:
        if not buf:
            return 0.0
        return float(sum(buf) / len(buf))

    def update(self, bar) -> np.ndarray:
        """Update state and return 29-element feature vector."""
        mid = self._safe(bar.get('weighted_mid') or bar.get('close'), 0.0)
        if mid <= 0:
            mid = self._last_mid or 1.0

        ts  = self._safe(bar.get('ts_sec', 0))
        ist = (ts + 19800) % 86400

        ret = 0.0
        if self._last_mid and self._last_mid > 0:
            ret = (mid - self._last_mid) / self._last_mid
        self._mid_buf.append(mid)
        self._ret_buf.append(ret)
        self._td_buf.append(1.0 if ret > 0 else 0.0)

        vol_60  = max(self._safe(bar.get('realized_vol_60s',  2.0), 2.0), 0.001)
        vol_300 = max(self._safe(bar.get('realized_vol_300s', 2.0), 2.0), 0.001)
        self._vc_buf.append(vol_60)
        self._vl_buf.append(vol_300)
        self._oi_buf.append(self._safe(bar.get('oi', 0)))

        # OFI + book imbalance
        wb = wa = ofi = 0.0
        for level, weight in enumerate(OFI_WEIGHTS, 1):
            bq = self._safe(bar.get(f'bid_q{level}', 0))
            aq = self._safe(bar.get(f'ask_q{level}', 0))
            wb  += weight * bq
            wa  += weight * aq
            ofi += weight * (bq - self._last_bid[level-1]
                             - aq + self._last_ask[level-1])
            self._last_bid[level-1] = bq
            self._last_ask[level-1] = aq

        total_w    = max(wb + wa, 1e-8)
        book_imb_w = (wb - wa) / total_w

        ofi_z = 0.0
        if len(self._mid_buf) >= 30:
            ofi_z = max(-3.0, min(3.0, ofi / max(abs(ofi) * 3.0, 1e-8)))

        # Tick direction z-score
        tick_dir_z = 0.0
        if len(self._td_buf) >= 30:
            td60   = self._buf_mean(list(self._td_buf)[-60:])
            td_std = self._buf_std(list(self._td_buf)[-300:])
            tick_dir_z = (td60 - 0.5) / max(td_std, 0.01)

        # Price autocorrelation (every 10 bars)
        if len(self._ret_buf) >= 60 and self._autocorr_i % 10 == 0:
            rets = list(self._ret_buf)[-60:]
            r1, r2 = np.array(rets[:-1]), np.array(rets[1:])
            if r1.std() > 1e-10 and r2.std() > 1e-10:
                self._autocorr = float(np.corrcoef(r1, r2)[0, 1])
        self._autocorr_i += 1

        vol_regime  = self._buf_mean(self._vc_buf) / max(
                          self._buf_mean(self._vl_buf), 0.01)
        tb          = max(self._safe(bar.get('total_bid_qty', 0)), 1e-8)
        ta          = max(self._safe(bar.get('total_ask_qty', 0)), 1e-8)
        depth_ratio = math.log(tb / ta)

        vwap = self._safe(bar.get('vwap', mid) or mid, mid)
        if vwap <= 0:
            vwap = mid
        mid_vs_vwap = (mid - vwap) / max(vwap, 1e-8)

        vol_10          = max(self._safe(bar.get('realized_vol_10s', 2.0), 2.0), 0.001)
        vol_ratio_10_60 = vol_10 / vol_60

        oi_now = self._safe(bar.get('oi', 0))
        oi_z   = 0.0
        if len(self._oi_buf) >= 2:
            oi_chg = oi_now - list(self._oi_buf)[-2]
            oi_std = self._buf_std(self._oi_buf)
            oi_z   = oi_chg / max(oi_std, 1e-8)

        self._last_mid = mid

        features = {
            'imbalance_last':          self._safe(bar.get('imbalance_last', 0)),
            'price_mom_10s':           self._safe(bar.get('price_mom_10s', 0)),
            'price_mom_30s':           self._safe(bar.get('price_mom_30s', 0)),
            'price_mom_60s':           self._safe(bar.get('price_mom_60s', 0)),
            'spread_zscore':           self._safe(bar.get('spread_zscore', 0)),
            'volume_ratio':            self._safe(bar.get('volume_ratio', 1.0), 1.0),
            'volume_delta':            self._safe(bar.get('volume_delta', 0)),
            'realized_vol_60s':        vol_60,
            'price_impact':            self._safe(bar.get('price_impact', 0)),
            'tick_count':              self._safe(bar.get('tick_count', 1), 1.0),
            'spread_bps':              self._safe(bar.get('spread_bps', 0)),
            'bid_q1':                  self._safe(bar.get('bid_q1', 0)),
            'ask_q1':                  self._safe(bar.get('ask_q1', 0)),
            'total_bid_qty':           tb,
            'total_ask_qty':           ta,
            'book_imbalance_weighted': book_imb_w,
            'ofi_weighted':            ofi,
            'tick_direction_z':        tick_dir_z,
            'price_autocorr_60':       self._autocorr,
            'vol_regime_ratio':        vol_regime,
            'ist_sin':                 math.sin(2 * math.pi * ist / 86400),
            'ist_cos':                 math.cos(2 * math.pi * ist / 86400),
            'mins_since_open':         max(0.0, min((ist - 33300) / 60, 375.0)),
            'mins_to_close':           max(0.0, min((55800 - ist) / 60, 375.0)),
            'spread_rel':              self._safe(bar.get('spread_bps', 0)),
            'depth_ratio':             depth_ratio,
            'mid_vs_vwap':             mid_vs_vwap,
            'vol_ratio_10_60':         vol_ratio_10_60,
            'oi_change_z':             oi_z,
        }

        return np.array(
            [features.get(f, 0.0) for f in FEATURE_COLS],
            dtype=np.float32
        )


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation — batch (training)
# ─────────────────────────────────────────────────────────────────────────────

def compute_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    df  = df.copy().reset_index(drop=True)
    n   = len(df)
    mid = df['weighted_mid'].fillna(df['close'])

    wb = wa = np.zeros(n), np.zeros(n)
    wb, wa = np.zeros(n), np.zeros(n)
    ofi = np.zeros(n)
    for level, w in enumerate(OFI_WEIGHTS, 1):
        bc, ac = f'bid_q{level}', f'ask_q{level}'
        if bc in df.columns and ac in df.columns:
            bv = df[bc].fillna(0).values
            av = df[ac].fillna(0).values
            wb  += w * bv
            wa  += w * av
            ofi += w * (df[bc].diff().fillna(0).values -
                        df[ac].diff().fillna(0).values)
    tot = np.where(wb + wa == 0, 1, wb + wa)
    df['book_imbalance_weighted'] = (wb - wa) / tot
    df['ofi_weighted']            = ofi

    pd_diff = mid.diff().fillna(0)
    uptick  = (pd_diff > 0).astype(float)
    td60    = uptick.rolling(60).mean().fillna(0.5)
    td_std  = td60.rolling(300).std().replace(0, 0.1).fillna(0.1)
    df['tick_direction_z'] = (td60 - 0.5) / td_std

    rets     = mid.pct_change().fillna(0)
    autocorr = np.zeros(n)
    for i in range(60, n):
        w = rets.iloc[i-60:i].values
        if w.std() > 1e-10:
            autocorr[i] = np.corrcoef(w[:-1], w[1:])[0, 1]
    df['price_autocorr_60'] = autocorr

    vc = df['realized_vol_60s'].fillna(2.0).rolling(60).mean()
    vl = df['realized_vol_300s'].fillna(2.0).rolling(300).mean()
    df['vol_regime_ratio'] = (vc / vl.replace(0, 2.0)).fillna(1.0)

    ist = (df['ts_sec'] + 19800) % 86400
    df['ist_sin']         = np.sin(2 * np.pi * ist / 86400)
    df['ist_cos']         = np.cos(2 * np.pi * ist / 86400)
    df['mins_since_open'] = ((ist - 33300) / 60).clip(0, 375)
    df['mins_to_close']   = ((55800 - ist) / 60).clip(0, 375)

    df['spread_rel'] = df['spread_bps'].fillna(0)
    tb  = df['total_bid_qty'].replace(0, np.nan)
    ta  = df['total_ask_qty'].replace(0, np.nan)
    df['depth_ratio'] = np.log((tb / ta).fillna(1.0).clip(0.1, 10))

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


def build_labels(df: pd.DataFrame) -> np.ndarray:
    mid = df['weighted_mid'].fillna(df['close'])
    fwd = mid.shift(-FORWARD_HORIZON)
    ret = (fwd - mid) / mid
    lbl = np.zeros(len(df), dtype=np.int8)
    lbl[ret >  FLAT_THRESHOLD] =  1
    lbl[ret < -FLAT_THRESHOLD] = -1
    return lbl


# ─────────────────────────────────────────────────────────────────────────────
# Main predictor class
# ─────────────────────────────────────────────────────────────────────────────

class LGBMPredictor:
    """
    Per-symbol LightGBM price direction predictor.
    Handles contract roll automatically via strip_contract_suffix().

    Model files are stored as {SYMBOL}_up.lgbm / {SYMBOL}_dn.lgbm
    where SYMBOL is whatever name was used during training (e.g. CHOLAFIN26MAYFUT).

    When predicting for CHOLAFIN26JUNFUT:
      1. Try exact model: CHOLAFIN26JUNFUT_up.lgbm — not found
      2. Try base name:   CHOLAFIN_up.lgbm — not found
      3. Try any model with base CHOLAFIN: finds CHOLAFIN26MAYFUT_up.lgbm ✓
    """

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.models_up: dict = {}
        self.models_dn: dict = {}
        self.states:    dict = {}
        self._loaded         = False
        self._inference_times: list = []

        # base_name → symbol key in models_up
        # Built during load_models() for fast roll-aware lookup
        self._base_to_sym: dict = {}

    def _resolve_model_key(self, symbol: str) -> Optional[str]:
        """
        Find the model key for a symbol, handling contract roll.

        Returns the key used in self.models_up, or None if not found.
        """
        # Exact match
        if symbol in self.models_up:
            return symbol

        # Base name lookup via pre-built index
        base = strip_contract_suffix(symbol)
        if base in self._base_to_sym:
            return self._base_to_sym[base]

        return None

    # ── Training ──────────────────────────────────────────────────────────

    def train_symbol(self, symbol: str, dates: list) -> dict:
        from src.backtest.data_loader import load_day

        frames = []
        for date in dates:
            try:
                df_d = load_day(symbol, date, market_hours_only=True)
                if not df_d.empty and len(df_d) >= 500:
                    frames.append(df_d)
            except Exception:
                pass

        if len(frames) < 2:
            return {'ok': False, 'error': 'insufficient data'}

        df = pd.concat(frames, ignore_index=True).sort_values('ts_sec')
        df = compute_features_batch(df)
        y  = build_labels(df)

        df = df.iloc[:-FORWARD_HORIZON]
        y  = y[:-FORWARD_HORIZON]

        avail = [f for f in FEATURE_COLS if f in df.columns]
        X     = df[avail].fillna(0).values.astype(np.float32)

        non_flat = y != 0
        if non_flat.sum() < 200:
            return {'ok': False, 'error': 'too few non-flat bars'}

        Xf, yf = X[non_flat], y[non_flat]

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

        self.models_up[symbol] = m_up
        self.models_dn[symbol] = m_dn
        self._base_to_sym[strip_contract_suffix(symbol)] = symbol

        return {
            'ok': True, 'n_bars': len(X),
            'n_non_flat': int(non_flat.sum()), 'n_dates': len(frames),
        }

    def train_all(self, symbols: list, dates: list,
                  verbose: bool = True) -> dict:
        if not LGBM_AVAILABLE:
            raise ImportError("pip install lightgbm")

        results = {}
        t0      = time.perf_counter()
        print(f"Training LightGBM models for {len(symbols)} symbols...")
        print(f"Dates: {dates}")
        print(f"{'─'*60}")

        for i, symbol in enumerate(symbols, 1):
            t_sym  = time.perf_counter()
            result = self.train_symbol(symbol, dates)
            results[symbol] = result

            if verbose:
                status = (f"n={result['n_bars']:,} "
                          f"non_flat={result['n_non_flat']:,}"
                          if result['ok'] else result['error'])
                print(f"  {i:>3}/{len(symbols)} {symbol:<30} "
                      f"{'✓' if result['ok'] else '✗'}  "
                      f"{status}  ({time.perf_counter()-t_sym:.1f}s)")

        n_ok = sum(1 for r in results.values() if r['ok'])
        print(f"\n  Trained: {n_ok}/{len(symbols)} "
              f"({time.perf_counter()-t0:.0f}s total)")
        return results

    # ── Save / Load ────────────────────────────────────────────────────────

    def save_models(self):
        saved = 0
        for symbol, m_up in self.models_up.items():
            m_dn = self.models_dn.get(symbol)
            if m_dn is None:
                continue
            m_up.save_model(str(self.model_dir / f"{symbol}_up.lgbm"))
            m_dn.save_model(str(self.model_dir / f"{symbol}_dn.lgbm"))
            saved += 1

        meta = {
            'symbols':         list(self.models_up.keys()),
            'feature_cols':    FEATURE_COLS,
            'forward_horizon': FORWARD_HORIZON,
            'flat_threshold':  FLAT_THRESHOLD,
            'n_models':        saved,
        }
        with open(self.model_dir / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Saved {saved} model pairs to {self.model_dir}")

    def load_models(self, symbols: list = None):
        if not LGBM_AVAILABLE:
            logger.warning("LightGBM not available")
            return

        meta_path = self.model_dir / 'metadata.json'
        if not meta_path.exists():
            logger.warning(f"No models at {self.model_dir}")
            return

        with open(meta_path) as f:
            meta = json.load(f)

        to_load  = symbols or meta['symbols']
        n_loaded = 0

        for symbol in to_load:
            up_path = self.model_dir / f"{symbol}_up.lgbm"
            dn_path = self.model_dir / f"{symbol}_dn.lgbm"
            if up_path.exists() and dn_path.exists():
                self.models_up[symbol] = lgb.Booster(
                    model_file=str(up_path))
                self.models_dn[symbol] = lgb.Booster(
                    model_file=str(dn_path))
                # Build base → symbol index for contract roll
                base = strip_contract_suffix(symbol)
                self._base_to_sym[base] = symbol
                n_loaded += 1

        self._loaded = n_loaded > 0
        print(f"[LGBMPredictor] Loaded {n_loaded} models")

    # ── Inference ──────────────────────────────────────────────────────────

    def get_state(self, symbol: str) -> FeatureState:
        """Get or create feature state. Uses base name for roll-safe keying."""
        key = strip_contract_suffix(symbol) or symbol
        if key not in self.states:
            self.states[key] = FeatureState()
        return self.states[key]

    def predict(self, symbol: str, bar) -> MLPrediction:
        """
        Full pipeline: update feature state → predict.
        Handles contract roll — CHOLAFIN26JUNFUT uses CHOLAFIN26MAYFUT model.
        """
        state    = self.get_state(symbol)
        features = state.update(bar)
        return self.predict_features(symbol, features)

    def predict_features(self, symbol: str,
                         features: np.ndarray) -> MLPrediction:
        """Predict from pre-computed feature vector (~0.5ms)."""
        model_key = self._resolve_model_key(symbol)

        if not self._loaded or model_key is None:
            return MLPrediction(
                direction=0, confidence=0.5,
                p_up=0.5, p_down=0.5, signal=False
            )

        t0    = time.perf_counter()
        x     = features.reshape(1, -1)
        p_up  = float(self.models_up[model_key].predict(x)[0])
        p_dn  = float(self.models_dn[model_key].predict(x)[0])
        self._inference_times.append(time.perf_counter() - t0)

        threshold  = get_threshold(symbol)
        confidence = max(p_up, p_dn)

        if p_up > threshold:
            direction = 1
        elif p_dn > threshold:
            direction = -1
        else:
            direction = 0

        return MLPrediction(
            direction  = direction,
            confidence = round(confidence, 4),
            p_up       = round(p_up, 4),
            p_down     = round(p_dn, 4),
            signal     = direction != 0,
        )

    def avg_inference_ms(self) -> float:
        if not self._inference_times:
            return 0.0
        return float(np.mean(self._inference_times)) * 1000

    def reset_states(self):
        self.states.clear()

    @property
    def available_symbols(self) -> list:
        return list(self.models_up.keys())

    @property
    def n_models(self) -> int:
        return len(self.models_up)


# ─────────────────────────────────────────────────────────────────────────────
# Training entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_and_save(dates: list = None, symbols: list = None):
    """Train all models and save to disk."""
    if dates is None:
        dates = [
            '2026-05-13', '2026-05-14', '2026-05-15',
            '2026-05-18', '2026-05-19', '2026-05-20',
            '2026-05-21', '2026-05-22', '2026-05-25',
            '2026-05-27', '2026-05-28',
        ]

    if symbols is None:
        symbols = list(SYMBOL_THRESHOLDS.keys())

    predictor = LGBMPredictor()
    predictor.train_all(symbols, dates)
    predictor.save_models()
    print(f"\nModels saved to {MODEL_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if '--train' in sys.argv:
        train_and_save()
    else:
        print("LGBMPredictor — Quick Test")
        print("=" * 50)

        # Test get_threshold contract roll
        print("\nThreshold lookup (contract roll test):")
        for sym in ['CHOLAFIN26MAYFUT', 'CHOLAFIN26JUNFUT',
                    'HAVELLS26MAYFUT', 'HAVELLS26JUNFUT',
                    'UNKNOWN26JUNFUT']:
            print(f"  {sym:<30} → {get_threshold(sym)}")

        # Test FeatureState
        state = FeatureState()
        dummy = {
            'weighted_mid': 1520.0, 'close': 1520.0,
            'ts_sec': 1716000000,
            'bid_q1': 625, 'ask_q1': 625,
            'bid_q2': 500, 'ask_q2': 500,
            'bid_q3': 400, 'ask_q3': 400,
            'bid_q4': 300, 'ask_q4': 300,
            'bid_q5': 200, 'ask_q5': 200,
            'total_bid_qty': 20000, 'total_ask_qty': 21000,
            'imbalance_last': 0.05, 'price_mom_10s': 0.1,
            'price_mom_30s': 0.2,  'price_mom_60s': 0.15,
            'spread_zscore': 0.5,  'volume_ratio': 1.2,
            'volume_delta': 100,   'realized_vol_60s': 2.5,
            'realized_vol_10s': 2.2, 'realized_vol_300s': 2.3,
            'price_impact': 0.01,  'tick_count': 5,
            'spread_bps': 1.5,     'vwap': 1519.5,
            'oi': 100000,
        }
        for _ in range(100):
            features = state.update(dummy)
        print(f"\nFeature vector shape: {features.shape}")

        # Test predictor with JUNFUT (contract roll)
        predictor = LGBMPredictor()
        predictor.load_models()

        for sym in ['CHOLAFIN26MAYFUT', 'CHOLAFIN26JUNFUT']:
            pred = predictor.predict(sym, dummy)
            key  = predictor._resolve_model_key(sym)
            print(f"\n  {sym}")
            print(f"    model_key={key}  dir={pred.direction}  "
                  f"conf={pred.confidence:.3f}  signal={pred.signal}")

        print(f"\nAvg inference: {predictor.avg_inference_ms():.3f}ms")
        print(f"\nRun training: python src/ml/models/lgbm_predictor.py --train")
