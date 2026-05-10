"""
Lead-Lag Model — Cross-Futures Predictive Signals
===================================================
Measures which futures lead (predict) which others using
cross-correlation analysis and Lasso regression.

Core insight: NIFTY moves first, then stocks follow with a lag.
When NIFTY drops 10pts, you can adjust HDFCBANK quotes 15-30
seconds BEFORE the move hits — a genuine structural edge.

Mathematics:
    Cross-correlation at lag k:
        CCF(X, Y, k) = corr(X_t, Y_{t+k})

    If CCF(NIFTY, HDFCBANK, k=3) is high:
        → NIFTY at time t predicts HDFCBANK at time t+3 bars

    Lasso regression (sparse leader detection):
        y_t = β_0 + Σ_i Σ_k β_{i,k} × X_{i,t-k} + ε_t
        with L1 penalty: min ||y - Xβ||² + α||β||₁

    The L1 penalty forces most β to zero — only true leaders
    get non-zero coefficients. More robust than plain correlation.

    Predictive signal (live trading):
        signal_t = Σ_{leaders} β_{i,k} × ΔX_{i,t-k}
        If signal > threshold → expect upward move → skew ask tighter
        If signal < threshold → expect downward move → skew bid tighter

Usage:
    # Offline analysis — find leaders for all symbols
    python src/models/lead_lag.py --date 2026-04-30 --target HDFCBANK26MAYFUT

    # Analyse all symbols, find universal leaders
    python src/models/lead_lag.py --date 2026-04-30 --all

    # In live trading (import)
    from src.models.lead_lag import LeadLagSignal
    signal = LeadLagSignal(target='HDFCBANK26MAYFUT', leaders=['NIFTY26MAYFUT'])
    signal.update(bars_dict)   # dict of {symbol: latest_bar}
    skew = signal.quote_skew() # positive = skew ask, negative = skew bid

Structure:
    src/models/lead_lag.py   ← this file
"""

import argparse
import io
import json
import logging
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from google.cloud import storage

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

GCS_BUCKET       = "hedge-fund-494103-marketdata"
PROCESSED_PREFIX = "processed/features"
MODELS_PREFIX    = "models/lead_lag"

# ── Constants ─────────────────────────────────────────────────────────────────

# Max lag to test in bars (minutes). 10 = look back 10 minutes.
MAX_LAG = 10

# Known index futures — always check these as potential leaders
INDEX_FUTURES = [
    "NIFTY26MAYFUT",
    "BANKNIFTY26MAYFUT",
    "FINNIFTY26MAYFUT",
    "MIDCPNIFTY26MAYFUT",
]

# Lasso regularization strength — higher = sparser (fewer leaders selected)
LASSO_ALPHA = 0.001


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class LeadLagResult:
    """Cross-correlation result between a leader and target."""
    leader:       str
    target:       str
    best_lag:     int        # lag in bars where correlation is highest
    best_corr:    float      # peak cross-correlation coefficient
    corr_by_lag:  list       # correlation at each lag 0..MAX_LAG
    is_leader:    bool       # True if correlation is significant
    direction:    str        # POSITIVE / NEGATIVE / MIXED


@dataclass
class LassoLeader:
    """A leader selected by Lasso regression."""
    symbol:      str
    lag:         int
    coefficient: float       # Lasso β — magnitude = predictive strength
    abs_coef:    float


@dataclass
class TargetAnalysis:
    """Full lead-lag analysis for one target symbol."""
    target:          str
    date:            str
    n_bars:          int
    ccf_results:     list[LeadLagResult]
    lasso_leaders:   list[LassoLeader]
    top_leader:      Optional[str]
    top_leader_lag:  int
    signal_strength: float    # 0-1, how predictable this target is
    recommendation:  str


# ─── GCS helpers ──────────────────────────────────────────────────────────────

def load_symbol(client: storage.Client, symbol: str, date: str) -> Optional[pd.DataFrame]:
    path  = f"{PROCESSED_PREFIX}/{symbol}/{date}.parquet"
    blob  = client.bucket(GCS_BUCKET).blob(path)
    if not blob.exists():
        return None
    return pd.read_parquet(io.BytesIO(blob.download_as_bytes()))


def list_futures(client: storage.Client, date: str) -> list[str]:
    prefix = f"{PROCESSED_PREFIX}/"
    blobs  = client.list_blobs(GCS_BUCKET, prefix=prefix)
    syms   = []
    for blob in blobs:
        parts = blob.name.split("/")
        if len(parts) == 4 and parts[3] == f"{date}.parquet":
            sym = parts[2]
            if "CE" not in sym and "PE" not in sym:
                syms.append(sym)
    return sorted(syms)


def save_results(client: storage.Client, date: str, results: dict):
    path = f"{MODELS_PREFIX}/{date}/lead_lag.json"
    client.bucket(GCS_BUCKET).blob(path).upload_from_string(
        json.dumps(results, indent=2, default=str),
        content_type="application/json",
    )
    log.info(f"Saved → gs://{GCS_BUCKET}/{path}")


# ─── Feature extraction ───────────────────────────────────────────────────────

def extract_returns(df: pd.DataFrame) -> pd.Series:
    """
    Extract 1-bar log returns aligned to ts_sec.
    Uses close price. Returns pd.Series indexed by ts_sec.
    """
    df = df.copy()
    df = df.sort_values("ts_sec").drop_duplicates("ts_sec")
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    return df.set_index("ts_sec")["ret"].dropna()


def align_series(s1: pd.Series, s2: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Align two return series on common timestamps."""
    common = s1.index.intersection(s2.index)
    return s1.loc[common], s2.loc[common]


# ─── Cross-correlation analysis ───────────────────────────────────────────────

def cross_correlation(
    leader_ret: pd.Series,
    target_ret: pd.Series,
    max_lag: int = MAX_LAG,
) -> LeadLagResult:
    """
    Compute CCF(leader, target, lag=k) for k in 0..max_lag.

    CCF(k) = corr(leader_t, target_{t+k})
    Positive k means leader predicts target k bars in the future.
    """
    corrs = []
    for lag in range(0, max_lag + 1):
        if lag == 0:
            l, t = align_series(leader_ret, target_ret)
        else:
            # Shift target forward by lag (leader at t predicts target at t+lag)
            l = leader_ret.iloc[:-lag].values
            t = target_ret.iloc[lag:].values
            min_len = min(len(l), len(t))
            l, t = l[:min_len], t[:min_len]

        if len(l) < 30:
            corrs.append(0.0)
            continue

        # Pearson correlation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = float(np.corrcoef(l, t)[0, 1])
        corrs.append(c if not np.isnan(c) else 0.0)

    abs_corrs  = [abs(c) for c in corrs]
    best_lag   = int(np.argmax(abs_corrs))
    best_corr  = corrs[best_lag]
    is_leader  = abs(best_corr) > 0.05 and best_lag > 0  # lag=0 is contemporaneous

    if best_corr > 0.03:
        direction = "POSITIVE"
    elif best_corr < -0.03:
        direction = "NEGATIVE"
    else:
        direction = "MIXED"

    # Extract symbol names from series names if available
    leader_name = str(leader_ret.name) if leader_ret.name else "LEADER"
    target_name = str(target_ret.name) if target_ret.name else "TARGET"

    return LeadLagResult(
        leader      = leader_name,
        target      = target_name,
        best_lag    = best_lag,
        best_corr   = round(best_corr, 4),
        corr_by_lag = [round(c, 4) for c in corrs],
        is_leader   = is_leader,
        direction   = direction,
    )


# ─── Lasso regression ─────────────────────────────────────────────────────────

def lasso_leader_detection(
    target_ret:  pd.Series,
    leader_rets: dict[str, pd.Series],
    max_lag:     int   = MAX_LAG,
    alpha:       float = LASSO_ALPHA,
) -> list[LassoLeader]:
    """
    Sparse leader detection via Lasso regression.

    Build feature matrix X where each column is leader_i at lag_k.
    Fit: target_t = Σ β_{i,k} × leader_{i,t-k}
    L1 penalty forces most β to zero — only true leaders survive.

    Returns list of LassoLeader with non-zero coefficients.
    """
    try:
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        log.warning("sklearn not installed — skipping Lasso. pip install scikit-learn")
        return []

    # Build feature matrix
    feature_cols = []
    feature_names = []

    common_idx = target_ret.index
    for sym, ret in leader_rets.items():
        common_idx = common_idx.intersection(ret.index)

    if len(common_idx) < 50:
        return []

    y = target_ret.loc[common_idx].values

    for sym, ret in leader_rets.items():
        r = ret.loc[common_idx]
        for lag in range(1, max_lag + 1):
            lagged = r.shift(lag).fillna(0).values
            feature_cols.append(lagged)
            feature_names.append(f"{sym}__lag{lag}")

    if not feature_cols:
        return []

    X = np.column_stack(feature_cols)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Lasso
    model = Lasso(alpha=alpha, max_iter=5000, fit_intercept=True)
    model.fit(X_scaled, y)

    # Extract non-zero coefficients
    leaders = []
    for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
        if abs(coef) > 1e-8:
            parts = name.split("__lag")
            sym   = parts[0]
            lag   = int(parts[1])
            leaders.append(LassoLeader(
                symbol  = sym,
                lag     = lag,
                coefficient = round(float(coef), 6),
                abs_coef    = round(abs(float(coef)), 6),
            ))

    return sorted(leaders, key=lambda x: x.abs_coef, reverse=True)


# ─── Full target analysis ─────────────────────────────────────────────────────

def analyse_target(
    target:      str,
    date:        str,
    client:      storage.Client,
    candidates:  Optional[list[str]] = None,
) -> Optional[TargetAnalysis]:
    """
    Full lead-lag analysis for one target symbol.
    Tests all candidate leaders (defaults to index futures).
    """
    target_df = load_symbol(client, target, date)
    if target_df is None or len(target_df) < 50:
        log.warning(f"Insufficient data for {target}")
        return None

    target_ret      = extract_returns(target_df)
    target_ret.name = target

    # Default candidates: index futures + any explicitly passed
    if candidates is None:
        candidates = INDEX_FUTURES
    candidates = [c for c in candidates if c != target]

    # Load all candidate leader data
    leader_rets = {}
    for sym in candidates:
        df = load_symbol(client, sym, date)
        if df is not None and len(df) >= 50:
            ret      = extract_returns(df)
            ret.name = sym
            leader_rets[sym] = ret

    if not leader_rets:
        log.warning(f"No leader data found for {target}")
        return None

    # ── Cross-correlation ──
    ccf_results = []
    for sym, ret in leader_rets.items():
        result = cross_correlation(ret, target_ret)
        result.leader = sym
        result.target = target
        ccf_results.append(result)

    # Sort by absolute correlation
    ccf_results.sort(key=lambda x: abs(x.best_corr), reverse=True)

    # ── Lasso ──
    lasso_leaders = lasso_leader_detection(target_ret, leader_rets)

    # ── Summarize ──
    significant = [r for r in ccf_results if r.is_leader]
    top_leader  = ccf_results[0].leader if ccf_results else None
    top_lag     = ccf_results[0].best_lag if ccf_results else 0

    # Signal strength: average abs correlation of significant leaders
    if significant:
        signal_strength = float(np.mean([abs(r.best_corr) for r in significant]))
    else:
        signal_strength = 0.0

    if signal_strength > 0.15:
        rec = "STRONG signal — integrate lead-lag into A-S quoting"
    elif signal_strength > 0.07:
        rec = "MODERATE signal — useful as confirmation filter"
    elif signal_strength > 0.03:
        rec = "WEAK signal — monitor, not yet tradeable"
    else:
        rec = "NO signal — target moves independently"

    return TargetAnalysis(
        target          = target,
        date            = date,
        n_bars          = len(target_ret),
        ccf_results     = ccf_results,
        lasso_leaders   = lasso_leaders,
        top_leader      = top_leader,
        top_leader_lag  = top_lag,
        signal_strength = round(signal_strength, 4),
        recommendation  = rec,
    )


# ─── Live signal (plugs into A-S) ────────────────────────────────────────────

class LeadLagSignal:
    """
    Real-time lead-lag signal for live trading.
    Maintains rolling return buffer per symbol.
    Outputs a quote skew: positive = expect up = tighten ask,
                          negative = expect down = tighten bid.

    Usage in mm_strategy.py (integration — week 2):
        from src.models.lead_lag import LeadLagSignal

        # In __init__:
        self.lead_lag = LeadLagSignal(
            target  = 'HDFCBANK26MAYFUT',
            leaders = {'NIFTY26MAYFUT': {'lag': 3, 'coef': 0.12}},
        )

        # In on_bar:
        self.lead_lag.update('NIFTY26MAYFUT', nifty_bar)
        skew = self.lead_lag.quote_skew()

        # Apply skew to reservation price:
        r = self._reservation_price(mid, q, sigma, T)
        r += skew * self.min_spread * 0.5   # shift r by fraction of spread
    """

    def __init__(
        self,
        target:  str,
        leaders: dict,          # {symbol: {'lag': k, 'coef': β}}
        buffer:  int = 20,      # how many bars to keep per symbol
        threshold: float = 0.02 # min signal to act on
    ):
        self.target    = target
        self.leaders   = leaders
        self.threshold = threshold

        # Rolling price buffer per leader
        self._prices: dict[str, deque] = {
            sym: deque(maxlen=buffer)
            for sym in leaders
        }
        self._signal = 0.0

    def update(self, symbol: str, bar: dict) -> None:
        """Feed a new bar for a leader symbol."""
        if symbol not in self._prices:
            return
        price = bar.get("close") or bar.get("weighted_mid")
        if price:
            self._prices[symbol].append(float(price))
        self._recompute()

    def _recompute(self) -> None:
        """Recompute signal from current leader buffers."""
        total = 0.0
        count = 0
        for sym, config in self.leaders.items():
            lag  = config.get("lag", 1)
            coef = config.get("coef", 0.0)
            buf  = list(self._prices[sym])
            if len(buf) < lag + 2:
                continue
            # Log return at the relevant lag
            p_now  = buf[-(lag)]
            p_prev = buf[-(lag + 1)]
            if p_prev > 0:
                ret    = np.log(p_now / p_prev)
                total += coef * ret
                count += 1
        self._signal = total / count if count > 0 else 0.0

    def quote_skew(self) -> float:
        """
        Returns skew in [-1, 1].
        Positive → expect price up → tighten ask (more aggressive selling)
        Negative → expect price down → tighten bid (more aggressive buying)
        Apply as: r_adjusted = r + skew * spread * 0.5
        """
        if abs(self._signal) < self.threshold:
            return 0.0
        return float(np.clip(self._signal * 100, -1.0, 1.0))

    @classmethod
    def from_analysis(cls, analysis: TargetAnalysis, threshold: float = 0.02):
        """
        Build a LeadLagSignal directly from a TargetAnalysis result.
        Use after running analyse_target() offline.

        Example:
            analysis = analyse_target('HDFCBANK26MAYFUT', '2026-05-12', client)
            signal   = LeadLagSignal.from_analysis(analysis)
        """
        leaders = {}
        # Use Lasso leaders if available (more reliable)
        if analysis.lasso_leaders:
            for ll in analysis.lasso_leaders[:3]:  # top 3 only
                leaders[ll.symbol] = {
                    "lag":  ll.lag,
                    "coef": ll.coefficient,
                }
        # Fall back to CCF leaders
        elif analysis.ccf_results:
            for r in analysis.ccf_results[:2]:
                if r.is_leader:
                    leaders[r.leader] = {
                        "lag":  r.best_lag,
                        "coef": r.best_corr,
                    }
        return cls(
            target    = analysis.target,
            leaders   = leaders,
            threshold = threshold,
        )


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_analysis(analysis: TargetAnalysis):
    print(f"\n{'=' * 65}")
    print(f"  LEAD-LAG ANALYSIS — {analysis.target}")
    print(f"  Date: {analysis.date}  |  Bars: {analysis.n_bars}")
    print(f"{'=' * 65}")
    print(f"  Signal Strength: {analysis.signal_strength:.4f}")
    print(f"  Recommendation:  {analysis.recommendation}")

    print(f"\n── Cross-Correlation Results ─────────────────────────────────")
    print(f"  {'Leader':<30} {'Best Lag':>8} {'Corr':>8} {'Direction':<12} {'Leader?'}")
    print(f"  {'-'*62}")
    for r in analysis.ccf_results:
        flag = "✓" if r.is_leader else " "
        print(
            f"  {r.leader:<30} {r.best_lag:>8} "
            f"{r.best_corr:>8.4f} {r.direction:<12} {flag}"
        )

    if analysis.lasso_leaders:
        print(f"\n── Lasso Selected Leaders ────────────────────────────────────")
        print(f"  {'Symbol':<30} {'Lag':>5} {'Coefficient':>12}")
        print(f"  {'-'*50}")
        for ll in analysis.lasso_leaders[:10]:
            print(f"  {ll.symbol:<30} {ll.lag:>5} {ll.coefficient:>12.6f}")
    else:
        print(f"\n── Lasso: no significant leaders found ───────────────────────")

    print(f"\n── Integration Code ──────────────────────────────────────────")
    if analysis.lasso_leaders:
        leaders_dict = {
            ll.symbol: {"lag": ll.lag, "coef": ll.coefficient}
            for ll in analysis.lasso_leaders[:3]
        }
        print(f"  signal = LeadLagSignal(")
        print(f"      target  = '{analysis.target}',")
        print(f"      leaders = {json.dumps(leaders_dict, indent=8)},")
        print(f"  )")
    elif analysis.top_leader:
        print(f"  signal = LeadLagSignal(")
        print(f"      target  = '{analysis.target}',")
        print(f"      leaders = {{'{analysis.top_leader}': "
              f"{{'lag': {analysis.top_leader_lag}, 'coef': {analysis.ccf_results[0].best_corr}}}}},")
        print(f"  )")
    print(f"{'=' * 65}\n")


def print_summary(analyses: list[TargetAnalysis]):
    print(f"\n{'=' * 75}")
    print(f"  LEAD-LAG SUMMARY — {analyses[0].date if analyses else ''}")
    print(f"{'=' * 75}")
    print(f"  {'Target':<30} {'Signal':>8} {'Top Leader':<25} {'Lag':>5}  Rec")
    print(f"  {'-'*73}")

    for a in sorted(analyses, key=lambda x: x.signal_strength, reverse=True):
        leader = a.top_leader or "none"
        rec    = a.recommendation.split(" — ")[0]
        print(
            f"  {a.target:<30} {a.signal_strength:>8.4f} "
            f"{leader:<25} {a.top_leader_lag:>5}  {rec}"
        )
    print(f"{'=' * 75}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lead-Lag Cross-Futures Model")
    parser.add_argument("--date",     required=True, help="Date YYYY-MM-DD")
    parser.add_argument("--target",   type=str,      help="Single target symbol")
    parser.add_argument("--all",      action="store_true", help="Analyse all futures")
    parser.add_argument("--leaders",  type=str, nargs="+",
                        help="Override leader candidates (default: index futures)")
    parser.add_argument("--max-lag",  type=int, default=MAX_LAG,
                        help=f"Max lag in bars (default {MAX_LAG})")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    client = storage.Client()

    if args.target:
        targets = [args.target]
    elif args.all:
        targets = list_futures(client, args.date)
        targets = [t for t in targets if t not in INDEX_FUTURES]
        log.info(f"Analysing {len(targets)} stock futures against index leaders")
    else:
        parser.error("Provide --target SYMBOL or --all")

    candidates = args.leaders or INDEX_FUTURES

    analyses = []
    for target in targets:
        log.info(f"Analysing {target}...")
        result = analyse_target(target, args.date, client, candidates)
        if result:
            analyses.append(result)
            if args.target:
                print_analysis(result)

    if not analyses:
        log.error("No results — check data availability")
        return

    if args.all or len(analyses) > 1:
        print_summary(analyses)

    if not args.no_upload:
        payload = {
            "date":         args.date,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "max_lag":      args.max_lag,
            "targets": [
                {
                    "target":          a.target,
                    "signal_strength": a.signal_strength,
                    "top_leader":      a.top_leader,
                    "top_leader_lag":  a.top_leader_lag,
                    "recommendation":  a.recommendation,
                    "lasso_leaders":   [
                        {"symbol": ll.symbol, "lag": ll.lag, "coef": ll.coefficient}
                        for ll in a.lasso_leaders[:5]
                    ],
                    "ccf_results":     [
                        {"leader": r.leader, "lag": r.best_lag,
                         "corr": r.best_corr, "is_leader": r.is_leader}
                        for r in a.ccf_results
                    ],
                }
                for a in analyses
            ],
        }
        save_results(client, args.date, payload)


if __name__ == "__main__":
    main()