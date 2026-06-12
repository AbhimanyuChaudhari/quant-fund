"""
Kyle's Lambda — Adverse Selection Model
========================================
Measures price impact per unit of signed order flow.

    Δprice = λ × signed_volume + ε

A high λ means informed traders are active — every unit of volume
moves price more. For market making:
  - High λ  → widen spread, reduce size (toxic flow)
  - Low λ   → tighten spread, increase size (uninformed flow)

This plugs directly into Avellaneda-Stoikov as a spread multiplier.

Usage:
    # Analyse a single symbol from GCS
    python src/models/kyle_lambda.py --symbol NIFTY26MAYFUT --date 2026-05-12

    # Analyse all symbols for a date
    python src/models/kyle_lambda.py --date 2026-05-12 --all

    # Use in live trading (import)
    from src.models.kyle_lambda import KyleLambdaModel
    model = KyleLambdaModel(window=30)
    model.update(bar)
    multiplier = model.spread_multiplier()   # feed into A-S

Structure:
    src/models/kyle_lambda.py   ← this file
"""

import argparse
import io
import json
import logging
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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

GCS_BUCKET         = "hedge-fund-494103-marketdata-mumbai"
PROCESSED_PREFIX   = "processed/features"
MODELS_PREFIX      = "models/kyle_lambda"


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Bar:
    """Single 1-minute bar from processed features."""
    symbol:       str
    ts:           datetime
    close:        float
    volume_delta: float   # signed: positive = net buying, negative = net selling
    tick_count:   float
    imbalance:    float   # order book imbalance (bid_qty - ask_qty) / total
    spread_bps:   float


@dataclass
class LambdaEstimate:
    """Kyle's Lambda estimate for a rolling window."""
    symbol:        str
    timestamp:     datetime
    window_bars:   int          # how many bars used
    lambda_val:    float        # price impact per unit signed volume
    lambda_robust: float        # Theil-Sen robust estimate
    r_squared:     float        # OLS fit quality
    avg_spread_bps: float
    toxicity_score: float       # 0-1 normalized
    spread_multiplier: float    # plug into A-S: bid = r - spread/2 * multiplier
    regime:        str          # LOW / MEDIUM / HIGH / EXTREME toxicity


# ─── Core model ───────────────────────────────────────────────────────────────

class KyleLambdaModel:
    """
    Rolling Kyle's Lambda estimator.

    Designed for two use cases:
      1. Offline analysis — load a full day parquet, compute lambda per window
      2. Live trading    — call update(bar) every tick, query spread_multiplier()

    Parameters
    ----------
    window : int
        Rolling window in bars (minutes). Default 30 = 30 min window.
    min_bars : int
        Minimum bars needed before estimating. Default 10.
    multiplier_clip : tuple
        Clamp spread multiplier to this range. Default (0.5, 3.0).
        0.5 = tighten spread to half, 3.0 = widen to 3x.
    """

    def __init__(
        self,
        window: int = 30,
        min_bars: int = 10,
        multiplier_clip: tuple = (0.5, 3.0),
    ):
        self.window          = window
        self.min_bars        = min_bars
        self.multiplier_clip = multiplier_clip

        # Rolling buffer
        self._bars: deque[Bar] = deque(maxlen=window)

        # Last estimate (cached)
        self._last_estimate: Optional[LambdaEstimate] = None

        # Percentile thresholds (calibrated from data, updated daily)
        self._p25 = None
        self._p50 = None
        self._p75 = None

    def update(self, bar: Bar) -> Optional[LambdaEstimate]:
        """
        Add a new bar and recompute lambda.
        Returns estimate if enough data, else None.
        """
        self._bars.append(bar)
        if len(self._bars) < self.min_bars:
            return None

        estimate = self._estimate(list(self._bars))
        self._last_estimate = estimate
        return estimate

    def spread_multiplier(self) -> float:
        """
        Returns multiplier for A-S spread.
        Call this from mm_strategy.py instead of fixed spread.

        Usage in A-S:
            base_spread = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/kappa)
            adjusted_spread = base_spread * model.spread_multiplier()
        """
        if self._last_estimate is None:
            return 1.0
        return self._last_estimate.spread_multiplier

    def _estimate(self, bars: list[Bar]) -> LambdaEstimate:
        """Core OLS + Theil-Sen estimation."""
        # Build price changes and signed volume
        closes       = np.array([b.close        for b in bars])
        signed_vols  = np.array([b.volume_delta for b in bars])
        spread_bps   = np.array([b.spread_bps   for b in bars])

        # Δprice = price(t) - price(t-1)
        delta_price = np.diff(closes)
        signed_vol  = signed_vols[1:]   # align with delta_price

        # Filter zero-volume bars (no trades = no information)
        mask = signed_vol != 0
        if mask.sum() < self.min_bars // 2:
            return self._null_estimate(bars[-1])

        dp = delta_price[mask]
        sv = signed_vol[mask]

        # ── OLS: Δp = λ × sv ──
        # No intercept — Kyle's model assumes zero drift over short windows
        lambda_ols, r2 = self._ols(dp, sv)

        # ── Theil-Sen robust estimate ──
        # More robust to outlier ticks (news spikes, fat fingers)
        lambda_robust = self._theil_sen(dp, sv)

        # ── Toxicity score (0-1) ──
        # Normalize lambda against its own recent distribution
        # Use robust estimate as it's more stable
        toxicity = self._toxicity_score(lambda_robust, spread_bps.mean())

        # ── Spread multiplier ──
        # Linear mapping: toxicity 0 → 0.5x, toxicity 1 → 3.0x
        lo, hi = self.multiplier_clip
        multiplier = lo + toxicity * (hi - lo)

        # ── Regime ──
        if toxicity < 0.25:
            regime = "LOW"
        elif toxicity < 0.50:
            regime = "MEDIUM"
        elif toxicity < 0.75:
            regime = "HIGH"
        else:
            regime = "EXTREME"

        return LambdaEstimate(
            symbol           = bars[-1].symbol,
            timestamp        = bars[-1].ts,
            window_bars      = len(bars),
            lambda_val       = round(lambda_ols, 6),
            lambda_robust    = round(lambda_robust, 6),
            r_squared        = round(r2, 4),
            avg_spread_bps   = round(spread_bps.mean(), 2),
            toxicity_score   = round(toxicity, 4),
            spread_multiplier= round(multiplier, 4),
            regime           = regime,
        )

    def _ols(self, dp: np.ndarray, sv: np.ndarray) -> tuple[float, float]:
        """OLS regression Δp ~ sv, no intercept. Returns (lambda, r2)."""
        sv2 = sv ** 2
        denom = sv2.sum()
        if denom == 0:
            return 0.0, 0.0
        lam = (dp * sv).sum() / denom

        # R² = 1 - SS_res / SS_tot
        residuals = dp - lam * sv
        ss_res = (residuals ** 2).sum()
        ss_tot = ((dp - dp.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return lam, max(0.0, r2)

    def _theil_sen(self, dp: np.ndarray, sv: np.ndarray) -> float:
        """
        Theil-Sen estimator — median of pairwise slopes.
        Robust to up to 29% outliers.
        Capped at 1000 pairs for performance.
        """
        n = len(dp)
        slopes = []
        indices = np.arange(n)
        if n > 45:
            # Subsample for speed — still robust
            rng = np.random.default_rng(42)
            indices = rng.choice(n, 45, replace=False)

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                dsv = sv[indices[j]] - sv[indices[i]]
                if abs(dsv) > 1e-10:
                    slopes.append((dp[indices[j]] - dp[indices[i]]) / dsv)

        return float(np.median(slopes)) if slopes else 0.0

    def _toxicity_score(self, lambda_val: float, avg_spread_bps: float) -> float:
        """
        Normalize lambda to 0-1 toxicity score.

        Key insight: lambda matters relative to spread.
        If lambda >> spread, informed flow is eating your quotes.
        Ratio: lambda × typical_volume / spread_in_price_terms
        """
        if avg_spread_bps <= 0 or lambda_val <= 0:
            return 0.0

        # Use percentile thresholds if calibrated, else use heuristic
        if self._p75 is not None and self._p75 > self._p25:
            score = (lambda_val - self._p25) / (self._p75 - self._p25)
        else:
            # Heuristic: lambda > 0.001 is moderately toxic for futures
            # This gets replaced once you calibrate on real data
            score = min(lambda_val / 0.005, 1.0)

        return float(np.clip(score, 0.0, 1.0))

    def calibrate_thresholds(self, lambda_history: list[float]):
        """
        Calibrate toxicity percentiles from historical lambda values.
        Call this daily after loading previous day's data.

        Usage:
            estimates = model.analyse_day(df)
            lambdas = [e.lambda_robust for e in estimates]
            model.calibrate_thresholds(lambdas)
        """
        arr = np.array([x for x in lambda_history if x > 0])
        if len(arr) < 10:
            return
        self._p25 = float(np.percentile(arr, 25))
        self._p50 = float(np.percentile(arr, 50))
        self._p75 = float(np.percentile(arr, 75))
        log.info(
            f"Calibrated thresholds: p25={self._p25:.6f} "
            f"p50={self._p50:.6f} p75={self._p75:.6f}"
        )

    def analyse_day(self, df: pd.DataFrame) -> list[LambdaEstimate]:
        """
        Run rolling lambda over a full day parquet.
        Returns list of LambdaEstimate, one per bar (after warmup).
        """
        self._bars.clear()
        estimates = []

        for _, row in df.iterrows():
            bar = self._row_to_bar(row)
            est = self.update(bar)
            if est is not None:
                estimates.append(est)

        return estimates

    def _row_to_bar(self, row: pd.Series) -> Bar:
        ts = pd.to_datetime(row.get("ts_ist") or row.get("ts_sec"))
        return Bar(
            symbol       = str(row.get("symbol", "")),
            ts           = ts,
            close        = float(row.get("close", 0)),
            volume_delta = float(row.get("volume_delta", 0)),
            tick_count   = float(row.get("tick_count", 0)),
            imbalance    = float(row.get("imbalance_last", 0)),
            spread_bps   = float(row.get("spread_bps", 1)),
        )

    def _null_estimate(self, bar: Bar) -> LambdaEstimate:
        return LambdaEstimate(
            symbol            = bar.symbol,
            timestamp         = bar.ts,
            window_bars       = len(self._bars),
            lambda_val        = 0.0,
            lambda_robust     = 0.0,
            r_squared         = 0.0,
            avg_spread_bps    = bar.spread_bps,
            toxicity_score    = 0.0,
            spread_multiplier = 1.0,
            regime            = "LOW",
        )


# ─── GCS helpers ──────────────────────────────────────────────────────────────

def load_from_gcs(symbol: str, date: str) -> Optional[pd.DataFrame]:
    client = storage.Client()
    path   = f"{PROCESSED_PREFIX}/{symbol}/{date}.parquet"
    bucket = client.bucket(GCS_BUCKET)
    blob   = bucket.blob(path)
    if not blob.exists():
        log.warning(f"Not found: gs://{GCS_BUCKET}/{path}")
        return None
    data = blob.download_as_bytes()
    return pd.read_parquet(io.BytesIO(data))


def list_symbols(date: str) -> list[str]:
    client  = storage.Client()
    prefix  = f"{PROCESSED_PREFIX}/"
    blobs   = client.list_blobs(GCS_BUCKET, prefix=prefix)
    symbols = []
    for blob in blobs:
        parts = blob.name.split("/")
        if len(parts) == 4 and parts[3] == f"{date}.parquet":
            # Futures only for now (no CE/PE)
            sym = parts[2]
            if "CE" not in sym and "PE" not in sym:
                symbols.append(sym)
    return sorted(symbols)


def save_results_to_gcs(date: str, results: dict):
    client  = storage.Client()
    bucket  = client.bucket(GCS_BUCKET)
    path    = f"{MODELS_PREFIX}/{date}/lambda_estimates.json"
    bucket.blob(path).upload_from_string(
        json.dumps(results, indent=2, default=str),
        content_type="application/json",
    )
    log.info(f"Saved → gs://{GCS_BUCKET}/{path}")


# ─── Analysis helpers ─────────────────────────────────────────────────────────

def analyse_symbol(symbol: str, date: str, window: int = 30) -> Optional[dict]:
    df = load_from_gcs(symbol, date)
    if df is None or len(df) < 15:
        return None

    model     = KyleLambdaModel(window=window)
    estimates = model.analyse_day(df)

    if not estimates:
        return None

    lambdas   = [e.lambda_robust for e in estimates]
    toxicity  = [e.toxicity_score for e in estimates]
    regimes   = [e.regime for e in estimates]

    # Calibrate and re-run for better toxicity scores
    model.calibrate_thresholds(lambdas)
    model._bars.clear()
    estimates = model.analyse_day(df)
    if estimates:
        lambdas  = [e.lambda_robust for e in estimates]
        toxicity = [e.toxicity_score for e in estimates]
        regimes  = [e.regime for e in estimates]

    regime_counts = {r: regimes.count(r) for r in ["LOW", "MEDIUM", "HIGH", "EXTREME"]}
    dominant      = max(regime_counts, key=regime_counts.get)

    return {
        "symbol":            symbol,
        "date":              date,
        "bars_analysed":     len(estimates),
        "lambda_mean":       round(float(np.mean(lambdas)), 6),
        "lambda_median":     round(float(np.median(lambdas)), 6),
        "lambda_p75":        round(float(np.percentile(lambdas, 75)), 6),
        "lambda_p95":        round(float(np.percentile(lambdas, 95)), 6),
        "avg_toxicity":      round(float(np.mean(toxicity)), 4),
        "max_toxicity":      round(float(np.max(toxicity)), 4),
        "dominant_regime":   dominant,
        "regime_breakdown":  regime_counts,
        "avg_spread_multiplier": round(
            float(np.mean([e.spread_multiplier for e in estimates])), 4
        ),
        "recommendation":    _recommendation(dominant, float(np.mean(lambdas))),
        "per_bar": [
            {
                "ts":               str(e.timestamp),
                "lambda":           e.lambda_robust,
                "toxicity":         e.toxicity_score,
                "spread_multiplier":e.spread_multiplier,
                "regime":           e.regime,
                "r2":               e.r_squared,
            }
            for e in estimates
        ],
    }


def _recommendation(regime: str, lambda_mean: float) -> str:
    recs = {
        "LOW":     "GOOD for MM — low adverse selection, tighten spread",
        "MEDIUM":  "ACCEPTABLE — use base A-S spread",
        "HIGH":    "CAUTION — widen spread 1.5-2x, reduce size",
        "EXTREME": "AVOID — informed flow dominant, pause MM",
    }
    return recs.get(regime, "UNKNOWN")


def print_summary(results: list[dict]):
    print("\n" + "=" * 70)
    print("  KYLE'S LAMBDA — ADVERSE SELECTION REPORT")
    print("=" * 70)
    print(f"  {'Symbol':<30} {'λ median':>10} {'Toxicity':>10} {'Regime':<10} {'Recommendation'}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["avg_toxicity"]):
        print(
            f"  {r['symbol']:<30} "
            f"{r['lambda_median']:>10.6f} "
            f"{r['avg_toxicity']:>10.4f} "
            f"{r['dominant_regime']:<10} "
            f"{r['recommendation']}"
        )
    print("=" * 70)

    # Best MM candidates
    low = [r for r in results if r["dominant_regime"] == "LOW"]
    if low:
        print(f"\n  ✓ Best MM candidates ({len(low)} symbols):")
        for r in low[:10]:
            print(f"    {r['symbol']}")

    extreme = [r for r in results if r["dominant_regime"] == "EXTREME"]
    if extreme:
        print(f"\n  ✗ Avoid for MM ({len(extreme)} symbols — high toxicity):")
        for r in extreme[:10]:
            print(f"    {r['symbol']}")
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kyle's Lambda Adverse Selection Model")
    parser.add_argument("--symbol", type=str, help="Single symbol to analyse")
    parser.add_argument("--date",   type=str, required=True, help="Date YYYY-MM-DD")
    parser.add_argument("--all",    action="store_true", help="Analyse all symbols")
    parser.add_argument("--window", type=int, default=30, help="Rolling window in bars (default 30)")
    parser.add_argument("--no-upload", action="store_true", help="Skip GCS upload")
    args = parser.parse_args()

    if args.symbol:
        symbols = [args.symbol]
    elif args.all:
        symbols = list_symbols(args.date)
        log.info(f"Found {len(symbols)} futures symbols for {args.date}")
    else:
        parser.error("Provide --symbol SYMBOL or --all")

    results = []
    for sym in symbols:
        log.info(f"Analysing {sym}...")
        r = analyse_symbol(sym, args.date, window=args.window)
        if r:
            results.append(r)
        else:
            log.warning(f"  Skipped {sym} — insufficient data")

    if not results:
        log.error("No results — check data availability")
        return

    print_summary(results)

    if not args.no_upload:
        payload = {
            "date":        args.date,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "window_bars": args.window,
            "symbols":     results,
        }
        save_results_to_gcs(args.date, payload)


if __name__ == "__main__":
    main()
