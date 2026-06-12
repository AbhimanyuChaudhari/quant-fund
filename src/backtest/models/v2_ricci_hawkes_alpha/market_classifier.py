"""
Market Classifier — Ricci Chapter 3, Section 3.2.2
====================================================
Detects whether each arriving market order is "influential" (probability ρ).

From the paper:
    Market orders arrive in two types:
    1. Influential orders — excite the market, cause λ±t to jump
    2. Non-influential orders — no effect on arrival rates

    "The type indicator of an order is not observable. Rather, all one
     can observe is whether the market became more active after that trade."
     — Ricci (2014), p.27

Since we can't observe type directly, we estimate P(influential | observations)
from features already in our processed parquet files:
    - price_impact    → large = informed/influential MO
    - volume_delta    → signed flow imbalance
    - tick_count      → burst in activity
    - spread_zscore   → spread widening after MO (market reacting)
    - volume_ratio    → volume spike vs recent average

Calibration:
    ρ (base probability) is fit per symbol from historical data.
    Default ρ = 0.3 (Ricci uses ρ = 0.7 for benchmark HFT in simulations).

Usage:
    clf = MarketClassifier(rho=0.3, window=60)
    result = clf.classify(bar)
    if result.is_influential:
        # update Hawkes, alpha, kappa
"""

import math
import collections
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ClassifierResult:
    is_influential:  bool    # binary classification
    prob_influential: float  # P(influential | bar)
    price_impact_z:  float  # normalized price impact
    volume_spike:    float  # volume vs recent average
    flow_imbalance:  float  # signed order flow
    tick_burst:      float  # tick count vs recent average


class MarketClassifier:
    """
    Estimates whether each 1-second bar contains an influential market order.

    Strategy:
        We compute a composite "influence score" from multiple signals.
        Score > threshold → influential.

        This approximates the latent ρ parameter in Ricci's model.

    Parameters:
        rho:       base probability of influential order (calibrated per symbol)
        window:    lookback bars for computing z-scores (default 60 = 1 min)
        threshold: influence score threshold (default 0.5)
    """

    def __init__(self,
                 rho:       float = 0.30,
                 window:    int   = 60,
                 threshold: float = 0.50):
        self.rho       = rho
        self.window    = window
        self.threshold = threshold

        # Rolling buffers for z-score computation
        self._impact_buf  = collections.deque(maxlen=window)
        self._volume_buf  = collections.deque(maxlen=window)
        self._tick_buf    = collections.deque(maxlen=window)
        self._spread_buf  = collections.deque(maxlen=window)

        # Stats
        self.n_classified    = 0
        self.n_influential   = 0

    def _safe(self, val, default: float = 0.0) -> float:
        """Safe float extraction."""
        if val is None:
            return default
        try:
            f = float(val)
            return default if math.isnan(f) or math.isinf(f) else f
        except (TypeError, ValueError):
            return default

    def _zscore(self, buf: collections.deque, val: float) -> float:
        """Z-score of val relative to buffer history."""
        if len(buf) < 5:
            return 0.0
        arr  = np.array(buf)
        mean = arr.mean()
        std  = arr.std()
        if std < 1e-8:
            return 0.0
        return (val - mean) / std

    def classify(self, bar) -> ClassifierResult:
        """
        Classify whether this bar contains an influential market order.

        Args:
            bar: pd.Series or dict with processed feature columns

        Returns:
            ClassifierResult with classification and component scores
        """
        # ── Extract features ───────────────────────────────────────────────────
        price_impact  = abs(self._safe(bar.get("price_impact", 0)))
        volume_ratio  = self._safe(bar.get("volume_ratio", 1.0), 1.0)
        tick_count    = self._safe(bar.get("tick_count", 1), 1.0)
        spread_z      = self._safe(bar.get("spread_zscore", 0))
        vol_delta     = abs(self._safe(bar.get("volume_delta", 0)))
        imb_last      = abs(self._safe(bar.get("imbalance_last", 0)))

        # ── Update buffers ─────────────────────────────────────────────────────
        self._impact_buf.append(price_impact)
        self._volume_buf.append(volume_ratio)
        self._tick_buf.append(tick_count)
        self._spread_buf.append(spread_z)

        # ── Compute z-scores ───────────────────────────────────────────────────
        impact_z  = self._zscore(self._impact_buf, price_impact)
        vol_spike = self._zscore(self._volume_buf, volume_ratio)
        tick_z    = self._zscore(self._tick_buf,   tick_count)

        # ── Composite influence score (weighted sum, normalized to [0,1]) ──────
        # Weights from Ricci's intuition:
        #   price impact is strongest signal of informed/influential trade
        #   volume spike confirms large MO
        #   tick burst = self-excitation starting
        #   spread widening = market reacting to influential order
        #   flow imbalance = one-sided pressure
        score = (
            0.35 * min(max(impact_z  / 3.0, 0), 1.0) +   # price impact z
            0.25 * min(max(vol_spike / 2.0, 0), 1.0) +   # volume spike
            0.20 * min(max(tick_z    / 2.0, 0), 1.0) +   # tick burst
            0.10 * min(max(spread_z  / 2.0, 0), 1.0) +   # spread widening
            0.10 * min(imb_last / 0.5, 1.0)               # flow imbalance
        )

        # ── Scale by base rate ρ ───────────────────────────────────────────────
        # P(influential) = ρ when score=0.5, higher/lower otherwise
        # Sigmoid-like scaling around ρ
        prob = self.rho * (1 + (score - 0.5) * 2)
        prob = min(max(prob, 0.0), 1.0)

        is_influential = score >= self.threshold

        self.n_classified += 1
        if is_influential:
            self.n_influential += 1

        return ClassifierResult(
            is_influential   = is_influential,
            prob_influential = round(prob, 4),
            price_impact_z   = round(impact_z, 4),
            volume_spike     = round(vol_spike, 4),
            flow_imbalance   = round(imb_last, 4),
            tick_burst       = round(tick_z, 4),
        )

    @property
    def empirical_rho(self) -> float:
        """Observed proportion of influential bars so far."""
        if self.n_classified == 0:
            return self.rho
        return self.n_influential / self.n_classified

    def reset_day(self):
        """Reset daily counters but keep buffers (they span days)."""
        self.n_classified  = 0
        self.n_influential = 0

    def status(self) -> dict:
        return {
            "rho_prior":    self.rho,
            "rho_empirical": self.empirical_rho,
            "n_classified": self.n_classified,
            "n_influential": self.n_influential,
            "threshold":    self.threshold,
        }
