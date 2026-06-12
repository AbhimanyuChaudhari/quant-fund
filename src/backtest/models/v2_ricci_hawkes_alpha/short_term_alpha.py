"""
Short-Term Alpha — Ricci Chapter 3, Section 3.4
================================================
Stochastic drift component of the midprice.

From the paper (Assumption 3.4.1):
    dαt = -ζ αt dt + σα dBt + ε⁺ dM⁺t - ε⁻ dM⁻t

Where:
    αt  = predictable short-term drift (zero-mean reverting)
    ζ   = mean reversion rate of alpha
    σα  = diffusion of alpha
    Bt  = Brownian motion (independent of market orders)
    ε⁺  = impact of influential buy MO on drift (positive jump)
    ε⁻  = impact of influential sell MO on drift (negative jump)
    M⁺t = influential buy MO arrivals
    M⁻t = influential sell MO arrivals

Intuition:
    When an influential buy MO arrives:
        αt jumps UP → price expected to continue rising
        → widen ask (avoid being picked off)
        → tighten bid (capture the upward move)
    
    When an influential sell MO arrives:
        αt jumps DOWN → price expected to continue falling
        → widen bid (avoid being picked off)
        → tighten ask (capture the downward move)

    Alpha mean reverts at rate ζ — the effect is temporary.
    This is the key adverse selection protection mechanism.

Proxy Implementation (Phase 1 — no calibration needed):
    We estimate αt from existing features:
        alpha_t ≈ price_mom_60s     (primary — 60s momentum)
                + f(volume_delta)   (flow direction confirms α)
                + f(imbalance_last) (LOB leading indicator)

Usage:
    alpha = ShortTermAlpha(zeta=0.5, sigma_alpha=0.001,
                            epsilon_plus=0.002, epsilon_minus=0.002)
    alpha_t = alpha.update(bar, is_buy_influential=True)
    
    # Use alpha_t to adjust quotes:
    delta_ask += B * alpha_t / zeta * (1 - exp(-zeta * T))
    delta_bid -= B * alpha_t / zeta * (1 - exp(-zeta * T))
"""

import math
import collections
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class AlphaState:
    alpha:          float   # current αt
    alpha_integral: float   # Et[∫t^T αu du] — used directly in quote formula
    direction:      str     # 'up', 'down', or 'neutral'
    strength:       float   # |αt| normalized to [0,1]
    source:         str     # 'momentum', 'flow', 'combined'


class ShortTermAlpha:
    """
    Estimates and tracks short-term alpha (αt) — the predictable drift.
    
    Phase 1 (current): Uses existing processed features as proxies.
    Phase 2 (after 30 days): Replace with calibrated OU process + MLE fit.

    Parameters:
        zeta:          mean reversion rate of alpha (higher = faster decay)
                       typical: 0.3 - 2.0 per second
                       
        sigma_alpha:   diffusion coefficient (controls noise level)
                       typical: 0.001 - 0.01 in price point space
                       
        epsilon_plus:  jump size on influential buy MO
        epsilon_minus: jump size on influential sell MO
                       typical: 0.001 - 0.005 price points
                       
        mom_weight:    weight on price_mom_60s in proxy estimate
        flow_weight:   weight on volume_delta in proxy estimate
        imb_weight:    weight on imbalance_last in proxy estimate
    """

    def __init__(self,
                 zeta:          float = 0.5,
                 sigma_alpha:   float = 0.002,
                 epsilon_plus:  float = 0.002,
                 epsilon_minus: float = 0.002,
                 mom_weight:    float = 0.60,
                 flow_weight:   float = 0.25,
                 imb_weight:    float = 0.15,
                 window:        int   = 60):
        self.zeta          = zeta
        self.sigma_alpha   = sigma_alpha
        self.epsilon_plus  = epsilon_plus
        self.epsilon_minus = epsilon_minus
        self.mom_weight    = mom_weight
        self.flow_weight   = flow_weight
        self.imb_weight    = imb_weight
        self.window        = window

        # Current alpha state
        self.alpha = 0.0

        # Buffers for normalization
        self._mom_buf    = collections.deque(maxlen=window)
        self._flow_buf   = collections.deque(maxlen=window)
        self._imb_buf    = collections.deque(maxlen=window)
        self._alpha_buf  = collections.deque(maxlen=window)

    def _safe(self, val, default: float = 0.0) -> float:
        if val is None:
            return default
        try:
            f = float(val)
            return default if math.isnan(f) or math.isinf(f) else f
        except (TypeError, ValueError):
            return default

    def _normalize(self, buf: collections.deque, val: float) -> float:
        """Normalize val to [-1, +1] using buffer min/max."""
        if len(buf) < 5:
            return 0.0
        arr  = np.array(buf)
        rng  = arr.max() - arr.min()
        if rng < 1e-10:
            return 0.0
        return 2 * (val - arr.min()) / rng - 1.0

    def update(self,
               bar,
               is_buy_influential:  bool = False,
               is_sell_influential: bool = False,
               dt: float = 1.0) -> AlphaState:
        """
        Update αt given current bar and influential MO arrivals.
        
        Process:
        1. Decay: αt → αt * e^(-ζ dt)  [OU mean reversion to 0]
        2. Jump:  αt += ε+ if buy influential, αt -= ε- if sell influential
        3. Proxy: blend with feature-based estimate for stability

        Args:
            bar:                 current processed bar
            is_buy_influential:  influential buy MO in this bar
            is_sell_influential: influential sell MO in this bar
            dt:                  bar duration (1 second)

        Returns:
            AlphaState with current alpha and pre-computed integral
        """
        # ── Extract features ───────────────────────────────────────────────────
        mom_60s    = self._safe(bar.get("price_mom_60s", 0))
        vol_delta  = self._safe(bar.get("volume_delta", 0))
        imb_last   = self._safe(bar.get("imbalance_last", 0))
        tick_count = max(self._safe(bar.get("tick_count", 1), 1.0), 1.0)

        # ── Update normalization buffers ───────────────────────────────────────
        self._mom_buf.append(mom_60s)
        self._flow_buf.append(vol_delta / tick_count)  # per-tick flow
        self._imb_buf.append(imb_last)

        # ── Feature-based alpha proxy ──────────────────────────────────────────
        mom_norm  = self._normalize(self._mom_buf,  mom_60s)
        flow_norm = self._normalize(self._flow_buf, vol_delta / tick_count)
        imb_norm  = self._normalize(self._imb_buf,  imb_last)

        # Composite proxy (scale to price points)
        alpha_proxy = (
            self.mom_weight  * mom_norm  +
            self.flow_weight * flow_norm +
            self.imb_weight  * imb_norm
        ) * self.sigma_alpha * 10   # scale to price point space

        # ── OU mean reversion ──────────────────────────────────────────────────
        # Exact solution: α(t+dt) = α(t) * e^(-ζ dt)
        self.alpha *= math.exp(-self.zeta * dt)

        # ── Jumps from influential orders ──────────────────────────────────────
        # Influential buy → αt jumps up (expect price rise)
        if is_buy_influential:
            self.alpha += self.epsilon_plus

        # Influential sell → αt jumps down (expect price fall)
        if is_sell_influential:
            self.alpha -= self.epsilon_minus

        # ── Blend OU model with proxy (Phase 1 approach) ───────────────────────
        # Weight: 0.7 model + 0.3 proxy
        # This provides stability before calibration
        blend_weight = 0.3
        self.alpha = (1 - blend_weight) * self.alpha + blend_weight * alpha_proxy

        # ── Track history ──────────────────────────────────────────────────────
        self._alpha_buf.append(self.alpha)

        # ── Compute alpha integral Et[∫t^T αu du] ─────────────────────────────
        # From Ricci Eq 3.15a simplified:
        # bα(t) = (1/ζ)(1 - e^(-ζT))
        # So integral ≈ αt * bα(t) = αt/ζ * (1 - e^(-ζT))
        # This is the key quantity that adjusts quote depths
        T_remaining = 0.5   # will be overridden by strategy — default mid-day
        alpha_integral = self._compute_integral(self.alpha, T_remaining)

        # ── Classify direction ─────────────────────────────────────────────────
        if len(self._alpha_buf) > 5:
            alpha_std = np.std(list(self._alpha_buf))
            threshold = max(alpha_std * 0.5, 1e-6)
        else:
            threshold = self.sigma_alpha

        if self.alpha > threshold:
            direction = 'up'
        elif self.alpha < -threshold:
            direction = 'down'
        else:
            direction = 'neutral'

        # Strength: normalized |α| in [0,1]
        max_alpha = max(abs(a) for a in self._alpha_buf) if self._alpha_buf else 1e-6
        strength  = min(abs(self.alpha) / max(max_alpha, 1e-8), 1.0)

        return AlphaState(
            alpha          = round(self.alpha, 6),
            alpha_integral = round(alpha_integral, 6),
            direction      = direction,
            strength       = round(strength, 4),
            source         = 'combined',
        )

    def _compute_integral(self, alpha: float, T: float) -> float:
        """
        Et[∫t^T αu du] = αt/ζ * (1 - e^(-ζT))
        
        From Ricci Theorem 3.5.2, Equation 3.15a.
        This is the quantity that appears in the optimal spread formula.
        """
        if self.zeta < 1e-8:
            return alpha * T
        return alpha / self.zeta * (1 - math.exp(-self.zeta * T))

    def compute_integral(self, T: float) -> float:
        """Public method for strategy to call with current T."""
        return self._compute_integral(self.alpha, T)

    def quote_adjustments(self, T: float, B: float) -> tuple[float, float]:
        """
        Compute δ± adjustments from alpha.
        
        From Ricci Equation 3.16:
            δ⁺* += B * (-alpha_integral)   ← widen ask if α > 0 (price rising)
            δ⁻* += B * (+alpha_integral)   ← tighten bid if α > 0

        Args:
            T: time remaining (fraction)
            B: coefficient B(δ0±; κ) from Proposition 3.5.1

        Returns:
            (delta_plus_adj, delta_minus_adj) — add to base spread
        """
        integral = self.compute_integral(T)
        delta_plus_adj  = B * (-integral)   # widen ask when α > 0
        delta_minus_adj = B * (+integral)   # tighten bid when α > 0
        return delta_plus_adj, delta_minus_adj

    def reset_day(self):
        """Reset alpha to zero at start of each trading day."""
        self.alpha = 0.0
        self._alpha_buf.clear()

    def status(self) -> dict:
        return {
            "alpha":    round(self.alpha, 6),
            "zeta":     self.zeta,
            "epsilon+": self.epsilon_plus,
            "epsilon-": self.epsilon_minus,
            "direction": 'up' if self.alpha > 0 else 'down' if self.alpha < 0 else 'neutral',
        }
