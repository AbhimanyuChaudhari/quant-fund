"""
Stochastic Fill Rate — Ricci Chapter 3, Section 3.3
====================================================
Replaces constant kappa assumption with stochastic κ±t.

From the paper (Assumption 3.3.1 and 3.3.2):
    Fill rate:  Λ±t = λ±t · h±(δ; κt)
    
    Exponential class (Example 3.5.5):
        h±(δ; κt) = e^(-κ±t · δ)
    
    κt dynamics (Assumption 3.3.2):
        dκ⁻t = βκ(θκ - κ⁻t)dt + ηκ dM⁺t + νκ dM⁻t
        dκ⁺t = βκ(θκ - κ⁺t)dt + νκ dM⁺t + ηκ dM⁻t

Where:
    κ±t = fill rate decay parameter (higher = fewer fills far from mid)
    βκ  = mean reversion rate of κ (typically slower than λ reversion)
    θκ  = long-run average κ (same as your current kappa=1.5)
    ηκ  = self-excitation of κ (same side)
    νκ  = cross-excitation of κ

Intuition:
    After an influential buy MO arrives:
        κ⁺t jumps (sell side book thins — harder to fill sell limits)
        κ⁻t also jumps slightly (book adjusts on both sides)
    
    Both κ values mean revert to θκ at rate βκ.
    This means fill probability is LOWER right after a large MO.
    
    "Since many market participants react in similar way, the probability
     of limit orders being filled, conditional on a market order arriving,
     decreases." — Ricci (2014), p.32

Key formula (optimal risk-neutral depth from Equation 3.13):
    δ0± satisfies: δ0± h'±(δ0±; κ) + h±(δ0±; κ) = 0
    
    For exponential h± = e^(-κδ):
        δ0± = 1/κ±t   ← optimal depth is reciprocal of κ

B coefficient (from Proposition 3.5.1):
    B(δ0±; κ) = h0(δ0±; κ) / (2h0(δ0±; κ) + δ0± h0''(δ0±; κ))
    
    For exponential: B = e^(-1) / (2e^(-1) + e^(-1)) = 1/3... simplifies
    → B = 1 / (2κ±t)   ← this is what multiplies alpha and phi terms

Usage:
    fill_rate = FillRate(beta_kappa=0.5, theta_kappa=1.5,
                          eta_kappa=0.3, nu_kappa=0.1)
    state = fill_rate.update(bar, is_buy_influential=True)
    
    delta0      = state.optimal_depth_buy    # 1/κ⁻t
    B_coeff     = state.B_coefficient_buy    # 1/(2κ⁻t)
    fill_prob   = state.fill_probability(delta=delta0, side='buy')
"""

import math
import collections
import numpy as np
from dataclasses import dataclass


@dataclass
class FillRateState:
    kappa_buy:          float   # κ⁺t — buy-side fill rate shape
    kappa_sell:         float   # κ⁻t — sell-side fill rate shape
    optimal_depth_buy:  float   # δ0⁻ = 1/κ⁻t (optimal bid depth)
    optimal_depth_sell: float   # δ0⁺ = 1/κ⁺t (optimal ask depth)
    B_coeff_buy:        float   # B coefficient for bid side
    B_coeff_sell:       float   # B coefficient for ask side
    fill_prob_at_opt:   float   # h±(δ0±) = e^(-1) ≈ 0.368 always


class FillRate:
    """
    Stochastic fill rate dynamics (κ±t process).
    
    Implements Assumptions 3.3.1 and 3.3.2 from Ricci (2014).
    Uses exponential fill rate class (Example 3.5.5).

    Key result from Example 3.5.5:
        h±(δ0±; κ) = e^(-1) ≈ 0.368 (constant, independent of κt!)
        
        This means the FILL PROBABILITY at optimal depth is always ~36.8%
        regardless of κt. But the OPTIMAL DEPTH δ0± = 1/κt changes.
        
        When κt is high (book is thin): post closer to mid (smaller δ)
        When κt is low (book is deep):  post further from mid (larger δ)

    Parameters:
        beta_kappa:  mean reversion rate for κ (typically slower than β for λ)
                     typical: 0.2 - 1.0
                     
        theta_kappa: long-run average κ (your current kappa=1.5 calibrated value)
                     typical: 1.0 - 3.0 per price point
                     
        eta_kappa:   same-side jump in κ on influential MO
                     typical: 0.1 - 0.5
                     
        nu_kappa:    cross-side jump in κ on influential MO
                     typical: 0.05 - 0.2
    """

    def __init__(self,
                 beta_kappa:  float = 0.5,
                 theta_kappa: float = 1.5,
                 eta_kappa:   float = 0.3,
                 nu_kappa:    float = 0.1):
        self.beta_kappa  = beta_kappa
        self.theta_kappa = theta_kappa
        self.eta_kappa   = eta_kappa
        self.nu_kappa    = nu_kappa

        # Current state
        self.kappa_buy  = theta_kappa
        self.kappa_sell = theta_kappa

        # History
        self._kappa_buy_hist  = collections.deque(maxlen=300)
        self._kappa_sell_hist = collections.deque(maxlen=300)

    def _decay(self, kappa: float, dt: float = 1.0) -> float:
        """
        Mean reversion: κt = θκ + (κ0 - θκ) * e^(-βκ dt)
        """
        return self.theta_kappa + (kappa - self.theta_kappa) * math.exp(
            -self.beta_kappa * dt
        )

    def fill_probability(self, delta: float, kappa: float) -> float:
        """
        h±(δ; κ) = e^(-κ · δ)   [exponential class]
        
        Args:
            delta: posting depth (price distance from mid)
            kappa: current κ±t
        """
        return math.exp(-kappa * delta)

    def optimal_depth(self, kappa: float) -> float:
        """
        δ0± = 1/κ±t  [from solving δ h'(δ;κ) + h(δ;κ) = 0 for exponential]
        
        This is the risk-neutral optimal posting depth.
        Closer to mid when book is thin (κ high).
        """
        return 1.0 / max(kappa, 1e-6)

    def B_coefficient(self, kappa: float) -> float:
        """
        B(δ0±; κ) = h0 / (2h0 + δ0 h0'')
        
        For exponential h = e^(-κδ):
            h0   = e^(-κ · 1/κ) = e^(-1)
            h0'  = -κ · e^(-κδ)|δ=1/κ = -κ · e^(-1)
            h0'' = κ² · e^(-κδ)|δ=1/κ = κ² · e^(-1)
            δ0   = 1/κ
            
        B = e^(-1) / (2·e^(-1) + (1/κ)·κ²·e^(-1))
          = 1 / (2 + κ)   ← simplified
          
        But using the full Proposition 3.5.1 coefficient:
        B = h0 / (2h0 + δ0 h0'')
          = e^(-1) / (2e^(-1) + (1/κ)·κ²·e^(-1))
          = 1 / (2 + κ)
        """
        return 1.0 / (2.0 + kappa)

    def update(self,
               bar,
               is_buy_influential:  bool = False,
               is_sell_influential: bool = False,
               dt: float = 1.0) -> FillRateState:
        """
        Update κ±t given current bar.

        Steps:
        1. Decay κ toward θκ (LOB recovers from recent MOs)
        2. Jump if influential MO detected (book temporarily thins)

        Args:
            bar:                current bar (pd.Series or dict)
            is_buy_influential: influential buy MO → sell-side book thins
            is_sell_influential: influential sell MO → buy-side book thins
            dt:                 bar duration in seconds

        Returns:
            FillRateState with current κ±t and derived quantities
        """
        # ── Step 1: Decay toward θκ ────────────────────────────────────────────
        self.kappa_buy  = self._decay(self.kappa_buy,  dt)
        self.kappa_sell = self._decay(self.kappa_sell, dt)

        # ── Step 2: Adjust from actual LOB depth if available ─────────────────
        # Use bid_q1/ask_q1 as a proxy for κ adjustment
        # Thinner book (lower q1) → higher κ (fill probability decays faster)
        try:
            bid_q1 = float(bar.get("bid_q1", 0) or 0)
            ask_q1 = float(bar.get("ask_q1", 0) or 0)
            lot_size = float(bar.get("lot_size", 75) or 75)

            if bid_q1 > 0 and ask_q1 > 0:
                # Normalize queue to lots
                bid_lots = bid_q1 / lot_size
                ask_lots = ask_q1 / lot_size

                # Thin book = higher κ, deep book = lower κ
                # Calibrated so κ = θκ when queue ≈ 2 lots
                ref_lots = 2.0
                kappa_from_book_buy  = self.theta_kappa * (ref_lots / max(bid_lots, 0.1))
                kappa_from_book_sell = self.theta_kappa * (ref_lots / max(ask_lots, 0.1))

                # Blend: 70% model, 30% LOB observation
                self.kappa_buy  = 0.7 * self.kappa_buy  + 0.3 * kappa_from_book_buy
                self.kappa_sell = 0.7 * self.kappa_sell + 0.3 * kappa_from_book_sell
        except Exception:
            pass

        # ── Step 3: Jumps from influential orders ──────────────────────────────
        # Influential buy MO eats sell-side book → κ⁺t jumps (sell side thins)
        if is_buy_influential:
            self.kappa_sell += self.eta_kappa   # sell side thins
            self.kappa_buy  += self.nu_kappa    # buy side slight effect

        # Influential sell MO eats buy-side book → κ⁻t jumps
        if is_sell_influential:
            self.kappa_buy  += self.eta_kappa   # buy side thins
            self.kappa_sell += self.nu_kappa    # sell side slight effect

        # ── Clip to reasonable range ───────────────────────────────────────────
        self.kappa_buy  = max(0.1, min(self.kappa_buy,  10.0))
        self.kappa_sell = max(0.1, min(self.kappa_sell, 10.0))

        # ── Track history ──────────────────────────────────────────────────────
        self._kappa_buy_hist.append(self.kappa_buy)
        self._kappa_sell_hist.append(self.kappa_sell)

        # ── Compute derived quantities ─────────────────────────────────────────
        d0_buy  = self.optimal_depth(self.kappa_buy)    # 1/κ⁻t
        d0_sell = self.optimal_depth(self.kappa_sell)   # 1/κ⁺t
        B_buy   = self.B_coefficient(self.kappa_buy)
        B_sell  = self.B_coefficient(self.kappa_sell)

        return FillRateState(
            kappa_buy          = round(self.kappa_buy, 4),
            kappa_sell         = round(self.kappa_sell, 4),
            optimal_depth_buy  = round(d0_buy, 4),
            optimal_depth_sell = round(d0_sell, 4),
            B_coeff_buy        = round(B_buy, 4),
            B_coeff_sell       = round(B_sell, 4),
            fill_prob_at_opt   = round(math.exp(-1), 4),   # always e^(-1) ≈ 0.368
        )

    def reset_day(self):
        """Reset κ to long-run mean at start of each trading day."""
        self.kappa_buy  = self.theta_kappa
        self.kappa_sell = self.theta_kappa

    def status(self) -> dict:
        return {
            "kappa_buy":    round(self.kappa_buy, 4),
            "kappa_sell":   round(self.kappa_sell, 4),
            "delta0_buy":   round(self.optimal_depth(self.kappa_buy), 4),
            "delta0_sell":  round(self.optimal_depth(self.kappa_sell), 4),
            "theta_kappa":  self.theta_kappa,
            "beta_kappa":   self.beta_kappa,
        }