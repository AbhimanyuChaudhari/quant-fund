"""
Hawkes Process — Ricci Chapter 3, Section 3.2.2
================================================
Self-exciting market order arrival dynamics.

From the paper (Assumption 3.2.1):
    dλ⁻t = β(θ - λ⁻t)dt + η dM⁺t + ν dM⁻t
    dλ⁺t = β(θ - λ⁺t)dt + η dM⁻t + ν dM⁺t

Where:
    λ⁻t = intensity of sell market orders
    λ⁺t = intensity of buy market orders
    M⁺t = influential buy MO arrivals (counting process)
    M⁻t = influential sell MO arrivals (counting process)
    β   = mean reversion rate (how fast λ decays back to θ)
    θ   = long-run average intensity
    η   = self-excitation jump (same side)
    ν   = cross-excitation jump (opposite side)
    ρ   = probability trade is influential

Stability condition (Lemma 3.2.2):
    β > ρ(η + ν)
    
    If violated: λ±t → ∞ (explosive) → stop trading
    Long-run mean: lim E[λ±t] = A⁻¹ζ

Key insight:
    - Influential buy MO → λ⁺t jumps by η (more buys coming)
                         → λ⁻t jumps by ν (some sells too — reaction)
    - Typically η > ν (same-side excitation > cross-side)
    - This captures trade clustering we proved in Notebook 01

Usage:
    hawkes = HawkesProcess(beta=1.0, theta=2.0, eta=0.5, nu=0.2, rho=0.3)
    state  = hawkes.update(bar, is_buy_influential=True, is_sell_influential=False)
    
    if not state.is_stable:
        # Stop trading — unstable market
        pass
    
    lambda_imbalance = state.lambda_buy - state.lambda_sell
"""

import math
import numpy as np
import collections
from dataclasses import dataclass
from typing import Optional


@dataclass
class HawkesState:
    lambda_buy:       float   # λ⁺t — buy MO intensity
    lambda_sell:      float   # λ⁻t — sell MO intensity
    lambda_imbalance: float   # λ⁺t - λ⁻t (key signal for quote adjustment)
    is_stable:        bool    # β > ρ(η + ν)
    branching_ratio:  float   # ρ(η + ν) / β (must be < 1)
    long_run_mean:    float   # θ (long-run level)
    burst_intensity:  float   # how far above baseline we are


class HawkesProcess:
    """
    Bivariate Hawkes process for buy/sell market order intensities.
    
    Implements Assumption 3.2.1 and Lemma 3.2.2 from Ricci (2014).

    Parameters (calibrate per symbol using calibrator.py):
        beta:  mean reversion rate — how fast bursts die out
               typical range: 0.5 - 5.0
               higher β = faster reversion to baseline
               
        theta: long-run intensity (baseline MO arrival rate per second)
               typical range: 0.5 - 5.0
               from your data: avg tick_count per bar ≈ 1.5
               
        eta:   self-excitation coefficient (same-side jump)
               typical range: 0.1 - 0.8
               η > ν always (same-side excitation is stronger)
               
        nu:    cross-excitation coefficient (opposite-side jump)
               typical range: 0.05 - 0.4
               
        rho:   probability each MO is influential (from MarketClassifier)
               typical range: 0.2 - 0.7
               
    Stability requirement (Lemma 3.2.2):
        β > ρ(η + ν)
        
        With defaults: 1.0 > 0.3 * (0.5 + 0.2) = 0.21 ✓
    """

    def __init__(self,
                 beta:  float = 1.0,
                 theta: float = 2.0,
                 eta:   float = 0.5,
                 nu:    float = 0.2,
                 rho:   float = 0.3,
                 dt:    float = 1.0):   # bar duration in seconds
        self.beta  = beta
        self.theta = theta
        self.eta   = eta
        self.nu    = nu
        self.rho   = rho
        self.dt    = dt

        # Current state
        self.lambda_buy  = theta   # start at long-run mean
        self.lambda_sell = theta

        # Track last update time for decay computation
        self._last_ts: Optional[float] = None

        # History for diagnostics
        self._buy_history  = collections.deque(maxlen=300)
        self._sell_history = collections.deque(maxlen=300)

    @property
    def is_stable(self) -> bool:
        """
        Lemma 3.2.2 stability condition: β > ρ(η + ν)
        If False: λ±t → ∞, market is in explosive regime → stop trading.
        """
        return self.beta > self.rho * (self.eta + self.nu)

    @property
    def branching_ratio(self) -> float:
        """
        ρ(η + ν) / β
        < 1.0: stable
        = 1.0: critical
        > 1.0: explosive
        """
        return self.rho * (self.eta + self.nu) / max(self.beta, 1e-8)

    @property
    def long_run_mean(self) -> float:
        """
        lim E[λ±t] = βθ / (β - ρ(η + ν))   [symmetric case, no news]
        From Lemma 3.2.2 with µ± = 0.
        """
        denom = self.beta - self.rho * (self.eta + self.nu)
        if denom <= 0:
            return float('inf')
        return self.beta * self.theta / denom

    def _decay(self, lambda_val: float, elapsed: float) -> float:
        """
        Continuous-time decay: λt = θ + (λ0 - θ) * e^(-β * elapsed)
        Exact solution to dλt = β(θ - λt)dt between jumps.
        """
        return self.theta + (lambda_val - self.theta) * math.exp(
            -self.beta * elapsed
        )

    def update(self,
               bar,
               is_buy_influential:  bool = False,
               is_sell_influential: bool = False) -> HawkesState:
        """
        Update λ±t given the current bar and whether influential MOs arrived.

        Steps:
        1. Decay toward θ (continuous mean reversion)
        2. Jump if influential MO detected (self + cross excitation)

        Args:
            bar:                  current bar (pd.Series or dict)
            is_buy_influential:   influential buy MO in this bar
            is_sell_influential:  influential sell MO in this bar

        Returns:
            HawkesState with updated λ±t and stability info
        """
        # ── Step 1: Decay toward long-run mean ────────────────────────────────
        elapsed = self.dt   # 1 second bar
        self.lambda_buy  = self._decay(self.lambda_buy,  elapsed)
        self.lambda_sell = self._decay(self.lambda_sell, elapsed)

        # ── Step 2: Jumps from influential orders ─────────────────────────────
        # Influential buy MO:
        #   λ⁺t += η (buy side excited — more buys expected)
        #   λ⁻t += ν (sell side also excited — cross effect)
        if is_buy_influential:
            self.lambda_buy  += self.eta   # same side
            self.lambda_sell += self.nu    # cross side

        # Influential sell MO:
        #   λ⁻t += η (sell side excited)
        #   λ⁺t += ν (buy side cross effect)
        if is_sell_influential:
            self.lambda_sell += self.eta   # same side
            self.lambda_buy  += self.nu    # cross side

        # ── Enforce non-negativity ────────────────────────────────────────────
        self.lambda_buy  = max(self.lambda_buy,  1e-6)
        self.lambda_sell = max(self.lambda_sell, 1e-6)

        # ── Track history ─────────────────────────────────────────────────────
        self._buy_history.append(self.lambda_buy)
        self._sell_history.append(self.lambda_sell)

        # ── Burst intensity ───────────────────────────────────────────────────
        # How far above baseline are we?
        baseline     = self.long_run_mean
        burst        = (self.lambda_buy + self.lambda_sell) / 2 / max(baseline, 1e-6)

        return HawkesState(
            lambda_buy       = round(self.lambda_buy, 6),
            lambda_sell      = round(self.lambda_sell, 6),
            lambda_imbalance = round(self.lambda_buy - self.lambda_sell, 6),
            is_stable        = self.is_stable,
            branching_ratio  = round(self.branching_ratio, 4),
            long_run_mean    = round(self.long_run_mean, 4),
            burst_intensity  = round(burst, 4),
        )

    def alpha_integral(self,
                       alpha_t: float,
                       T: float,
                       zeta: float,
                       epsilon: float) -> float:
        """
        Expected integral of alpha from Ricci Equation 3.19a (simplified):
        
        Et[∫t^T αu du] ≈ ε(λ⁺t - λ⁻t) * [
            ρ/(β̂) * (1 - e^(-β̂T)) / β̂  +
            (e^(-β̂T) - e^(-ζT)) / (β̂ - ζ) * (1 - e^(-ζT)) / ζ
        ]
        
        Where β̂ = β - ρ(η - ν) is the effective decay rate.
        
        This tells us: if buy activity > sell activity (λ⁺ > λ⁻),
        we expect positive price drift → adjust quotes accordingly.

        Args:
            alpha_t: current short-term alpha estimate
            T:       time remaining (fraction)
            zeta:    alpha mean reversion rate
            epsilon: average influential trade impact on alpha

        Returns:
            Expected integrated alpha (used to shift δ±)
        """
        beta_hat = self.beta - self.rho * (self.eta - self.nu)
        beta_hat = max(beta_hat, 1e-6)   # avoid division by zero

        lambda_imb = self.lambda_buy - self.lambda_sell

        # Term 1: from lambda imbalance
        if abs(beta_hat - zeta) > 1e-6:
            term1 = epsilon * lambda_imb * (
                self.rho / beta_hat * (1 - math.exp(-beta_hat * T)) / beta_hat +
                (math.exp(-beta_hat * T) - math.exp(-zeta * T)) / (beta_hat - zeta) *
                (1 - math.exp(-zeta * T)) / max(zeta, 1e-6)
            )
        else:
            # Degenerate case: β̂ ≈ ζ
            term1 = epsilon * lambda_imb * T * math.exp(-zeta * T)

        # Term 2: from current alpha (mean-reverting contribution)
        term2 = alpha_t / max(zeta, 1e-6) * (1 - math.exp(-zeta * T))

        return term1 + term2

    def lambda_adjustment(self, T: float, h: float, phi: float) -> float:
        """
        Inventory-management adjustment from lambda imbalance.
        Ricci Equation 3.19b:
        
        bφ = 2h(λ⁺t - λ⁻t) * [(T-t)/β̂ - (1 - e^(-β̂T)) / β̂²]
        
        Where h = h±(δ0±, κ) = e^(-1) for exponential fill rate.
        
        This adjusts spread based on market order flow imbalance.
        When buy orders dominate (λ⁺ > λ⁻): widen ask, tighten bid.

        Args:
            T:   time remaining
            h:   fill probability at risk-neutral depth (e^-1 ≈ 0.368)
            phi: inventory penalty parameter

        Returns:
            Lambda-based spread adjustment
        """
        beta_hat = max(self.beta - self.rho * (self.eta - self.nu), 1e-6)
        lambda_imb = self.lambda_buy - self.lambda_sell

        adj = 2 * h * lambda_imb * (
            T / beta_hat -
            (1 - math.exp(-beta_hat * T)) / (beta_hat ** 2)
        )
        return phi * adj

    def reset_day(self):
        """Reset to long-run mean at start of each trading day."""
        self.lambda_buy  = self.theta
        self.lambda_sell = self.theta
        self._last_ts    = None

    def status(self) -> dict:
        return {
            "lambda_buy":      round(self.lambda_buy, 4),
            "lambda_sell":     round(self.lambda_sell, 4),
            "lambda_imb":      round(self.lambda_buy - self.lambda_sell, 4),
            "is_stable":       self.is_stable,
            "branching_ratio": round(self.branching_ratio, 4),
            "long_run_mean":   round(self.long_run_mean, 4),
            "beta":  self.beta,
            "theta": self.theta,
            "eta":   self.eta,
            "nu":    self.nu,
            "rho":   self.rho,
        }