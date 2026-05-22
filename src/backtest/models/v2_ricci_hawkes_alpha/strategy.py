"""
Ricci Hawkes-Alpha Market Making Strategy
==========================================
Implementation of Chapter 3: "Market Making in a Single Asset"
Ricci, J. (2014). Applied Stochastic Control in High Frequency Trading.
PhD Thesis, University of Toronto.

Key improvements over v1 (Avellaneda-Stoikov):

1. Self-exciting order arrivals (Section 3.2.2)
   λ±t follows bivariate Hawkes process instead of constant Poisson rates.
   Each influential MO causes λ to jump then mean revert.
   Stability check: β > ρ(η+ν) (Lemma 3.2.2) — stop trading if violated.

2. Short-term alpha (Section 3.4)
   αt = predictable drift component (mean-reverting OU process).
   Jumps on influential MO arrival → adjusts quotes asymmetrically.
   Positive α → widen ask, tighten bid (price expected to rise).
   Negative α → widen bid, tighten ask (price expected to fall).

3. Stochastic fill rate κ±t (Section 3.3)
   κt jumps when influential MO arrives (book thins) then recovers.
   Optimal depth δ0± = 1/κ±t adapts to current LOB state.
   B coefficient B = 1/(2+κ) scales alpha and phi adjustments.

4. Optimal quotes (Corollary 3.5.3):
   δ±* = δ0± + B * {∓ Et[∫αu du] + φ * bφ}

   Where:
   - δ0± = 1/κ±t                        (risk-neutral base)
   - Et[∫αu du] ≈ αt/ζ*(1-e^(-ζT))     (alpha integral)
   - bφ = lambda imbalance adjustment    (inventory + flow)

Parameters (calibrate with calibrator.py after 30 days data):
    Hawkes: beta=1.0, theta=2.0, eta=0.5, nu=0.2, rho=0.3
    Alpha:  zeta=0.5, epsilon_plus=0.002, epsilon_minus=0.002
    Fill:   beta_kappa=0.5, theta_kappa=1.5
    Risk:   phi=0.001, max_inventory=5

Compatibility:
    Same interface as v1 AvellanedaStoikovStrategy.
    Drop-in replacement in backtest engine.
"""

import math
import collections
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from src.backtest.models.v1_avellaneda_stoikov.strategy import BaseStrategy, StrategyConfig
from src.backtest.order_book import SimulatedOrderBook, Side, Fill

# V2 components
from src.backtest.models.v2_ricci_hawkes_alpha.market_classifier import (
    MarketClassifier, ClassifierResult
)
from src.backtest.models.v2_ricci_hawkes_alpha.hawkes_process import (
    HawkesProcess, HawkesState
)
from src.backtest.models.v2_ricci_hawkes_alpha.short_term_alpha import (
    ShortTermAlpha, AlphaState
)
from src.backtest.models.v2_ricci_hawkes_alpha.fill_rate import (
    FillRate, FillRateState
)


@dataclass
class V2Params:
    """
    All parameters for the Ricci v2 model.
    Calibrate per symbol using calibrator.py after 30 days data.
    Defaults are reasonable starting points from Ricci's simulations.
    """
    # ── Hawkes process (Section 3.2.2) ─────────────────────────────────────────
    beta:  float = 1.0    # mean reversion rate of λ±t
    theta: float = 2.0    # long-run intensity (MOs per second)
    eta:   float = 0.5    # self-excitation (same-side jump)
    nu:    float = 0.2    # cross-excitation (opposite-side jump)
    rho:   float = 0.30   # P(influential | MO arrives)

    # ── Short-term alpha (Section 3.4) ─────────────────────────────────────────
    zeta:          float = 0.5    # alpha mean reversion rate
    epsilon_plus:  float = 0.002  # alpha jump on influential buy
    epsilon_minus: float = 0.002  # alpha jump on influential sell

    # ── Stochastic fill rate κt (Section 3.3) ─────────────────────────────────
    beta_kappa:  float = 0.5    # mean reversion rate of κ±t
    theta_kappa: float = 1.5    # long-run κ (same as v1 kappa)
    eta_kappa:   float = 0.3    # κ jump (same-side)
    nu_kappa:    float = 0.1    # κ jump (cross-side)

    # ── Risk and inventory ─────────────────────────────────────────────────────
    phi:        float = 0.001   # inventory penalty parameter
    min_spread: float = 0.10    # minimum spread floor
    max_spread: float = 10.0    # maximum spread cap
    open_mult:  float = 2.0     # open period spread multiplier

    # ── Market classifier ──────────────────────────────────────────────────────
    classifier_window:    int   = 60    # lookback for z-scores
    classifier_threshold: float = 0.50  # influence score threshold


class RicciHawkesAlphaStrategy(BaseStrategy):
    """
    Chapter 3 Ricci market making strategy.

    Optimal quotes (Corollary 3.5.3):
        δ⁺* = δ0⁺ + B * (-α_integral + φ * bφ)   [ask depth]
        δ⁻* = δ0⁻ + B * (+α_integral + φ * bφ)   [bid depth]

    Quotes:
        bid = mid - δ⁻*
        ask = mid + δ⁺*
    """

    # ── Session constants ──────────────────────────────────────────────────────
    SESSION_SECONDS   = 22500.0
    SESSION_START_UTC = 13500
    OPEN_START_IST    = 33300    # 09:15 IST
    OPEN_END_IST      = 35100    # 09:45 IST
    FLATTEN_IST       = 54720    # 15:12 IST
    MARKET_CLOSE_IST  = 55800    # 15:30 IST

    def __init__(self,
                 config: StrategyConfig,
                 params: Optional[V2Params] = None):
        super().__init__(config)

        self.params = params or V2Params()
        p = self.params

        # ── Initialize V2 components ───────────────────────────────────────────
        self.classifier = MarketClassifier(
            rho       = p.rho,
            window    = p.classifier_window,
            threshold = p.classifier_threshold,
        )
        self.hawkes = HawkesProcess(
            beta  = p.beta,
            theta = p.theta,
            eta   = p.eta,
            nu    = p.nu,
            rho   = p.rho,
        )
        self.alpha_model = ShortTermAlpha(
            zeta          = p.zeta,
            epsilon_plus  = p.epsilon_plus,
            epsilon_minus = p.epsilon_minus,
        )
        self.fill_rate = FillRate(
            beta_kappa  = p.beta_kappa,
            theta_kappa = p.theta_kappa,
            eta_kappa   = p.eta_kappa,
            nu_kappa    = p.nu_kappa,
        )

        # ── Order tracking ─────────────────────────────────────────────────────
        self.bid_id: Optional[int] = None
        self.ask_id: Optional[int] = None

        # ── Rolling vol for sigma (kept from v1 — still needed) ────────────────
        self._vol_buffer = collections.deque(maxlen=300)
        self.default_vol = 2.0

        # ── Current timestamp — used inside _compute_quotes ────────────────────
        # Fix: store ts_sec here so _compute_quotes doesn't need bar reference
        self._current_ts: int = 0

        # ── Diagnostics ────────────────────────────────────────────────────────
        self.n_stable_halts = 0
        self.n_influential  = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Time helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _ist_tod(self, ts_sec: int) -> int:
        return (ts_sec + 19800) % 86400

    def _time_remaining(self, ts_sec: int) -> float:
        secs_since_midnight = ts_sec % 86400
        elapsed   = max(0.0, secs_since_midnight - self.SESSION_START_UTC)
        remaining = max(60.0, self.SESSION_SECONDS - elapsed)
        return remaining / self.SESSION_SECONDS

    def _is_open_period(self, ts_sec: int) -> bool:
        ist = self._ist_tod(ts_sec)
        return self.OPEN_START_IST <= ist <= self.OPEN_END_IST

    def _should_flatten(self, ts_sec: int, inventory: int) -> bool:
        ist = self._ist_tod(ts_sec)
        return ist >= self.FLATTEN_IST and abs(inventory) > 1

    def _get_sigma(self, bar: pd.Series) -> float:
        """Rolling 5-min vol — kept from v1 for urgency term."""
        raw_vol = bar.get("realized_vol_60s")
        if raw_vol is not None and not pd.isna(raw_vol) and float(raw_vol) > 0:
            self._vol_buffer.append(float(raw_vol))
        if len(self._vol_buffer) >= 30:
            return float(sum(self._vol_buffer) / len(self._vol_buffer))
        elif len(self._vol_buffer) > 0:
            return float(self._vol_buffer[-1])
        return self.default_vol

    # ─────────────────────────────────────────────────────────────────────────
    # Core quote formula (Corollary 3.5.3)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_quotes(self,
                        mid:          float,
                        T:            float,
                        q:            int,
                        hawkes_state: HawkesState,
                        alpha_state:  AlphaState,
                        fill_state:   FillRateState) -> tuple[float, float]:
        """
        Optimal quotes from Ricci Corollary 3.5.3:
            δ±* = δ0± + B * {∓ Et[∫αu du] + φ * bφ}

        Returns:
            (bid_price, ask_price)

        Note: uses self._current_ts (set in on_bar/on_bar_live before calling)
        for the open period check — no bar reference needed here.
        """
        p = self.params

        # ── Risk-neutral base depth (δ0± = 1/κ±t) ─────────────────────────────
        d0_bid = fill_state.optimal_depth_buy    # 1/κ⁻t
        d0_ask = fill_state.optimal_depth_sell   # 1/κ⁺t

        # ── B coefficients (scale alpha and phi adjustments) ───────────────────
        B_bid = fill_state.B_coeff_buy    # 1/(2 + κ⁻t)
        B_ask = fill_state.B_coeff_sell   # 1/(2 + κ⁺t)

        # ── Alpha integral Et[∫t^T αu du] ─────────────────────────────────────
        alpha_integral = self.hawkes.alpha_integral(
            alpha_t = alpha_state.alpha,
            T       = T,
            zeta    = p.zeta,
            epsilon = (p.epsilon_plus + p.epsilon_minus) / 2,
        )

        # ── Lambda imbalance term (bφ from Equation 3.19b) ────────────────────
        h = math.exp(-1)   # e^(-1) ≈ 0.368 — always constant for exponential class
        lambda_adj = self.hawkes.lambda_adjustment(T=T, h=h, phi=p.phi)

        # ── Inventory management (urgency toward zero) ─────────────────────────
        max_inv     = max(1, self.config.max_inventory)
        urgency     = 1.0 + (abs(q) / max_inv) * (1.0 - T)
        inv_adj_bid = p.phi * (1 + 2 * q) * T * urgency    # widen bid if long
        inv_adj_ask = p.phi * (1 - 2 * q) * T * urgency    # widen ask if short

        # ── Optimal spreads (Equation 3.16) ───────────────────────────────────
        delta_ask = d0_ask + B_ask * (
            -alpha_integral   # widen ask if α > 0 (price rising)
            + inv_adj_ask     # widen ask if short
            + lambda_adj      # flow imbalance
        )
        delta_bid = d0_bid + B_bid * (
            +alpha_integral   # tighten bid if α > 0 (price rising)
            + inv_adj_bid     # widen bid if long
            + lambda_adj      # flow imbalance
        )

        # ── Open period multiplier (09:15-09:45 IST) ──────────────────────────
        # Uses self._current_ts set in on_bar/on_bar_live — no bar needed here
        if self._is_open_period(self._current_ts):
            delta_ask = min(delta_ask * p.open_mult, p.max_spread / 2)
            delta_bid = min(delta_bid * p.open_mult, p.max_spread / 2)

        # ── Clip to [min_spread/2, max_spread/2] ──────────────────────────────
        half_min  = p.min_spread / 2
        half_max  = p.max_spread / 2
        delta_ask = max(half_min, min(delta_ask, half_max))
        delta_bid = max(half_min, min(delta_bid, half_max))

        bid_price = round(mid - delta_bid, 2)
        ask_price = round(mid + delta_ask, 2)

        return bid_price, ask_price

    # ─────────────────────────────────────────────────────────────────────────
    # Main on_bar — backtest
    # ─────────────────────────────────────────────────────────────────────────

    def on_bar(self, bar: pd.Series, book: SimulatedOrderBook) -> None:
        self.bars_processed += 1

        mid    = float(bar["weighted_mid"]) if not pd.isna(bar["weighted_mid"]) \
                 else float(bar["close"])
        ts_sec = int(bar["ts_sec"])
        q      = book.inventory

        # ── Imbalance spike filter ─────────────────────────────────────────────
        imbalance_last = float(bar.get("imbalance_last", 0) or 0)
        imbalance_ma30 = float(bar.get("imbalance_ma_30s", 0) or 0)
        volume_ratio   = float(bar.get("volume_ratio", 1.0) or 1.0)

        if abs(imbalance_last - imbalance_ma30) > 0.3 and volume_ratio < 0.5:
            if self.bid_id is not None:
                book.cancel_order(self.bid_id)
                self.bid_id = None
            if self.ask_id is not None:
                book.cancel_order(self.ask_id)
                self.ask_id = None
            return

        # ── Terminal flatten ───────────────────────────────────────────────────
        if self._should_flatten(ts_sec, q):
            if self.bid_id is not None:
                book.cancel_order(self.bid_id)
                self.bid_id = None
            if self.ask_id is not None:
                book.cancel_order(self.ask_id)
                self.ask_id = None
            if q > 0:
                book.post_order(Side.SELL, mid - 0.05, abs(q), ts_sec)
            elif q < 0:
                book.post_order(Side.BUY,  mid + 0.05, abs(q), ts_sec)
            return

        # ── Step 1: Classify incoming MO ──────────────────────────────────────
        clf_result = self.classifier.classify(bar)
        vol_delta  = float(bar.get("volume_delta", 0) or 0)

        is_buy_influential  = clf_result.is_influential and vol_delta > 0
        is_sell_influential = clf_result.is_influential and vol_delta < 0

        if clf_result.is_influential:
            self.n_influential += 1

        # ── Step 2: Update Hawkes λ±t ──────────────────────────────────────────
        hawkes_state = self.hawkes.update(
            bar,
            is_buy_influential  = is_buy_influential,
            is_sell_influential = is_sell_influential,
        )

        # ── Step 3: Stability check (Lemma 3.2.2) ─────────────────────────────
        if not hawkes_state.is_stable:
            self.n_stable_halts += 1
            if self.bid_id is not None:
                book.cancel_order(self.bid_id)
                self.bid_id = None
            if self.ask_id is not None:
                book.cancel_order(self.ask_id)
                self.ask_id = None
            return

        # ── Step 4: Update short-term alpha αt ────────────────────────────────
        alpha_state = self.alpha_model.update(
            bar,
            is_buy_influential  = is_buy_influential,
            is_sell_influential = is_sell_influential,
        )

        # ── Step 5: Update fill rate κ±t ──────────────────────────────────────
        fill_state = self.fill_rate.update(
            bar,
            is_buy_influential  = is_buy_influential,
            is_sell_influential = is_sell_influential,
        )

        # ── Step 6: Time remaining ─────────────────────────────────────────────
        T = self._time_remaining(ts_sec)

        # ── Step 7: Store ts then compute optimal quotes ───────────────────────
        # _current_ts is used inside _compute_quotes for open period check
        self._current_ts = ts_sec

        bid_price, ask_price = self._compute_quotes(
            mid          = mid,
            T            = T,
            q            = q,
            hawkes_state = hawkes_state,
            alpha_state  = alpha_state,
            fill_state   = fill_state,
        )

        # ── Step 8: Cancel and repost ─────────────────────────────────────────
        if self.bid_id is not None:
            book.cancel_order(self.bid_id)
        if self.ask_id is not None:
            book.cancel_order(self.ask_id)

        self.bid_id = book.post_order(Side.BUY,  bid_price, 1, ts_sec)
        self.ask_id = book.post_order(Side.SELL, ask_price, 1, ts_sec)

    # ─────────────────────────────────────────────────────────────────────────
    # Fill / day-end handlers
    # ─────────────────────────────────────────────────────────────────────────

    def on_fill(self, fill: Fill) -> None:
        if fill.order_id == self.bid_id:
            self.bid_id = None
        elif fill.order_id == self.ask_id:
            self.ask_id = None

    def on_day_end(self, date: str, book: SimulatedOrderBook) -> None:
        book.cancel_all()
        self.bid_id = None
        self.ask_id = None

        # Force close remaining inventory at mid
        q = book.inventory
        if q != 0:
            mid = book.last_mid if hasattr(book, 'last_mid') else 0
            if mid > 0:
                side = Side.SELL if q > 0 else Side.BUY
                book.post_order(side, mid, abs(q), 0)

        # Reset all components for new day
        self._vol_buffer.clear()
        self._current_ts = 0
        self.hawkes.reset_day()
        self.alpha_model.reset_day()
        self.fill_rate.reset_day()
        self.classifier.reset_day()

    # ─────────────────────────────────────────────────────────────────────────
    # Live trading
    # ─────────────────────────────────────────────────────────────────────────

    def on_bar_live(self, bar: dict, engine) -> None:
        """Live trading — same logic as on_bar."""
        self.bars_processed += 1

        def get(key, default=None):
            val = bar.get(key, default)
            if val is None:
                return default
            try:
                if math.isnan(float(val)):
                    return default
            except (TypeError, ValueError):
                pass
            return val

        mid    = get("weighted_mid") or get("close")
        ts_sec = get("ts_sec")

        if not mid or not ts_sec or mid <= 0:
            return

        mid    = float(mid)
        ts_sec = int(ts_sec)
        q      = engine.portfolio.get_position(engine.symbol)

        # Imbalance filter
        if (abs((get("imbalance_last", 0) or 0) -
                (get("imbalance_ma_30s", 0) or 0)) > 0.3
                and (get("volume_ratio", 1.0) or 1.0) < 0.5):
            return

        # Terminal flatten
        if self._should_flatten(ts_sec, q):
            if q > 0:
                engine.post_market_order("SELL", abs(q))
            elif q < 0:
                engine.post_market_order("BUY", abs(q))
            return

        bar_series  = pd.Series(bar)
        clf_result  = self.classifier.classify(bar_series)
        vol_delta   = float(bar.get("volume_delta", 0) or 0)
        is_buy_inf  = clf_result.is_influential and vol_delta > 0
        is_sell_inf = clf_result.is_influential and vol_delta < 0

        hawkes_state = self.hawkes.update(bar_series, is_buy_inf, is_sell_inf)

        if not hawkes_state.is_stable:
            return

        alpha_state = self.alpha_model.update(bar_series, is_buy_inf, is_sell_inf)
        fill_state  = self.fill_rate.update(bar_series, is_buy_inf, is_sell_inf)
        T           = self._time_remaining(ts_sec)

        # Store ts before calling _compute_quotes
        self._current_ts = ts_sec

        bid_price, ask_price = self._compute_quotes(
            mid          = mid,
            T            = T,
            q            = q,
            hawkes_state = hawkes_state,
            alpha_state  = alpha_state,
            fill_state   = fill_state,
        )

        if math.isnan(bid_price) or math.isnan(ask_price):
            return

        self.bid_id = engine.post_quote("BUY",  bid_price, 1)
        self.ask_id = engine.post_quote("SELL", ask_price, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        """Return current state of all v2 components."""
        return {
            "hawkes":         self.hawkes.status(),
            "alpha":          self.alpha_model.status(),
            "fill_rate":      self.fill_rate.status(),
            "classifier":     self.classifier.status(),
            "n_stable_halts": self.n_stable_halts,
            "n_influential":  self.n_influential,
            "bars_processed": self.bars_processed,
        }