"""
Ricci Hawkes-Alpha Market Making Strategy — V2 with ML Alpha (Option B)
========================================================================
ML integration uses spread multiplier approach instead of alpha blending.

ML signal directly adjusts bid/ask depths:
    Price predicted UP   → tighter bid (capture upward move)
                         → wider ask   (avoid selling cheap)
    Price predicted DOWN → wider bid   (avoid buying into fall)
                         → tighter ask (sell before price drops)

This is more impactful than alpha blending because it directly
changes fill probability rather than a tiny downstream alpha adjustment.
"""

import math
import collections
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from src.backtest.models.v1_avellaneda_stoikov.strategy import BaseStrategy, StrategyConfig
from src.backtest.order_book import SimulatedOrderBook, Side, Fill

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
    """
    # ── Hawkes process (Section 3.2.2) ────────────────────────────────────────
    beta:  float = 1.0
    theta: float = 2.0
    eta:   float = 0.5
    nu:    float = 0.2
    rho:   float = 0.30

    # ── Short-term alpha (Section 3.4) ────────────────────────────────────────
    zeta:          float = 0.5
    epsilon_plus:  float = 0.002
    epsilon_minus: float = 0.002

    # ── Stochastic fill rate κt (Section 3.3) ─────────────────────────────────
    beta_kappa:  float = 0.5
    theta_kappa: float = 1.5
    eta_kappa:   float = 0.3
    nu_kappa:    float = 0.1

    # ── Risk and inventory ────────────────────────────────────────────────────
    phi:        float = 0.001
    min_spread: float = 0.10
    max_spread: float = 10.0
    open_mult:  float = 2.0

    # ── Market classifier ─────────────────────────────────────────────────────
    classifier_window:    int   = 60
    classifier_threshold: float = 0.50

    # ── ML spread adjustment ──────────────────────────────────────────────────
    # When ML is confident about price direction, adjust bid/ask depths:
    #   ml_tight_mult: multiply depth by this to tighten (more aggressive)
    #   ml_wide_mult:  multiply depth by this to widen (less aggressive)
    #
    # Example with defaults (15% adjustment):
    #   Price UP predicted:  bid × 0.85 (tighter), ask × 1.15 (wider)
    #   Price DOWN predicted: bid × 1.15 (wider), ask × 0.85 (tighter)
    ml_tight_mult: float = 0.85   # tighten side (more fills)
    ml_wide_mult:  float = 1.15   # widen side (fewer adverse fills)


class RicciHawkesAlphaStrategy(BaseStrategy):
    """
    Chapter 3 Ricci market making strategy with ML spread adjustment.

    ML integration (Option B — spread multiplier):
        When ML predicts price direction with sufficient confidence,
        directly adjust bid/ask depths to exploit the prediction.

        Price UP   → tighter bid, wider ask
        Price DOWN → wider bid, tighter ask

    Call enable_ml() after __init__ to activate ML.
    Falls back gracefully to pure Hawkes if ML unavailable.
    """

    SESSION_SECONDS   = 22500.0
    SESSION_START_UTC = 13500
    OPEN_START_IST    = 33300
    OPEN_END_IST      = 35100
    FLATTEN_IST       = 54720
    MARKET_CLOSE_IST  = 55800

    def __init__(self,
                 config: StrategyConfig,
                 params: Optional[V2Params] = None):
        super().__init__(config)

        self.params = params or V2Params()
        p = self.params

        # ── V2 components ──────────────────────────────────────────────────────
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

        # ── Rolling vol ────────────────────────────────────────────────────────
        self._vol_buffer = collections.deque(maxlen=300)
        self.default_vol = 2.0
        self._current_ts: int = 0

        # ── ML predictor (disabled by default) ────────────────────────────────
        self._ml_predictor  = None
        self._ml_enabled    = False
        self._ml_symbol_ok  = False

        # Current bar ML multipliers — set by _get_ml_multipliers()
        # Reset to (1.0, 1.0) each bar
        self._ml_bid_mult: float = 1.0
        self._ml_ask_mult: float = 1.0

        # ── Diagnostics ────────────────────────────────────────────────────────
        self.n_stable_halts  = 0
        self.n_influential   = 0
        self.n_ml_signals    = 0
        self.n_ml_up         = 0
        self.n_ml_down       = 0

    # ─────────────────────────────────────────────────────────────────────────
    # ML integration
    # ─────────────────────────────────────────────────────────────────────────

    def enable_ml(self, predictor=None) -> bool:
        """
        Enable ML spread adjustment for this strategy.

        Args:
            predictor: pre-loaded LGBMPredictor instance (optional).
                       If None, loads from disk automatically.

        Returns:
            True if ML successfully enabled for this symbol.

        Usage:
            strat = RicciHawkesAlphaStrategy(config, params)
            strat.enable_ml()

            # Or share one predictor across many strategies (faster):
            from src.ml.models.lgbm_predictor import LGBMPredictor
            shared_predictor = LGBMPredictor()
            shared_predictor.load_models()
            strat.enable_ml(predictor=shared_predictor)
        """
        try:
            from src.ml.models.lgbm_predictor import LGBMPredictor

            if predictor is not None:
                self._ml_predictor = predictor
            else:
                self._ml_predictor = LGBMPredictor()
                self._ml_predictor.load_models()

            self._ml_symbol_ok = (
                self.config.symbol in self._ml_predictor.available_symbols
            )
            self._ml_enabled = self._ml_symbol_ok

            if self._ml_enabled:
                print(f"[V2+ML] ML enabled for {self.config.symbol}")
            else:
                print(f"[V2+ML] No ML model for {self.config.symbol} "
                      f"— using pure Hawkes")

            return self._ml_enabled

        except Exception as e:
            print(f"[V2+ML] ML not available ({e}) — using pure Hawkes")
            self._ml_enabled   = False
            self._ml_symbol_ok = False
            return False

    def _get_ml_multipliers(self, bar) -> tuple[float, float]:
        """
        Get bid/ask depth multipliers from ML prediction.

        Returns:
            (bid_mult, ask_mult)
            1.0 = no change (default when ML disabled or no signal)

        Logic:
            Price predicted UP:
                bid_mult = ml_tight_mult (0.85) → tighter bid → more bid fills
                ask_mult = ml_wide_mult  (1.15) → wider ask   → fewer ask fills
                (we want to BUY more and SELL less when price is rising)

            Price predicted DOWN:
                bid_mult = ml_wide_mult  (1.15) → wider bid   → fewer bid fills
                ask_mult = ml_tight_mult (0.85) → tighter ask → more ask fills
                (we want to SELL more and BUY less when price is falling)
        """
        if not self._ml_enabled or self._ml_predictor is None:
            return 1.0, 1.0

        try:
            pred = self._ml_predictor.predict(self.config.symbol, bar)

            if not pred.signal:
                return 1.0, 1.0

            p = self.params
            self.n_ml_signals += 1

            if pred.direction == 1:   # price predicted UP
                self.n_ml_up += 1
                return p.ml_tight_mult, p.ml_wide_mult

            else:                      # price predicted DOWN
                self.n_ml_down += 1
                return p.ml_wide_mult, p.ml_tight_mult

        except Exception:
            return 1.0, 1.0

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
        raw_vol = bar.get("realized_vol_60s")
        if raw_vol is not None and not pd.isna(raw_vol) and float(raw_vol) > 0:
            self._vol_buffer.append(float(raw_vol))
        if len(self._vol_buffer) >= 30:
            return float(sum(self._vol_buffer) / len(self._vol_buffer))
        elif len(self._vol_buffer) > 0:
            return float(self._vol_buffer[-1])
        return self.default_vol

    # ─────────────────────────────────────────────────────────────────────────
    # Core quote formula (Corollary 3.5.3) + ML spread adjustment
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_quotes(self,
                        mid:          float,
                        T:            float,
                        q:            int,
                        hawkes_state: HawkesState,
                        alpha_state:  AlphaState,
                        fill_state:   FillRateState,
                        bid_mult:     float = 1.0,
                        ask_mult:     float = 1.0) -> tuple[float, float]:
        """
        Optimal quotes from Ricci Corollary 3.5.3 with ML adjustment.

        bid_mult / ask_mult: from _get_ml_multipliers()
            < 1.0 → tighten (closer to mid, more fills)
            > 1.0 → widen  (further from mid, fewer fills)
            = 1.0 → no change (ML disabled or no signal)
        """
        p = self.params

        d0_bid = fill_state.optimal_depth_buy
        d0_ask = fill_state.optimal_depth_sell
        B_bid  = fill_state.B_coeff_buy
        B_ask  = fill_state.B_coeff_sell

        alpha_integral = self.hawkes.alpha_integral(
            alpha_t = alpha_state.alpha,
            T       = T,
            zeta    = p.zeta,
            epsilon = (p.epsilon_plus + p.epsilon_minus) / 2,
        )

        h          = math.exp(-1)
        lambda_adj = self.hawkes.lambda_adjustment(T=T, h=h, phi=p.phi)

        max_inv     = max(1, self.config.max_inventory)
        urgency     = 1.0 + (abs(q) / max_inv) * (1.0 - T)
        inv_adj_bid = p.phi * (1 + 2 * q) * T * urgency
        inv_adj_ask = p.phi * (1 - 2 * q) * T * urgency

        # ── Ricci optimal depths ───────────────────────────────────────────────
        delta_ask = d0_ask + B_ask * (
            -alpha_integral + inv_adj_ask + lambda_adj
        )
        delta_bid = d0_bid + B_bid * (
            +alpha_integral + inv_adj_bid + lambda_adj
        )

        # ── Open period multiplier ─────────────────────────────────────────────
        if self._is_open_period(self._current_ts):
            delta_ask = min(delta_ask * p.open_mult, p.max_spread / 2)
            delta_bid = min(delta_bid * p.open_mult, p.max_spread / 2)

        # ── ML spread adjustment ───────────────────────────────────────────────
        # Applied AFTER Ricci formula, BEFORE clipping.
        # Multiplying depth by < 1 = tighten (quote closer to mid)
        # Multiplying depth by > 1 = widen  (quote further from mid)
        delta_bid *= bid_mult
        delta_ask *= ask_mult

        # ── Dynamic min_spread (5 bps of mid) ─────────────────────────────────
        dyn_min  = max(p.min_spread, mid * 0.0005)
        half_min = dyn_min / 2
        half_max = p.max_spread / 2

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

        # ── Step 1: Classify ───────────────────────────────────────────────────
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

        # ── Step 3: Stability check ────────────────────────────────────────────
        if not hawkes_state.is_stable:
            self.n_stable_halts += 1
            if self.bid_id is not None:
                book.cancel_order(self.bid_id)
                self.bid_id = None
            if self.ask_id is not None:
                book.cancel_order(self.ask_id)
                self.ask_id = None
            return

        # ── Step 4: Update alpha αt ────────────────────────────────────────────
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

        # ── Step 7: ML spread multipliers ─────────────────────────────────────
        # Get bid/ask depth multipliers from ML prediction.
        # (1.0, 1.0) when ML disabled — no effect on quotes.
        self._current_ts = ts_sec
        bid_mult, ask_mult = self._get_ml_multipliers(bar)

        # ── Step 8: Compute optimal quotes ────────────────────────────────────
        bid_price, ask_price = self._compute_quotes(
            mid          = mid,
            T            = T,
            q            = q,
            hawkes_state = hawkes_state,
            alpha_state  = alpha_state,
            fill_state   = fill_state,
            bid_mult     = bid_mult,
            ask_mult     = ask_mult,
        )

        # ── Step 9: Cancel and repost ─────────────────────────────────────────
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

        # Reset ML feature state rolling buffers
        if self._ml_enabled and self._ml_predictor:
            try:
                state = self._ml_predictor.get_state(self.config.symbol)
                state.__init__()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # Live trading
    # ─────────────────────────────────────────────────────────────────────────

    def on_bar_live(self, bar: dict, engine) -> None:
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

        if (abs((get("imbalance_last", 0) or 0) -
                (get("imbalance_ma_30s", 0) or 0)) > 0.3
                and (get("volume_ratio", 1.0) or 1.0) < 0.5):
            return

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

        self._current_ts = ts_sec
        bid_mult, ask_mult = self._get_ml_multipliers(bar)

        bid_price, ask_price = self._compute_quotes(
            mid=mid, T=T, q=q,
            hawkes_state=hawkes_state,
            alpha_state=alpha_state,
            fill_state=fill_state,
            bid_mult=bid_mult,
            ask_mult=ask_mult,
        )

        if math.isnan(bid_price) or math.isnan(ask_price):
            return

        self.bid_id = engine.post_quote("BUY",  bid_price, 1)
        self.ask_id = engine.post_quote("SELL", ask_price, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        base = {
            "hawkes":         self.hawkes.status(),
            "alpha":          self.alpha_model.status(),
            "fill_rate":      self.fill_rate.status(),
            "classifier":     self.classifier.status(),
            "n_stable_halts": self.n_stable_halts,
            "n_influential":  self.n_influential,
            "bars_processed": self.bars_processed,
        }
        if self._ml_enabled:
            base["ml"] = {
                "enabled":     True,
                "n_signals":   self.n_ml_signals,
                "n_up":        self.n_ml_up,
                "n_down":      self.n_ml_down,
                "signal_rate": round(self.n_ml_signals /
                                     max(self.bars_processed, 1), 4),
                "avg_ms":      round(self._ml_predictor.avg_inference_ms(), 3)
                               if self._ml_predictor else 0,
            }
        return base
