"""
Avellaneda-Stoikov Market Making Strategy (2008)
Paper: "High-Frequency Trading in a Limit Order Book"

We work entirely in PRICE POINT space (not return space).

Key formulas (price point version):
    Reservation price:
        r = mid - q * gamma * sigma^2 * T

    Optimal spread:
        spread = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/kappa)

Where:
    sigma = realized_vol_60s in price points (e.g. 2.5 pts)
    T     = fraction of session remaining (0.0 → 1.0)
    gamma = risk aversion, scaled to price point space (~0.001 for NIFTY)
    kappa = order arrival rate (~1.5)
    q     = inventory in lots

Calibration note:
    gamma must be calibrated to sigma's scale.
    For NIFTY with sigma ~2pts, gamma ~0.001 gives spread ~1-5pts.
    Rule of thumb: gamma ≈ 1 / (sigma^2 * 10) for ~1pt spread at mid-session.
"""

import math
import pandas as pd
from typing import Optional

from src.backtest import engine
from src.backtest.strategy import BaseStrategy, StrategyConfig
from src.backtest.order_book import SimulatedOrderBook, Side, Fill


class AvellanedaStoikovStrategy(BaseStrategy):

    SESSION_SECONDS   = 22500.0   # 09:15 to 15:30 IST
    SESSION_START_UTC = 13500     # 09:15 IST in UTC seconds since midnight

    def __init__(self, config: StrategyConfig,
                 gamma: float      = 0.1,
                 kappa: float      = 1.5,
                 min_spread: float = 0.5,
                 max_spread: float = 10.0):
        super().__init__(config)
        self.gamma      = gamma
        self.kappa      = kappa
        self.min_spread = min_spread
        self.max_spread = max_spread

        self.bid_id: Optional[int] = None
        self.ask_id: Optional[int] = None
        self.default_vol = 2.0    # NIFTY typical 60s vol in price points

    # ─────────────────────────────────────
    # Core formulas (price point space)
    # ─────────────────────────────────────

    def _time_remaining(self, ts_sec: int) -> float:
        """Fraction of session remaining (0.0 → 1.0). Min 60s worth."""
        secs_since_midnight = ts_sec % 86400
        elapsed   = max(0.0, secs_since_midnight - self.SESSION_START_UTC)
        remaining = max(60.0, self.SESSION_SECONDS - elapsed)
        return remaining / self.SESSION_SECONDS

    def _get_sigma(self, bar: pd.Series) -> float:
        """Realized vol in price points. Use 60s window."""
        sigma = bar.get("realized_vol_60s")
        if sigma is None or pd.isna(sigma) or sigma <= 0:
            return self.default_vol
        return float(sigma)

    def _reservation_price(self, mid: float, inventory: int,
                           sigma: float, T: float) -> float:
        """r = mid - q * gamma * sigma^2 * T  (all in price points)"""
        return mid - inventory * self.gamma * (sigma ** 2) * T

    def _optimal_spread(self, sigma: float, T: float) -> float:
        """
        spread = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/kappa)
        All terms in price points.
        """
        term1  = self.gamma * (sigma ** 2) * T
        term2  = (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)
        spread = term1 + term2
        return max(self.min_spread, min(spread, self.max_spread))

    # ─────────────────────────────────────
    # Strategy logic
    # ─────────────────────────────────────

    def on_bar(self, bar: pd.Series, book: SimulatedOrderBook) -> None:
        self.bars_processed += 1

        mid    = float(bar["weighted_mid"]) if not pd.isna(bar["weighted_mid"]) \
                 else float(bar["close"])
        ts_sec = int(bar["ts_sec"])
        imbalance_last = float(bar.get("imbalance_last", 0) or 0)
        imbalance_ma30 = float(bar.get("imbalance_ma_30s", 0) or 0)
        volume_ratio   = float(bar.get("volume_ratio", 1.0) or 1.0)

        if (abs(imbalance_last - imbalance_ma30) > 0.3
                and volume_ratio < 0.5):
            # Suspected fake order spike — skip
            if self.bid_id is not None:
                book.cancel_order(self.bid_id)
                self.bid_id = None
            if self.ask_id is not None:
                book.cancel_order(self.ask_id)
                self.ask_id = None
            return
        sigma  = self._get_sigma(bar)
        T      = self._time_remaining(ts_sec)
        q      = book.inventory

        r      = self._reservation_price(mid, q, sigma, T)
        spread = self._optimal_spread(sigma, T)

        bid_price = round(r - spread / 2, 2)
        ask_price = round(r + spread / 2, 2)

        # Cancel and repost every bar
        if self.bid_id is not None:
            book.cancel_order(self.bid_id)
        if self.ask_id is not None:
            book.cancel_order(self.ask_id)

        self.bid_id = book.post_order(Side.BUY,  bid_price, 1, ts_sec)
        self.ask_id = book.post_order(Side.SELL, ask_price, 1, ts_sec)

    def on_bar_live(self, bar: dict, engine) -> None:
        self.bars_processed += 1

        import pandas as pd
        import math

        def get(key, default=None):
            val = bar.get(key, default) if isinstance(bar, dict) \
                else bar[key] if key in bar.index else default
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

        # ── Imbalance persistence filter ──────────────────
        # Only trust imbalance if it agrees with 30s moving average
        # Spike that doesn't persist = likely fake orders
        imbalance_last = get("imbalance_last", 0) or 0
        imbalance_ma30 = get("imbalance_ma_30s", 0) or 0
        volume_ratio   = get("volume_ratio", 1.0) or 1.0

        imbalance_spike = (
            abs(imbalance_last - imbalance_ma30) > 0.3  # large deviation
            and volume_ratio < 0.5                        # without volume
        )

        if imbalance_spike:
            # Suspected fake order — skip this bar
            return

        # ── Rest of strategy logic unchanged ──────────────
        bar_series = pd.Series(bar) if isinstance(bar, dict) else bar
        sigma  = self._get_sigma(bar_series)
        T      = self._time_remaining(ts_sec)
        q      = engine.portfolio.get_position(engine.symbol)

        r      = self._reservation_price(mid, q, sigma, T)
        spread = self._optimal_spread(sigma, T)

        bid_price = round(r - spread / 2, 2)
        ask_price = round(r + spread / 2, 2)

        if math.isnan(bid_price) or math.isnan(ask_price):
            return

        self.bid_id = engine.post_quote("BUY",  bid_price, 1)
        self.ask_id = engine.post_quote("SELL", ask_price, 1)

        def get(key, default=None):
            val = bar.get(key, default) if isinstance(bar, dict) \
                else bar[key] if key in bar.index else default
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

        # Skip bar if no valid price or timestamp
        if not mid or not ts_sec or mid <= 0:
            return

        mid    = float(mid)
        ts_sec = int(ts_sec)

        bar_series = pd.Series(bar) if isinstance(bar, dict) else bar
        sigma  = self._get_sigma(bar_series)
        T      = self._time_remaining(ts_sec)
        q      = engine.portfolio.get_position(engine.symbol)

        r      = self._reservation_price(mid, q, sigma, T)
        spread = self._optimal_spread(sigma, T)

        bid_price = round(r - spread / 2, 2)
        ask_price = round(r + spread / 2, 2)

        # Sanity check prices
        if math.isnan(bid_price) or math.isnan(ask_price):
            return

        self.bid_id = engine.post_quote("BUY",  bid_price, 1)
        self.ask_id = engine.post_quote("SELL", ask_price, 1)

    def on_fill(self, fill: Fill) -> None:
        if fill.order_id == self.bid_id:
            self.bid_id = None
        elif fill.order_id == self.ask_id:
            self.ask_id = None

    def on_day_end(self, date: str, book: SimulatedOrderBook) -> None:
        book.cancel_all()
        self.bid_id = None
        self.ask_id = None


if __name__ == "__main__":
    config = StrategyConfig(
        symbol        = "NIFTY26MAYFUT",
        lot_size      = 75,
        max_inventory = 5,
    )
    strat = AvellanedaStoikovStrategy(config, gamma=0.1, kappa=1.5,
                                      min_spread=0.5, max_spread=10.0)

    mid   = 23880.0
    sigma = 2.5      # price points (from realized_vol_60s)

    print("=== Formula diagnostics ===")
    for T, label in [(1.0, "Start of day"), (0.5, "Mid session"), (0.01, "End of day")]:
        r      = strat._reservation_price(mid, inventory=0, sigma=sigma, T=T)
        spread = strat._optimal_spread(sigma=sigma, T=T)
        term1  = strat.gamma * sigma**2 * T
        term2  = (2/strat.gamma) * math.log(1 + strat.gamma/strat.kappa)
        print(f"\n{label} (T={T}):")
        print(f"  term1 (vol):      {term1:.4f} pts")
        print(f"  term2 (arrival):  {term2:.4f} pts")
        print(f"  spread:           {spread:.4f} pts")
        print(f"  bid:              {r - spread/2:.2f}")
        print(f"  ask:              {r + spread/2:.2f}")

    print("\n=== Inventory shift (mid session, T=0.5) ===")
    T = 0.5
    for q, label in [(0, "flat"), (3, "long +3"), (-3, "short -3")]:
        r      = strat._reservation_price(mid, inventory=q, sigma=sigma, T=T)
        spread = strat._optimal_spread(sigma=sigma, T=T)
        print(f"  inventory={q:+d} ({label}): r={r:.2f} | "
              f"bid={r-spread/2:.2f} | ask={r+spread/2:.2f}")