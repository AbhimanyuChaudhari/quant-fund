"""
Avellaneda-Stoikov Market Making Strategy (2008)
Paper: "High-Frequency Trading in a Limit Order Book"

Improvements over vanilla A-S (from research notebooks 03 & 04):
    1. Urgency multiplier  — amplifies inventory skewing as time runs out
    2. Terminal flatten    — market order to close ALL inventory in last 18 mins
    3. Dynamic sigma       — rolling 5-min mean of realized_vol_60s
    4. Open period mult    — wider spread during 09:15-09:45 IST (high vol window)
    5. Dynamic min_spread  — 5 basis points of mid price (prevents over-trading
                             on high-price symbols like BAJAJ-AUTO, APOLLOHOSP)
    6. Percentage flatten  — uses 2 bps offset instead of hardcoded 0.05

We work entirely in PRICE POINT space (not return space).

Key formulas (price point version):
    Reservation price:
        r = mid - q * gamma * sigma^2 * T * urgency

    Urgency:
        urgency = 1.0 + (|q| / max_inventory) * (1 - T)

    Optimal spread:
        spread = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/kappa)
        effective_min_spread = max(min_spread, mid * 0.0005)  ← dynamic floor

    Open period multiplier (09:15-09:45 IST):
        spread *= open_mult   (default 2.0x)

    Kyle's Lambda adjustment (adverse selection):
        adjusted_spread = spread * kyle_model.spread_multiplier()

Where:
    sigma = rolling 5-min mean of realized_vol_60s in price points
    T     = fraction of session remaining (0.0 → 1.0)
    gamma = risk aversion, scaled to price point space (~0.001 for NIFTY)
    kappa = order arrival rate (~1.5)
    q     = inventory in lots
"""

import math
import collections
import pandas as pd
from typing import Optional

from src.backtest.strategy import BaseStrategy, StrategyConfig
from src.backtest.order_book import SimulatedOrderBook, Side, Fill
from src.models.kyle_lambda import KyleLambdaModel, Bar as KyleBar


class AvellanedaStoikovStrategy(BaseStrategy):

    SESSION_SECONDS   = 22500.0   # 09:15 to 15:30 IST
    SESSION_START_UTC = 13500     # 09:15 IST in UTC seconds since midnight

    # ── IST time-of-day constants ──────────────────────────────────────────────
    OPEN_START_IST   = 9 * 3600 + 15 * 60   # 09:15 IST = 33300s
    OPEN_END_IST     = 9 * 3600 + 45 * 60   # 09:45 IST = 35100s
    FLATTEN_IST      = 15 * 3600 + 12 * 60  # 15:12 IST = 54720s (18 mins before close)
    MARKET_CLOSE_IST = 15 * 3600 + 30 * 60  # 15:30 IST = 55800s

    # ── Dynamic min_spread: 5 basis points of mid price ───────────────────────
    # Prevents over-trading on high-price symbols (BAJAJ-AUTO ~9000, APOLLOHOSP ~7000)
    # E.g. BAJAJ-AUTO: max(0.10, 9000 * 0.0005) = max(0.10, 4.50) = 4.50 pts
    #      CHOLAFIN:   max(0.10, 1520 * 0.0005) = max(0.10, 0.76) = 0.76 pts
    #      SAIL:       max(0.10,  150 * 0.0005) = max(0.10, 0.075) = 0.10 pts
    MIN_SPREAD_BPS = 0.0005   # 5 basis points

    def __init__(self, config: StrategyConfig,
                 gamma: float      = 0.001,
                 kappa: float      = 1.5,
                 min_spread: float = 0.10,
                 max_spread: float = 10.0,
                 open_mult: float  = 2.0,
                 use_kyle: bool    = False,
                 kyle_window: int  = 30):
        super().__init__(config)
        self.gamma      = gamma
        self.kappa      = kappa
        self.min_spread = min_spread
        self.max_spread = max_spread
        self.open_mult  = open_mult

        self.bid_id: Optional[int] = None
        self.ask_id: Optional[int] = None
        self.default_vol = 2.0

        # ── Rolling sigma buffer (5-min = 300 bars at 1s) ─────────────────────
        self._vol_buffer: collections.deque = collections.deque(maxlen=300)

        # ── Kyle's Lambda ─────────────────────────────────────────────────────
        self.use_kyle   = use_kyle
        self.kyle_model = KyleLambdaModel(window=kyle_window) if use_kyle else None

    # ─────────────────────────────────────
    # Core formulas
    # ─────────────────────────────────────

    def _ist_tod(self, ts_sec: int) -> int:
        """UTC epoch → IST time-of-day in seconds."""
        return (ts_sec + 19800) % 86400

    def _time_remaining(self, ts_sec: int) -> float:
        """Fraction of session remaining (1.0 at open → 0.0 at close)."""
        secs_since_midnight = ts_sec % 86400
        elapsed   = max(0.0, secs_since_midnight - self.SESSION_START_UTC)
        remaining = max(60.0, self.SESSION_SECONDS - elapsed)
        return remaining / self.SESSION_SECONDS

    def _get_sigma(self, bar: pd.Series) -> float:
        """Dynamic sigma: rolling 5-min mean of realized_vol_60s."""
        raw_vol = bar.get("realized_vol_60s")
        if raw_vol is not None and not pd.isna(raw_vol) and float(raw_vol) > 0:
            self._vol_buffer.append(float(raw_vol))
        if len(self._vol_buffer) >= 30:
            return float(sum(self._vol_buffer) / len(self._vol_buffer))
        elif len(self._vol_buffer) > 0:
            return float(self._vol_buffer[-1])
        return self.default_vol

    def _dynamic_min_spread(self, mid: float) -> float:
        """
        Dynamic minimum spread = max(param min_spread, 5 bps of mid price).
        Prevents over-trading on high-price symbols.

        Examples:
            BAJAJ-AUTO @ 9000: max(0.10, 4.50) = 4.50 pts
            APOLLOHOSP @ 7000: max(0.10, 3.50) = 3.50 pts
            CHOLAFIN   @ 1520: max(0.10, 0.76) = 0.76 pts
            SAIL       @  150: max(0.10, 0.08) = 0.10 pts
        """
        return max(self.min_spread, mid * self.MIN_SPREAD_BPS)

    def _urgency(self, inventory: int, T: float) -> float:
        """urgency = 1.0 + (|q| / max_inventory) * (1 - T)"""
        max_inv = max(1, self.config.max_inventory)
        return 1.0 + (abs(inventory) / max_inv) * (1.0 - T)

    def _reservation_price(self, mid: float, inventory: int,
                            sigma: float, T: float) -> float:
        """r = mid - q * gamma * sigma^2 * T * urgency"""
        u = self._urgency(inventory, T)
        return mid - inventory * self.gamma * (sigma ** 2) * T * u

    def _optimal_spread(self, sigma: float, T: float,
                         min_spread: float = None) -> float:
        """
        spread = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/kappa)
        Uses dynamic min_spread if provided, otherwise falls back to self.min_spread.
        """
        min_s  = min_spread if min_spread is not None else self.min_spread
        term1  = self.gamma * (sigma ** 2) * T
        term2  = (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)
        spread = term1 + term2
        return max(min_s, min(spread, self.max_spread))

    def _is_open_period(self, ts_sec: int) -> bool:
        """09:15-09:45 IST — high vol window, quote wider."""
        ist = self._ist_tod(ts_sec)
        return self.OPEN_START_IST <= ist <= self.OPEN_END_IST

    def _should_flatten(self, ts_sec: int, inventory: int) -> bool:
        """
        Flatten ALL inventory (not just >1) after 15:12 IST.
        Changed from abs(inventory) > 1 to abs(inventory) > 0.
        """
        ist = self._ist_tod(ts_sec)
        return ist >= self.FLATTEN_IST and abs(inventory) > 0

    def _flatten_offset(self, mid: float) -> float:
        """
        Market order offset for terminal flatten.
        Uses 2 bps of mid price (not hardcoded 0.05).
        Ensures aggressive enough fill even for high-price symbols.

        Examples:
            BAJAJ-AUTO @ 9000: max(0.05, 1.80) = 1.80 pts
            CHOLAFIN   @ 1520: max(0.05, 0.30) = 0.30 pts
            SAIL       @  150: max(0.05, 0.03) = 0.05 pts
        """
        return max(0.05, mid * 0.0002)

    def _kyle_bar(self, bar) -> KyleBar:
        """Convert a pd.Series or dict bar to KyleBar."""
        def get(key, default=0.0):
            val = bar.get(key, default) if isinstance(bar, dict) \
                else bar[key] if key in bar.index else default
            try:
                return float(val) if val is not None \
                    and not math.isnan(float(val)) else default
            except (TypeError, ValueError):
                return default

        ts_raw = get("ts_sec")
        ts = pd.to_datetime(ts_raw, unit="s", utc=True).to_pydatetime() \
            if ts_raw else pd.Timestamp.now().to_pydatetime()

        return KyleBar(
            symbol       = str(bar.get("symbol", "") if isinstance(bar, dict)
                               else bar.get("symbol", "")),
            ts           = ts,
            close        = get("close"),
            volume_delta = get("volume_delta"),
            tick_count   = get("tick_count"),
            imbalance    = get("imbalance_last"),
            spread_bps   = get("spread_bps", 1.0),
        )

    # ─────────────────────────────────────
    # Strategy logic — backtest
    # ─────────────────────────────────────

    def on_bar(self, bar: pd.Series, book: SimulatedOrderBook) -> None:
        self.bars_processed += 1

        mid    = float(bar["weighted_mid"]) if not pd.isna(bar["weighted_mid"]) \
                 else float(bar["close"])
        ts_sec = int(bar["ts_sec"])
        q      = book.inventory

        # ── Imbalance spike filter (fake order detection) ──────────────────────
        imbalance_last = float(bar.get("imbalance_last", 0) or 0)
        imbalance_ma30 = float(bar.get("imbalance_ma_30s", 0) or 0)
        volume_ratio   = float(bar.get("volume_ratio", 1.0) or 1.0)

        if (abs(imbalance_last - imbalance_ma30) > 0.3
                and volume_ratio < 0.5):
            if self.bid_id is not None:
                book.cancel_order(self.bid_id)
                self.bid_id = None
            if self.ask_id is not None:
                book.cancel_order(self.ask_id)
                self.ask_id = None
            return

        # ── Terminal flatten — last 18 mins, flatten ALL inventory ────────────
        if self._should_flatten(ts_sec, q):
            if self.bid_id is not None:
                book.cancel_order(self.bid_id)
                self.bid_id = None
            if self.ask_id is not None:
                book.cancel_order(self.ask_id)
                self.ask_id = None
            offset = self._flatten_offset(mid)
            if q > 0:
                book.post_order(Side.SELL, mid - offset, abs(q), ts_sec)
            elif q < 0:
                book.post_order(Side.BUY,  mid + offset, abs(q), ts_sec)
            return

        sigma = self._get_sigma(bar)
        T     = self._time_remaining(ts_sec)

        # ── Dynamic min_spread — 5 bps of mid price ───────────────────────────
        dyn_min_spread = self._dynamic_min_spread(mid)

        r           = self._reservation_price(mid, q, sigma, T)
        base_spread = self._optimal_spread(sigma, T, dyn_min_spread)

        # ── Open period multiplier (09:15-09:45 IST) ──────────────────────────
        if self._is_open_period(ts_sec):
            base_spread = min(base_spread * self.open_mult, self.max_spread)

        # ── Kyle's Lambda adjustment ───────────────────────────────────────────
        if self.use_kyle and self.kyle_model is not None:
            self.kyle_model.update(self._kyle_bar(bar))
            multiplier = self.kyle_model.spread_multiplier()
            spread     = max(dyn_min_spread,
                             min(base_spread * multiplier, self.max_spread))
        else:
            spread = base_spread

        bid_price = round(r - spread / 2, 2)
        ask_price = round(r + spread / 2, 2)

        # Cancel and repost every bar
        if self.bid_id is not None:
            book.cancel_order(self.bid_id)
        if self.ask_id is not None:
            book.cancel_order(self.ask_id)

        self.bid_id = book.post_order(Side.BUY,  bid_price, 1, ts_sec)
        self.ask_id = book.post_order(Side.SELL, ask_price, 1, ts_sec)

    # ─────────────────────────────────────
    # Strategy logic — live trading
    # ─────────────────────────────────────

    def on_bar_live(self, bar: dict, engine) -> None:
        self.bars_processed += 1

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
        q      = engine.portfolio.get_position(engine.symbol)

        # ── Imbalance spike filter ─────────────────────────────────────────────
        imbalance_last = get("imbalance_last", 0) or 0
        imbalance_ma30 = get("imbalance_ma_30s", 0) or 0
        volume_ratio   = get("volume_ratio", 1.0) or 1.0

        if (abs(imbalance_last - imbalance_ma30) > 0.3
                and volume_ratio < 0.5):
            return

        # ── Terminal flatten ───────────────────────────────────────────────────
        if self._should_flatten(ts_sec, q):
            offset = self._flatten_offset(mid)
            if q > 0:
                engine.post_market_order("SELL", abs(q))
            elif q < 0:
                engine.post_market_order("BUY", abs(q))
            return

        bar_series     = pd.Series(bar) if isinstance(bar, dict) else bar
        sigma          = self._get_sigma(bar_series)
        T              = self._time_remaining(ts_sec)
        dyn_min_spread = self._dynamic_min_spread(mid)

        r           = self._reservation_price(mid, q, sigma, T)
        base_spread = self._optimal_spread(sigma, T, dyn_min_spread)

        # ── Open period multiplier ─────────────────────────────────────────────
        if self._is_open_period(ts_sec):
            base_spread = min(base_spread * self.open_mult, self.max_spread)

        # ── Kyle's Lambda ──────────────────────────────────────────────────────
        if self.use_kyle and self.kyle_model is not None:
            self.kyle_model.update(self._kyle_bar(bar))
            multiplier = self.kyle_model.spread_multiplier()
            spread     = max(dyn_min_spread,
                             min(base_spread * multiplier, self.max_spread))
        else:
            spread = base_spread

        bid_price = round(r - spread / 2, 2)
        ask_price = round(r + spread / 2, 2)

        if math.isnan(bid_price) or math.isnan(ask_price):
            return

        self.bid_id = engine.post_quote("BUY",  bid_price, 1)
        self.ask_id = engine.post_quote("SELL", ask_price, 1)

    # ─────────────────────────────────────
    # Fill / day-end handlers
    # ─────────────────────────────────────

    def on_fill(self, fill: Fill) -> None:
        if fill.order_id == self.bid_id:
            self.bid_id = None
        elif fill.order_id == self.ask_id:
            self.ask_id = None

    def on_day_end(self, date: str, book: SimulatedOrderBook) -> None:
        """Cancel all orders. Force close any remaining inventory at mid."""
        book.cancel_all()
        self.bid_id = None
        self.ask_id = None

        q = book.inventory
        if q != 0:
            mid = book.last_mid if hasattr(book, 'last_mid') else 0
            if mid > 0:
                side = Side.SELL if q > 0 else Side.BUY
                book.post_order(side, mid, abs(q), 0)

        self._vol_buffer.clear()


# ─────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────

if __name__ == "__main__":
    config = StrategyConfig(
        symbol        = "NIFTY26MAYFUT",
        lot_size      = 75,
        max_inventory = 5,
    )
    strat = AvellanedaStoikovStrategy(
        config, gamma=0.001, kappa=1.5,
        min_spread=0.10, max_spread=10.0, open_mult=2.0
    )

    print("=== Dynamic min_spread by price level ===")
    for sym, price in [("SAIL", 150), ("CHOLAFIN", 1520),
                       ("NIFTY", 23880), ("APOLLOHOSP", 7000),
                       ("BAJAJ-AUTO", 9000)]:
        dms = strat._dynamic_min_spread(price)
        off = strat._flatten_offset(price)
        print(f"  {sym:<12} @ {price:>6}: "
              f"min_spread={dms:.3f}  flatten_offset={off:.3f}")

    mid   = 23880.0
    sigma = 2.5

    print("\n=== Formula diagnostics ===")
    for T, label in [(1.0, "Start of day"), (0.5, "Mid session"),
                     (0.05, "End of day")]:
        dms    = strat._dynamic_min_spread(mid)
        r      = strat._reservation_price(mid, inventory=0, sigma=sigma, T=T)
        spread = strat._optimal_spread(sigma=sigma, T=T, min_spread=dms)
        print(f"\n{label} (T={T:.2f}):")
        print(f"  dynamic_min_spread: {dms:.4f} pts")
        print(f"  spread:             {spread:.4f} pts")
        print(f"  bid:                {r - spread/2:.2f}")
        print(f"  ask:                {r + spread/2:.2f}")

    print("\n=== Inventory skewing (T=0.5) ===")
    T = 0.5
    for q, label in [(0, "flat"), (3, "long +3"), (-3, "short -3")]:
        r       = strat._reservation_price(mid, inventory=q, sigma=sigma, T=T)
        dms     = strat._dynamic_min_spread(mid)
        spread  = strat._optimal_spread(sigma=sigma, T=T, min_spread=dms)
        urgency = strat._urgency(q, T)
        print(f"  q={q:+d} ({label}): "
              f"urgency={urgency:.3f} | r={r:.2f} | "
              f"bid={r-spread/2:.2f} | ask={r+spread/2:.2f}")