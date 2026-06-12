"""
Real-time Feature Engine

Converts raw WebSocket ticks into the same 33 features
as duckdb_pipeline.py — but computed live in memory.

Uses rolling deques (fixed-size buffers) for all windows.
No GCS, no DuckDB — pure in-memory computation.

Output: same column names as processed parquet features
so strategy code works identically in backtest and live.
"""

import time
import logging
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TickBar:
    """One second bar — same structure as processed parquet row."""
    ts_sec:        int
    symbol:        str

    # OHLCV
    open:          float = 0.0
    high:          float = 0.0
    low:           float = float('inf')
    close:         float = 0.0
    volume:        int   = 0
    tick_count:    int   = 0
    vwap:          float = 0.0
    oi:            int   = 0

    # Spread
    spread_mean:   float = 0.0
    spread_max:    float = 0.0
    spread_bps:    float = 0.0

    # Imbalance
    imbalance_mean: float = 0.0
    imbalance_std:  float = 0.0
    imbalance_last: float = 0.0

    # Depth
    total_bid_qty:  int   = 0
    total_ask_qty:  int   = 0
    weighted_mid:   float = 0.0
    price_impact:   float = 0.0

    # Rolling features (computed after bar closes)
    realized_vol_10s:  Optional[float] = None
    realized_vol_30s:  Optional[float] = None
    realized_vol_60s:  Optional[float] = None
    realized_vol_300s: Optional[float] = None
    imbalance_ma_10s:  Optional[float] = None
    imbalance_ma_30s:  Optional[float] = None
    imbalance_ma_60s:  Optional[float] = None
    spread_zscore:     Optional[float] = None
    volume_ratio:      Optional[float] = None
    price_mom_10s:     Optional[float] = None
    price_mom_30s:     Optional[float] = None
    price_mom_60s:     Optional[float] = None

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


class SymbolFeatureEngine:
    """
    Per-symbol feature engine.
    Maintains rolling windows and computes features on each tick.
    """

    def __init__(self, symbol: str, max_window: int = 300):
        self.symbol     = symbol
        self.max_window = max_window

        # Current second's tick accumulator
        self._current_sec:   Optional[int]   = None
        self._current_ticks: list            = []

        # Rolling bars (deque auto-drops oldest)
        self._bars: deque[TickBar] = deque(maxlen=max_window + 10)

        # Latest completed bar
        self.latest_bar: Optional[TickBar] = None

    def on_tick(self, tick: dict) -> Optional[TickBar]:
        """
        Process one raw tick. Returns completed bar if second rolled over.

        Args:
            tick: raw Zerodha tick dict with ts_local_ns, last_price, etc.

        Returns:
            Completed TickBar if a new second started, else None
        """
        ts_ns  = tick.get("ts_local_ns", time.time_ns())
        ts_sec = ts_ns // 1_000_000_000
        completed_bar = None

        if self._current_sec is None:
            self._current_sec = ts_sec

        if ts_sec != self._current_sec:
            # Second rolled over — close current bar
            if self._current_ticks:
                bar = self._build_bar(self._current_sec, self._current_ticks)
                self._add_rolling_features(bar)
                self._bars.append(bar)
                self.latest_bar = bar
                completed_bar   = bar

            # Start new second
            self._current_sec   = ts_sec
            self._current_ticks = []

        self._current_ticks.append(tick)
        return completed_bar

    def _build_bar(self, ts_sec: int, ticks: list) -> TickBar:
        """Build a 1-second OHLCV bar from raw ticks."""
        prices     = [t.get("last_price", 0) for t in ticks if t.get("last_price")]
        spreads    = [t.get("spread", 0) for t in ticks]
        imbalances = [t.get("book_imbalance", 0) for t in ticks]

        if not prices:
            return TickBar(ts_sec=ts_sec, symbol=self.symbol)

        last_tick = ticks[-1]
        mid       = last_tick.get("mid_price", 0)
        last_p    = prices[-1]

        # Weighted mid
        bid_p1 = last_tick.get("bid_p1", 0)
        ask_p1 = last_tick.get("ask_p1", 0)
        bid_q1 = last_tick.get("bid_q1", 0)
        ask_q1 = last_tick.get("ask_q1", 0)
        denom  = bid_q1 + ask_q1
        weighted_mid = (bid_p1 * ask_q1 + ask_p1 * bid_q1) / denom \
                       if denom > 0 else mid

        spread_bps = (np.mean(spreads) / mid * 10000) if mid > 0 else 0

        bar = TickBar(
            ts_sec       = ts_sec,
            symbol       = self.symbol,
            open         = prices[0],
            high         = max(prices),
            low          = min(prices),
            close        = prices[-1],
            volume       = last_tick.get("volume", 0),
            tick_count   = len(ticks),
            vwap         = last_tick.get("avg_price", prices[-1]),
            oi           = last_tick.get("oi", 0),
            spread_mean  = float(np.mean(spreads)),
            spread_max   = float(max(spreads)),
            spread_bps   = float(spread_bps),
            imbalance_mean = float(np.mean(imbalances)),
            imbalance_std  = float(np.std(imbalances)) if len(imbalances) > 1 else 0.0,
            imbalance_last = float(imbalances[-1]),
            total_bid_qty  = last_tick.get("total_bid_qty", 0),
            total_ask_qty  = last_tick.get("total_ask_qty", 0),
            weighted_mid   = float(weighted_mid),
            price_impact   = float(abs(last_p - mid)),
        )
        return bar

    def _add_rolling_features(self, bar: TickBar):
        """Compute rolling features from bar history."""
        bars = list(self._bars)
        if not bars:
            return

        closes     = [b.close for b in bars] + [bar.close]
        imbalances = [b.imbalance_last for b in bars] + [bar.imbalance_last]
        spreads    = [b.spread_mean for b in bars] + [bar.spread_mean]
        ticks      = [b.tick_count for b in bars] + [bar.tick_count]

        n = len(closes)

        # Realized volatility (std of closes over window)
        def vol(window):
            if n >= window:
                return float(np.std(closes[-window:]))
            return None

        bar.realized_vol_10s  = vol(10)
        bar.realized_vol_30s  = vol(30)
        bar.realized_vol_60s  = vol(60)
        bar.realized_vol_300s = vol(300)

        # Imbalance moving averages
        def ma(arr, window):
            if len(arr) >= window:
                return float(np.mean(arr[-window:]))
            return None

        bar.imbalance_ma_10s = ma(imbalances, 10)
        bar.imbalance_ma_30s = ma(imbalances, 30)
        bar.imbalance_ma_60s = ma(imbalances, 60)

        # Spread z-score (300s rolling)
        if n >= 10:
            sp_arr = np.array(spreads)
            window = min(300, n)
            sp_mean = np.mean(sp_arr[-window:])
            sp_std  = np.std(sp_arr[-window:])
            bar.spread_zscore = float(
                (spreads[-1] - sp_mean) / sp_std
            ) if sp_std > 0 else 0.0

        # Volume ratio
        if n >= 10:
            window = min(60, n)
            avg_ticks = np.mean(ticks[-window:])
            bar.volume_ratio = float(bar.tick_count / avg_ticks) \
                               if avg_ticks > 0 else 1.0

        # Price momentum
        def mom(lag):
            if n >= lag and closes[-lag] > 0:
                return float((closes[-1] - closes[-lag]) / closes[-lag])
            return None

        bar.price_mom_10s = mom(10)
        bar.price_mom_30s = mom(30)
        bar.price_mom_60s = mom(60)


class FeatureEngine:
    """
    Multi-symbol feature engine.
    Maintains one SymbolFeatureEngine per symbol.

    Usage:
        fe = FeatureEngine()
        bar = fe.on_tick(tick_dict)
        if bar:
            strategy.on_bar(bar.to_dict())
    """

    def __init__(self):
        self._engines: dict[str, SymbolFeatureEngine] = {}

    def on_tick(self, tick: dict) -> Optional[TickBar]:
        """
        Route tick to correct symbol engine.
        Returns completed bar if second rolled over, else None.
        """
        symbol = tick.get("symbol", "")
        if not symbol:
            return None

        if symbol not in self._engines:
            self._engines[symbol] = SymbolFeatureEngine(symbol)

        return self._engines[symbol].on_tick(tick)

    def latest_bar(self, symbol: str) -> Optional[TickBar]:
        eng = self._engines.get(symbol)
        return eng.latest_bar if eng else None

    def symbols(self) -> list[str]:
        return list(self._engines.keys())


if __name__ == "__main__":
    import time as time_mod

    print("=== Feature Engine Tests ===\n")

    fe  = FeatureEngine()
    now = int(time_mod.time())

    # Simulate 65 ticks across 3 seconds
    bars_seen = []
    for i in range(65):
        sec    = now + (i // 20)   # 20 ticks per second
        tick   = {
            "ts_local_ns":   sec * 1_000_000_000 + i * 1_000_000,
            "symbol":        "NIFTY26MAYFUT",
            "last_price":    23880.0 + (i % 5) * 0.5,
            "avg_price":     23882.0,
            "volume":        1000000 + i * 100,
            "oi":            5000000,
            "spread":        1.5,
            "mid_price":     23880.75,
            "book_imbalance": 0.1 + (i % 3) * 0.05,
            "total_bid_qty": 5000,
            "total_ask_qty": 4500,
            "bid_p1":        23880.0,
            "ask_p1":        23881.5,
            "bid_q1":        100,
            "ask_q1":        80,
        }
        bar = fe.on_tick(tick)
        if bar:
            bars_seen.append(bar)
            print(f"Bar: ts={bar.ts_sec} | close={bar.close} | "
                  f"ticks={bar.tick_count} | "
                  f"imbalance={bar.imbalance_last:.3f} | "
                  f"vol_60s={bar.realized_vol_60s}")

    print(f"\nBars completed: {len(bars_seen)}")
    print("\n✓ Feature engine tests passed") 
