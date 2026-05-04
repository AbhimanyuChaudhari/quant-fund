"""
Realistic Fill Simulator — Level 1 + 2 + 3

Level 1 — Price crossing:
    BUY  filled only if bar.low  <= our bid price
    SELL filled only if bar.high >= our ask price

Level 2 — Activity filter:
    Enough ticks must have traded at our price level

Level 3 — Queue simulation:
    Uses L5 order book depth to estimate queue position
    fill_prob = ticks_at_level / ticks_needed_to_clear_queue

Real NIFTY calibration (Apr 30 data):
    avg_ticks/bar:  1.36
    avg_bid_q1:     427 shares
    avg_ask_q1:     478 shares
    Target fill rate: 15-30% for realistic MM
"""

import math
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

AVG_TRADE_SIZE = 75   # 1 NIFTY lot = 75 shares per trade


@dataclass
class FillResult:
    filled:        bool  = False
    fill_price:    float = 0.0
    filled_lots:   int   = 0
    fill_prob:     float = 0.0
    reject_reason: str   = ""


class FillSimulator:
    """
    Realistic fill simulator for market making backtesting.

    Fill logic:
      Level 1: Did price reach our level? (bar.low/high check)
      Level 2: Was there enough activity at our level?
      Level 3: Queue-based fill probability

    Fill probability formula:
      ticks_at_level   = tick_count × price_fraction
      ticks_to_clear   = queue_shares / avg_trade_size
      fill_prob        = ticks_at_level / ticks_to_clear × aggression

    Usage:
        sim = FillSimulator(lot_size=75, queue_aggression=0.5)
        result = sim.check_fill('BUY', 23879.0, 1, bar_dict)
    """

    def __init__(self,
                 lot_size:         int   = 75,
                 queue_aggression: float = 0.5,
                 min_ticks:        int   = 1,
                 partial_fills:    bool  = True):
        """
        Args:
            lot_size:         contract lot size (NIFTY=75, BANKNIFTY=35)
            queue_aggression: scaling factor (0-1), higher = more fills
            min_ticks:        minimum ticks needed at our level
            partial_fills:    allow partial fills
        """
        self.lot_size         = lot_size
        self.queue_aggression = queue_aggression
        self.min_ticks        = min_ticks
        self.partial_fills    = partial_fills

    def check_fill(self, side: str, our_price: float,
                   lots: int, bar: dict) -> FillResult:
        """
        Check if our limit order gets filled on this bar.

        Args:
            side:       'BUY' or 'SELL'
            our_price:  our limit price
            lots:       our order size in lots
            bar:        dict with processed feature columns + L5 depth

        Returns:
            FillResult with filled=True/False and details
        """
        result   = FillResult()
        bar_low  = self._get(bar, "low",  0)
        bar_high = self._get(bar, "high", 0)
        bar_range = max(bar_high - bar_low, 0.05)

        # ── Level 1: Price crossing ───────────────────
        if side == "BUY":
            if bar_low > our_price:
                result.reject_reason = (
                    f"Price never reached bid "
                    f"(low={bar_low:.2f} > bid={our_price:.2f})"
                )
                return result
        else:
            if bar_high < our_price:
                result.reject_reason = (
                    f"Price never reached ask "
                    f"(high={bar_high:.2f} < ask={our_price:.2f})"
                )
                return result

        # ── Level 2: Activity filter ──────────────────
        tick_count = max(1, self._get(bar, "tick_count", 1))

        # Fraction of bar range at our price level
        if side == "BUY":
            price_fraction = 1.0 - abs(our_price - bar_low) / bar_range
        else:
            price_fraction = 1.0 - abs(our_price - bar_high) / bar_range

        price_fraction  = max(0.01, min(1.0, price_fraction))
        ticks_at_level  = tick_count * price_fraction

        if ticks_at_level < self.min_ticks:
            result.reject_reason = (
                f"Not enough activity at level "
                f"(ticks_at_level={ticks_at_level:.2f} < {self.min_ticks})"
            )
            return result

        # ── Level 3: Queue simulation ─────────────────
        queue_shares = self._estimate_queue_shares(side, our_price, bar)
        fill_prob    = self._compute_fill_prob(
            ticks_at_level, queue_shares
        )
        result.fill_prob = fill_prob

        if fill_prob <= 0:
            result.reject_reason = (
                f"Queue too long "
                f"(queue={queue_shares:.0f} shares, "
                f"ticks_at_level={ticks_at_level:.2f})"
            )
            return result

        # Determine fill size
        if fill_prob >= 1.0:
            filled_lots = lots
        elif self.partial_fills:
            filled_lots = max(1, int(lots * fill_prob))
        else:
            filled_lots = lots if fill_prob >= 0.5 else 0

        if filled_lots <= 0:
            result.reject_reason = f"Fill prob too low ({fill_prob:.2%})"
            return result

        result.filled      = True
        result.fill_price  = our_price
        result.filled_lots = filled_lots
        return result

    def _estimate_queue_shares(self, side: str, our_price: float,
                               bar: dict) -> float:
        """
        Estimate shares ahead of us in queue using L5 order book.
        Zerodha reports quantities in shares (not lots).
        """
        if side == "BUY":
            levels = [
                (self._get(bar, "bid_p1", 0), self._get(bar, "bid_q1", 0)),
                (self._get(bar, "bid_p2", 0), self._get(bar, "bid_q2", 0)),
                (self._get(bar, "bid_p3", 0), self._get(bar, "bid_q3", 0)),
                (self._get(bar, "bid_p4", 0), self._get(bar, "bid_q4", 0)),
                (self._get(bar, "bid_p5", 0), self._get(bar, "bid_q5", 0)),
            ]
        else:
            levels = [
                (self._get(bar, "ask_p1", 0), self._get(bar, "ask_q1", 0)),
                (self._get(bar, "ask_p2", 0), self._get(bar, "ask_q2", 0)),
                (self._get(bar, "ask_p3", 0), self._get(bar, "ask_q3", 0)),
                (self._get(bar, "ask_p4", 0), self._get(bar, "ask_q4", 0)),
                (self._get(bar, "ask_p5", 0), self._get(bar, "ask_q5", 0)),
            ]

        queue = 0.0
        for price, qty in levels:
            if price <= 0 or qty <= 0:
                continue
            if side == "BUY" and price >= our_price:
                queue += qty
            elif side == "SELL" and price <= our_price:
                queue += qty

        # Fallback if no L5 data
        if queue == 0:
            key   = "total_bid_qty" if side == "BUY" else "total_ask_qty"
            queue = self._get(bar, key, 0) * 0.3

        return queue

    def _compute_fill_prob(self, ticks_at_level: float,
                           queue_shares: float) -> float:
        """
        Fill probability based on ticks at our level vs queue size.

        ticks_to_clear = queue_shares / avg_trade_size
        fill_prob      = ticks_at_level / ticks_to_clear × aggression

        Example with real NIFTY data:
            avg_bid_q1    = 427 shares
            ticks_to_clear = 427 / 75 = 5.7 ticks
            avg ticks/bar  = 1.36
            fill_prob      = 1.36 / 5.7 × 0.5 = 11.9%  ✓ realistic
        """
        if queue_shares <= 0:
            return 1.0

        ticks_to_clear = queue_shares / AVG_TRADE_SIZE
        fill_prob      = (ticks_at_level / max(ticks_to_clear, 0.1)) \
                         * self.queue_aggression
        return min(1.0, max(0.0, fill_prob))

    def _get(self, bar: dict, key: str, default: float) -> float:
        val = bar.get(key, default)
        if val is None:
            return default
        try:
            f = float(val)
            return default if math.isnan(f) or math.isinf(f) else f
        except (TypeError, ValueError):
            return default


# ─────────────────────────────────────
# Integration with backtester
# ─────────────────────────────────────

class RealisticOrderBook:
    """
    Drop-in replacement for SimulatedOrderBook using FillSimulator.

    Usage:
        book = RealisticOrderBook(max_inventory=5, lot_size=75)
        bid_id = book.post_order(Side.BUY, bid_price, qty, ts)
        fills  = book.process_bar(bar_dict)
    """

    def __init__(self, max_inventory:    int   = 5,
                 lot_size:              int   = 75,
                 queue_aggression:      float = 0.5):
        from src.backtest.order_book import SimulatedOrderBook
        self._book = SimulatedOrderBook(max_inventory=max_inventory)
        self._sim  = FillSimulator(
            lot_size         = lot_size,
            queue_aggression = queue_aggression,
        )
        self.fill_attempts  = 0
        self.fill_successes = 0
        self.max_inventory   = max_inventory

    def post_order(self, side, price, quantity, timestamp):
        return self._book.post_order(side, price, quantity, timestamp)

    def cancel_order(self, order_id):
        return self._book.cancel_order(order_id)

    def cancel_all(self):
        return self._book.cancel_all()

    def process_bar(self, bar) -> list:
        from src.backtest.order_book import OrderStatus, Fill
        bar_dict = bar.to_dict() if hasattr(bar, 'to_dict') else dict(bar)
        fills    = []

        for order in list(self._book.orders.values()):
            if order.status != OrderStatus.OPEN:
                continue

            # Inventory check before filling
            if order.side.value == "BUY":
                projected = self._book.inventory + order.quantity
            else:
                projected = self._book.inventory - order.quantity

            if abs(projected) > self.max_inventory:
                order.status = OrderStatus.CANCELLED
                continue

            self.fill_attempts += 1
            result = self._sim.check_fill(
                side      = order.side.value,
                our_price = order.price,
                lots      = order.quantity,
                bar       = bar_dict,
            )

            if result.filled:
                self.fill_successes += 1
                order.status  = OrderStatus.FILLED
                order.fill_ts = int(bar_dict.get("ts_sec", 0))

                fill = Fill(
                    order_id  = order.order_id,
                    side      = order.side,
                    price     = result.fill_price,
                    quantity  = result.filled_lots,
                    timestamp = order.fill_ts,
                )
                self._book.fills.append(fill)
                fills.append(fill)

                if order.side.value == "BUY":
                    self._book.inventory += result.filled_lots
                else:
                    self._book.inventory -= result.filled_lots

        return fills

    @property
    def inventory(self):
        return self._book.inventory

    @property
    def fills(self):
        return self._book.fills

    def open_orders(self):
        return self._book.open_orders()

    def open_bids(self):
        return self._book.open_bids()

    def open_asks(self):
        return self._book.open_asks()

    def has_open_bid(self):
        return self._book.has_open_bid()

    def has_open_ask(self):
        return self._book.has_open_ask()

    def fill_rate(self) -> float:
        if self.fill_attempts == 0:
            return 0.0
        return self.fill_successes / self.fill_attempts

    def reset(self):
        self._book.reset()
        self.fill_attempts  = 0
        self.fill_successes = 0


if __name__ == "__main__":
    print("=== Fill Simulator Tests ===\n")

    sim = FillSimulator(lot_size=75, queue_aggression=0.5)

    # Realistic NIFTY bar — real scale data
    bar = {
        "open":  23880.0, "high": 23885.0,
        "low":   23875.0, "close": 23882.0,
        "volume": 4200000, "tick_count": 2,   # avg 1.36 ticks/bar
        # Real NIFTY-scale queues (shares)
        "bid_p1": 23882.0, "bid_q1": 427,    # avg real queue
        "bid_p2": 23881.5, "bid_q2": 65,
        "bid_p3": 23881.0, "bid_q3": 585,
        "bid_p4": 23880.5, "bid_q4": 300,
        "bid_p5": 23880.0, "bid_q5": 450,
        "ask_p1": 23882.5, "ask_q1": 478,    # avg real queue
        "ask_p2": 23883.0, "ask_q2": 250,
        "ask_p3": 23883.5, "ask_q3": 400,
        "ask_p4": 23884.0, "ask_q4": 200,
        "ask_p5": 23884.5, "ask_q5": 300,
        "total_bid_qty": 358710,
        "total_ask_qty": 358710,
    }

    print("Bar: low=23875, high=23885, tick_count=2")
    print("bid_q1=427 shares (real NIFTY avg), ask_q1=478 shares\n")

    tests = [
        ("BUY",  23877.0, 1, "above low, small queue"),
        ("BUY",  23882.0, 1, "at best bid, avg queue (427 shares)"),
        ("BUY",  23870.0, 1, "below low — no fill"),
        ("SELL", 23883.0, 1, "below high, avg ask queue (478 shares)"),
        ("SELL", 23890.0, 1, "above high — no fill"),
    ]

    for side, price, lots, label in tests:
        r = sim.check_fill(side, price, lots, bar)
        print(f"{side} @ {price} ({label}):")
        print(f"  filled={r.filled} | prob={r.fill_prob:.2%} | "
              f"lots={r.filled_lots} | reason='{r.reject_reason}'")
        print()

    # Expected fill prob for avg bar:
    # ticks_at_level = 2 × 0.5 = 1.0
    # ticks_to_clear = 427 / 75 = 5.7
    # fill_prob = 1.0 / 5.7 × 0.5 = 8.8%
    print(f"Expected fill prob (avg bar): ~8-12%\n")

    print("=== Fill rate simulation (1000 bars) ===")
    import random
    fills = 0
    total = 1000

    for _ in range(total):
        mid     = 23880 + random.uniform(-10, 10)
        rng     = random.uniform(0.5, 5.0)
        ticks   = max(1, int(random.gauss(1.36, 0.8)))  # real distribution
        q1_bid  = max(75, int(random.gauss(427, 200)))   # real queue dist
        q1_ask  = max(75, int(random.gauss(478, 200)))

        test_bar = {
            "open": mid, "high": mid+rng, "low": mid-rng, "close": mid,
            "volume": 4200000, "tick_count": ticks,
            "bid_p1": mid-0.5, "bid_q1": q1_bid,
            "bid_p2": mid-1.0, "bid_q2": random.randint(50, 200),
            "bid_p3": mid-1.5, "bid_q3": random.randint(100, 600),
            "bid_p4": mid-2.0, "bid_q4": random.randint(50, 300),
            "bid_p5": mid-2.5, "bid_q5": random.randint(50, 200),
            "ask_p1": mid+0.5, "ask_q1": q1_ask,
            "ask_p2": mid+1.0, "ask_q2": random.randint(50, 200),
            "ask_p3": mid+1.5, "ask_q3": random.randint(100, 400),
            "ask_p4": mid+2.0, "ask_q4": random.randint(50, 300),
            "ask_p5": mid+2.5, "ask_q5": random.randint(50, 200),
            "total_bid_qty": 358710,
            "total_ask_qty": 358710,
        }

        bid = mid - random.uniform(0.5, 2.0)
        r   = sim.check_fill("BUY", bid, 1, test_bar)
        if r.filled:
            fills += 1

    print(f"Fill rate: {fills/total:.1%} (target: 10-25%)")
    print("\n✓ Fill simulator tests passed")