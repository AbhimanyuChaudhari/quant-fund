from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class Side(Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    OPEN      = "OPEN"
    FILLED    = "FILLED"
    CANCELLED = "CANCELLED"


@dataclass
class Order:
    order_id:  int
    side:      Side
    price:     float
    quantity:  int
    timestamp: int              # ts_sec when submitted
    status:    OrderStatus = OrderStatus.OPEN
    fill_ts:   Optional[int] = None


@dataclass
class Fill:
    order_id:  int
    side:      Side
    price:     float            # limit price (we are maker, we get our price)
    quantity:  int
    timestamp: int              # ts_sec when filled


class SimulatedOrderBook:
    """
    Simulated order book for backtesting market making strategies.

    Fill logic (simple — maker fills):
        BUY  limit @ P → filled if bar.low  <= P
        SELL limit @ P → filled if bar.high >= P

    As a market maker we always post limit orders.
    Fill price = our limit price (no slippage for maker orders).

    Usage:
        book = SimulatedOrderBook()
        bid_id = book.post_order(Side.BUY,  bid_price, qty, ts)
        ask_id = book.post_order(Side.SELL, ask_price, qty, ts)
        fills  = book.process_bar(bar)   ← call every bar
    """

    def __init__(self, max_inventory: int = 50):
        self.max_inventory  = max_inventory  # max net position in lots
        self.orders: dict[int, Order] = {}
        self.fills:  List[Fill]       = []
        self.inventory: int           = 0   # net position (+ long, - short)
        self._next_id: int            = 1

    # ─────────────────────────────────────
    # Order management
    # ─────────────────────────────────────

    def post_order(self, side: Side, price: float,
                   quantity: int, timestamp: int) -> Optional[int]:
        """
        Post a limit order. Returns order_id, or None if rejected.
        Rejects if posting would breach max_inventory limit.
        """
        # Inventory check before posting
        projected = self.inventory + (quantity if side == Side.BUY else -quantity)
        if abs(projected) > self.max_inventory:
            return None  # rejected — would breach inventory limit

        order = Order(
            order_id  = self._next_id,
            side      = side,
            price     = price,
            quantity  = quantity,
            timestamp = timestamp,
        )
        self.orders[self._next_id] = order
        self._next_id += 1
        return order.order_id

    def cancel_order(self, order_id: int) -> bool:
        """Cancel an open order. Returns True if cancelled, False if not found/already filled."""
        order = self.orders.get(order_id)
        if order and order.status == OrderStatus.OPEN:
            order.status = OrderStatus.CANCELLED
            return True
        return False

    def cancel_all(self):
        """Cancel all open orders — called at end of day or on risk breach."""
        for order in self.orders.values():
            if order.status == OrderStatus.OPEN:
                order.status = OrderStatus.CANCELLED

    # ─────────────────────────────────────
    # Fill simulation
    # ─────────────────────────────────────

    def process_bar(self, bar) -> List[Fill]:
        bar_fills = []
        ts   = int(bar["ts_sec"])
        low  = bar["low"]
        high = bar["high"]

        for order in self.orders.values():
            if order.status != OrderStatus.OPEN:
                continue

            filled = False
            if order.side == Side.BUY and low <= order.price:
                filled = True
            elif order.side == Side.SELL and high >= order.price:
                filled = True

            if filled:
                # ── Check inventory AFTER this fill would apply ────────────────
                projected = self.inventory + (
                    order.quantity if order.side == Side.BUY
                    else -order.quantity
                )
                if abs(projected) > self.max_inventory:
                    order.status = OrderStatus.CANCELLED
                    continue

                order.status  = OrderStatus.FILLED
                order.fill_ts = ts
                fill = Fill(
                    order_id  = order.order_id,
                    side      = order.side,
                    price     = order.price,
                    quantity  = order.quantity,
                    timestamp = ts,
                )
                self.fills.append(fill)
                bar_fills.append(fill)
                if order.side == Side.BUY:
                    self.inventory += order.quantity
                else:
                    self.inventory -= order.quantity

        return bar_fills

    # ─────────────────────────────────────
    # State queries
    # ─────────────────────────────────────

    def open_orders(self) -> List[Order]:
        """Return all open orders."""
        return [o for o in self.orders.values() if o.status == OrderStatus.OPEN]

    def open_bids(self) -> List[Order]:
        return [o for o in self.open_orders() if o.side == Side.BUY]

    def open_asks(self) -> List[Order]:
        return [o for o in self.open_orders() if o.side == Side.SELL]

    def has_open_bid(self) -> bool:
        return len(self.open_bids()) > 0

    def has_open_ask(self) -> bool:
        return len(self.open_asks()) > 0

    def reset(self):
        """Full reset — call between backtest runs."""
        self.orders    = {}
        self.fills     = []
        self.inventory = 0
        self._next_id  = 1


if __name__ == "__main__":
    # Quick unit test — no GCS needed
    import pandas as pd

    book = SimulatedOrderBook(max_inventory=10)

    # Post a bid at 23880 and ask at 23885
    bid_id = book.post_order(Side.BUY,  23880.0, 1, timestamp=1000)
    ask_id = book.post_order(Side.SELL, 23885.0, 1, timestamp=1000)
    print(f"Posted bid #{bid_id} @ 23880 | ask #{ask_id} @ 23885")
    print(f"Open orders: {len(book.open_orders())}")

    # Bar where neither fills (high=23883, low=23881 — doesn't touch our levels)
    bar1 = pd.Series({"ts_sec": 1001, "low": 23881.0, "high": 23883.0})
    fills = book.process_bar(bar1)
    print(f"\nBar1 (low=23881, high=23883) → fills: {len(fills)} | inventory: {book.inventory}")

    # Bar where bid fills (low=23879 crosses our 23880 bid)
    bar2 = pd.Series({"ts_sec": 1002, "low": 23879.0, "high": 23884.0})
    fills = book.process_bar(bar2)
    print(f"Bar2 (low=23879, high=23884) → fills: {len(fills)} | inventory: {book.inventory}")
    for f in fills:
        print(f"  FILLED: {f.side.value} {f.quantity} @ {f.price}")

    # Bar where ask fills (high=23886 crosses our 23885 ask)
    bar3 = pd.Series({"ts_sec": 1003, "low": 23882.0, "high": 23886.0})
    fills = book.process_bar(bar3)
    print(f"Bar3 (low=23882, high=23886) → fills: {len(fills)} | inventory: {book.inventory}")
    for f in fills:
        print(f"  FILLED: {f.side.value} {f.quantity} @ {f.price}")

    print(f"\nTotal fills: {len(book.fills)}")
    print(f"Final inventory: {book.inventory} (should be 0 — bought and sold 1)")
