"""
Order Manager

Paper mode:  logs orders to file, simulates fills
Live mode:   sends real orders to Zerodha API

Switch between modes with PAPER_TRADING flag.
Everything else stays the same.
"""

import logging
import time
import csv
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING   = "PENDING"
    FILLED    = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED  = "REJECTED"


class OrderType(Enum):
    LIMIT  = "LIMIT"
    MARKET = "MARKET"


@dataclass
class Order:
    order_id:    str
    symbol:      str
    side:        str          # BUY or SELL
    lots:        int
    price:       float        # limit price
    order_type:  str          = "LIMIT"
    status:      str          = "PENDING"
    fill_price:  float        = 0.0
    fill_time:   str          = ""
    paper:       bool         = True
    zerodha_id:  str          = ""   # real order ID from Zerodha


class OrderManager:
    """
    Handles order placement, cancellation and tracking.

    Paper mode (PAPER_TRADING=True):
        - Orders logged to logs/orders/YYYY-MM-DD.csv
        - Fills simulated immediately at limit price
        - No Zerodha API calls

    Live mode (PAPER_TRADING=False):
        - Real orders sent to Zerodha via Kite API
        - Order status tracked via Zerodha order updates
        - Fills reported back via on_fill callback

    Usage:
        om = OrderManager(kite, portfolio, paper_trading=True)
        order_id = om.place_limit_order('NIFTY26MAYFUT', 'BUY', 1, 23880.0)
        om.cancel_order(order_id)
    """

    def __init__(self, kite=None,
                 portfolio=None,
                 paper_trading: bool = True):
        self.kite          = kite
        self.portfolio     = portfolio
        self.paper_trading = paper_trading
        self.orders:       dict[str, Order] = {}
        self._order_count  = 0

        # Log file
        log_dir = Path("logs/orders")
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.csv"
        self._init_log()

        mode = "PAPER" if paper_trading else "LIVE"
        logger.info(f"OrderManager initialized — {mode} mode")

    # ─────────────────────────────────────
    # Order placement
    # ─────────────────────────────────────
    def place_limit_order(self, symbol: str, side: str,
                          lots: int, price: float) -> Optional[str]:
        """
        Place a limit order.

        Returns:
            order_id if successful, None if failed
        """
        self._order_count += 1
        order_id = f"{'P' if self.paper_trading else 'L'}" \
                   f"{datetime.now().strftime('%H%M%S')}" \
                   f"{self._order_count:04d}"

        order = Order(
            order_id   = order_id,
            symbol     = symbol,
            side       = side,
            lots       = lots,
            price      = price,
            paper      = self.paper_trading,
        )

        if self.paper_trading:
            self._paper_fill(order)
        else:
            self._live_place(order)

        self.orders[order_id] = order
        self._log_order(order)
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Returns:
            True if cancelled, False if not found or already filled
        """
        order = self.orders.get(order_id)
        if not order or order.status != OrderStatus.PENDING.value:
            return False

        if self.paper_trading:
            order.status = OrderStatus.CANCELLED.value
            logger.debug(f"[PAPER] Cancelled {order_id}")
            return True
        else:
            return self._live_cancel(order)

    def cancel_all(self, symbol: Optional[str] = None):
        """Cancel all open orders, optionally filtered by symbol."""
        cancelled = 0
        for order_id, order in self.orders.items():
            if order.status == OrderStatus.PENDING.value:
                if symbol is None or order.symbol == symbol:
                    self.cancel_order(order_id)
                    cancelled += 1
        if cancelled:
            logger.info(f"Cancelled {cancelled} open orders")

    # ─────────────────────────────────────
    # Paper trading
    # ─────────────────────────────────────
    def _paper_fill(self, order: Order):
        """
        Simulate immediate fill at limit price.
        In reality fills depend on market — this is optimistic.
        For more realistic simulation, use the backtester's
        order book fill logic.
        """
        order.status     = OrderStatus.FILLED.value
        order.fill_price = order.price
        order.fill_time  = datetime.now().strftime("%H:%M:%S")

        logger.info(
            f"[PAPER] FILL {order.side} {order.lots}L "
            f"{order.symbol} @ {order.fill_price:.2f}"
        )

        # Update portfolio
        if self.portfolio:
            self.portfolio.on_fill(
                symbol   = order.symbol,
                side     = order.side,
                lots     = order.lots,
                price    = order.fill_price,
                order_id = order.order_id,
            )

    # ─────────────────────────────────────
    # Live trading
    # ─────────────────────────────────────
    def _live_place(self, order: Order):
        """Send real order to Zerodha."""
        if not self.kite:
            logger.error("No Kite client — cannot place live order")
            order.status = OrderStatus.REJECTED.value
            return

        try:
            zerodha_id = self.kite.place_order(
                variety          = self.kite.VARIETY_REGULAR,
                exchange         = "NFO",
                tradingsymbol    = order.symbol,
                transaction_type = self.kite.TRANSACTION_TYPE_BUY
                                   if order.side == "BUY"
                                   else self.kite.TRANSACTION_TYPE_SELL,
                quantity         = order.lots,
                product          = self.kite.PRODUCT_MIS,
                order_type       = self.kite.ORDER_TYPE_LIMIT,
                price            = order.price,
            )
            order.zerodha_id = str(zerodha_id)
            order.status     = OrderStatus.PENDING.value
            logger.info(
                f"[LIVE] Placed {order.side} {order.lots}L "
                f"{order.symbol} @ {order.price:.2f} "
                f"| Zerodha ID: {zerodha_id}"
            )
        except Exception as e:
            order.status = OrderStatus.REJECTED.value
            logger.error(f"[LIVE] Order rejected: {e}")

    def _live_cancel(self, order: Order) -> bool:
        """Cancel real order on Zerodha."""
        if not self.kite or not order.zerodha_id:
            return False
        try:
            self.kite.cancel_order(
                variety  = self.kite.VARIETY_REGULAR,
                order_id = order.zerodha_id,
            )
            order.status = OrderStatus.CANCELLED.value
            logger.info(f"[LIVE] Cancelled {order.zerodha_id}")
            return True
        except Exception as e:
            logger.error(f"[LIVE] Cancel failed: {e}")
            return False

    # ─────────────────────────────────────
    # Logging
    # ─────────────────────────────────────
    def _init_log(self):
        if not self._log_path.exists():
            with open(self._log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "order_id", "symbol", "side",
                    "lots", "price", "status", "fill_price",
                    "fill_time", "paper", "zerodha_id"
                ])

    def _log_order(self, order: Order):
        with open(self._log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                order.order_id, order.symbol, order.side,
                order.lots, order.price, order.status,
                order.fill_price, order.fill_time,
                order.paper, order.zerodha_id,
            ])

    def open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        return [o for o in self.orders.values()
                if o.status == OrderStatus.PENDING.value
                and (symbol is None or o.symbol == symbol)]

    def summary(self) -> dict:
        statuses = {}
        for o in self.orders.values():
            statuses[o.status] = statuses.get(o.status, 0) + 1
        return {
            "total_orders": len(self.orders),
            "by_status":    statuses,
            "paper_mode":   self.paper_trading,
        }


if __name__ == "__main__":
    from unittest.mock import MagicMock

    print("=== Order Manager Tests ===\n")

    portfolio = MagicMock()
    om = OrderManager(portfolio=portfolio, paper_trading=True)

    # Test 1: place and fill
    oid = om.place_limit_order("NIFTY26MAYFUT", "BUY", 1, 23880.0)
    print(f"Placed order: {oid}")
    order = om.orders[oid]
    print(f"Status: {order.status} | Fill: {order.fill_price}")
    assert order.status == "FILLED"

    # Test 2: cancel pending (need to create pending manually)
    order2 = Order(order_id="TEST001", symbol="NIFTY26MAYFUT",
                   side="SELL", lots=1, price=23885.0)
    order2.status = "PENDING"
    om.orders["TEST001"] = order2
    cancelled = om.cancel_order("TEST001")
    print(f"Cancel result: {cancelled} | Status: {order2.status}")
    assert cancelled

    print(f"\nSummary: {om.summary()}")
    print("\n✓ Order manager tests passed")