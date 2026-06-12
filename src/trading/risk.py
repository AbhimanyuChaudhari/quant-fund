"""
Risk Management System

Hard limits that cannot be overridden by any strategy.
Every order must pass risk checks before being sent to broker.
"""

import time
import logging
from dataclasses import dataclass
from datetime import datetime, date
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class RiskViolation(Enum):
    DAILY_LOSS_LIMIT   = "Daily loss limit breached"
    MAX_POSITION       = "Max position size exceeded"
    MAX_GROSS_EXPOSURE = "Max gross exposure exceeded"
    MAX_ORDER_SIZE     = "Max order size exceeded"
    KILL_SWITCH        = "Kill switch activated"
    MARKET_CLOSED      = "Market is closed"
    COOLDOWN           = "In cooldown period after loss limit"


@dataclass
class RiskConfig:
    daily_loss_limit:   float = 10_000.0   # hard stop in rupees
    drawdown_warn:      float = 7_000.0    # warn at this level
    max_position_lots:  int   = 2          # per symbol
    max_gross_exposure: int   = 4          # total across all symbols
    max_order_lots:     int   = 2          # single order max
    cooldown_seconds:   int   = 300        # pause after loss limit
    market_open_utc:    int   = 13500      # 09:15 IST in UTC seconds
    market_close_utc:   int   = 55800      # 15:30 IST in UTC seconds


class RiskManager:
    """
    Central risk gate. All orders must pass check_order() first.

    Usage:
        risk = RiskManager(RiskConfig())
        ok, reason = risk.check_order(symbol, side, lots, portfolio)
        if ok:
            order_manager.place_order(...)
        else:
            logger.warning(f"Risk blocked: {reason}")
    """

    def __init__(self, config: RiskConfig):
        self.config          = config
        self.kill_switch     = False
        self.trading_date    = date.today()
        self._cooldown_until: Optional[float] = None

        # Daily counters — reset each morning
        self.daily_pnl       = 0.0
        self.daily_orders    = 0
        self.daily_fills     = 0
        self.peak_daily_pnl  = 0.0

    # ─────────────────────────────────────
    # Core risk gate
    # ─────────────────────────────────────
    def check_order(self, symbol: str, side: str,
                    lots: int, portfolio) -> tuple[bool, str]:
        """
        Run all risk checks for a proposed order.

        Returns:
            (True, '')       → order allowed
            (False, reason)  → order blocked
        """
        # 1. Kill switch — hard stop, no override
        if self.kill_switch:
            return False, RiskViolation.KILL_SWITCH.value

        # 2. Cooldown period after loss limit
        if self._cooldown_until and time.time() < self._cooldown_until:
            remaining = int(self._cooldown_until - time.time())
            return False, f"{RiskViolation.COOLDOWN.value} ({remaining}s left)"

        # 3. Market hours
        if not self._is_market_open():
            return False, RiskViolation.MARKET_CLOSED.value

        # 4. Daily loss limit
        if self.daily_pnl <= -self.config.daily_loss_limit:
            self._trigger_cooldown()
            return False, (f"{RiskViolation.DAILY_LOSS_LIMIT.value} "
                           f"(loss: Rs.{abs(self.daily_pnl):,.0f})")

        # 5. Single order size
        if lots > self.config.max_order_lots:
            return False, (f"{RiskViolation.MAX_ORDER_SIZE.value} "
                           f"({lots} > {self.config.max_order_lots})")

        # 6. Per-symbol position limit
        current_pos = abs(portfolio.get_position(symbol))
        projected   = current_pos + lots
        if projected > self.config.max_position_lots:
            return False, (f"{RiskViolation.MAX_POSITION.value} "
                           f"({projected} > {self.config.max_position_lots})")

        # 7. Total gross exposure
        gross = portfolio.gross_exposure() + lots
        if gross > self.config.max_gross_exposure:
            return False, (f"{RiskViolation.MAX_GROSS_EXPOSURE.value} "
                           f"({gross} > {self.config.max_gross_exposure})")

        # 8. Drawdown warning (non-blocking)
        if self.daily_pnl <= -self.config.drawdown_warn:
            logger.warning(
                f"Drawdown warning: Rs.{abs(self.daily_pnl):,.0f} "
                f"/ Rs.{self.config.daily_loss_limit:,.0f} limit"
            )

        self.daily_orders += 1
        return True, ""

    # ─────────────────────────────────────
    # State updates
    # ─────────────────────────────────────
    def update_pnl(self, pnl_change: float):
        """Call after every fill with incremental PnL change."""
        self.daily_pnl     += pnl_change
        self.peak_daily_pnl = max(self.peak_daily_pnl, self.daily_pnl)
        self.daily_fills   += 1

        if self.daily_pnl <= -self.config.daily_loss_limit:
            self._trigger_cooldown()
            logger.critical(
                f"DAILY LOSS LIMIT HIT: Rs.{abs(self.daily_pnl):,.0f}. "
                f"Cooldown {self.config.cooldown_seconds}s."
            )

    def on_day_start(self):
        """Reset all daily counters. Call at market open each day."""
        self.daily_pnl       = 0.0
        self.daily_orders    = 0
        self.daily_fills     = 0
        self.peak_daily_pnl  = 0.0
        self._cooldown_until = None
        self.trading_date    = date.today()
        logger.info(f"Risk reset for {self.trading_date}")

    def activate_kill_switch(self, reason: str = "Manual"):
        """Hard stop — no more trading until manually reset."""
        self.kill_switch = True
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

    def reset_kill_switch(self):
        self.kill_switch = False
        logger.info("Kill switch reset")

    def _trigger_cooldown(self):
        self._cooldown_until = time.time() + self.config.cooldown_seconds

    def _is_market_open(self) -> bool:
        now_utc_sec = int(time.time()) % 86400
        return (self.config.market_open_utc
                <= now_utc_sec
                <= self.config.market_close_utc)

    def status(self) -> dict:
        in_cooldown = bool(self._cooldown_until and
                           time.time() < self._cooldown_until)
        return {
            "kill_switch":  self.kill_switch,
            "daily_pnl":    round(self.daily_pnl, 2),
            "daily_orders": self.daily_orders,
            "daily_fills":  self.daily_fills,
            "loss_limit":   self.config.daily_loss_limit,
            "pct_used":     round(abs(min(self.daily_pnl, 0)) /
                            self.config.daily_loss_limit * 100, 1),
            "in_cooldown":  in_cooldown,
            "market_open":  self._is_market_open(),
        }


if __name__ == "__main__":
    # Quick sanity test
    from unittest.mock import MagicMock

    config = RiskConfig(daily_loss_limit=10_000, max_position_lots=2)
    risk   = RiskManager(config)

    # Mock portfolio
    portfolio = MagicMock()
    portfolio.get_position.return_value = 0
    portfolio.gross_exposure.return_value = 0

    print("=== Risk Manager Tests ===\n")

    # Test 1: normal order passes
    ok, reason = risk.check_order("NIFTY26MAYFUT", "BUY", 1, portfolio)
    print(f"Normal order:     ok={ok}  reason='{reason}'")
    assert ok

    # Test 2: order too large
    ok, reason = risk.check_order("NIFTY26MAYFUT", "BUY", 5, portfolio)
    print(f"Order too large:  ok={ok}  reason='{reason}'")
    assert not ok

    # Test 3: kill switch
    risk.activate_kill_switch("Test")
    ok, reason = risk.check_order("NIFTY26MAYFUT", "BUY", 1, portfolio)
    print(f"Kill switch:      ok={ok}  reason='{reason}'")
    assert not ok
    risk.reset_kill_switch()

    # Test 4: daily loss limit
    risk.update_pnl(-10_001)
    ok, reason = risk.check_order("NIFTY26MAYFUT", "BUY", 1, portfolio)
    print(f"Loss limit hit:   ok={ok}  reason='{reason}'")
    assert not ok

    print("\n✓ All risk tests passed")
    print(f"\nStatus: {risk.status()}")
