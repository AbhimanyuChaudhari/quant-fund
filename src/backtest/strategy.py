from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
from src.backtest.order_book import SimulatedOrderBook, Fill
from typing import List


@dataclass
class StrategyConfig:
    """Base config all strategies share."""
    symbol:        str
    lot_size:      int    # NIFTY = 75, BANKNIFTY = 35 etc.
    max_inventory: int    # max lots to hold at once


class BaseStrategy(ABC):
    """
    Abstract base class for all backtesting strategies.

    Every strategy must implement:
        on_bar(bar, book) → called every second with latest bar + order book

    The engine calls on_bar() in a loop. The strategy posts/cancels
    orders on the book. The engine checks fills after each bar.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.bars_processed = 0

    @abstractmethod
    def on_bar(self, bar: pd.Series, book: SimulatedOrderBook) -> None:
        """
        Called once per bar (second). Strategy logic goes here.
        Post or cancel orders on the book based on current bar data.

        Args:
            bar:  pd.Series — current bar with all 33 features
            book: SimulatedOrderBook — post/cancel orders here
        """
        pass

    def on_fill(self, fill: Fill) -> None:
        """
        Optional — called when one of our orders gets filled.
        Override to track fills, update internal state etc.
        """
        pass

    def on_day_start(self, date: str) -> None:
        """Optional — called at the start of each trading day."""
        pass

    def on_day_end(self, date: str, book: SimulatedOrderBook) -> None:
        """Optional — called at end of day. Default: cancel all open orders."""
        book.cancel_all()
