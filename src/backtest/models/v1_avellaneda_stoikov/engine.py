"""
Backtest Engine — wires data loader, order book, strategy, and metrics together.

Flow:
    data_loader  → iter_bars()
        ↓
    engine       → feeds each bar to strategy
        ↓
    strategy     → posts/cancels orders on book
        ↓
    order_book   → checks fills
        ↓
    metrics      → tracks PnL, Sharpe, drawdown
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Type

from src.backtest.data_loader import iter_bars
from src.backtest.order_book import SimulatedOrderBook, Fill
from src.backtest.models.v1_avellaneda_stoikov.strategy import BaseStrategy, StrategyConfig
from src.backtest.metrics import compute_metrics, print_metrics, BacktestMetrics
from src.backtest.transaction_costs import TransactionCosts


@dataclass
class BacktestConfig:
    symbol:        str
    start:         str          # 'YYYY-MM-DD'
    end:           str          # 'YYYY-MM-DD'
    lot_size:      int  = 75    # NIFTY=75, BANKNIFTY=35
    max_inventory: int  = 5     # max lots at any time
    market_hours:  bool = True  # filter to 09:15-15:30 IST


class BacktestEngine:
    """
    Core simulation engine.

    Usage:
        config   = BacktestConfig('NIFTY26MAYFUT', '2026-04-30', '2026-04-30')
        strategy = AvellanedaStoikovStrategy(strat_config, gamma=0.1)
        engine   = BacktestEngine(config, strategy)
        metrics  = engine.run()
    """

    def __init__(self, config: BacktestConfig, strategy: BaseStrategy, order_book=None):
        self.config   = config
        self.strategy = strategy
        self.book = order_book or SimulatedOrderBook(max_inventory=config.max_inventory)
        self.tc       = TransactionCosts(lot_size=config.lot_size)

        self._all_fills:   list[Fill] = []
        self._current_day: str        = ""

    def run(self) -> BacktestMetrics:
        """
        Run the full backtest. Returns metrics.
        """
        print(f"\nRunning backtest: {self.config.symbol} "
              f"| {self.config.start} → {self.config.end}")
        print(f"Strategy: {self.strategy.__class__.__name__}")
        print(f"Max inventory: {self.config.max_inventory} lots\n")

        bars_seen = 0

        for bar in iter_bars(self.config.symbol,
                             self.config.start,
                             self.config.end,
                             self.config.market_hours):

            # ── Day boundary handling ──────────────────
            bar_date = str(bar.get("ts_ist", "")) [:10] if "ts_ist" in bar.index \
                else str(bar.get("ts_sec", ""))[:10] if "ts_sec" in bar.index else ""  # 'YYYY-MM-DD'
            if bar_date != self._current_day:
                if self._current_day:
                    # End of previous day
                    self.strategy.on_day_end(self._current_day, self.book)

                # Start of new day
                self._current_day = bar_date
                self.strategy.on_day_start(bar_date)
                print(f"  Day: {bar_date}")

            # ── Feed bar to strategy ───────────────────
            self.strategy.on_bar(bar, self.book)

            # ── Process fills ──────────────────────────
            fills = self.book.process_bar(bar)
            for fill in fills:
                self.strategy.on_fill(fill)
                self._all_fills.append(fill)

            bars_seen += 1

        # ── End of last day ────────────────────────────
        if self._current_day:
            self.strategy.on_day_end(self._current_day, self.book)

        # ── Compute metrics ────────────────────────────
        metrics = compute_metrics(
            fills          = self._all_fills,
            tc             = self.tc,
            bars_processed = bars_seen,
        )

        # ── Print summary ──────────────────────────────
        print_metrics(metrics, self.config.symbol,
                      self.config.start, self.config.end)

        print(f"  Bars processed:  {bars_seen:,}")
        if hasattr(self.strategy, 'bars_quoted'):
            quote_rate = self.strategy.bars_quoted / bars_seen * 100 if bars_seen else 0
            print(f"  Quote rate:      {quote_rate:.1f}%")
            print(f"  Bars quoted:     {self.strategy.bars_quoted:,}")
            print(f"  Bars filtered:   {self.strategy.bars_filtered:,}")
        print()

        return metrics