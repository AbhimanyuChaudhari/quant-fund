"""
Backtest performance metrics.

Computes PnL, Sharpe, drawdown, fill rate, and other
statistics from a list of fills and the transaction cost engine.
"""

import math
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List

from src.backtest.order_book import Fill, Side
from src.backtest.transaction_costs import TransactionCosts


@dataclass
class BacktestMetrics:
    # PnL
    gross_pnl:      float = 0.0
    total_costs:    float = 0.0
    net_pnl:        float = 0.0

    # Trade stats
    total_fills:    int   = 0
    buy_fills:      int   = 0
    sell_fills:     int   = 0

    # Spread capture
    spread_captured:     float = 0.0   # gross per round trip
    avg_spread_captured: float = 0.0

    # Risk metrics
    sharpe_ratio:   float = 0.0
    max_drawdown:   float = 0.0
    win_rate:       float = 0.0

    # Inventory
    max_inventory:  int   = 0
    final_inventory:int   = 0


def compute_metrics(fills: List[Fill],
                    tc: TransactionCosts,
                    bars_processed: int) -> BacktestMetrics:
    """
    Compute full backtest metrics from a list of fills.

    Args:
        fills:          all fills from the backtest run
        tc:             TransactionCosts instance
        bars_processed: total bars the strategy ran over

    Returns:
        BacktestMetrics dataclass
    """
    m = BacktestMetrics()

    if not fills:
        return m

    m.total_fills = len(fills)
    m.buy_fills   = sum(1 for f in fills if f.side == Side.BUY)
    m.sell_fills  = sum(1 for f in fills if f.side == Side.SELL)

    # ── PnL calculation ──────────────────────────────
    # Track running inventory and cost basis
    inventory   = 0
    cost_basis  = 0.0
    realized    = 0.0
    total_costs = 0.0
    pnl_series  = []   # realized PnL after each round trip

    for fill in sorted(fills, key=lambda f: f.timestamp):
        cost = tc.compute(fill.price, fill.quantity, fill.side.value)
        total_costs += cost.total

        if fill.side == Side.BUY:
            cost_basis += fill.price * fill.quantity
            inventory  += fill.quantity
        else:
            # Realize PnL on sell
            if inventory > 0:
                avg_cost  = cost_basis / inventory
                trade_pnl = (fill.price - avg_cost) * fill.quantity * tc.lot_size
                realized  += trade_pnl
                pnl_series.append(realized)

                cost_basis -= avg_cost * fill.quantity
                inventory  -= fill.quantity

    m.gross_pnl      = realized
    m.total_costs    = total_costs
    m.net_pnl        = realized - total_costs
    m.final_inventory = inventory

    # ── Sharpe ratio ─────────────────────────────────
    if len(pnl_series) > 1:
        pnl_arr    = np.array(pnl_series)
        pnl_diff   = np.diff(pnl_arr)   # incremental PnL per trade
        if pnl_diff.std() > 0:
            m.sharpe_ratio = (pnl_diff.mean() / pnl_diff.std()) * math.sqrt(252)
        else:
            m.sharpe_ratio = 0.0

    # ── Max drawdown ─────────────────────────────────
    if pnl_series:
        pnl_arr  = np.array(pnl_series)
        peak     = np.maximum.accumulate(pnl_arr)
        drawdown = pnl_arr - peak
        m.max_drawdown = float(drawdown.min())

    # ── Win rate ─────────────────────────────────────
    if len(pnl_series) > 1:
        wins       = sum(1 for i in range(1, len(pnl_series))
                         if pnl_series[i] > pnl_series[i-1])
        m.win_rate = wins / (len(pnl_series) - 1)

    # ── Max inventory ─────────────────────────────────
    inv = 0
    max_inv = 0
    for fill in sorted(fills, key=lambda f: f.timestamp):
        if fill.side == Side.BUY:
            inv += fill.quantity
        else:
            inv -= fill.quantity
        max_inv = max(max_inv, abs(inv))
    m.max_inventory = max_inv
    m.final_inventory = inv

    return m


def print_metrics(m: BacktestMetrics, symbol: str,
                  start: str, end: str) -> None:
    """Pretty print backtest results to terminal."""
    print()
    print(f"{'─'*45}")
    print(f"  Backtest Results — {symbol}")
    print(f"  {start} → {end}")
    print(f"{'─'*45}")
    print(f"  Gross PnL:          ₹{m.gross_pnl:>12,.2f}")
    print(f"  Total Costs:       -₹{m.total_costs:>12,.2f}")
    print(f"  Net PnL:            ₹{m.net_pnl:>12,.2f}")
    print(f"{'─'*45}")
    print(f"  Total Fills:        {m.total_fills:>12,}")
    print(f"  Buy Fills:          {m.buy_fills:>12,}")
    print(f"  Sell Fills:         {m.sell_fills:>12,}")
    print(f"{'─'*45}")
    print(f"  Sharpe Ratio:       {m.sharpe_ratio:>12.2f}")
    print(f"  Max Drawdown:      -₹{abs(m.max_drawdown):>12,.2f}")
    print(f"  Win Rate:           {m.win_rate*100:>11.1f}%")
    print(f"{'─'*45}")
    print(f"  Max Inventory:      {m.max_inventory:>12,} lots")
    print(f"  Final Inventory:    {m.final_inventory:>12,} lots")
    print(f"{'─'*45}")
    print()