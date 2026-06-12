"""
Fast Backtest Engine
====================
Drop-in replacement for BacktestEngine using vectorized NumPy + Numba.

Speed: 20-100x faster than original bar-by-bar Python loop.

Usage (same as original):
    from src.backtest.engine_fast import FastBacktestEngine, BacktestConfig
    from src.backtest.mm_strategy import AvellanedaStoikovStrategy

    config   = BacktestConfig('ADANIPORTS26MAYFUT', '2026-05-18', '2026-05-18',
                               lot_size=475)
    strategy = AvellanedaStoikovStrategy(strat_config, gamma=0.001)
    engine   = FastBacktestEngine(config, strategy)
    metrics  = engine.run()

Or use run_fast_backtest() directly for even faster batch runs:
    from src.backtest.engine_fast import run_fast_backtest_batch
    results = run_fast_backtest_batch(
        symbols=['ADANIPORTS26MAYFUT', 'HAVELLS26MAYFUT'],
        dates=['2026-05-18', '2026-05-19'],
        gamma=0.001, min_spread=0.10
    )
"""

import time
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.backtest.simulators.fill_simulator_fast import run_fast_backtest, FastBacktestResult
from src.backtest.data_loader import load_day
from src.backtest.metrics import BacktestMetrics


# ─────────────────────────────────────────────────────────────────────────────
# BacktestConfig — same as original for compatibility
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    symbol:        str
    start:         str
    end:           str
    lot_size:      int  = 75
    max_inventory: int  = 5
    market_hours:  bool = True


# ─────────────────────────────────────────────────────────────────────────────
# FastBacktestEngine — compatible with original BacktestEngine interface
# ─────────────────────────────────────────────────────────────────────────────

class FastBacktestEngine:
    """
    Fast vectorized backtest engine.
    Compatible with original BacktestEngine interface — just swap the import.

    Key difference: reads params directly from strategy object instead of
    calling on_bar() bar by bar.
    """

    def __init__(self, config: BacktestConfig, strategy, order_book=None):
        self.config   = config
        self.strategy = strategy
        self._result: Optional[FastBacktestResult] = None

    def run(self) -> BacktestMetrics:
        """Run fast vectorized backtest. Returns BacktestMetrics."""
        t0 = time.perf_counter()

        # ── Detect instrument type ────────────────────────────────────────────
        sym = self.config.symbol.upper()
        if any(x in sym for x in ['USDINR', 'EURINR', 'GBPINR', 'JPYINR']):
            instrument_type = 'currency_futures'
        else:
            instrument_type = 'equity_futures'

        # ── Extract strategy params ────────────────────────────────────────────
        gamma      = getattr(self.strategy, 'gamma',      0.001)
        kappa      = getattr(self.strategy, 'kappa',      1.5)
        min_spread = getattr(self.strategy, 'min_spread', 0.10)
        max_spread = getattr(self.strategy, 'max_spread', 10.0)
        open_mult  = getattr(self.strategy, 'open_mult',  2.0)

        # ── Load all days ─────────────────────────────────────────────────────
        from datetime import datetime, timedelta
        start_dt = datetime.strptime(self.config.start, '%Y-%m-%d')
        end_dt   = datetime.strptime(self.config.end,   '%Y-%m-%d')

        frames = []
        current = start_dt
        while current <= end_dt:
            date_str = current.strftime('%Y-%m-%d')
            df = load_day(self.config.symbol, date_str,
                          market_hours_only=self.config.market_hours)
            if not df.empty:
                df['_date'] = date_str
                frames.append(df)
            current += timedelta(days=1)

        if not frames:
            print(f"[fast] No data: {self.config.symbol} "
                  f"{self.config.start}→{self.config.end}")
            return self._empty_metrics()

        full_df = pd.concat(frames, ignore_index=True).sort_values('ts_sec')

        # ── Run fast backtest ─────────────────────────────────────────────────
        result = run_fast_backtest(
            df               = full_df,
            gamma            = gamma,
            kappa            = kappa,
            min_spread       = min_spread,
            max_spread       = max_spread,
            open_mult        = open_mult,
            lot_size         = self.config.lot_size,
            max_inventory    = self.config.max_inventory,
            queue_aggression = 0.3,
            instrument_type  = instrument_type,
        )

        self._result = result
        elapsed = time.perf_counter() - t0

        # ── Print summary ──────────────────────────────────────────────────────
        sign = '+' if result.net_pnl >= 0 else ''
        print(f"[fast] {self.config.symbol:30s} "
              f"Rs.{sign}{result.net_pnl:>10,.0f}  "
              f"fills={result.total_fills:>5}  "
              f"fill%={result.fill_rate:>5.1f}%  "
              f"win%={result.win_rate:>5.1f}%  "
              f"sharpe={result.sharpe_ratio:>6.2f}  "
              f"[{elapsed:.2f}s]")

        return self._to_backtest_metrics(result)

    def _to_backtest_metrics(self, r: FastBacktestResult) -> BacktestMetrics:
        """Convert FastBacktestResult to BacktestMetrics for compatibility."""
        m = BacktestMetrics()
        m.gross_pnl       = r.gross_pnl
        m.total_costs     = r.total_costs
        m.net_pnl         = r.net_pnl
        m.total_fills     = r.total_fills
        m.buy_fills       = r.buy_fills
        m.sell_fills      = r.sell_fills
        m.sharpe_ratio    = r.sharpe_ratio
        m.max_drawdown    = r.max_drawdown
        m.win_rate        = r.win_rate / 100.0
        m.max_inventory   = r.max_inventory
        m.final_inventory = r.final_inventory
        return m

    def _empty_metrics(self) -> BacktestMetrics:
        return BacktestMetrics()


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner — parallel across symbols and dates
# ─────────────────────────────────────────────────────────────────────────────

def run_fast_backtest_batch(
    symbols:          list,
    dates:            list,
    lot_sizes:        dict,
    gamma:            float = 0.001,
    kappa:            float = 1.5,
    min_spread:       float = 0.10,
    max_spread:       float = 10.0,
    open_mult:        float = 2.0,
    max_inventory:    int   = 5,
    queue_aggression: float = 0.3,
    workers:          int   = 8,
    min_bars:         int   = 100,
) -> list[dict]:
    """
    Run fast backtest for all symbols × dates in parallel.
    Uses ThreadPoolExecutor — GCS reads are I/O bound so threads work well.

    Args:
        symbols:   list of tradingsymbol strings
        dates:     list of 'YYYY-MM-DD' strings
        lot_sizes: dict of {base_name: lot_size} e.g. {'ADANIPORTS': 475}
        ...

    Returns:
        List of result dicts with symbol, date, net_pnl, etc.
    """
    t0     = time.perf_counter()
    tasks  = [(sym, date) for sym in symbols for date in dates]
    results = []

    def run_one_task(sym_date):
        sym, date = sym_date
        name = sym.replace('26MAYFUT', '').replace('26JUNFUT', '')
        lot  = lot_sizes.get(name, lot_sizes.get(sym, 75))

        # Detect instrument type
        if any(x in sym.upper() for x in ['USDINR', 'EURINR']):
            inst = 'currency_futures'
        else:
            inst = 'equity_futures'

        try:
            df = load_day(sym, date, market_hours_only=True)
            if df.empty or len(df) < min_bars:
                return {'symbol': sym, 'date': date, 'ok': False,
                        'error': f'insufficient bars ({len(df)})',
                        'net_pnl': 0, 'bars': len(df)}

            result = run_fast_backtest(
                df               = df,
                gamma            = gamma,
                kappa            = kappa,
                min_spread       = min_spread,
                max_spread       = max_spread,
                open_mult        = open_mult,
                lot_size         = lot,
                max_inventory    = max_inventory,
                queue_aggression = queue_aggression,
                instrument_type  = inst,
            )
            return {
                'symbol':    sym,
                'date':      date,
                'lot_size':  lot,
                'net_pnl':   round(result.net_pnl, 2),
                'gross_pnl': round(result.gross_pnl, 2),
                'costs':     round(result.total_costs, 2),
                'fills':     result.total_fills,
                'fill_rate': result.fill_rate,
                'win_rate':  result.win_rate,
                'sharpe':    result.sharpe_ratio,
                'max_dd':    result.max_drawdown,
                'bars':      result.bars_processed,
                'ok':        True,
            }
        except Exception as e:
            return {'symbol': sym, 'date': date, 'ok': False,
                    'error': str(e)[:80], 'net_pnl': 0, 'bars': 0}

    print(f"[fast_batch] {len(tasks)} tasks | {workers} workers | "
          f"{len(symbols)} symbols × {len(dates)} dates")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one_task, t): t for t in tasks}
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.perf_counter() - t0
    ok      = sum(1 for r in results if r.get('ok'))
    print(f"[fast_batch] Done: {ok}/{len(tasks)} succeeded | {elapsed:.1f}s total "
          f"({elapsed/len(tasks):.2f}s/task)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Speed benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(symbol: str = 'ADANIPORTS26MAYFUT',
              date:   str = '2026-05-18',
              lot_size: int = 475):
    """Compare fast vs original engine speed."""
    import time

    print(f"\nBenchmarking: {symbol} | {date}")
    print("=" * 55)

    # ── Fast engine ────────────────────────────────────────────────────────────
    df = load_day(symbol, date, market_hours_only=True)
    if df.empty:
        print("No data available for benchmark")
        return

    print(f"Bars loaded: {len(df):,}")
    print(f"Bars loaded: {len(df):,}")

# # ── DEBUG BLOCK — remove after fixing ─────────────────────────────────────
#     import numpy as np
#     from src.backtest.fill_simulator_fast import compute_quotes_vectorized

#     ts_sec  = df['ts_sec'].values.astype(np.float64)
#     mid     = df['weighted_mid'].fillna(df['close']).values.astype(np.float64)
#     vol_60s = df['realized_vol_60s'].fillna(2.0).values.astype(np.float64)
#     lows    = df['low'].values.astype(np.float64)
#     highs   = df['high'].values.astype(np.float64)
#     bid_q1  = df['bid_q1'].fillna(100).values.astype(np.float64)
#     ist_tod = (ts_sec + 19800) % 86400

#     inv  = np.zeros(len(df), dtype=np.int32)
#     bids, asks = compute_quotes_vectorized(
#         ts_sec, mid, vol_60s, inv,
#         0.001, 1.5, 0.10, 10.0, 5, 2.0
#     )

#     print(f"\n--- DEBUG ---")
#     print(f"Sample mid:   {mid[100:105]}")
#     print(f"Sample bids:  {bids[100:105]}")
#     print(f"Sample asks:  {asks[100:105]}")
#     print(f"NaN bids:     {np.isnan(bids).sum()} / {len(bids)}")
#     print(f"low<=bid:     {(lows <= bids).sum()} bars")
#     print(f"high>=ask:    {(highs >= asks).sum()} bars")
#     print(f"IST range:    {ist_tod.min():.0f} to {ist_tod.max():.0f}")
#     print(f"Mkt hrs bars: {((ist_tod >= 33300) & (ist_tod <= 55800)).sum()}")
#     print(f"Terminal bars:{(ist_tod >= 54720).sum()}")
#     print(f"bid_q1 sample:{bid_q1[100:105]}")
#     print(f"--- END DEBUG ---\n")
# # ── END DEBUG BLOCK ────────────────────────────────────────────────────────

    t0     = time.perf_counter()
    result = run_fast_backtest(df, gamma=0.001, min_spread=0.10,
                                lot_size=lot_size)
    t_fast = time.perf_counter() - t0

    print(f"\nFast engine:     {t_fast:.3f}s")
    print(f"Net PnL:         Rs.{result.net_pnl:,.0f}")
    print(f"Fills:           {result.total_fills}")
    print(f"Fill rate:       {result.fill_rate:.1f}%")
    print(f"Bars/second:     {len(df)/t_fast:,.0f}")

    # ── Original engine ────────────────────────────────────────────────────────
    print("\nRunning original engine for comparison...")
    try:
        from src.backtest.models.v1_avellaneda_stoikov.engine import BacktestEngine
        from src.backtest.models.v1_avellaneda_stoikov.engine import BacktestConfig as OrigConfig
        from src.backtest.models.v1_avellaneda_stoikov.strategy import StrategyConfig
        from src.backtest.mm_strategy import AvellanedaStoikovStrategy
        from src.backtest.models.v1_avellaneda_stoikov.fill_simulator import RealisticOrderBook

        sc   = StrategyConfig(symbol=symbol, lot_size=lot_size, max_inventory=5)
        strat = AvellanedaStoikovStrategy(sc, gamma=0.001, min_spread=0.10)
        cfg  = OrigConfig(symbol=symbol, start=date, end=date, lot_size=lot_size)
        book = RealisticOrderBook(max_inventory=5, lot_size=lot_size)

        t0   = time.perf_counter()
        orig_metrics = BacktestEngine(cfg, strat, order_book=book).run()
        t_orig = time.perf_counter() - t0

        print(f"\nOriginal engine: {t_orig:.3f}s")
        print(f"Net PnL:         Rs.{orig_metrics.net_pnl:,.0f}")
        print(f"\nSpeedup:         {t_orig/t_fast:.1f}x faster")
    except Exception as e:
        print(f"Could not run original engine: {e}")


if __name__ == "__main__":
    benchmark()
