"""
Live Trading Engine

Three modes:
  --paper-live:  Real WebSocket ticks, simulated fills (best for testing)
  --paper:       Historical GCS data, simulated fills (offline testing)
  --live:        Real WebSocket ticks, real Zerodha orders (production)
"""

import logging
import time
import threading
import pathlib
import sys
from datetime import datetime, date
from typing import Optional

# Create logs dir BEFORE logging setup
pathlib.Path("logs").mkdir(exist_ok=True)

# Fix Windows console Unicode issue
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from src.trading.risk import RiskManager, RiskConfig
from src.trading.portfolio import Portfolio
from src.trading.order_manager import OrderManager
from src.trading.feature_engine import FeatureEngine, TickBar
from src.backtest.strategy import BaseStrategy
from src.utils.auth import get_kite_client

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers= [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"logs/engine_{date.today()}.log",
            mode     = "a",
            encoding = "utf-8",
        ),
    ]
)
logger = logging.getLogger(__name__)


class LiveEngine:
    """
    Live trading engine.

    Modes:
        paper_live=True,  live=False  → WebSocket ticks, simulated fills
        paper_live=False, live=False  → Historical GCS data, simulated fills
        paper_live=False, live=True   → WebSocket ticks, real Zerodha orders
    """

    def __init__(self,
                 strategy:    BaseStrategy,
                 symbol:      str,
                 paper_live:  bool      = False,
                 live:        bool      = False,
                 lot_size:    int       = 75,
                 risk_config: RiskConfig = None):

        self.strategy    = strategy
        self.symbol      = symbol
        self.paper_live  = paper_live
        self.live        = live
        self.lot_size    = lot_size
        self.running     = False

        # Determine mode
        if live:
            self.mode = "LIVE"
        elif paper_live:
            self.mode = "PAPER-LIVE"
        else:
            self.mode = "PAPER"

        self.risk      = RiskManager(risk_config or RiskConfig())
        self.portfolio = Portfolio(
            lot_sizes     = {symbol: lot_size},
            paper_trading = not live,
        )

        # Only connect Kite for live mode
        self.kite = None
        if live:
            self.kite = get_kite_client()
            logger.warning("LIVE MODE - real orders will be placed")
        else:
            logger.info(f"{self.mode} MODE - no real orders")

        self.order_manager  = OrderManager(
            kite          = self.kite,
            portfolio     = self.portfolio,
            paper_trading = not live,
        )
        self.feature_engine = FeatureEngine()

        self._bid_order_id: Optional[str] = None
        self._ask_order_id: Optional[str] = None
        self._bars_processed = 0
        self._start_time     = None

    # ─────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────
    def run(self):
        print(f"Engine started - mode={self.mode}", flush=True)
        logger.info(
            f"Starting engine | symbol={self.symbol} | mode={self.mode}"
        )

        self._start_time = time.time()
        self.running     = True
        self.risk.on_day_start()

        threading.Thread(
            target = self._status_loop,
            daemon = True,
            name   = "StatusThread"
        ).start()

        try:
            if self.mode == "PAPER":
                self._run_paper_historical()
            elif self.mode == "PAPER-LIVE":
                self._run_paper_live()
            else:
                self._run_live()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt - shutting down")
        finally:
            self._shutdown()

    # ─────────────────────────────────────
    # Mode 1: Historical paper trading
    # ─────────────────────────────────────
    def _run_paper_historical(self):
        """
        Offline paper trading on GCS processed features.
        Finds most recent date with real data.
        Good for: strategy testing, debugging, weekend work.
        """
        from src.backtest.data_loader import load_day
        import gcsfs
        from pathlib import Path

        logger.info("Loading historical data from GCS...")

        fs    = gcsfs.GCSFileSystem(project="hedge-fund-494103")
        files = fs.glob(
            f"hedge-fund-494103-marketdata/processed/features/"
            f"{self.symbol}/*.parquet"
        )
        if not files:
            logger.error(f"No processed data found for {self.symbol}")
            return

        # Find most recent date with enough bars
        dates = sorted([Path(f).stem for f in files], reverse=True)
        use_date = None
        df       = None

        for d in dates:
            test = load_day(self.symbol, d, market_hours_only=True)
            if not test.empty and len(test) > 100:
                use_date = d
                df       = test
                break

        if df is None:
            logger.error(f"No usable data found for {self.symbol}")
            return

        logger.info(f"Using: {use_date} | bars: {len(df):,}")

        for _, bar in df.iterrows():
            if not self.running:
                break
            self._on_bar(bar.to_dict())

        logger.info(
            f"Complete | bars={self._bars_processed:,} | "
            f"PnL=Rs.{self.portfolio.total_pnl():+,.0f}"
        )

    # ─────────────────────────────────────
    # Mode 2: Live paper trading (best mode)
    # ─────────────────────────────────────
    def _run_paper_live(self):
        """
        Paper trading on LIVE WebSocket ticks.
        Real market data, simulated fills — no real orders sent.

        Fill simulation:
          BUY  limit filled if tick.last_price <= our bid
          SELL limit filled if tick.last_price >= our ask
          Same logic as backtester's order_book.py

        Best mode for validating strategy before going live.
        Run this during market hours for realistic results.
        """
        logger.info("Connecting to Zerodha WebSocket for live paper trading...")

        kite        = get_kite_client()
        instruments = kite.instruments("NFO")
        token_map   = {i["tradingsymbol"]: i["instrument_token"]
                       for i in instruments}
        token       = token_map.get(self.symbol)

        if not token:
            logger.error(f"Token not found for {self.symbol}")
            return

        from kiteconnect import KiteTicker
        ticker = KiteTicker(kite.api_key, kite.access_token)

        def on_ticks(ws, ticks):
            for tick in ticks:
                if tick["instrument_token"] == token:
                    parsed = self._parse_tick(tick)

                    # Check paper fills against live price
                    self._check_paper_fills(tick.get("last_price", 0))

                    # Compute features
                    bar = self.feature_engine.on_tick(parsed)
                    if bar:
                        self._on_bar(bar.to_dict())

        def on_connect(ws, response):
            ws.subscribe([token])
            ws.set_mode(ws.MODE_FULL, [token])
            logger.info(
                f"WebSocket connected | "
                f"subscribed to {self.symbol} | "
                f"mode=PAPER-LIVE"
            )
            print(f"\nPaper trading live on {self.symbol}", flush=True)
            print("Press Ctrl+C to stop\n", flush=True)

        def on_error(ws, code, reason):
            logger.error(f"WebSocket error {code}: {reason}")

        def on_close(ws, code, reason):
            logger.warning(f"WebSocket closed {code}: {reason}")
            self.running = False

        def on_reconnect(ws, attempts):
            logger.info(f"Reconnecting... attempt {attempts}")

        def on_noreconnect(ws):
            logger.error("Max reconnect attempts reached")
            self.running = False

        ticker.on_ticks      = on_ticks
        ticker.on_connect    = on_connect
        ticker.on_error      = on_error
        ticker.on_close      = on_close
        ticker.on_reconnect  = on_reconnect
        ticker.on_noreconnect = on_noreconnect
        ticker.connect(threaded=False)

    # ─────────────────────────────────────
    # Mode 3: Live trading (real orders)
    # ─────────────────────────────────────
    def _run_live(self):
        """
        Real live trading — sends actual orders to Zerodha.
        Only use after extensive paper-live validation.
        """
        logger.info("Connecting to Zerodha WebSocket for LIVE trading...")

        kite        = get_kite_client()
        instruments = kite.instruments("NFO")
        token_map   = {i["tradingsymbol"]: i["instrument_token"]
                       for i in instruments}
        token       = token_map.get(self.symbol)

        if not token:
            logger.error(f"Token not found for {self.symbol}")
            return

        from kiteconnect import KiteTicker
        ticker = KiteTicker(kite.api_key, kite.access_token)

        def on_ticks(ws, ticks):
            for tick in ticks:
                if tick["instrument_token"] == token:
                    parsed = self._parse_tick(tick)
                    bar    = self.feature_engine.on_tick(parsed)
                    if bar:
                        self._on_bar(bar.to_dict())

        def on_connect(ws, response):
            ws.subscribe([token])
            ws.set_mode(ws.MODE_FULL, [token])
            logger.info(f"LIVE WebSocket connected | {self.symbol}")

        def on_error(ws, code, reason):
            logger.error(f"WebSocket error {code}: {reason}")

        def on_close(ws, code, reason):
            logger.warning(f"WebSocket closed: {reason}")
            self.running = False

        ticker.on_ticks   = on_ticks
        ticker.on_connect = on_connect
        ticker.on_error   = on_error
        ticker.on_close   = on_close
        ticker.connect(threaded=False)

    # ─────────────────────────────────────
    # Paper fill simulation
    # ─────────────────────────────────────
    def _check_paper_fills(self, last_price: float):
        """
        Simulate fills based on live price crossing our limit.
        Called on every tick in paper-live mode.

        BUY  filled if last_price <= bid price
        SELL filled if last_price >= ask price
        """
        if not last_price:
            return

        # Check bid
        if self._bid_order_id:
            order = self.order_manager.orders.get(self._bid_order_id)
            if order and order.status == "PENDING":
                if last_price <= order.price:
                    # Fill it
                    order.status     = "FILLED"
                    order.fill_price = order.price
                    order.fill_time  = datetime.now().strftime("%H:%M:%S")
                    pnl = self.portfolio.on_fill(
                        order.symbol, order.side, order.lots,
                        order.price, order.order_id
                    )
                    self.risk.update_pnl(pnl)
                    self._bid_order_id = None
                    logger.info(
                        f"[PAPER-LIVE] FILLED BUY @ {order.price:.2f} "
                        f"(market={last_price:.2f})"
                    )

        # Check ask
        if self._ask_order_id:
            order = self.order_manager.orders.get(self._ask_order_id)
            if order and order.status == "PENDING":
                if last_price >= order.price:
                    order.status     = "FILLED"
                    order.fill_price = order.price
                    order.fill_time  = datetime.now().strftime("%H:%M:%S")
                    pnl = self.portfolio.on_fill(
                        order.symbol, order.side, order.lots,
                        order.price, order.order_id
                    )
                    self.risk.update_pnl(pnl)
                    self._ask_order_id = None
                    logger.info(
                        f"[PAPER-LIVE] FILLED SELL @ {order.price:.2f} "
                        f"(market={last_price:.2f})"
                    )

    # ─────────────────────────────────────
    # Per-bar logic
    # ─────────────────────────────────────
    def _on_bar(self, bar: dict):
        self._bars_processed += 1
        self.portfolio.update_price(self.symbol, bar.get("close", 0))
        self._cancel_quotes()
        self.strategy.on_bar_live(bar, self)

    def post_quote(self, side: str, price: float, lots: int):
        """Called by strategy. Routes through risk checks."""
        ok, reason = self.risk.check_order(
            self.symbol, side, lots, self.portfolio
        )
        if not ok:
            logger.debug(f"Risk blocked {side}: {reason}")
            return None

        # In paper-live mode, place as PENDING (fill check happens on tick)
        # In paper mode, order_manager fills immediately
        order_id = self.order_manager.place_limit_order(
            self.symbol, side, lots, price
        )

        # For paper-live: override the immediate fill — keep as PENDING
        if self.mode == "PAPER-LIVE" and order_id:
            order = self.order_manager.orders.get(order_id)
            if order:
                order.status     = "PENDING"
                order.fill_price = 0.0
                order.fill_time  = ""

        if order_id:
            if side == "BUY":
                self._bid_order_id = order_id
            else:
                self._ask_order_id = order_id

        return order_id

    def _cancel_quotes(self):
        if self._bid_order_id:
            self.order_manager.cancel_order(self._bid_order_id)
            self._bid_order_id = None
        if self._ask_order_id:
            self.order_manager.cancel_order(self._ask_order_id)
            self._ask_order_id = None

    # ─────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────
    def _parse_tick(self, tick: dict) -> dict:
        depth    = tick.get("depth", {})
        bids     = depth.get("buy",  [{}])
        asks     = depth.get("sell", [{}])
        best_bid = bids[0].get("price", 0) if bids else 0
        best_ask = asks[0].get("price", 0) if asks else 0
        mid      = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        spread   = best_ask - best_bid if best_bid and best_ask else 0
        tbq      = tick.get("total_buy_quantity", 0)
        taq      = tick.get("total_sell_quantity", 0)
        tq       = tbq + taq
        imbal    = (tbq - taq) / tq if tq else 0

        return {
            "ts_local_ns":    time.time_ns(),
            "symbol":         self.symbol,
            "last_price":     tick.get("last_price", 0),
            "avg_price":      tick.get("average_traded_price", 0),
            "volume":         tick.get("volume_traded", 0),
            "oi":             tick.get("oi", 0),
            "spread":         spread,
            "mid_price":      mid,
            "book_imbalance": imbal,
            "total_bid_qty":  tbq,
            "total_ask_qty":  taq,
            "bid_p1":         bids[0].get("price", 0) if bids else 0,
            "ask_p1":         asks[0].get("price", 0) if asks else 0,
            "bid_q1":         bids[0].get("quantity", 0) if bids else 0,
            "ask_q1":         asks[0].get("quantity", 0) if asks else 0,
        }

    def _status_loop(self):
        while self.running:
            time.sleep(60)
            risk   = self.risk.status()
            port   = self.portfolio.summary()
            uptime = int(time.time() - self._start_time) // 60
            logger.info(
                f"STATUS | uptime={uptime}m | "
                f"bars={self._bars_processed:,} | "
                f"pnl=Rs.{port['total_pnl']:+,.0f} | "
                f"loss_used={risk['pct_used']:.1f}% | "
                f"positions={port['open_positions']}"
            )

    def _shutdown(self):
        logger.info("Shutting down...")
        self.running = False
        self.order_manager.cancel_all()
        self.portfolio.on_day_end()
        logger.info(
            f"Final PnL: Rs.{self.portfolio.total_pnl():+,.0f} | "
            f"Trades: {len(self.portfolio.trades)}"
        )