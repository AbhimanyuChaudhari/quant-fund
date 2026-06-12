"""
Signal Aggregator
=================
Combines all intelligence sources into a single IntelSignal per symbol.

Sources combined:
    1. Event calendar  (event_calendar.py) — scheduled events
    2. News scores     (processor.py)      — real-time LLM scored news

Output:
    One IntelSignal per symbol saved to GCS
    mm_strategy.py reads this to adjust spreads

Aggregation logic:
    Take the WORST (most cautious) signal across all sources.
    Event calendar PAUSE overrides news WIDEN.
    News PAUSE overrides event calendar WIDEN.
    Multiple WIDEN signals compound (up to max_widen cap).

Runs:
    Every 10 minutes during market hours
    Cron: */10 3-10 * * 1-5 (UTC)

Usage:
    # Aggregate all symbols
    python intelligence/signals/aggregator.py --all

    # Single symbol
    python intelligence/signals/aggregator.py --symbol RELIANCE26MAYFUT

    # Show current signals without saving
    python intelligence/signals/aggregator.py --all --dry-run

    # Import
    from intelligence.signals.aggregator import SignalAggregator
    agg    = SignalAggregator()
    signal = agg.aggregate(symbol)
    signal.save()
"""

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to path
root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if root not in sys.path:
    sys.path.insert(0, root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

GCS_BUCKET = "hedge-fund-494103-marketdata-mumbai"
GCP_PROJECT = "hedge-fund-494103"

# Signal TTL — how long before signal expires and mm_strategy uses defaults
SIGNAL_TTL_MINUTES = 15   # re-aggregate every 15 mins during market hours

# Max widen factor — never widen more than this regardless of how many signals
MAX_WIDEN_FACTOR = 4.0

# Minimum uncertainty to trigger WIDEN from news
NEWS_UNCERTAINTY_THRESHOLD = 0.4

# Action priority (higher = more cautious = wins)
ACTION_PRIORITY = {
    "NORMAL": 0,
    "WIDEN":  1,
    "PAUSE":  2,
    "AVOID":  3,
}

# Futures symbols to aggregate (loaded from config or hardcoded)
DEFAULT_FUTURES = [
    "NIFTY26MAYFUT", "BANKNIFTY26MAYFUT", "FINNIFTY26MAYFUT",
    "MIDCPNIFTY26MAYFUT", "RELIANCE26MAYFUT", "HDFCBANK26MAYFUT",
    "ICICIBANK26MAYFUT", "INFY26MAYFUT", "TCS26MAYFUT",
    "AXISBANK26MAYFUT", "KOTAKBANK26MAYFUT", "SBIN26MAYFUT",
    "BHARTIARTL26MAYFUT", "BAJFINANCE26MAYFUT", "ADANIPORTS26MAYFUT",
    "ADANIENT26MAYFUT", "HAL26MAYFUT", "BEL26MAYFUT",
    "TATASTEEL26MAYFUT", "JSWSTEEL26MAYFUT", "HINDALCO26MAYFUT",
    "ONGC26MAYFUT", "BPCL26MAYFUT", "IOC26MAYFUT",
    "DRREDDY26MAYFUT", "SUNPHARMA26MAYFUT", "CIPLA26MAYFUT",
    "LUPIN26MAYFUT", "DIVISLAB26MAYFUT", "MARUTI26MAYFUT",
    "TVSMOTOR26MAYFUT", "EICHERMOT26MAYFUT", "HEROMOTOCO26MAYFUT",
    "ULTRACEMCO26MAYFUT", "GRASIM26MAYFUT", "DLF26MAYFUT",
    "GODREJPROP26MAYFUT", "CHOLAFIN26MAYFUT", "BAJAJFINSV26MAYFUT",
    "MUTHOOTFIN26MAYFUT", "LICHSGFIN26MAYFUT", "LT26MAYFUT",
    "POWERGRID26MAYFUT", "NTPC26MAYFUT", "PFC26MAYFUT",
    "RECLTD26MAYFUT", "HCLTECH26MAYFUT", "WIPRO26MAYFUT",
    "TECHM26MAYFUT", "TITAN26MAYFUT", "NESTLEIND26MAYFUT",
    "HINDUNILVR26MAYFUT", "BRITANNIA26MAYFUT", "TATACONSUM26MAYFUT",
    "COALINDIA26MAYFUT", "SAIL26MAYFUT", "VEDL26MAYFUT",
    "USDINR26508FUT", "USDINR26515FUT",
]


# ─── Helper ───────────────────────────────────────────────────────────────────

def load_futures_symbols() -> list[str]:
    """Load futures symbols from config if available, else use defaults."""
    try:
        from config.symbols import get_all_futures
        return get_all_futures()
    except ImportError:
        return DEFAULT_FUTURES


# ─── Component signal builders ────────────────────────────────────────────────

class EventSignalBuilder:
    """Builds signal component from event calendar."""

    def __init__(self):
        self._calendar = None

    def _get_calendar(self):
        if self._calendar is None:
            from intelligence.news.event_calendar import EventCalendar
            self._calendar = EventCalendar()
        return self._calendar

    def build(self, futures_symbol: str) -> dict:
        """
        Returns component dict:
        {source, mm_action, widen_factor, pause_until,
         has_event, event_type, days_to_event, reason}
        """
        try:
            cal    = self._get_calendar()
            signal = cal.to_intel_signal(futures_symbol, ttl_hours=4)

            if hasattr(signal, "mm_action"):
                return {
                    "source":        "EVENTS",
                    "mm_action":     signal.mm_action,
                    "widen_factor":  signal.widen_factor,
                    "pause_until":   signal.pause_until,
                    "has_event":     signal.has_upcoming_event,
                    "event_type":    signal.event_type,
                    "days_to_event": signal.days_to_event,
                    "reason":        signal.reason,
                    "confidence":    0.95,
                }
        except Exception as e:
            log.debug(f"Event signal build failed for {futures_symbol}: {e}")

        return {
            "source":       "EVENTS",
            "mm_action":    "NORMAL",
            "widen_factor": 1.0,
            "pause_until":  None,
            "has_event":    False,
            "event_type":   None,
            "days_to_event":None,
            "reason":       "No upcoming events",
            "confidence":   1.0,
        }


class NewsSignalBuilder:
    """Builds signal component from scored news items."""

    def __init__(self):
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from intelligence.news.processor import NewsProcessor
            self._processor = NewsProcessor()
        return self._processor

    def build(self, futures_symbol: str) -> dict:
        """
        Returns component dict built from worst scored news item.
        """
        try:
            processor = self._get_processor()
            worst     = processor.worst_signal(futures_symbol)

            if worst is None:
                return self._normal("No relevant news today")

            uncertainty  = worst.get("uncertainty", 0.0)
            mm_action    = worst.get("mm_action", "NORMAL")
            widen_factor = worst.get("widen_factor", 1.0)
            pause_mins   = worst.get("pause_minutes", 0)

            # Only act on news if uncertainty exceeds threshold
            if uncertainty < NEWS_UNCERTAINTY_THRESHOLD and mm_action == "WIDEN":
                mm_action    = "NORMAL"
                widen_factor = 1.0

            pause_until = None
            if mm_action == "PAUSE" and pause_mins > 0:
                pause_until = (
                    datetime.utcnow() + timedelta(minutes=pause_mins)
                ).isoformat() + "Z"

            return {
                "source":             "NEWS",
                "mm_action":          mm_action,
                "widen_factor":       widen_factor,
                "pause_until":        pause_until,
                "sentiment":          worst.get("sentiment", 0.0),
                "uncertainty":        uncertainty,
                "informed_flow_risk": worst.get("informed_flow_risk", "LOW"),
                "event_type":         worst.get("event_type", "noise"),
                "reason":             worst.get("llm_reasoning", "")[:100],
                "confidence":         min(uncertainty + 0.3, 1.0),
                "headline":           worst.get("headline", "")[:80],
            }

        except Exception as e:
            log.debug(f"News signal build failed for {futures_symbol}: {e}")
            return self._normal("News processor unavailable")

    def _normal(self, reason: str) -> dict:
        return {
            "source":             "NEWS",
            "mm_action":          "NORMAL",
            "widen_factor":       1.0,
            "pause_until":        None,
            "sentiment":          0.0,
            "uncertainty":        0.0,
            "informed_flow_risk": "LOW",
            "event_type":         "noise",
            "reason":             reason,
            "confidence":         1.0,
            "headline":           "",
        }


# ─── Core aggregator ──────────────────────────────────────────────────────────

class SignalAggregator:
    """
    Combines all signal components into one IntelSignal per symbol.

    Aggregation rules:
    1. Take the most cautious mm_action across all sources
       (AVOID > PAUSE > WIDEN > NORMAL)
    2. Compound widen factors (multiply, cap at MAX_WIDEN_FACTOR)
    3. Take the earliest pause_until across sources
    4. Carry forward metadata from the most impactful source
    """

    def __init__(self):
        self.event_builder = EventSignalBuilder()
        self.news_builder  = NewsSignalBuilder()

    def aggregate(
        self,
        futures_symbol: str,
        ttl_minutes:    int = SIGNAL_TTL_MINUTES,
    ):
        """
        Build aggregated IntelSignal for one futures symbol.
        Returns IntelSignal object.
        """
        from intelligence.signals.schema import (
            IntelSignal, SignalSource, FinancialHealth
        )

        now     = datetime.utcnow()
        expires = now + timedelta(minutes=ttl_minutes)

        # ── Build component signals ──
        event_sig = self.event_builder.build(futures_symbol)
        news_sig  = self.news_builder.build(futures_symbol)

        components = [event_sig, news_sig]

        # ── Aggregate mm_action (most cautious wins) ──
        best_action = "NORMAL"
        for comp in components:
            action = comp.get("mm_action", "NORMAL")
            if ACTION_PRIORITY.get(action, 0) > ACTION_PRIORITY.get(best_action, 0):
                best_action = action

        # ── Aggregate widen factor (compound, capped) ──
        compound_widen = 1.0
        for comp in components:
            factor = comp.get("widen_factor", 1.0)
            if factor > 1.0:
                compound_widen *= factor
        compound_widen = round(min(compound_widen, MAX_WIDEN_FACTOR), 2)

        # ── Aggregate pause_until (earliest) ──
        pause_until = None
        for comp in components:
            pu = comp.get("pause_until")
            if pu:
                if pause_until is None or pu < pause_until:
                    pause_until = pu

        # ── Build reason string ──
        reasons = []
        for comp in components:
            if comp.get("mm_action", "NORMAL") != "NORMAL":
                reasons.append(
                    f"[{comp['source']}] {comp.get('reason', '')}"
                )
        reason = " | ".join(reasons) if reasons else "All clear"

        # ── Determine dominant source ──
        dominant_src = SignalSource.DEFAULT
        for comp in components:
            if comp.get("mm_action", "NORMAL") != "NORMAL":
                src = comp.get("source", "")
                if src == "EVENTS":
                    dominant_src = (
                        SignalSource.EARNINGS
                        if comp.get("event_type") == "EARNINGS"
                        else SignalSource.CORPORATE
                    )
                elif src == "NEWS":
                    dominant_src = SignalSource.NEWS

        # ── Pull metadata from components ──
        news_sentiment   = news_sig.get("sentiment", 0.0)
        news_uncertainty = news_sig.get("uncertainty", 0.0)
        active_news      = news_sig.get("mm_action", "NORMAL") != "NORMAL"

        has_event    = event_sig.get("has_event", False)
        event_type   = event_sig.get("event_type")
        event_date   = None
        days_to_event= event_sig.get("days_to_event")

        info_risk = news_sig.get("informed_flow_risk", "LOW")
        if has_event and event_sig.get("mm_action") in ("PAUSE", "AVOID"):
            info_risk = "HIGH"
        elif has_event and event_sig.get("mm_action") == "WIDEN":
            if info_risk == "LOW":
                info_risk = "MEDIUM"

        # ── Build component scores dict ──
        component_scores = {
            "event": {
                "action":       event_sig.get("mm_action", "NORMAL"),
                "widen_factor": event_sig.get("widen_factor", 1.0),
                "has_event":    has_event,
                "event_type":   event_type,
                "days_to_event":days_to_event,
                "reason":       event_sig.get("reason", ""),
            },
            "news": {
                "action":       news_sig.get("mm_action", "NORMAL"),
                "widen_factor": news_sig.get("widen_factor", 1.0),
                "uncertainty":  news_uncertainty,
                "sentiment":    news_sentiment,
                "flow_risk":    news_sig.get("informed_flow_risk", "LOW"),
                "headline":     news_sig.get("headline", ""),
                "reason":       news_sig.get("reason", ""),
            },
        }

        # ── Assemble final IntelSignal ──
        signal = IntelSignal(
            symbol             = futures_symbol,
            generated_at       = now.isoformat() + "Z",
            expires_at         = expires.isoformat() + "Z",
            mm_action          = best_action,
            widen_factor       = compound_widen,
            pause_until        = pause_until,
            confidence         = 0.9,
            source             = dominant_src,
            reason             = reason[:200],
            news_sentiment     = news_sentiment,
            news_uncertainty   = news_uncertainty,
            active_news        = active_news,
            has_upcoming_event = has_event,
            event_type         = event_type,
            event_date         = event_date,
            days_to_event      = days_to_event,
            mm_suitability     = 0.5,
            financial_health   = FinancialHealth.MODERATE,
            information_asymmetry_risk = info_risk,
            component_scores   = component_scores,
        )

        return signal

    def aggregate_all(
        self,
        symbols:     list[str],
        dry_run:     bool = False,
        verbose:     bool = False,
    ) -> dict:
        """
        Aggregate signals for all symbols.
        Returns summary dict with counts.
        """
        results = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total":        len(symbols),
            "saved":        0,
            "pause":        [],
            "widen":        [],
            "normal":       0,
            "errors":       [],
        }

        for sym in symbols:
            try:
                signal = self.aggregate(sym)

                if verbose or signal.mm_action != "NORMAL":
                    self._print_signal(signal)

                if not dry_run:
                    signal.save()
                    results["saved"] += 1

                if signal.mm_action in ("PAUSE", "AVOID"):
                    results["pause"].append({
                        "symbol": sym,
                        "reason": signal.reason[:80],
                        "factor": signal.widen_factor,
                    })
                elif signal.mm_action == "WIDEN":
                    results["widen"].append({
                        "symbol": sym,
                        "factor": signal.widen_factor,
                        "reason": signal.reason[:80],
                    })
                else:
                    results["normal"] += 1

            except Exception as e:
                log.warning(f"Aggregation failed for {sym}: {e}")
                results["errors"].append(sym)

        return results

    def _print_signal(self, signal):
        action_icon = {
            "PAUSE":  "🔴",
            "AVOID":  "⛔",
            "WIDEN":  "⚠️ ",
            "NORMAL": "✓ ",
        }.get(signal.mm_action, "  ")

        print(
            f"  {action_icon} {signal.symbol:<30} "
            f"{signal.mm_action:<8} "
            f"widen={signal.effective_widen_factor():.1f}x  "
            f"{signal.reason[:60]}"
        )


# ─── Summary printer ──────────────────────────────────────────────────────────

def print_summary(results: dict):
    print(f"\n{'=' * 70}")
    print(f"  SIGNAL AGGREGATION SUMMARY")
    print(f"  {results['generated_at'][:19].replace('T', ' ')} UTC")
    print(f"{'=' * 70}")
    print(f"  Total symbols:  {results['total']}")
    print(f"  Saved to GCS:   {results['saved']}")
    print(f"  🔴 PAUSE/AVOID: {len(results['pause'])}")
    print(f"  ⚠️  WIDEN:       {len(results['widen'])}")
    print(f"  ✓  NORMAL:      {results['normal']}")
    print(f"  ✗  Errors:      {len(results['errors'])}")

    if results["pause"]:
        print(f"\n── Paused Symbols ───────────────────────────────────────────")
        for s in results["pause"]:
            print(f"  {s['symbol']:<32} {s['reason'][:60]}")

    if results["widen"]:
        print(f"\n── Widened Symbols ──────────────────────────────────────────")
        for s in results["widen"]:
            print(f"  {s['symbol']:<32} {s['factor']:.1f}x  {s['reason'][:50]}")

    if results["errors"]:
        print(f"\n── Errors ───────────────────────────────────────────────────")
        for sym in results["errors"]:
            print(f"  {sym}")

    print(f"{'=' * 70}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Signal Aggregator — combines all intelligence into IntelSignal"
    )
    parser.add_argument("--all",      action="store_true",
                        help="Aggregate signals for all futures symbols")
    parser.add_argument("--symbol",   type=str,
                        help="Aggregate signal for single symbol")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Aggregate but don't save to GCS")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print every symbol including NORMAL ones")
    args = parser.parse_args()

    agg = SignalAggregator()

    if args.symbol:
        log.info(f"Aggregating signal for {args.symbol}...")
        signal = agg.aggregate(args.symbol)

        print(f"\n{'=' * 60}")
        print(f"  IntelSignal — {signal.symbol}")
        print(f"{'=' * 60}")
        print(signal.to_json())

        if not args.dry_run:
            signal.save()
            log.info(f"Saved to GCS")

    elif args.all:
        symbols = load_futures_symbols()
        log.info(f"Aggregating signals for {len(symbols)} symbols...")

        results = agg.aggregate_all(
            symbols = symbols,
            dry_run = args.dry_run,
            verbose = args.verbose,
        )
        print_summary(results)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
