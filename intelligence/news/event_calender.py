"""
Event Calendar
==============
Fetches and caches corporate events from NSE and BSE.
Produces IntelSignal with PAUSE/WIDEN actions for upcoming events.

Events tracked:
    Quarterly results (earnings)
    Board meetings
    AGM / EGM
    Dividend announcements
    Bonus / stock splits
    Rights issues

Runs:
    Daily at 6:00am IST (before market open)
    Cron: 30 0 * * 1-5 cd /home/abhim/quant-fund && venv/bin/python intelligence/news/event_calendar.py

Usage:
    # Fetch and cache all events
    python intelligence/news/event_calendar.py --fetch

    # Check upcoming events for a symbol
    python intelligence/news/event_calendar.py --symbol RELIANCE26MAYFUT

    # Check all symbols, generate signals
    python intelligence/news/event_calendar.py --all --generate-signals

    # Import in other modules
    from intelligence.news.event_calendar import EventCalendar
    cal = EventCalendar()
    events = cal.get_upcoming(symbol, days_ahead=7)
    signal = cal.get_signal(symbol)
"""

import argparse
import hashlib
import io
import json
import logging
import time
import warnings
from datetime import datetime, timedelta
from typing import Optional

import requests

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

GCS_BUCKET         = "hedge-fund-494103-marketdata-mumbai"
EVENTS_PREFIX      = "intelligence/events"
SIGNALS_PREFIX     = "signals"

# NSE API endpoints
NSE_EVENT_CALENDAR = "https://www.nseindia.com/api/event-calendar"
NSE_CORP_ACTIONS   = "https://www.nseindia.com/api/corporates-corporateActions"
NSE_RESULTS_CAL    = "https://www.nseindia.com/api/corporates-financial-results"

# BSE API endpoints
BSE_CORP_ACTIONS   = "https://api.bseindia.com/BseIndiaAPI/api/DefaultData/w"
BSE_ANNOUNCEMENTS  = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"

# Request headers — NSE requires browser-like headers
NSE_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/",
    "Connection":      "keep-alive",
}

# How many days ahead to look for events
LOOKAHEAD_DAYS = 30

# Symbol mapping — NSE futures symbol → NSE equity symbol
# Used to look up corporate events (events are filed under equity symbol)
def futures_to_equity(symbol: str) -> str:
    """
    Convert futures symbol to equity symbol.
    RELIANCE26MAYFUT → RELIANCE
    NIFTY26MAYFUT    → NIFTY 50
    USDINR26508FUT   → None (currency, no equity events)
    """
    if "USDINR" in symbol:
        return None
    # Strip expiry suffix: RELIANCE26MAYFUT → RELIANCE
    # Pattern: remove trailing digits + month abbreviation + FUT
    import re
    equity = re.sub(r'\d{2}[A-Z]{3}FUT$', '', symbol)
    return equity if equity else None


# ─── Event data structures (local, mirrors schema.py) ─────────────────────────

class EventType:
    EARNINGS      = "EARNINGS"
    DIVIDEND      = "DIVIDEND"
    BOARD_MEETING = "BOARD_MEETING"
    AGM           = "AGM"
    BONUS_SPLIT   = "BONUS_SPLIT"
    RIGHTS_ISSUE  = "RIGHTS_ISSUE"
    MERGER        = "MERGER_ACQUIRE"
    OTHER         = "OTHER"


def classify_event(description: str) -> str:
    """Classify event type from description string."""
    desc = description.upper()
    if any(x in desc for x in ["RESULT", "EARNING", "FINANCIAL", "QUARTERLY", "Q1", "Q2", "Q3", "Q4"]):
        return EventType.EARNINGS
    if any(x in desc for x in ["DIVIDEND", "INTERIM DIV", "FINAL DIV"]):
        return EventType.DIVIDEND
    if any(x in desc for x in ["BOARD MEETING", "BOARD MTG"]):
        return EventType.BOARD_MEETING
    if any(x in desc for x in ["AGM", "EGM", "ANNUAL GENERAL"]):
        return EventType.AGM
    if any(x in desc for x in ["BONUS", "SPLIT", "SUBDIVISION"]):
        return EventType.BONUS_SPLIT
    if any(x in desc for x in ["RIGHTS", "RIGHT ISSUE"]):
        return EventType.RIGHTS_ISSUE
    if any(x in desc for x in ["MERGER", "AMALGAMATION", "ACQUISITION", "TAKEOVER"]):
        return EventType.MERGER
    return EventType.OTHER


def event_mm_impact(event_type: str, days_until: int) -> tuple:
    """
    Returns (mm_action, widen_factor, pause_minutes).
    This mirrors CorporateEvent.mm_impact() from schema.py.
    """
    if event_type == EventType.EARNINGS:
        if days_until == 0:
            return "PAUSE", 3.0, 120
        elif days_until == 1:
            return "WIDEN", 2.0, 0
        elif days_until <= 3:
            return "WIDEN", 1.5, 0
        elif days_until <= 7:
            return "WIDEN", 1.2, 0

    elif event_type == EventType.MERGER:
        if days_until <= 1:
            return "AVOID", 5.0, 0
        elif days_until <= 7:
            return "WIDEN", 2.5, 0

    elif event_type == EventType.BOARD_MEETING:
        if days_until == 0:
            return "WIDEN", 1.5, 0
        elif days_until <= 2:
            return "WIDEN", 1.2, 0

    elif event_type == EventType.BONUS_SPLIT:
        if days_until <= 1:
            return "WIDEN", 1.3, 0

    elif event_type == EventType.DIVIDEND:
        if days_until == 0:
            return "WIDEN", 1.2, 0

    return "NORMAL", 1.0, 0


# ─── NSE fetcher ──────────────────────────────────────────────────────────────

class NSEFetcher:
    """Fetches corporate events from NSE India."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)
        self._init_session()

    def _init_session(self):
        """NSE requires a cookie from the homepage first."""
        try:
            self.session.get(
                "https://www.nseindia.com",
                timeout=10
            )
            time.sleep(1)
        except Exception as e:
            log.warning(f"NSE session init failed: {e}")

    def fetch_event_calendar(self, days_ahead: int = LOOKAHEAD_DAYS) -> list[dict]:
        """Fetch upcoming events from NSE event calendar."""
        today     = datetime.utcnow().date()
        end_date  = today + timedelta(days=days_ahead)

        params = {
            "index": "equities",
        }

        try:
            resp = self.session.get(
                NSE_EVENT_CALENDAR,
                params=params,
                timeout=15,
            )
            if resp.status_code != 200:
                log.warning(f"NSE event calendar returned {resp.status_code}")
                return []

            data = resp.json()
            events = []

            for item in data:
                try:
                    event_date_str = item.get("date", "")
                    if not event_date_str:
                        continue

                    # Parse NSE date format (DD-MMM-YYYY)
                    try:
                        event_date = datetime.strptime(
                            event_date_str, "%d-%b-%Y"
                        ).date()
                    except ValueError:
                        try:
                            event_date = datetime.strptime(
                                event_date_str, "%Y-%m-%d"
                            ).date()
                        except ValueError:
                            continue

                    if event_date < today or event_date > end_date:
                        continue

                    symbol      = item.get("symbol", "")
                    description = item.get("purpose", "") or item.get("description", "")
                    event_type  = classify_event(description)
                    days_until  = (event_date - today).days

                    events.append({
                        "symbol":      symbol,
                        "event_type":  event_type,
                        "event_date":  str(event_date),
                        "description": description,
                        "days_until":  days_until,
                        "source":      "NSE",
                        "confirmed":   True,
                    })

                except Exception as e:
                    log.debug(f"Error parsing NSE event: {e}")
                    continue

            log.info(f"NSE: fetched {len(events)} events")
            return events

        except Exception as e:
            log.warning(f"NSE fetch failed: {e}")
            return []

    def fetch_corp_actions(self, symbol: str) -> list[dict]:
        """Fetch corporate actions for a specific equity symbol."""
        try:
            params = {
                "symbol":   symbol,
                "segment":  "equities",
                "series":   "EQ",
            }
            resp = self.session.get(
                NSE_CORP_ACTIONS,
                params=params,
                timeout=10,
            )
            if resp.status_code != 200:
                return []

            data   = resp.json()
            today  = datetime.utcnow().date()
            events = []

            for item in data:
                try:
                    date_str = item.get("exDate", "") or item.get("recordDate", "")
                    if not date_str:
                        continue
                    event_date = datetime.strptime(date_str, "%d-%b-%Y").date()
                    if event_date < today:
                        continue

                    desc       = item.get("subject", "") or item.get("purpose", "")
                    event_type = classify_event(desc)
                    days_until = (event_date - today).days

                    events.append({
                        "symbol":      symbol,
                        "event_type":  event_type,
                        "event_date":  str(event_date),
                        "description": desc,
                        "days_until":  days_until,
                        "source":      "NSE_ACTIONS",
                        "confirmed":   True,
                    })
                except Exception:
                    continue

            return events

        except Exception as e:
            log.debug(f"NSE corp actions failed for {symbol}: {e}")
            return []


# ─── BSE fetcher ──────────────────────────────────────────────────────────────

class BSEFetcher:
    """Fetches corporate events from BSE India."""

    BSE_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept":     "application/json",
        "Referer":    "https://www.bseindia.com/",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.BSE_HEADERS)

    def fetch_results_calendar(self, days_ahead: int = LOOKAHEAD_DAYS) -> list[dict]:
        """Fetch upcoming quarterly results from BSE."""
        today    = datetime.utcnow().date()
        end_date = today + timedelta(days=days_ahead)

        try:
            params = {
                "strCat":      "BP",
                "strPrevDate": str(today),
                "strScrip":    "",
                "strSearch":   "P",
                "strToDate":   str(end_date),
                "strType":     "C",
            }
            resp = self.session.get(
                "https://api.bseindia.com/BseIndiaAPI/api/DefaultData/w",
                params=params,
                timeout=15,
            )
            if resp.status_code != 200:
                return []

            data   = resp.json()
            events = []

            for item in data.get("Table", []):
                try:
                    date_str = item.get("Meeting_Date", "")
                    if not date_str:
                        continue

                    try:
                        event_date = datetime.strptime(
                            date_str[:10], "%Y-%m-%d"
                        ).date()
                    except ValueError:
                        continue

                    if event_date < today or event_date > end_date:
                        continue

                    symbol     = item.get("scrip_cd", "") or item.get("SCRIP_CD", "")
                    desc       = item.get("MEETING_TYPE", "") or "Board Meeting"
                    event_type = classify_event(desc)
                    days_until = (event_date - today).days

                    events.append({
                        "symbol":      str(symbol),
                        "event_type":  event_type,
                        "event_date":  str(event_date),
                        "description": desc,
                        "days_until":  days_until,
                        "source":      "BSE",
                        "confirmed":   True,
                    })
                except Exception:
                    continue

            log.info(f"BSE: fetched {len(events)} events")
            return events

        except Exception as e:
            log.warning(f"BSE fetch failed: {e}")
            return []


# ─── Event Calendar (main class) ──────────────────────────────────────────────

class EventCalendar:
    """
    Main interface for corporate event data.

    Usage:
        cal    = EventCalendar()
        events = cal.get_upcoming('RELIANCE', days_ahead=7)
        signal = cal.to_intel_signal('RELIANCE26MAYFUT')
        signal.save()
    """

    def __init__(self, use_cache: bool = True):
        self.use_cache  = use_cache
        self._cache:    dict[str, list] = {}   # equity_symbol → events
        self._loaded:   bool = False
        self._nse       = NSEFetcher()
        self._bse       = BSEFetcher()

    def fetch_all(self) -> dict[str, list]:
        """
        Fetch all events from NSE + BSE.
        Returns dict: equity_symbol → list of events.
        Saves raw events to GCS.
        """
        log.info("Fetching events from NSE...")
        nse_events = self._nse.fetch_event_calendar()
        time.sleep(2)

        log.info("Fetching results calendar from BSE...")
        bse_events = self._bse.fetch_results_calendar()

        all_events = nse_events + bse_events

        # Group by equity symbol
        by_symbol: dict[str, list] = {}
        for event in all_events:
            sym = event.get("symbol", "")
            if not sym:
                continue
            if sym not in by_symbol:
                by_symbol[sym] = []
            # Deduplicate by date + type
            key = f"{event['event_date']}_{event['event_type']}"
            existing_keys = [
                f"{e['event_date']}_{e['event_type']}"
                for e in by_symbol[sym]
            ]
            if key not in existing_keys:
                by_symbol[sym].append(event)

        self._cache  = by_symbol
        self._loaded = True

        log.info(f"Total: {len(all_events)} events across {len(by_symbol)} symbols")

        # Save to GCS
        self._save_to_gcs(by_symbol)

        return by_symbol

    def load_from_gcs(self) -> bool:
        """Load cached events from GCS (faster than re-fetching)."""
        try:
            from google.cloud import storage
            client = storage.Client()
            today  = datetime.utcnow().date()
            path   = f"{EVENTS_PREFIX}/{today}/all_events.json"
            blob   = client.bucket(GCS_BUCKET).blob(path)

            if not blob.exists():
                log.info("No cached events found for today — will fetch fresh")
                return False

            data         = json.loads(blob.download_as_text())
            self._cache  = data
            self._loaded = True
            log.info(f"Loaded events from GCS cache ({len(data)} symbols)")
            return True

        except Exception as e:
            log.warning(f"GCS load failed: {e}")
            return False

    def _save_to_gcs(self, by_symbol: dict):
        """Save events to GCS for today."""
        try:
            from google.cloud import storage
            client = storage.Client()
            today  = datetime.utcnow().date()
            path   = f"{EVENTS_PREFIX}/{today}/all_events.json"
            client.bucket(GCS_BUCKET).blob(path).upload_from_string(
                json.dumps(by_symbol, indent=2),
                content_type="application/json",
            )
            log.info(f"Saved events → gs://{GCS_BUCKET}/{path}")
        except Exception as e:
            log.warning(f"GCS save failed: {e}")

    def _ensure_loaded(self):
        """Load from GCS cache or fetch fresh if needed."""
        if not self._loaded:
            if not self.load_from_gcs():
                self.fetch_all()

    def get_upcoming(
        self,
        equity_symbol: str,
        days_ahead:    int = 7,
    ) -> list[dict]:
        """
        Get upcoming events for an equity symbol.

        Args:
            equity_symbol: NSE equity ticker (e.g. 'RELIANCE', not futures)
            days_ahead:    how many days to look ahead

        Returns:
            List of event dicts sorted by date
        """
        self._ensure_loaded()
        events = self._cache.get(equity_symbol, [])
        today  = datetime.utcnow().date()

        upcoming = []
        for e in events:
            try:
                event_date = datetime.strptime(e["event_date"], "%Y-%m-%d").date()
                days_until = (event_date - today).days
                if 0 <= days_until <= days_ahead:
                    e = dict(e)
                    e["days_until"] = days_until
                    upcoming.append(e)
            except Exception:
                continue

        return sorted(upcoming, key=lambda x: x["event_date"])

    def get_upcoming_for_future(
        self,
        futures_symbol: str,
        days_ahead:     int = 7,
    ) -> list[dict]:
        """
        Get upcoming events for a futures symbol.
        Automatically converts futures → equity symbol.
        """
        equity = futures_to_equity(futures_symbol)
        if not equity:
            return []
        return self.get_upcoming(equity, days_ahead)

    def most_impactful_event(
        self,
        futures_symbol: str,
        days_ahead:     int = 7,
    ) -> Optional[dict]:
        """
        Returns the single most impactful upcoming event.
        Priority: EARNINGS > MERGER > BOARD_MEETING > others
        """
        events = self.get_upcoming_for_future(futures_symbol, days_ahead)
        if not events:
            return None

        priority = {
            EventType.MERGER:       0,
            EventType.EARNINGS:     1,
            EventType.BOARD_MEETING:2,
            EventType.AGM:          3,
            EventType.BONUS_SPLIT:  4,
            EventType.DIVIDEND:     5,
            EventType.RIGHTS_ISSUE: 6,
            EventType.OTHER:        7,
        }

        return min(
            events,
            key=lambda e: (
                priority.get(e["event_type"], 99),
                e["days_until"]
            )
        )

    def to_intel_signal(
        self,
        futures_symbol: str,
        ttl_hours:      int = 4,
    ):
        """
        Generate an IntelSignal from upcoming events.
        This is what mm_strategy.py ultimately uses.

        Returns an IntelSignal object (imports from schema.py).
        Falls back gracefully if schema import fails.
        """
        try:
            import sys
            import os
            # Add project root to path
            root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            if root not in sys.path:
                sys.path.insert(0, root)

            from intelligence.signals.schema import IntelSignal, SignalSource

        except ImportError:
            log.warning("Could not import IntelSignal — returning raw dict")
            return self._raw_signal(futures_symbol)

        now     = datetime.utcnow()
        expires = now + timedelta(hours=ttl_hours)

        event = self.most_impactful_event(futures_symbol)

        if not event:
            return IntelSignal.default(futures_symbol, ttl_minutes=ttl_hours * 60)

        event_type = event["event_type"]
        days_until = event["days_until"]
        action, widen, pause_mins = event_mm_impact(event_type, days_until)

        pause_until = None
        if action == "PAUSE" and pause_mins > 0:
            pause_until = (now + timedelta(minutes=pause_mins)).isoformat() + "Z"

        signal = IntelSignal(
            symbol             = futures_symbol,
            generated_at       = now.isoformat() + "Z",
            expires_at         = expires.isoformat() + "Z",
            mm_action          = action,
            widen_factor       = widen,
            pause_until        = pause_until,
            confidence         = 0.95,
            source             = SignalSource.EARNINGS
                                 if event_type == EventType.EARNINGS
                                 else SignalSource.CORPORATE,
            reason             = (
                f"{event_type} in {days_until} day(s): "
                f"{event.get('description', '')}"
            ),
            has_upcoming_event = True,
            event_type         = event_type,
            event_date         = event["event_date"],
            days_to_event      = days_until,
        )

        return signal

    def _raw_signal(self, futures_symbol: str) -> dict:
        """Fallback dict signal if schema.py import fails."""
        event = self.most_impactful_event(futures_symbol)
        if not event:
            return {
                "symbol":    futures_symbol,
                "mm_action": "NORMAL",
                "widen_factor": 1.0,
            }
        action, widen, pause = event_mm_impact(
            event["event_type"], event["days_until"]
        )
        return {
            "symbol":      futures_symbol,
            "mm_action":   action,
            "widen_factor":widen,
            "event_type":  event["event_type"],
            "days_until":  event["days_until"],
        }

    def generate_all_signals(self, futures_symbols: list[str]) -> dict:
        """
        Generate and save IntelSignals for all futures symbols.
        Call this daily after fetch_all().
        """
        self._ensure_loaded()
        results = {"saved": [], "normal": [], "errors": []}

        for sym in futures_symbols:
            try:
                signal = self.to_intel_signal(sym)
                if hasattr(signal, "save"):
                    signal.save()
                    if hasattr(signal, "mm_action") and signal.mm_action != "NORMAL":
                        results["saved"].append({
                            "symbol":    sym,
                            "action":    signal.mm_action,
                            "reason":    signal.reason,
                        })
                    else:
                        results["normal"].append(sym)
                else:
                    results["normal"].append(sym)
            except Exception as e:
                log.warning(f"Signal generation failed for {sym}: {e}")
                results["errors"].append(sym)

        return results

    def print_upcoming(self, days_ahead: int = 7):
        """Print all upcoming events across all symbols."""
        self._ensure_loaded()
        today = datetime.utcnow().date()

        all_events = []
        for sym, events in self._cache.items():
            for e in events:
                try:
                    event_date = datetime.strptime(
                        e["event_date"], "%Y-%m-%d"
                    ).date()
                    days_until = (event_date - today).days
                    if 0 <= days_until <= days_ahead:
                        all_events.append({**e, "days_until": days_until})
                except Exception:
                    continue

        all_events.sort(key=lambda x: (x["days_until"], x["symbol"]))

        print(f"\n{'=' * 75}")
        print(f"  UPCOMING CORPORATE EVENTS — Next {days_ahead} days")
        print(f"  As of {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'=' * 75}")
        print(f"  {'Symbol':<20} {'Date':<12} {'Days':>5} {'Type':<16} Description")
        print(f"  {'-' * 73}")

        for e in all_events:
            action, factor, _ = event_mm_impact(
                e["event_type"], e["days_until"]
            )
            flag = ""
            if action == "PAUSE":
                flag = " 🔴 PAUSE"
            elif action == "AVOID":
                flag = " ⛔ AVOID"
            elif action == "WIDEN":
                flag = f" ⚠️  WIDEN {factor}x"

            desc = e.get("description", "")[:35]
            print(
                f"  {e['symbol']:<20} {e['event_date']:<12} "
                f"{e['days_until']:>5} {e['event_type']:<16} "
                f"{desc}{flag}"
            )

        print(f"{'=' * 75}")
        print(f"  Total: {len(all_events)} events\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Event Calendar — Corporate event fetcher for quant fund"
    )
    parser.add_argument("--fetch",            action="store_true",
                        help="Fetch fresh events from NSE + BSE")
    parser.add_argument("--symbol",           type=str,
                        help="Check events for specific futures symbol")
    parser.add_argument("--all",              action="store_true",
                        help="Show all upcoming events")
    parser.add_argument("--generate-signals", action="store_true",
                        help="Generate and save IntelSignals for all symbols")
    parser.add_argument("--days",             type=int, default=7,
                        help="Days ahead to look (default 7)")
    parser.add_argument("--symbols-file",     type=str,
                        help="Path to file with futures symbols (one per line)")
    args = parser.parse_args()

    cal = EventCalendar()

    # Fetch fresh data
    if args.fetch:
        cal.fetch_all()
    else:
        # Try cache first
        if not cal.load_from_gcs():
            log.info("No cache found — fetching fresh")
            cal.fetch_all()

    # Single symbol check
    if args.symbol:
        equity = futures_to_equity(args.symbol)
        print(f"\nEvents for {args.symbol} (equity: {equity}):")

        events = cal.get_upcoming_for_future(args.symbol, days_ahead=args.days)
        if not events:
            print("  No upcoming events found")
        else:
            for e in events:
                action, factor, pause = event_mm_impact(
                    e["event_type"], e["days_until"]
                )
                print(
                    f"  {e['event_date']} ({e['days_until']}d) | "
                    f"{e['event_type']:<16} | "
                    f"Action: {action} {f'({factor}x)' if factor > 1 else ''} | "
                    f"{e.get('description', '')}"
                )

        signal = cal.to_intel_signal(args.symbol)
        print(f"\nIntelSignal for {args.symbol}:")
        if hasattr(signal, "to_json"):
            print(signal.to_json())
        else:
            print(json.dumps(signal, indent=2))

    # Show all events
    if args.all:
        cal.print_upcoming(days_ahead=args.days)

    # Generate signals for all symbols
    if args.generate_signals:
        symbols = []
        if args.symbols_file:
            with open(args.symbols_file) as f:
                symbols = [l.strip() for l in f if l.strip()]
        else:
            # Load from config
            try:
                import sys
                import os
                root = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                sys.path.insert(0, root)
                from config.symbols import get_all_futures
                symbols = get_all_futures()
            except ImportError:
                log.warning(
                    "Could not import config.symbols — "
                    "provide --symbols-file or --symbol"
                )
                return

        log.info(f"Generating signals for {len(symbols)} symbols...")
        results = cal.generate_all_signals(symbols)

        print(f"\n{'=' * 60}")
        print(f"  Signal Generation Results")
        print(f"{'=' * 60}")
        print(f"  Saved with action:  {len(results['saved'])}")
        print(f"  Normal (no event):  {len(results['normal'])}")
        print(f"  Errors:             {len(results['errors'])}")

        if results["saved"]:
            print(f"\n  Non-normal signals:")
            for s in results["saved"]:
                print(f"    {s['symbol']:<30} {s['action']:<8} {s['reason']}")
        print()


if __name__ == "__main__":
    main()
