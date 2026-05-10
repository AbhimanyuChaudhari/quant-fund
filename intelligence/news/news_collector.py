"""
News Collector
==============
Fetches news and announcements from:
    1. NSE corporate announcements (official filings)
    2. BSE corporate announcements (official filings)
    3. MoneyControl RSS (market rumors, analyst calls)
    4. Economic Times Markets RSS (broad market news)

Saves raw NewsItems to GCS for processor.py to score.
Deduplicates across sources using content hash.

Runs:
    Every 5 minutes during market hours via cron
    Cron: */5 3-10 * * 1-5 (UTC = 8:30am-4pm IST)

Usage:
    # Single fetch run
    python intelligence/news/news_collector.py --fetch

    # Fetch and show what was collected
    python intelligence/news/news_collector.py --fetch --verbose

    # Check what's in GCS for today
    python intelligence/news/news_collector.py --show-today

    # Import
    from intelligence.news.news_collector import NewsCollector
    collector = NewsCollector()
    items = collector.fetch_all()
"""

import argparse
import hashlib
import json
import logging
import re
import time
import warnings
from datetime import datetime, timedelta
from typing import Optional
from xml.etree import ElementTree as ET

import requests

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

GCS_BUCKET     = "hedge-fund-494103-marketdata"
NEWS_RAW_PREFIX = "intelligence/news/raw"
NEWS_IDX_PREFIX = "intelligence/news/index"

# NSE headers (required for API access)
NSE_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/",
    "Connection":      "keep-alive",
}

BSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept":     "application/json",
    "Referer":    "https://www.bseindia.com/",
}

# RSS feed URLs
RSS_FEEDS = {
    "moneycontrol_buzzing": "https://www.moneycontrol.com/rss/buzzingstocks.xml",
    "moneycontrol_news":    "https://www.moneycontrol.com/rss/latestnews.xml",
    "et_markets":           "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "et_stocks":            "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
}

# NSE equity symbols we care about (futures universe)
# Loaded from config if available, else hardcoded core set
CORE_SYMBOLS = {
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
    "AXISBANK", "KOTAKBANK", "SBIN", "BHARTIARTL", "BAJFINANCE",
    "ADANIPORTS", "ADANIENT", "HAL", "BEL", "TATASTEEL",
    "JSWSTEEL", "HINDALCO", "ONGC", "BPCL", "IOC",
    "DRREDDY", "SUNPHARMA", "CIPLA", "LUPIN", "DIVISLAB",
    "MARUTI", "BAJAJ-AUTO", "TVSMOTOR", "EICHERMOT", "HEROMOTOCO",
    "ULTRACEMCO", "GRASIM", "DLF", "GODREJPROP", "PRESTIGE",
    "CHOLAFIN", "BAJAJFINSV", "MUTHOOTFIN", "LICHSGFIN",
    "LT", "POWERGRID", "NTPC", "PFC", "RECLTD",
    "TCS", "HCLTECH", "WIPRO", "TECHM", "INFY",
    "TITAN", "NESTLEIND", "HINDUNILVR", "BRITANNIA", "TATACONSUM",
    "COALINDIA", "SAIL", "VEDL", "HINDALCO",
}

# How long to keep news items (days)
NEWS_RETENTION_DAYS = 7

# Request timeout
TIMEOUT = 10


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_news_id(source: str, headline: str) -> str:
    """Stable hash ID for deduplication."""
    content = f"{source}:{headline}".encode("utf-8")
    return hashlib.md5(content).hexdigest()[:12]


def extract_symbols(text: str, known_symbols: set = None) -> list[str]:
    """
    Extract NSE symbols mentioned in text.
    Looks for known symbols + common patterns.
    """
    if known_symbols is None:
        known_symbols = CORE_SYMBOLS

    found = set()
    text_upper = text.upper()

    # Direct symbol match
    for sym in known_symbols:
        if sym in text_upper:
            found.add(sym)

    # Pattern: "NSE: SYMBOL" or "BSE: SYMBOL"
    patterns = [
        r'NSE[:\s]+([A-Z]{2,10})',
        r'BSE[:\s]+([A-Z]{2,10})',
        r'\(NSE:\s*([A-Z]{2,10})\)',
        r'\(BSE:\s*([A-Z]{2,10})\)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text_upper)
        for m in matches:
            if len(m) >= 2:
                found.add(m)

    return sorted(found)


def parse_rss_date(date_str: str) -> str:
    """Parse RSS date to ISO UTC string."""
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.utctimetuple()
        except ValueError:
            continue
    return datetime.utcnow().isoformat() + "Z"


# ─── NSE Fetcher ──────────────────────────────────────────────────────────────

class NSENewsFetcher:
    """Fetches corporate announcements from NSE India."""

    ANNOUNCEMENTS_URL = "https://www.nseindia.com/api/corporate-announcements"
    INSIDER_URL       = "https://www.nseindia.com/api/corporates-insider-trading"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)
        self._init_session()

    def _init_session(self):
        try:
            self.session.get("https://www.nseindia.com", timeout=TIMEOUT)
            time.sleep(1)
        except Exception as e:
            log.warning(f"NSE session init: {e}")

    def fetch_announcements(self, index: str = "equities") -> list[dict]:
        """Fetch latest corporate announcements."""
        try:
            params = {"index": index}
            resp   = self.session.get(
                self.ANNOUNCEMENTS_URL,
                params=params,
                timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                log.warning(f"NSE announcements: HTTP {resp.status_code}")
                return []

            items = resp.json()
            if not isinstance(items, list):
                items = items.get("data", [])

            news_items = []
            now = datetime.utcnow().isoformat() + "Z"

            for item in items:
                try:
                    symbol   = item.get("symbol", "")
                    subject  = item.get("subject", "") or item.get("desc", "")
                    body     = item.get("body", "") or subject
                    url      = item.get("attchmntFile", "") or ""
                    date_str = item.get("exchdisstime", "") or now

                    if not symbol or not subject:
                        continue

                    news_id = make_news_id("NSE", f"{symbol}:{subject}")

                    news_items.append({
                        "id":         news_id,
                        "source":     "NSE",
                        "headline":   f"[{symbol}] {subject}",
                        "body":       body,
                        "url":        url,
                        "published":  date_str,
                        "symbols":    [symbol],
                        "fetched_at": now,
                        "scored":     False,
                    })
                except Exception as e:
                    log.debug(f"NSE item parse error: {e}")
                    continue

            log.info(f"NSE: {len(news_items)} announcements")
            return news_items

        except Exception as e:
            log.warning(f"NSE fetch failed: {e}")
            return []


# ─── BSE Fetcher ──────────────────────────────────────────────────────────────

class BSENewsFetcher:
    """Fetches corporate announcements from BSE India."""

    ANN_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(BSE_HEADERS)

    def fetch_announcements(self, days_back: int = 1) -> list[dict]:
        """Fetch recent BSE announcements."""
        try:
            today     = datetime.utcnow().date()
            from_date = today - timedelta(days=days_back)

            params = {
                "pageno":    "1",
                "strCat":    "-1",
                "strPrevDate": str(from_date),
                "strScrip":  "",
                "strSearch": "P",
                "strToDate": str(today),
                "strType":   "C",
                "subcategory": "-1",
            }

            resp = self.session.get(self.ANN_URL, params=params, timeout=TIMEOUT)
            if resp.status_code != 200:
                log.warning(f"BSE announcements: HTTP {resp.status_code}")
                return []

            data  = resp.json()
            items = data.get("Table", []) if isinstance(data, dict) else []

            news_items = []
            now = datetime.utcnow().isoformat() + "Z"

            for item in items:
                try:
                    scrip    = str(item.get("SCRIP_CD", "") or "")
                    subject  = item.get("NEWSSUB", "") or item.get("HEADLINE", "")
                    body     = item.get("HEADLINE", "") or subject
                    date_str = item.get("NEWS_DT", now)

                    if not subject:
                        continue

                    news_id  = make_news_id("BSE", f"{scrip}:{subject}")
                    symbols  = extract_symbols(f"{scrip} {subject}")

                    news_items.append({
                        "id":         news_id,
                        "source":     "BSE",
                        "headline":   subject,
                        "body":       body,
                        "url":        item.get("ATTACHMENTNAME", ""),
                        "published":  str(date_str),
                        "symbols":    symbols,
                        "fetched_at": now,
                        "scored":     False,
                    })
                except Exception as e:
                    log.debug(f"BSE item parse error: {e}")
                    continue

            log.info(f"BSE: {len(news_items)} announcements")
            return news_items

        except Exception as e:
            log.warning(f"BSE fetch failed: {e}")
            return []


# ─── RSS Fetcher ──────────────────────────────────────────────────────────────

class RSSFetcher:
    """Fetches news from RSS feeds."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; QuantFundBot/1.0)",
        })

    def fetch_feed(self, name: str, url: str) -> list[dict]:
        """Fetch and parse a single RSS feed."""
        try:
            resp = self.session.get(url, timeout=TIMEOUT)
            if resp.status_code != 200:
                log.warning(f"RSS {name}: HTTP {resp.status_code}")
                return []

            root  = ET.fromstring(resp.content)
            items = root.findall(".//item")

            news_items = []
            now = datetime.utcnow().isoformat() + "Z"

            for item in items:
                try:
                    headline = item.findtext("title", "").strip()
                    body     = item.findtext("description", "").strip()
                    url_item = item.findtext("link", "").strip()
                    pub_date = item.findtext("pubDate", now).strip()

                    if not headline:
                        continue

                    # Clean HTML from body
                    body_clean = re.sub(r'<[^>]+>', ' ', body).strip()
                    body_clean = re.sub(r'\s+', ' ', body_clean)

                    news_id = make_news_id(name, headline)
                    symbols = extract_symbols(f"{headline} {body_clean}")

                    # Only keep if relevant to our universe
                    # For RSS, filter loosely — processor.py does fine filtering
                    news_items.append({
                        "id":         news_id,
                        "source":     name.upper(),
                        "headline":   headline,
                        "body":       body_clean[:2000],   # cap length
                        "url":        url_item,
                        "published":  str(pub_date),
                        "symbols":    symbols,
                        "fetched_at": now,
                        "scored":     False,
                    })
                except Exception as e:
                    log.debug(f"RSS item parse error ({name}): {e}")
                    continue

            log.info(f"RSS {name}: {len(news_items)} items")
            return news_items

        except ET.ParseError as e:
            log.warning(f"RSS {name} parse error: {e}")
            return []
        except Exception as e:
            log.warning(f"RSS {name} fetch failed: {e}")
            return []

    def fetch_all_feeds(self) -> list[dict]:
        """Fetch all configured RSS feeds."""
        all_items = []
        for name, url in RSS_FEEDS.items():
            items = self.fetch_feed(name, url)
            all_items.extend(items)
            time.sleep(0.5)   # polite crawling
        return all_items


# ─── GCS Storage ──────────────────────────────────────────────────────────────

class NewsStorage:
    """Handles GCS persistence for news items."""

    def __init__(self, bucket: str = GCS_BUCKET):
        self.bucket_name = bucket
        self._client     = None
        self._seen_ids: set = set()
        self._loaded_today  = False

    def _client_lazy(self):
        if self._client is None:
            from google.cloud import storage
            self._client = storage.Client()
        return self._client

    def load_today_ids(self) -> set:
        """Load IDs of news items already saved today (for deduplication)."""
        if self._loaded_today:
            return self._seen_ids

        try:
            today  = datetime.utcnow().date()
            prefix = f"{NEWS_IDX_PREFIX}/{today}/"
            blobs  = self._client_lazy().list_blobs(
                self.bucket_name, prefix=prefix
            )
            for blob in blobs:
                name = blob.name.split("/")[-1].replace(".json", "")
                self._seen_ids.add(name)
            self._loaded_today = True
            log.info(f"Loaded {len(self._seen_ids)} existing news IDs for today")
        except Exception as e:
            log.warning(f"Could not load today's news IDs: {e}")

        return self._seen_ids

    def save_items(self, items: list[dict]) -> tuple[int, int]:
        """
        Save news items to GCS.
        Returns (saved_count, skipped_count).
        """
        existing = self.load_today_ids()
        saved    = 0
        skipped  = 0
        today    = datetime.utcnow().date()
        bkt      = self._client_lazy().bucket(self.bucket_name)

        for item in items:
            news_id = item.get("id", "")
            if not news_id:
                continue

            # Deduplicate
            if news_id in existing:
                skipped += 1
                continue

            try:
                # Save full item
                path = f"{NEWS_RAW_PREFIX}/{today}/{news_id}.json"
                bkt.blob(path).upload_from_string(
                    json.dumps(item, indent=2, default=str),
                    content_type="application/json",
                )

                # Save to index (lightweight, for fast ID lookup)
                idx_path = f"{NEWS_IDX_PREFIX}/{today}/{news_id}.json"
                bkt.blob(idx_path).upload_from_string(
                    json.dumps({"id": news_id, "headline": item.get("headline", "")}),
                    content_type="application/json",
                )

                existing.add(news_id)
                saved += 1

            except Exception as e:
                log.debug(f"Save failed for {news_id}: {e}")
                continue

        return saved, skipped

    def load_today_items(self) -> list[dict]:
        """Load all news items from today."""
        try:
            today  = datetime.utcnow().date()
            prefix = f"{NEWS_RAW_PREFIX}/{today}/"
            blobs  = self._client_lazy().list_blobs(
                self.bucket_name, prefix=prefix
            )
            items = []
            for blob in blobs:
                try:
                    data = json.loads(blob.download_as_text())
                    items.append(data)
                except Exception:
                    continue
            return items
        except Exception as e:
            log.warning(f"Could not load today's items: {e}")
            return []

    def load_unscored(self) -> list[dict]:
        """Load news items not yet scored by processor.py."""
        items = self.load_today_items()
        return [i for i in items if not i.get("scored", False)]

    def mark_scored(self, news_id: str, scores: dict):
        """Update a news item with LLM scores."""
        try:
            today = datetime.utcnow().date()
            path  = f"{NEWS_RAW_PREFIX}/{today}/{news_id}.json"
            bkt   = self._client_lazy().bucket(self.bucket_name)
            blob  = bkt.blob(path)

            if not blob.exists():
                return False

            item = json.loads(blob.download_as_text())
            item.update(scores)
            item["scored"] = True

            blob.upload_from_string(
                json.dumps(item, indent=2, default=str),
                content_type="application/json",
            )
            return True
        except Exception as e:
            log.debug(f"mark_scored failed for {news_id}: {e}")
            return False


# ─── Main collector ───────────────────────────────────────────────────────────

class NewsCollector:
    """
    Main news collection orchestrator.

    Usage:
        collector = NewsCollector()
        items     = collector.fetch_all()
        saved, skipped = collector.save(items)
    """

    def __init__(self):
        self.nse     = NSENewsFetcher()
        self.bse     = BSENewsFetcher()
        self.rss     = RSSFetcher()
        self.storage = NewsStorage()

    def fetch_all(self, verbose: bool = False) -> list[dict]:
        """Fetch from all sources, deduplicate, return combined list."""
        all_items = []

        # NSE announcements
        log.info("Fetching NSE announcements...")
        nse_items = self.nse.fetch_announcements()
        all_items.extend(nse_items)
        time.sleep(1)

        # BSE announcements
        log.info("Fetching BSE announcements...")
        bse_items = self.bse.fetch_announcements()
        all_items.extend(bse_items)
        time.sleep(1)

        # RSS feeds
        log.info("Fetching RSS feeds...")
        rss_items = self.rss.fetch_all_feeds()
        all_items.extend(rss_items)

        # Deduplicate by ID
        seen     = set()
        unique   = []
        for item in all_items:
            if item["id"] not in seen:
                seen.add(item["id"])
                unique.append(item)

        log.info(f"Total unique items: {len(unique)} "
                 f"(from {len(all_items)} raw)")

        if verbose:
            self._print_items(unique)

        return unique

    def save(self, items: list[dict]) -> tuple[int, int]:
        """Save items to GCS. Returns (saved, skipped)."""
        saved, skipped = self.storage.save_items(items)
        log.info(f"Saved: {saved} new | Skipped: {skipped} duplicates")
        return saved, skipped

    def fetch_and_save(self, verbose: bool = False) -> tuple[int, int]:
        """Convenience method: fetch all + save."""
        items = self.fetch_all(verbose=verbose)
        return self.save(items)

    def get_unscored(self) -> list[dict]:
        """Get items not yet processed by LLM."""
        return self.storage.load_unscored()

    def get_relevant(self, symbol: str) -> list[dict]:
        """
        Get today's news items relevant to a specific symbol.
        Used by aggregator.py.
        """
        equity = symbol.upper()
        # Strip futures suffix if present
        equity = re.sub(r'\d{2}[A-Z]{3}FUT$', '', equity)

        items = self.storage.load_today_items()
        return [
            i for i in items
            if equity in i.get("symbols", [])
            or equity in i.get("headline", "").upper()
        ]

    def _print_items(self, items: list[dict]):
        """Print items to terminal."""
        print(f"\n{'=' * 70}")
        print(f"  NEWS COLLECTED — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'=' * 70}")

        by_source: dict[str, list] = {}
        for item in items:
            src = item.get("source", "UNKNOWN")
            if src not in by_source:
                by_source[src] = []
            by_source[src].append(item)

        for source, source_items in sorted(by_source.items()):
            print(f"\n── {source} ({len(source_items)} items) ──────────────")
            for item in source_items[:10]:
                syms = ", ".join(item.get("symbols", [])[:3])
                syms = f" [{syms}]" if syms else ""
                print(f"  {item['headline'][:70]}{syms}")
            if len(source_items) > 10:
                print(f"  ... and {len(source_items) - 10} more")

        print(f"\n{'=' * 70}")
        print(f"  Total: {len(items)} items\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="News Collector — NSE/BSE/RSS news fetcher"
    )
    parser.add_argument("--fetch",      action="store_true",
                        help="Fetch news from all sources")
    parser.add_argument("--show-today", action="store_true",
                        help="Show today's collected news from GCS")
    parser.add_argument("--symbol",     type=str,
                        help="Filter news for specific symbol")
    parser.add_argument("--unscored",   action="store_true",
                        help="Show unscored items (not yet LLM processed)")
    parser.add_argument("--verbose",    action="store_true",
                        help="Print all collected items")
    parser.add_argument("--no-save",    action="store_true",
                        help="Fetch but don't save to GCS")
    args = parser.parse_args()

    collector = NewsCollector()

    if args.fetch:
        items = collector.fetch_all(verbose=args.verbose)
        if not args.no_save:
            saved, skipped = collector.save(items)
            print(f"\nSaved: {saved} new items | Skipped: {skipped} duplicates")
        else:
            print(f"\nFetched: {len(items)} items (not saved)")

    if args.show_today:
        items = collector.storage.load_today_items()
        print(f"\nToday's news: {len(items)} items")
        collector._print_items(items)

    if args.symbol:
        items = collector.get_relevant(args.symbol)
        print(f"\nNews for {args.symbol}: {len(items)} items")
        for item in items:
            print(f"  [{item['source']}] {item['headline'][:70]}")
            print(f"  Published: {item['published']}")
            print(f"  Scored: {item.get('scored', False)}")
            print()

    if args.unscored:
        items = collector.get_unscored()
        print(f"\nUnscored items: {len(items)}")
        for item in items[:20]:
            print(f"  {item['id']} | {item['source']} | {item['headline'][:60]}")


if __name__ == "__main__":
    main()