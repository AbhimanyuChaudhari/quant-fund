"""
News Processor
==============
Scores raw NewsItems using Claude API.
Produces mm_action signals (PAUSE/WIDEN/NORMAL) for each symbol.

Pipeline:
    news_collector.py → raw NewsItems (GCS)
    processor.py      → scores each item → updates GCS
    aggregator.py     → combines scores → IntelSignal per symbol

Claude API is used to:
    1. Classify event type (earnings/merger/management/macro/noise)
    2. Score sentiment (-1 to 1)
    3. Score uncertainty (0 to 1) — key for MM
    4. Determine mm_action (PAUSE/WIDEN/NORMAL)
    5. Identify affected symbols

Cost estimate:
    ~500 tokens/article × 30 relevant articles/day = 15,000 tokens/day
    ~$3/month — negligible

API key loading priority:
    1. GCP Secret Manager (VM / production)
    2. ANTHROPIC_API_KEY environment variable (local .env)
    3. Raises error if neither found

Runs:
    Every 10 minutes during market hours (after news_collector.py)
    Cron: */10 3-10 * * 1-5 (UTC)

Usage:
    # Process all unscored items
    python intelligence/news/processor.py --process

    # Process and show results
    python intelligence/news/processor.py --process --verbose

    # Dry run (score but don't save to GCS)
    python intelligence/news/processor.py --process --dry-run

    # Show scored news for a symbol
    python intelligence/news/processor.py --symbol RELIANCE26MAYFUT

    # Import
    from intelligence.news.processor import NewsProcessor
    processor = NewsProcessor()
    scored = processor.score_item(raw_item)
"""

import argparse
import json
import logging
import os
import time
import warnings
from datetime import datetime
from typing import Optional

# Load .env file if present (local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # dotenv not installed — fine, will use env var or Secret Manager

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

GCS_BUCKET      = "hedge-fund-494103-marketdata-mumbai"
GCP_PROJECT     = "hedge-fund-494103"
SECRET_NAME     = "ANTHROPIC_API_KEY"

CLAUDE_MODEL    = "claude-sonnet-4-6"      # current model — updated from deprecated
MAX_TOKENS      = 600
BATCH_SIZE      = 10                        # items per cron run
RATE_LIMIT_SECS = 0.5                       # pause between API calls

# Only score items with symbols in our futures universe
# Reduces cost by ~70% — filters out irrelevant news
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
    "HCLTECH", "WIPRO", "TECHM",
    "TITAN", "NESTLEIND", "HINDUNILVR", "BRITANNIA", "TATACONSUM",
    "COALINDIA", "SAIL", "VEDL",
}

# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a financial news classifier for an algorithmic market making system trading Indian equity and currency futures on NSE/BSE.

Market makers profit from uninformed (noise) traders and LOSE money to informed traders.
Your job is to identify news that increases informed trading activity so the MM system can widen spreads or pause trading.

Given a news headline and body, output ONLY a valid JSON object with NO markdown, NO explanation, NO preamble.

JSON schema:
{
  "symbols": ["RELIANCE", "HDFCBANK"],   // affected NSE equity symbols, empty if macro/general
  "event_type": "earnings",              // one of: earnings, merger, management, regulatory, macro, sector, dividend, insider, noise
  "sentiment": 0.7,                      // -1.0 (very negative) to 1.0 (very positive) for affected stocks
  "uncertainty": 0.8,                    // 0.0 (certain/predictable) to 1.0 (highly uncertain)
  "price_impact": "HIGH",               // HIGH / MEDIUM / LOW / NONE
  "informed_flow_risk": "HIGH",         // HIGH / MEDIUM / LOW — how much informed trading this attracts
  "mm_action": "WIDEN",                 // PAUSE / WIDEN / NORMAL
  "widen_factor": 1.5,                  // multiply spread by this (1.0 = no change, max 5.0)
  "pause_minutes": 0,                   // minutes to pause MM (0 if not pausing)
  "reasoning": "Brief explanation"      // max 100 chars
}

Rules for mm_action:
- PAUSE: earnings results released NOW, merger announced, trading halt, major fraud/regulatory action
- WIDEN: management change, analyst upgrade/downgrade, sector news, earnings approaching, fund raising
- NORMAL: routine filings, newspaper publications, AGM minutes, general market commentary, noise

Rules for informed_flow_risk:
- HIGH: insider-type events (merger, management exit, earnings beat/miss, regulatory probe)
- MEDIUM: analyst calls, sector rotation, fund raising
- LOW: dividends, routine filings, macro commentary

Be conservative — when in doubt use NORMAL. False positives (unnecessary pauses) cost more than false negatives."""


# ─── API key loader ───────────────────────────────────────────────────────────

def load_api_key() -> str:
    """
    Load Anthropic API key with fallback priority:
    1. GCP Secret Manager (VM / production)
    2. ANTHROPIC_API_KEY environment variable (local .env or shell)

    Raises RuntimeError if neither is available.
    """
    # Try Secret Manager first (works on VM with service account)
    try:
        from google.cloud import secretmanager
        client  = secretmanager.SecretManagerServiceClient()
        name    = f"projects/{GCP_PROJECT}/secrets/{SECRET_NAME}/versions/latest"
        payload = client.access_secret_version(
            request={"name": name}
        ).payload.data.decode("utf-8").strip()
        if payload:
            log.info("API key loaded from GCP Secret Manager")
            return payload
    except Exception:
        pass   # Not on VM or secret doesn't exist yet — fall through

    # Try environment variable (.env or shell export)
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        log.info("API key loaded from environment variable")
        return key

    raise RuntimeError(
        "ANTHROPIC_API_KEY not found.\n"
        "Options:\n"
        "  1. Add to .env file:  ANTHROPIC_API_KEY=sk-ant-...\n"
        "  2. PowerShell:        $env:ANTHROPIC_API_KEY='sk-ant-...'\n"
        "  3. GCP Secret Manager: gcloud secrets create ANTHROPIC_API_KEY"
    )


# ─── Claude API client ────────────────────────────────────────────────────────

class ClaudeScorer:
    """Scores news items using Claude API."""

    def __init__(self, model: str = CLAUDE_MODEL):
        self.model   = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package not installed.\n"
                    "Run: pip install anthropic"
                )
            api_key      = load_api_key()
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def score(
        self,
        headline: str,
        body:     str,
        source:   str = "",
    ) -> Optional[dict]:
        """
        Score a single news item.
        Returns dict with sentiment, uncertainty, mm_action etc.
        Returns None on API error.
        """
        body_truncated = body[:1500] if body else ""
        user_content   = (
            f"Source: {source}\n"
            f"Headline: {headline}\n"
            f"Body: {body_truncated}"
        )

        try:
            client   = self._get_client()
            response = client.messages.create(
                model      = self.model,
                max_tokens = MAX_TOKENS,
                system     = SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": user_content}],
            )

            raw_text = response.content[0].text.strip()

            # Strip any accidental markdown fences
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            raw_text = raw_text.strip()

            result = json.loads(raw_text)

            # Validate and clamp all values
            result["sentiment"]     = float(max(-1.0, min(1.0,
                                          result.get("sentiment", 0.0))))
            result["uncertainty"]   = float(max(0.0, min(1.0,
                                          result.get("uncertainty", 0.0))))
            result["widen_factor"]  = float(max(1.0, min(5.0,
                                          result.get("widen_factor", 1.0))))
            result["pause_minutes"] = int(max(0, min(240,
                                          result.get("pause_minutes", 0))))

            if result.get("mm_action") not in ("PAUSE", "WIDEN", "NORMAL"):
                result["mm_action"] = "NORMAL"

            return result

        except json.JSONDecodeError as e:
            log.warning(f"JSON parse error: {e}")
            return None
        except Exception as e:
            log.warning(f"Claude API error: {e}")
            return None


# ─── Relevance filter ─────────────────────────────────────────────────────────

def is_relevant(item: dict) -> bool:
    """
    Filter out items with no symbols in our trading universe.
    Reduces API cost by ~70%.
    """
    symbols  = item.get("symbols", [])
    headline = item.get("headline", "").upper()

    # Check extracted symbols
    for sym in symbols:
        if sym in CORE_SYMBOLS:
            return True

    # Check headline directly
    for sym in CORE_SYMBOLS:
        if sym in headline:
            return True

    # Always keep macro/index news
    macro_keywords = [
        "NIFTY", "SENSEX", "RBI", "SEBI", "FII", "FPI",
        "INFLATION", "GDP", "REPO RATE", "MONETARY POLICY",
        "BUDGET", "F&O", "DERIVATIVES",
    ]
    for kw in macro_keywords:
        if kw in headline:
            return True

    return False


# ─── Batch processor ──────────────────────────────────────────────────────────

class NewsProcessor:
    """
    Orchestrates LLM scoring of raw news items.
    Reads unscored items from GCS, scores with Claude, saves back to GCS.
    """

    def __init__(self):
        self.scorer   = ClaudeScorer()
        self._storage = None

    def _get_storage(self):
        if self._storage is None:
            from intelligence.news.news_collector import NewsStorage
            self._storage = NewsStorage()
        return self._storage

    def score_item(self, item: dict) -> Optional[dict]:
        """Score a single raw news item. Returns updated item or None."""
        headline = item.get("headline", "")
        body     = item.get("body", "")
        source   = item.get("source", "")

        if not headline:
            return None

        scores = self.scorer.score(headline, body, source)
        if scores is None:
            return None

        item = dict(item)
        item.update({
            "scored":             True,
            "scored_at":          datetime.utcnow().isoformat() + "Z",
            "event_type":         scores.get("event_type", "noise"),
            "sentiment":          scores.get("sentiment", 0.0),
            "uncertainty":        scores.get("uncertainty", 0.0),
            "price_impact":       scores.get("price_impact", "NONE"),
            "informed_flow_risk": scores.get("informed_flow_risk", "LOW"),
            "mm_action":          scores.get("mm_action", "NORMAL"),
            "widen_factor":       scores.get("widen_factor", 1.0),
            "pause_minutes":      scores.get("pause_minutes", 0),
            "llm_reasoning":      scores.get("reasoning", ""),
            "llm_symbols":        scores.get("symbols", []),
        })

        # Merge collector + LLM detected symbols
        all_symbols = list(set(
            item.get("symbols", []) + item.get("llm_symbols", [])
        ))
        item["symbols"] = all_symbols

        return item

    def process_batch(
        self,
        items:   list[dict],
        dry_run: bool = False,
        verbose: bool = False,
    ) -> list[dict]:
        """Score a batch of news items."""
        scored_items = []
        storage      = self._get_storage() if not dry_run else None

        for i, item in enumerate(items):
            news_id  = item.get("id", f"item_{i}")
            headline = item.get("headline", "")[:60]

            log.info(f"Scoring [{i+1}/{len(items)}]: {headline}...")

            scored = self.score_item(item)

            if scored is None:
                log.warning(f"  Failed to score {news_id}")
                continue

            scored_items.append(scored)

            if verbose:
                self._print_scored(scored)

            if not dry_run and storage:
                storage.mark_scored(news_id, {
                    k: scored[k] for k in [
                        "scored", "scored_at", "event_type",
                        "sentiment", "uncertainty", "price_impact",
                        "informed_flow_risk", "mm_action",
                        "widen_factor", "pause_minutes",
                        "llm_reasoning", "llm_symbols", "symbols",
                    ] if k in scored
                })

            time.sleep(RATE_LIMIT_SECS)

        return scored_items

    def process_unscored(
        self,
        max_items: int  = BATCH_SIZE,
        dry_run:   bool = False,
        verbose:   bool = False,
    ) -> list[dict]:
        """
        Fetch and process unscored items from today.
        Main entry point for cron job.
        Applies relevance filter before scoring to reduce API cost.
        """
        storage  = self._get_storage()
        unscored = storage.load_unscored()

        if not unscored:
            log.info("No unscored items to process")
            return []

        # Relevance filter
        relevant = [i for i in unscored if is_relevant(i)]
        filtered = len(unscored) - len(relevant)
        log.info(
            f"Relevance filter: {len(relevant)} relevant "
            f"/ {len(unscored)} total "
            f"({filtered} filtered out)"
        )

        if not relevant:
            log.info("No relevant items to score")
            return []

        # Prioritize NSE > BSE > RSS
        def priority(item):
            src = item.get("source", "")
            if src == "NSE":   return 0
            if src == "BSE":   return 1
            return 2

        relevant.sort(key=priority)
        batch = relevant[:max_items]

        log.info(
            f"Processing {len(batch)} items "
            f"({len(relevant) - len(batch)} remaining)"
        )

        return self.process_batch(batch, dry_run=dry_run, verbose=verbose)

    def get_symbol_signals(self, symbol: str) -> list[dict]:
        """Get all scored news items for a symbol from today."""
        import re
        storage = self._get_storage()
        items   = storage.load_today_items()
        equity  = re.sub(r'\d{2}[A-Z]{3}FUT$', '', symbol.upper())

        relevant = [
            i for i in items
            if i.get("scored", False)
            and (
                equity in i.get("symbols", [])
                or equity in i.get("llm_symbols", [])
                or equity in i.get("headline", "").upper()
            )
        ]

        return sorted(
            relevant,
            key=lambda x: x.get("uncertainty", 0),
            reverse=True,
        )

    def worst_signal(self, symbol: str) -> Optional[dict]:
        """Get highest-uncertainty scored news item for a symbol."""
        items = self.get_symbol_signals(symbol)
        if not items:
            return None
        return max(items, key=lambda x: x.get("uncertainty", 0))

    def _print_scored(self, item: dict):
        action      = item.get("mm_action", "NORMAL")
        action_icon = {
            "PAUSE":  "🔴",
            "WIDEN":  "⚠️ ",
            "NORMAL": "✓ ",
        }.get(action, "  ")

        print(f"\n  {action_icon} {item.get('headline', '')[:65]}")
        print(f"     Source:     {item.get('source', '')}")
        print(f"     Symbols:    {', '.join(item.get('symbols', []))}")
        print(f"     Event:      {item.get('event_type', '')}")
        print(f"     Sentiment:  {item.get('sentiment', 0):+.2f} | "
              f"Uncertainty: {item.get('uncertainty', 0):.2f} | "
              f"Flow risk: {item.get('informed_flow_risk', '')}")
        print(f"     Action:     {action} "
              f"(widen={item.get('widen_factor', 1.0)}x, "
              f"pause={item.get('pause_minutes', 0)}min)")
        print(f"     Reason:     {item.get('llm_reasoning', '')[:80]}")


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(scored_items: list[dict]):
    if not scored_items:
        print("\nNo items scored.")
        return

    pause_items  = [i for i in scored_items if i.get("mm_action") == "PAUSE"]
    widen_items  = [i for i in scored_items if i.get("mm_action") == "WIDEN"]
    normal_items = [i for i in scored_items if i.get("mm_action") == "NORMAL"]

    affected: dict[str, list] = {}
    for item in scored_items:
        action = item.get("mm_action", "NORMAL")
        if action == "NORMAL":
            continue
        for sym in item.get("symbols", []):
            if sym not in affected:
                affected[sym] = []
            affected[sym].append({
                "action":   action,
                "factor":   item.get("widen_factor", 1.0),
                "headline": item.get("headline", "")[:50],
            })

    print(f"\n{'=' * 65}")
    print(f"  NEWS PROCESSING SUMMARY")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'=' * 65}")
    print(f"  Scored:    {len(scored_items)} items")
    print(f"  🔴 PAUSE:  {len(pause_items)}")
    print(f"  ⚠️  WIDEN:  {len(widen_items)}")
    print(f"  ✓  NORMAL: {len(normal_items)}")

    if affected:
        print(f"\n── Affected Symbols ─────────────────────────────────────")
        for sym, signals in sorted(affected.items()):
            worst  = "PAUSE" if any(
                s["action"] == "PAUSE" for s in signals
            ) else "WIDEN"
            factor = max(s["factor"] for s in signals)
            print(f"  {sym:<25} {worst:<8} {factor:.1f}x  "
                  f"{signals[0]['headline']}")

    print(f"{'=' * 65}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="News Processor — LLM scoring pipeline"
    )
    parser.add_argument("--process",    action="store_true",
                        help="Process all unscored items from today")
    parser.add_argument("--max-items",  type=int, default=BATCH_SIZE,
                        help=f"Max items per run (default {BATCH_SIZE})")
    parser.add_argument("--symbol",     type=str,
                        help="Show scored news for specific symbol")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Score but don't save to GCS")
    parser.add_argument("--verbose",    action="store_true",
                        help="Print each scored item in detail")
    args = parser.parse_args()

    processor = NewsProcessor()

    if args.process:
        scored = processor.process_unscored(
            max_items = args.max_items,
            dry_run   = args.dry_run,
            verbose   = args.verbose,
        )
        print_summary(scored)

    if args.symbol:
        items = processor.get_symbol_signals(args.symbol)
        print(f"\nScored news for {args.symbol}: {len(items)} items")
        for item in items:
            processor._print_scored(item)

        worst = processor.worst_signal(args.symbol)
        if worst:
            print(f"\nWorst signal:")
            print(f"  mm_action:    {worst.get('mm_action')}")
            print(f"  uncertainty:  {worst.get('uncertainty')}")
            print(f"  widen_factor: {worst.get('widen_factor')}")


if __name__ == "__main__":
    main()
