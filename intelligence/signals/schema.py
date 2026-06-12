"""
Intelligence Layer — Shared Data Structures
============================================
This is the ONLY file that src/trading/ imports from intelligence/.
Everything else in intelligence/ is internal implementation detail.

The contract between intelligence/ and src/trading/ is:
    intelligence/ writes IntelSignal to GCS
    src/trading/  reads  IntelSignal from GCS
    They never call each other directly.

GCS paths:
    signals/{symbol}/latest.json     ← current signal
    signals/{symbol}/{date}.json     ← historical signals
    signals/_universe.json           ← MM suitability ranking

Usage in mm_strategy.py:
    from intelligence.signals.schema import IntelSignal

    signal = IntelSignal.load(symbol)
    if signal.mm_action == 'PAUSE':
        return
    spread *= signal.widen_factor
"""

import json
import io
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

# ─── Enums ────────────────────────────────────────────────────────────────────

class MMAction(str, Enum):
    """
    What should mm_strategy.py do with this signal?
    NORMAL  → trade as usual
    WIDEN   → multiply spread by widen_factor
    PAUSE   → don't post quotes until pause_until
    AVOID   → don't trade this symbol today at all
    """
    NORMAL  = "NORMAL"
    WIDEN   = "WIDEN"
    PAUSE   = "PAUSE"
    AVOID   = "AVOID"


class SignalSource(str, Enum):
    """Where did this signal come from?"""
    NEWS         = "NEWS"           # breaking news / RSS feed
    EARNINGS     = "EARNINGS"       # scheduled earnings event
    CORPORATE    = "CORPORATE"      # board meeting, dividend, AGM
    FUNDAMENTAL  = "FUNDAMENTAL"    # 10-K/10-Q scoring
    MANUAL       = "MANUAL"         # human override
    DEFAULT      = "DEFAULT"        # no signal — use defaults


class EventType(str, Enum):
    """Type of corporate event."""
    EARNINGS        = "EARNINGS"
    DIVIDEND        = "DIVIDEND"
    BOARD_MEETING   = "BOARD_MEETING"
    AGM             = "AGM"
    BONUS_SPLIT     = "BONUS_SPLIT"
    MERGER_ACQUIRE  = "MERGER_ACQUIRE"
    MANAGEMENT      = "MANAGEMENT"
    REGULATORY      = "REGULATORY"
    MACRO           = "MACRO"
    SECTOR          = "SECTOR"
    NOISE           = "NOISE"


class FinancialHealth(str, Enum):
    STRONG     = "STRONG"
    MODERATE   = "MODERATE"
    WEAK       = "WEAK"
    DISTRESSED = "DISTRESSED"


# ─── Core signal ──────────────────────────────────────────────────────────────

@dataclass
class IntelSignal:
    """
    The single object that mm_strategy.py reads.
    Everything in intelligence/ produces this.
    Everything in src/trading/ consumes this.

    This is the firewall between the two layers.
    """
    symbol:        str
    generated_at:  str                    # ISO UTC timestamp
    expires_at:    str                    # signal is stale after this

    # ── Trading action ──
    mm_action:     str = MMAction.NORMAL  # NORMAL / WIDEN / PAUSE / AVOID
    widen_factor:  float = 1.0            # multiply spread by this
    pause_until:   Optional[str] = None   # ISO UTC, None = not paused

    # ── Signal metadata ──
    confidence:    float = 1.0            # 0-1
    source:        str = SignalSource.DEFAULT
    reason:        str = ""

    # ── News scores ──
    news_sentiment:    float = 0.0        # -1 to 1
    news_uncertainty:  float = 0.0        # 0 to 1
    active_news:       bool  = False      # is there breaking news now?

    # ── Event awareness ──
    has_upcoming_event:  bool = False
    event_type:          Optional[str] = None
    event_date:          Optional[str] = None
    days_to_event:       Optional[int] = None

    # ── Fundamental score ──
    mm_suitability:      float = 0.5      # 0-1, higher = better for MM
    financial_health:    str = FinancialHealth.MODERATE
    information_asymmetry_risk: str = "MEDIUM"   # LOW / MEDIUM / HIGH

    # ── Raw scores for aggregator ──
    component_scores:    dict = field(default_factory=dict)

    # ─────────────────────────────────────────────────────────────────
    # Convenience methods
    # ─────────────────────────────────────────────────────────────────

    def is_paused(self) -> bool:
        """Is MM currently paused for this symbol?"""
        if self.mm_action not in (MMAction.PAUSE, MMAction.AVOID):
            return False
        if self.pause_until is None:
            return self.mm_action == MMAction.AVOID
        return datetime.utcnow().isoformat() < self.pause_until

    def is_stale(self) -> bool:
        """Has this signal expired?"""
        return datetime.utcnow().isoformat() > self.expires_at

    def effective_widen_factor(self) -> float:
        """
        What spread multiplier should we actually use?
        Combines news uncertainty + event proximity + fundamental risk.
        """
        if self.is_paused():
            return 999.0   # effectively infinite — don't trade

        factor = self.widen_factor

        # Boost widen factor if approaching an event
        if self.days_to_event is not None:
            if self.days_to_event == 0:
                factor = max(factor, 3.0)   # event day — very wide
            elif self.days_to_event == 1:
                factor = max(factor, 2.0)   # day before — wide
            elif self.days_to_event <= 3:
                factor = max(factor, 1.5)   # approaching — slightly wide

        # Boost for high information asymmetry
        if self.information_asymmetry_risk == "HIGH":
            factor = max(factor, 2.0)
        elif self.information_asymmetry_risk == "MEDIUM":
            factor = max(factor, 1.2)

        return round(factor, 2)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, d: dict) -> "IntelSignal":
        # Only pass fields that exist in the dataclass
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_json(cls, s: str) -> "IntelSignal":
        return cls.from_dict(json.loads(s))

    @classmethod
    def default(cls, symbol: str, ttl_minutes: int = 60) -> "IntelSignal":
        """
        Safe default signal — NORMAL action, no events, neutral scores.
        Used when no intelligence data is available for a symbol.
        mm_strategy.py should use this as fallback.
        """
        now     = datetime.utcnow()
        expires = now + timedelta(minutes=ttl_minutes)
        return cls(
            symbol       = symbol,
            generated_at = now.isoformat() + "Z",
            expires_at   = expires.isoformat() + "Z",
            mm_action    = MMAction.NORMAL,
            source       = SignalSource.DEFAULT,
            reason       = "No intelligence data available — using defaults",
        )

    # ─────────────────────────────────────────────────────────────────
    # GCS persistence
    # ─────────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        symbol:     str,
        bucket:     str = "hedge-fund-494103-marketdata-mumbai",
        fallback:   bool = True,
    ) -> "IntelSignal":
        """
        Load latest signal for a symbol from GCS.
        Falls back to default signal if not found or stale.

        Usage in mm_strategy.py:
            signal = IntelSignal.load(self.symbol)
            if signal.is_paused():
                return
            spread *= signal.effective_widen_factor()
        """
        try:
            from google.cloud import storage
            client = storage.Client()
            path   = f"signals/{symbol}/latest.json"
            blob   = client.bucket(bucket).blob(path)

            if not blob.exists():
                return cls.default(symbol) if fallback else None

            data   = blob.download_as_text()
            signal = cls.from_json(data)

            if signal.is_stale() and fallback:
                return cls.default(symbol)

            return signal

        except Exception:
            return cls.default(symbol) if fallback else None

    def save(
        self,
        bucket: str = "hedge-fund-494103-marketdata-mumbai",
        save_historical: bool = True,
    ) -> bool:
        """
        Save signal to GCS.
        Always writes to signals/{symbol}/latest.json
        Optionally writes to signals/{symbol}/{date}.json for history.
        """
        try:
            from google.cloud import storage
            client = storage.Client()
            bkt    = client.bucket(bucket)
            data   = self.to_json()

            # Latest
            bkt.blob(f"signals/{self.symbol}/latest.json").upload_from_string(
                data, content_type="application/json"
            )

            # Historical
            if save_historical:
                date = self.generated_at[:10]
                bkt.blob(f"signals/{self.symbol}/{date}.json").upload_from_string(
                    data, content_type="application/json"
                )
            return True

        except Exception as e:
            return False


# ─── Universe ranking ─────────────────────────────────────────────────────────

@dataclass
class MMCandidate:
    """
    MM suitability ranking for one symbol.
    universe.py produces this. aggregator.py reads it.
    Recomputed monthly from fundamental scores.
    """
    symbol:              str
    rank:                int           # 1 = best MM candidate
    mm_suitability:      float         # 0-1
    financial_health:    str
    avg_spread_bps:      float         # historical avg spread
    avg_fill_rate:       float         # historical fill rate
    adverse_selection:   float         # Kyle's lambda avg
    event_frequency:     int           # corporate events per year
    recommendation:      str           # TIER1 / TIER2 / AVOID
    last_updated:        str           # ISO date

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MMUniverse:
    """
    Full ranked universe of MM candidates.
    Saved to GCS: signals/_universe.json
    Updated monthly.
    """
    generated_at:  str
    total_symbols: int
    tier1:         list[MMCandidate]   # top MM candidates
    tier2:         list[MMCandidate]   # acceptable
    avoid:         list[MMCandidate]   # don't MM these

    def get_rank(self, symbol: str) -> Optional[MMCandidate]:
        for c in self.tier1 + self.tier2 + self.avoid:
            if c.symbol == symbol:
                return c
        return None

    def to_dict(self) -> dict:
        return {
            "generated_at":  self.generated_at,
            "total_symbols": self.total_symbols,
            "tier1":  [c.to_dict() for c in self.tier1],
            "tier2":  [c.to_dict() for c in self.tier2],
            "avoid":  [c.to_dict() for c in self.avoid],
        }

    def save(self, bucket: str = "hedge-fund-494103-marketdata-mumbai"):
        try:
            from google.cloud import storage
            client = storage.Client()
            client.bucket(bucket).blob("signals/_universe.json").upload_from_string(
                json.dumps(self.to_dict(), indent=2),
                content_type="application/json",
            )
            return True
        except Exception:
            return False

    @classmethod
    def load(cls, bucket: str = "hedge-fund-494103-marketdata-mumbai") -> Optional["MMUniverse"]:
        try:
            from google.cloud import storage
            client = storage.Client()
            blob   = client.bucket(bucket).blob("signals/_universe.json")
            if not blob.exists():
                return None
            d = json.loads(blob.download_as_text())
            return cls(
                generated_at  = d["generated_at"],
                total_symbols = d["total_symbols"],
                tier1  = [MMCandidate(**c) for c in d["tier1"]],
                tier2  = [MMCandidate(**c) for c in d["tier2"]],
                avoid  = [MMCandidate(**c) for c in d["avoid"]],
            )
        except Exception:
            return None


# ─── News item ────────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    """
    A single news article or announcement.
    news_collector.py produces this.
    processor.py scores this and produces IntelSignal.
    """
    id:          str            # hash of headline + source
    source:      str            # NSE / BSE / MONEYCONTROL / ET / RSS
    headline:    str
    body:        str
    url:         str
    published:   str            # ISO UTC
    symbols:     list[str]      # affected NSE symbols
    fetched_at:  str            # ISO UTC

    # Filled by processor.py
    scored:          bool  = False
    event_type:      Optional[str]   = None
    sentiment:       Optional[float] = None
    uncertainty:     Optional[float] = None
    price_impact:    Optional[str]   = None
    mm_action:       Optional[str]   = None
    widen_factor:    Optional[float] = None
    pause_minutes:   Optional[int]   = None
    llm_reasoning:   Optional[str]   = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# ─── Corporate event ──────────────────────────────────────────────────────────

@dataclass
class CorporateEvent:
    """
    A scheduled corporate event (earnings, AGM, board meeting).
    event_calendar.py produces this.
    aggregator.py reads this to set pause/widen signals.
    """
    symbol:      str
    event_type:  str            # EventType enum value
    event_date:  str            # YYYY-MM-DD
    description: str
    source:      str            # NSE / BSE
    confirmed:   bool = True

    # Computed fields
    days_until:  Optional[int] = None

    def compute_days_until(self):
        today = datetime.utcnow().date()
        event = datetime.strptime(self.event_date, "%Y-%m-%d").date()
        self.days_until = (event - today).days

    def mm_impact(self) -> tuple[str, float, int]:
        """
        Returns (mm_action, widen_factor, pause_minutes)
        based on event type and proximity.
        """
        self.compute_days_until()
        d = self.days_until

        if self.event_type == EventType.EARNINGS:
            if d == 0:
                return MMAction.PAUSE, 3.0, 120   # pause 2hrs around results
            elif d == 1:
                return MMAction.WIDEN, 2.0, 0
            elif d <= 3:
                return MMAction.WIDEN, 1.5, 0
            elif d <= 7:
                return MMAction.WIDEN, 1.2, 0

        elif self.event_type == EventType.MERGER_ACQUIRE:
            if d <= 1:
                return MMAction.AVOID, 5.0, 0
            elif d <= 7:
                return MMAction.WIDEN, 2.5, 0

        elif self.event_type == EventType.BOARD_MEETING:
            if d == 0:
                return MMAction.WIDEN, 1.5, 0
            elif d <= 2:
                return MMAction.WIDEN, 1.2, 0

        elif self.event_type == EventType.BONUS_SPLIT:
            if d <= 1:
                return MMAction.WIDEN, 1.3, 0

        return MMAction.NORMAL, 1.0, 0

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Fundamental score ────────────────────────────────────────────────────────

@dataclass
class FundamentalScore:
    """
    LLM-generated fundamental analysis of a company filing.
    filing_scorer.py produces this.
    universe.py aggregates these into MMCandidate rankings.
    """
    symbol:                      str
    filing_date:                 str
    filing_type:                 str        # ANNUAL / QUARTERLY
    mm_suitability_score:        float      # 0-1
    financial_health:            str        # FinancialHealth enum
    information_asymmetry_risk:  str        # LOW / MEDIUM / HIGH
    key_risks:                   list[str]
    upcoming_catalysts:          list[str]
    avoid_mm_until:              Optional[str]  # date
    llm_reasoning:               str
    scored_at:                   str        # ISO UTC

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test default signal
    sig = IntelSignal.default("NIFTY26MAYFUT")
    print("Default signal:")
    print(sig.to_json())
    print(f"Is paused: {sig.is_paused()}")
    print(f"Is stale:  {sig.is_stale()}")
    print(f"Widen factor: {sig.effective_widen_factor()}")

    # Test earnings event impact
    print("\nEarnings event (tomorrow):")
    event = CorporateEvent(
        symbol      = "RELIANCE26MAYFUT",
        event_type  = EventType.EARNINGS,
        event_date  = (datetime.utcnow().date() +
                       __import__('datetime').timedelta(days=1)).isoformat(),
        description = "Q4 FY26 Results",
        source      = "NSE",
    )
    action, factor, pause = event.mm_impact()
    print(f"  Action: {action}  Widen: {factor}x  Pause: {pause}min")

    # Test signal with event
    print("\nSignal with event:")
    sig2 = IntelSignal(
        symbol           = "RELIANCE26MAYFUT",
        generated_at     = datetime.utcnow().isoformat() + "Z",
        expires_at       = (datetime.utcnow() +
                           __import__('datetime').timedelta(hours=4)
                           ).isoformat() + "Z",
        mm_action        = MMAction.WIDEN,
        widen_factor     = 2.0,
        has_upcoming_event = True,
        event_type       = EventType.EARNINGS,
        days_to_event    = 1,
        source           = SignalSource.EARNINGS,
        reason           = "Q4 results tomorrow — widening spread",
        mm_suitability   = 0.6,
        financial_health = FinancialHealth.STRONG,
        information_asymmetry_risk = "MEDIUM",
    )
    print(f"  Effective widen factor: {sig2.effective_widen_factor()}")
    print(f"  Is paused: {sig2.is_paused()}")
