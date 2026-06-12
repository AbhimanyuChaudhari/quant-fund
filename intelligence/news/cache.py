"""
Intelligence Cache
==================
Fast in-memory cache for IntelSignal objects.
Used by mm_strategy.py to avoid GCS reads on every bar.

Without cache:
    Every bar → GCS read → 200ms latency → strategy slows down

With cache:
    First read → GCS → cached in memory
    Subsequent reads → memory → <1ms
    Cache refresh → every 15 minutes (matches signal TTL)

Usage in mm_strategy.py:
    from intelligence.news.cache import IntelCache

    # In __init__:
    self.intel_cache = IntelCache()

    # In on_bar:
    signal = self.intel_cache.get(self.symbol)
    if signal.is_paused():
        return
    spread *= signal.effective_widen_factor()
"""

import logging
import threading
import warnings
from datetime import datetime, timedelta
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)

log = logging.getLogger(__name__)

# Cache TTL — refresh signal from GCS after this many minutes
CACHE_TTL_MINUTES = 15

# If GCS read fails, use this as fallback TTL before retrying
ERROR_RETRY_MINUTES = 2


class IntelCache:
    """
    Thread-safe in-memory cache for IntelSignal objects.

    Automatically refreshes stale signals from GCS.
    Falls back to default signal (NORMAL) if GCS unavailable.

    Thread-safe: uses RLock for concurrent bar processing.
    """

    def __init__(
        self,
        ttl_minutes:   int = CACHE_TTL_MINUTES,
        bucket:        str = "hedge-fund-494103-marketdata-mumbai",
    ):
        self.ttl_minutes = ttl_minutes
        self.bucket      = bucket
        self._cache:     dict[str, dict] = {}   # symbol → {signal, cached_at}
        self._lock       = threading.RLock()

    def get(self, symbol: str):
        """
        Get IntelSignal for symbol.
        Loads from GCS if not cached or stale.
        Never raises — always returns a valid signal.

        Returns IntelSignal object.
        """
        with self._lock:
            cached = self._cache.get(symbol)

            if cached and not self._is_stale(cached):
                return cached["signal"]

            # Cache miss or stale — refresh from GCS
            signal = self._load(symbol)
            self._cache[symbol] = {
                "signal":    signal,
                "cached_at": datetime.utcnow(),
            }
            return signal

    def invalidate(self, symbol: str):
        """Force refresh on next get() for this symbol."""
        with self._lock:
            if symbol in self._cache:
                del self._cache[symbol]

    def invalidate_all(self):
        """Force refresh all symbols on next get()."""
        with self._lock:
            self._cache.clear()

    def preload(self, symbols: list[str]):
        """
        Preload signals for all symbols at startup.
        Call this once at strategy init to avoid cold-start latency.

        Usage in mm_strategy __init__:
            self.intel_cache.preload([self.symbol])
        """
        log.info(f"Preloading intel cache for {len(symbols)} symbols...")
        loaded = 0
        for sym in symbols:
            try:
                self.get(sym)
                loaded += 1
            except Exception as e:
                log.debug(f"Preload failed for {sym}: {e}")
        log.info(f"Intel cache preloaded: {loaded}/{len(symbols)} symbols")

    def summary(self) -> dict:
        """Return cache stats for monitoring."""
        with self._lock:
            total   = len(self._cache)
            stale   = sum(
                1 for v in self._cache.values()
                if self._is_stale(v)
            )
            paused  = sum(
                1 for v in self._cache.values()
                if not self._is_stale(v)
                and v["signal"].is_paused()
            )
            widened = sum(
                1 for v in self._cache.values()
                if not self._is_stale(v)
                and v["signal"].mm_action == "WIDEN"
                and not v["signal"].is_paused()
            )
            return {
                "cached":  total,
                "stale":   stale,
                "paused":  paused,
                "widened": widened,
                "normal":  total - paused - widened - stale,
            }

    def _is_stale(self, cached: dict) -> bool:
        """Check if cached entry is older than TTL."""
        cached_at = cached.get("cached_at")
        if not cached_at:
            return True
        age = (datetime.utcnow() - cached_at).total_seconds() / 60
        return age > self.ttl_minutes

    def _load(self, symbol: str):
        """Load signal from GCS. Returns default on failure."""
        try:
            from intelligence.signals.schema import IntelSignal
            signal = IntelSignal.load(symbol, bucket=self.bucket, fallback=True)
            log.debug(f"Cache loaded: {symbol} → {signal.mm_action}")
            return signal
        except Exception as e:
            log.debug(f"Cache load failed for {symbol}: {e}")
            from intelligence.signals.schema import IntelSignal
            return IntelSignal.default(symbol, ttl_minutes=ERROR_RETRY_MINUTES)


# ─── Global singleton ─────────────────────────────────────────────────────────

# Single shared cache instance — import this in mm_strategy.py
_global_cache: Optional[IntelCache] = None


def get_cache(
    ttl_minutes: int = CACHE_TTL_MINUTES,
    bucket:      str = "hedge-fund-494103-marketdata-mumbai",
) -> IntelCache:
    """
    Get or create the global cache singleton.

    Usage:
        from intelligence.news.cache import get_cache
        cache  = get_cache()
        signal = cache.get('RELIANCE26MAYFUT')
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = IntelCache(ttl_minutes=ttl_minutes, bucket=bucket)
    return _global_cache


# ─── Convenience function ─────────────────────────────────────────────────────

def get_signal(symbol: str):
    """
    One-liner signal lookup for mm_strategy.py.

    Usage:
        from intelligence.news.cache import get_signal

        signal = get_signal(self.symbol)
        if signal.is_paused():
            return
        spread *= signal.effective_widen_factor()
    """
    return get_cache().get(symbol)


# ─── Monitor ──────────────────────────────────────────────────────────────────

class CacheMonitor:
    """
    Logs cache health every N minutes.
    Useful for live trading monitoring.

    Usage in trade_runner.py:
        monitor = CacheMonitor(interval_minutes=5)
        monitor.start()
    """

    def __init__(self, interval_minutes: int = 5):
        self.interval = interval_minutes * 60
        self._thread  = None
        self._running = False

    def start(self):
        """Start background monitoring thread."""
        self._running = True
        self._thread  = threading.Thread(
            target=self._run,
            daemon=True,
            name="IntelCacheMonitor",
        )
        self._thread.start()
        log.info("Intel cache monitor started")

    def stop(self):
        """Stop monitoring thread."""
        self._running = False

    def _run(self):
        while self._running:
            try:
                cache = get_cache()
                stats = cache.summary()
                if stats["paused"] > 0 or stats["widened"] > 0:
                    log.info(
                        f"Intel cache: {stats['cached']} symbols | "
                        f"🔴 paused={stats['paused']} | "
                        f"⚠️  widened={stats['widened']} | "
                        f"✓ normal={stats['normal']}"
                    )
            except Exception as e:
                log.debug(f"Cache monitor error: {e}")

            import time
            time.sleep(self.interval)


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)

    print("Testing IntelCache...")

    cache = IntelCache(ttl_minutes=15)

    # Test default fallback (no GCS needed)
    from intelligence.signals.schema import IntelSignal
    IntelCache._load = lambda self, sym: IntelSignal.default(sym)

    signal = cache.get("NIFTY26MAYFUT")
    print(f"\nDefault signal for NIFTY26MAYFUT:")
    print(f"  mm_action:     {signal.mm_action}")
    print(f"  widen_factor:  {signal.effective_widen_factor()}")
    print(f"  is_paused:     {signal.is_paused()}")
    print(f"  is_stale:      {signal.is_stale()}")

    # Test cache hit
    signal2 = cache.get("NIFTY26MAYFUT")
    print(f"\nCache hit (same object): {signal is signal2}")

    # Test summary
    cache.get("RELIANCE26MAYFUT")
    cache.get("HDFCBANK26MAYFUT")
    stats = cache.summary()
    print(f"\nCache stats: {stats}")

    print("\nIntelCache working correctly.")
