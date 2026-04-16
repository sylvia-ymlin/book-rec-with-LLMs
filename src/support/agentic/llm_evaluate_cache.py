"""
In-process TTL cache for LLM evaluation results.

Design decisions
----------------
- Keyed by SHA-256 of (query, frozenset(isbn_list[:max_books])) → order-invariant.
- TTL default 60 s: short enough to reflect rapid state changes, long enough to
  avoid duplicate API calls for burst traffic on the same query.
- Thread-safe via threading.Lock; safe for FastAPI worker threads.
- No external dependency (Redis/Memcached) — suitable for single-process dev/demo.
  For multi-process production, swap the backing store with Redis using the same
  CacheEntry protocol.

Usage
-----
    from src.support.agentic.llm_evaluate_cache import llm_evaluate_cache

    cached = llm_evaluate_cache.get(query, isbn_list, max_books)
    if cached is None:
        result = await expensive_llm_call(...)
        llm_evaluate_cache.set(query, isbn_list, max_books, result)
"""

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CacheEntry:
    value: Dict[str, Any]
    expires_at: float          # monotonic clock seconds


class LLMEvaluateCache:
    """
    Bounded TTL cache for LLM evaluation results.

    Args:
        ttl_seconds: How long each entry lives.  Default: 60 s.
        max_size:    Maximum number of entries.  LRU eviction when full.
                     0 = unlimited (not recommended for production).
    """

    def __init__(self, ttl_seconds: float = 60.0, max_size: int = 1024) -> None:
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    @staticmethod
    def make_key(query: str, isbn_list: list[str], max_books: int) -> str:
        """
        Build a canonical cache key.

        The key is order-invariant with respect to isbn_list so that the same
        set of books retrieved in different orders hits the same cache entry.
        """
        canonical_isbns = "|".join(sorted(set(isbn_list[:max_books])))
        raw = f"{query.strip().lower()}::{canonical_isbns}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self,
        query: str,
        isbn_list: list[str],
        max_books: int,
    ) -> Optional[Dict[str, Any]]:
        """Return cached evaluation result or None on miss / expiry."""
        key = self.make_key(query, isbn_list, max_books)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.monotonic() > entry.expires_at:
                del self._store[key]
                return None
            return entry.value

    def set(
        self,
        query: str,
        isbn_list: list[str],
        max_books: int,
        value: Dict[str, Any],
    ) -> None:
        """Store an evaluation result with TTL."""
        key = self.make_key(query, isbn_list, max_books)
        entry = CacheEntry(
            value=value,
            expires_at=time.monotonic() + self._ttl,
        )
        with self._lock:
            # Evict oldest entry if at capacity (simple LRU approximation)
            if self._max_size > 0 and len(self._store) >= self._max_size:
                oldest_key = next(iter(self._store))
                del self._store[oldest_key]
            self._store[key] = entry

    def invalidate(self, query: str, isbn_list: list[str], max_books: int) -> bool:
        """Remove a specific entry.  Returns True if it existed."""
        key = self.make_key(query, isbn_list, max_books)
        with self._lock:
            return self._store.pop(key, None) is not None

    def clear(self) -> None:
        """Flush the entire cache.  Useful for testing."""
        with self._lock:
            self._store.clear()

    # ── Observability ──────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def stats(self) -> Dict[str, Any]:
        """Return a snapshot for health-check / metrics endpoints."""
        now = time.monotonic()
        with self._lock:
            live = sum(1 for e in self._store.values() if e.expires_at > now)
            return {
                "total_entries": len(self._store),
                "live_entries": live,
                "expired_entries": len(self._store) - live,
                "ttl_seconds": self._ttl,
                "max_size": self._max_size,
            }


# ── Module-level singleton ────────────────────────────────────────────────────

llm_evaluate_cache = LLMEvaluateCache(ttl_seconds=60.0, max_size=1024)
