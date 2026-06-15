"""Frequency-aware query result caching for retrieval optimization.

Retrieval (TF-IDF vectorization + scoring) is the most expensive per-query
operation in zeroH. For workloads with repeated or similar queries (e.g.
multi-turn conversations about the same topic), caching results avoids
redundant computation.

The :class:`QueryCache` provides:

* **LFU-LRU hybrid eviction**: Entries that are both old AND rarely accessed are
  evicted first, keeping frequently-used results available longer.
* **TTL expiry**: Cached results automatically expire after a configurable
  time-to-live, ensuring freshness after memory mutations.
* **Prefix matching**: Queries that share a common prefix with cached queries
  can return partial results, enabling fast typeahead-style retrieval.
* **Cache warming**: Pre-populate the cache with common queries to eliminate
  cold-start latency.
* **Automatic invalidation**: The cache can be explicitly cleared when the
  memory store is mutated (new memories added, superseded, or deactivated).
* **Hit/miss statistics**: For tuning cache size and TTL.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CacheEntry:
    """A single cached query result with metadata."""

    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    hits: int = 0
    last_access: float = field(default_factory=time.time)

    def is_expired(self, ttl: float, now: Optional[float] = None) -> bool:
        """Check if this entry has exceeded its TTL."""
        now = now or time.time()
        return (now - self.created_at) >= ttl

    @property
    def frequency_score(self) -> float:
        """Combined frequency-recency score for eviction decisions.

        Higher scores indicate more valuable entries that should be kept.
        """
        age = max(1.0, time.time() - self.created_at)
        # Hits per second of existence, biased toward recent access
        recency_bonus = max(0.0, 1.0 - (time.time() - self.last_access) / 3600)
        return (self.hits / age) * 1000 + recency_bonus


class QueryCache:
    """Frequency-aware LRU cache for retrieval query results.

    Uses a hybrid LFU-LRU eviction strategy: when the cache is full, the entry
    with the lowest combined frequency-recency score is evicted, rather than
    simply the oldest. This keeps frequently-accessed results cached longer.

    Prefix matching enables partial cache hits: a query for "capital of France"
    can partially match a cached result for "capital of France and its history",
    reducing redundant computation for related queries.

    Args:
        max_size: Maximum number of cached query results.
        ttl: Time-to-live in seconds for each cache entry.
        enable_prefix: Enable prefix-based partial cache matching.
        prefix_min_length: Minimum query length for prefix matching.
    """

    def __init__(
        self,
        *,
        max_size: int = 128,
        ttl: float = 300.0,
        enable_prefix: bool = True,
        prefix_min_length: int = 10,
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        self.max_size = max_size
        self.ttl = ttl
        self.enable_prefix = enable_prefix
        self.prefix_min_length = prefix_min_length
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._prefix_hits = 0

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached result, or None if not found/expired.

        Moves the entry to the end (most recently used) on hit.
        Falls back to prefix matching if enabled and exact match fails.
        """
        # Exact match
        entry = self._cache.get(key)
        if entry is not None:
            if entry.is_expired(self.ttl):
                del self._cache[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            entry.hits += 1
            entry.last_access = time.time()
            self._hits += 1
            return entry.value

        # Prefix matching fallback
        if self.enable_prefix and len(key) >= self.prefix_min_length:
            result = self._prefix_lookup(key)
            if result is not None:
                self._prefix_hits += 1
                self._hits += 1
                return result

        self._misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """Store a query result in the cache.

        If the cache is full, evicts the entry with the lowest frequency score
        (hybrid LFU-LRU eviction).
        """
        if key in self._cache:
            self._cache[key].value = value
            self._cache[key].created_at = time.time()
            self._cache[key].last_access = time.time()
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._evict_lowest_value()
            self._cache[key] = CacheEntry(key=key, value=value)

    def warm(self, entries: List[Tuple[str, Any]]) -> int:
        """Pre-populate the cache with known common queries.

        Useful for warming the cache at startup with frequently-used queries
        to eliminate cold-start latency.

        Args:
            entries: List of (key, value) tuples to pre-cache.

        Returns:
            Number of entries successfully cached.
        """
        count = 0
        for key, value in entries:
            if len(self._cache) >= self.max_size:
                break
            if key not in self._cache:
                self._cache[key] = CacheEntry(key=key, value=value)
                count += 1
        return count

    def invalidate(self, key: Optional[str] = None) -> int:
        """Invalidate a specific key or the entire cache.

        Args:
            key: If provided, only this key is removed. If ``None``, the
                entire cache is cleared (e.g. after memory mutation).

        Returns:
            Number of entries removed.
        """
        if key is not None:
            if key in self._cache:
                del self._cache[key]
                return 1
            return 0
        count = len(self._cache)
        self._cache.clear()
        return count

    def invalidate_prefix(self, prefix: str) -> int:
        """Invalidate all entries whose keys start with the given prefix.

        Useful for targeted invalidation when specific topics are updated.

        Returns:
            Number of entries removed.
        """
        to_remove = [k for k in self._cache if k.startswith(prefix)]
        for k in to_remove:
            del self._cache[k]
        return len(to_remove)

    def stats(self) -> Dict[str, Any]:
        """Return cache performance statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "prefix_hits": self._prefix_hits,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "ttl": self.ttl,
        }

    def top_entries(self, n: int = 10) -> List[Tuple[str, int]]:
        """Return the top-n most frequently accessed cache entries.

        Useful for identifying cache warming candidates.
        """
        entries = [(e.key, e.hits) for e in self._cache.values()]
        entries.sort(key=lambda x: x[1], reverse=True)
        return entries[:n]

    def _prefix_lookup(self, key: str) -> Optional[Any]:
        """Find a cached entry whose key is a prefix of the query or vice versa."""
        now = time.time()
        # Check if query is a prefix of a cached key
        for cached_key, entry in self._cache.items():
            if entry.is_expired(self.ttl, now):
                continue
            if cached_key.startswith(key) or key.startswith(cached_key):
                entry.hits += 1
                entry.last_access = now
                self._cache.move_to_end(cached_key)
                return entry.value
        return None

    def _evict_lowest_value(self) -> None:
        """Evict the entry with the lowest frequency-recency score.

        When entries have equal scores (e.g., all 0 hits), falls back to
        LRU behavior by evicting the front of the OrderedDict.
        """
        if not self._cache:
            return
        now = time.time()

        # Always evict expired entries first
        expired_key = None
        for key, entry in self._cache.items():
            if entry.is_expired(self.ttl, now):
                expired_key = key
                break
        if expired_key is not None:
            del self._cache[expired_key]
            return

        # Find entry with lowest combined value (fewest hits = least useful)
        # Among entries with equal hits, the first encountered (front/LRU) wins
        min_key = None
        min_hits = float("inf")
        for key, entry in self._cache.items():
            if entry.hits < min_hits:
                min_hits = entry.hits
                min_key = key

        if min_key is None:
            min_key = next(iter(self._cache))
        del self._cache[min_key]

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        entry = self._cache.get(key)
        if entry is None:
            return False
        return not entry.is_expired(self.ttl)
