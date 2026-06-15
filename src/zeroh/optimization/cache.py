"""LRU query result caching for retrieval optimization.

Retrieval (TF-IDF vectorization + scoring) is the most expensive per-query
operation in zeroH. For workloads with repeated or similar queries (e.g.
multi-turn conversations about the same topic), caching results avoids
redundant computation.

The :class:`QueryCache` provides:

* **LRU eviction**: Bounded memory usage with least-recently-used eviction.
* **TTL expiry**: Cached results automatically expire after a configurable
  time-to-live, ensuring freshness after memory mutations.
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

    def is_expired(self, ttl: float, now: Optional[float] = None) -> bool:
        """Check if this entry has exceeded its TTL."""
        now = now or time.time()
        return (now - self.created_at) >= ttl


class QueryCache:
    """LRU cache for retrieval query results.

    Designed to sit between the plugin and the retriever, caching scored
    retrieval results for recently-seen queries. The cache is invalidated
    when the underlying memory store changes.

    Args:
        max_size: Maximum number of cached query results.
        ttl: Time-to-live in seconds for each cache entry. After this time,
            the entry is considered stale and will be evicted on next access.
    """

    def __init__(self, *, max_size: int = 128, ttl: float = 300.0) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached result, or None if not found/expired.

        Moves the entry to the end (most recently used) on hit.
        """
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.is_expired(self.ttl):
            del self._cache[key]
            self._misses += 1
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.hits += 1
        self._hits += 1
        return entry.value

    def put(self, key: str, value: Any) -> None:
        """Store a query result in the cache.

        If the cache is full, evicts the least recently used entry.
        """
        if key in self._cache:
            # Update existing entry
            self._cache[key].value = value
            self._cache[key].created_at = time.time()
            self._cache.move_to_end(key)
        else:
            # Evict LRU if at capacity
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = CacheEntry(key=key, value=value)

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

    def stats(self) -> Dict[str, Any]:
        """Return cache performance statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "ttl": self.ttl,
        }

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        entry = self._cache.get(key)
        if entry is None:
            return False
        return not entry.is_expired(self.ttl)
