"""Tiered memory storage with automatic hot/warm/cold promotion and demotion.

This module implements a three-tier memory architecture that optimizes access
patterns by keeping frequently-accessed memories in a fast in-memory tier while
relegating rarely-used memories to a lower-priority tier:

* **Hot tier**: Recently or frequently accessed memories (fast dict lookup).
* **Warm tier**: Moderately accessed memories (indexed, quick retrieval).
* **Cold tier**: Infrequently accessed memories (still available but de-prioritized).

Automatic promotion/demotion happens based on access frequency and recency,
ensuring that the working set stays lean for retrieval while nothing is lost.
This is especially valuable for large memory stores where scanning all memories
on every query would be expensive.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from ..models import Memory


@dataclass
class AccessStats:
    """Track access patterns for a single memory."""

    memory_id: str
    access_count: int = 0
    last_accessed: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def access_frequency(self) -> float:
        """Accesses per hour since creation."""
        age_hours = max(0.001, (time.time() - self.created_at) / 3600)
        return self.access_count / age_hours


class TieredMemoryManager:
    """Manages memory across hot/warm/cold tiers based on access patterns.

    Memories are promoted to hotter tiers when accessed frequently and demoted
    to colder tiers when unused. This keeps the working set small and fast
    while retaining full recall capability.

    Args:
        hot_capacity: Maximum number of memories in the hot tier.
        warm_capacity: Maximum number of memories in the warm tier.
        promote_threshold: Access count to promote warm → hot.
        demote_after: Seconds of inactivity before demotion.
    """

    def __init__(
        self,
        *,
        hot_capacity: int = 50,
        warm_capacity: int = 200,
        promote_threshold: int = 3,
        demote_after: float = 3600.0,
    ) -> None:
        if hot_capacity <= 0 or warm_capacity <= 0:
            raise ValueError("Capacities must be positive")
        if promote_threshold <= 0:
            raise ValueError("promote_threshold must be positive")
        self.hot_capacity = hot_capacity
        self.warm_capacity = warm_capacity
        self.promote_threshold = promote_threshold
        self.demote_after = demote_after

        # Tier membership (sets of memory IDs)
        self._hot: Set[str] = set()
        self._warm: Set[str] = set()
        self._cold: Set[str] = set()

        # Access tracking
        self._stats: Dict[str, AccessStats] = {}

    def register(self, memory: Memory) -> str:
        """Register a new memory in the warm tier.

        New memories start warm (not hot) so they must earn their place in the
        hot tier through actual usage.
        """
        mid = memory.id
        if mid not in self._stats:
            self._stats[mid] = AccessStats(
                memory_id=mid, created_at=memory.created_at
            )
        if mid not in self._hot and mid not in self._warm:
            self._warm.add(mid)
            self._cold.discard(mid)
        self._enforce_warm_capacity()
        return self.tier_of(mid)

    def record_access(self, memory_id: str) -> str:
        """Record an access to a memory and potentially promote it.

        Returns the tier the memory is now in after any promotion.
        """
        stats = self._stats.get(memory_id)
        if stats is None:
            stats = AccessStats(memory_id=memory_id)
            self._stats[memory_id] = stats
        stats.access_count += 1
        stats.last_accessed = time.time()

        # Promote to hot if access count exceeds threshold
        if memory_id not in self._hot and stats.access_count >= self.promote_threshold:
            self._promote_to_hot(memory_id)

        return self.tier_of(memory_id)

    def tier_of(self, memory_id: str) -> str:
        """Return the current tier of a memory: 'hot', 'warm', or 'cold'."""
        if memory_id in self._hot:
            return "hot"
        if memory_id in self._warm:
            return "warm"
        return "cold"

    def get_hot_ids(self) -> List[str]:
        """Return IDs of all memories in the hot tier."""
        return list(self._hot)

    def get_warm_ids(self) -> List[str]:
        """Return IDs of all memories in the warm tier."""
        return list(self._warm)

    def get_cold_ids(self) -> List[str]:
        """Return IDs of all memories in the cold tier."""
        return list(self._cold)

    def demote_stale(self, now: Optional[float] = None) -> int:
        """Demote memories that haven't been accessed within ``demote_after``.

        Returns the number of memories demoted.
        """
        now = now or time.time()
        cutoff = now - self.demote_after
        demoted = 0

        # Hot → warm demotion
        stale_hot = [
            mid for mid in list(self._hot)
            if self._stats.get(mid) and self._stats[mid].last_accessed < cutoff
        ]
        for mid in stale_hot:
            self._hot.discard(mid)
            self._warm.add(mid)
            demoted += 1

        # Warm → cold demotion
        stale_warm = [
            mid for mid in list(self._warm)
            if self._stats.get(mid) and self._stats[mid].last_accessed < cutoff
        ]
        for mid in stale_warm:
            self._warm.discard(mid)
            self._cold.add(mid)
            demoted += 1

        self._enforce_warm_capacity()
        return demoted

    def remove(self, memory_id: str) -> None:
        """Remove a memory from all tiers and tracking."""
        self._hot.discard(memory_id)
        self._warm.discard(memory_id)
        self._cold.discard(memory_id)
        self._stats.pop(memory_id, None)

    def stats(self) -> Dict[str, int]:
        """Return counts per tier."""
        return {
            "hot": len(self._hot),
            "warm": len(self._warm),
            "cold": len(self._cold),
            "total_tracked": len(self._stats),
        }

    def _promote_to_hot(self, memory_id: str) -> None:
        """Move a memory to the hot tier, evicting the least-accessed if full."""
        if len(self._hot) >= self.hot_capacity:
            self._evict_from_hot()
        self._warm.discard(memory_id)
        self._cold.discard(memory_id)
        self._hot.add(memory_id)

    def _evict_from_hot(self) -> None:
        """Evict the least recently accessed memory from hot → warm."""
        if not self._hot:
            return
        # Find the memory with the oldest last_accessed in hot tier
        oldest_id = min(
            self._hot,
            key=lambda mid: self._stats[mid].last_accessed
            if mid in self._stats else 0.0,
        )
        self._hot.discard(oldest_id)
        self._warm.add(oldest_id)

    def _enforce_warm_capacity(self) -> None:
        """If warm tier exceeds capacity, move oldest to cold."""
        while len(self._warm) > self.warm_capacity:
            oldest_id = min(
                self._warm,
                key=lambda mid: self._stats[mid].last_accessed
                if mid in self._stats else 0.0,
            )
            self._warm.discard(oldest_id)
            self._cold.add(oldest_id)
