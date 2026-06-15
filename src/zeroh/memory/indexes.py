"""Inverted index for fast keyword-based memory lookup.

Instead of scanning all memories on every query (O(n) full scan), the inverted
index pre-maps tokens to the memory IDs that contain them, enabling O(1)
candidate set lookups per query term. Combined with the TF-IDF retriever, this
provides a two-phase retrieval strategy:

1. **Candidate selection** (this module): Fast set intersection to find memories
   that share query keywords.
2. **Scoring** (retriever): TF-IDF cosine + keyword blend on the candidate set.

This dramatically reduces retrieval latency for large stores (1000+ memories)
where scanning every memory for every query becomes a bottleneck.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set

from ..models import Memory
from ..text import tokenize


class InvertedIndex:
    """Token-to-memory-ID inverted index for fast candidate retrieval.

    This is a standard information retrieval inverted index: for each unique
    token in the corpus, it maps to the set of memory IDs containing that token.
    Query-time lookup is a simple set intersection across query terms.

    Args:
        min_token_length: Minimum token length to index (filters noise).
    """

    def __init__(self, *, min_token_length: int = 2) -> None:
        self.min_token_length = min_token_length
        # token → set of memory IDs containing that token
        self._index: Dict[str, Set[str]] = defaultdict(set)
        # memory_id → set of tokens (for efficient removal)
        self._memory_tokens: Dict[str, FrozenSet[str]] = {}

    def add(self, memory: Memory) -> None:
        """Index a memory's content tokens."""
        tokens = self._extract_tokens(memory.content)
        self._memory_tokens[memory.id] = tokens
        for token in tokens:
            self._index[token].add(memory.id)

    def remove(self, memory_id: str) -> None:
        """Remove a memory from the index."""
        tokens = self._memory_tokens.pop(memory_id, frozenset())
        for token in tokens:
            self._index[token].discard(memory_id)
            if not self._index[token]:
                del self._index[token]

    def update(self, memory: Memory) -> None:
        """Re-index a memory (remove old tokens, add new)."""
        self.remove(memory.id)
        self.add(memory)

    def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        min_overlap: int = 1,
    ) -> List[str]:
        """Find memory IDs matching query tokens.

        Args:
            query: The search query text.
            top_k: Maximum number of candidate IDs to return. ``None`` returns all.
            min_overlap: Minimum number of query tokens a memory must contain to
                be included in results.

        Returns:
            Memory IDs sorted by number of matching tokens (descending).
        """
        query_tokens = self._extract_tokens(query)
        if not query_tokens:
            return []

        # Count how many query tokens each memory matches
        hit_counts: Dict[str, int] = defaultdict(int)
        for token in query_tokens:
            for memory_id in self._index.get(token, set()):
                hit_counts[memory_id] += 1

        # Filter by minimum overlap
        candidates = [
            (mid, count) for mid, count in hit_counts.items()
            if count >= min_overlap
        ]

        # Sort by hit count descending (most relevant first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            candidates = candidates[:top_k]

        return [mid for mid, _ in candidates]

    def rebuild(self, memories: List[Memory]) -> None:
        """Clear and rebuild the index from a list of memories."""
        self._index.clear()
        self._memory_tokens.clear()
        for memory in memories:
            self.add(memory)

    def vocabulary_size(self) -> int:
        """Number of unique tokens in the index."""
        return len(self._index)

    def memory_count(self) -> int:
        """Number of indexed memories."""
        return len(self._memory_tokens)

    def token_frequency(self, token: str) -> int:
        """Number of memories containing this token."""
        return len(self._index.get(token, set()))

    def stats(self) -> Dict[str, int]:
        """Return index statistics."""
        return {
            "vocabulary_size": self.vocabulary_size(),
            "memory_count": self.memory_count(),
            "total_postings": sum(len(s) for s in self._index.values()),
        }

    def _extract_tokens(self, text: str) -> FrozenSet[str]:
        """Extract indexable tokens from text."""
        tokens = tokenize(text, remove_stopwords=True)
        return frozenset(
            t for t in tokens if len(t) >= self.min_token_length
        )
