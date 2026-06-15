"""Retrieval over the memory store (the *R* in RAG).

Combines TF-IDF cosine similarity (semantic-ish) with exact keyword overlap, so
relevant grounding facts can be surfaced for a query. Retrieval is the
foundation of zeroH's anti-hallucination strategy: the agent only answers from
what it can actually retrieve.

Beyond raw textual relevance the retriever can optionally fold two extra signals
into the final score, giving the agent richer, more trustworthy memory context:

* **Confidence** – a caller-supplied trust score per memory. Down-weighting
  low-confidence memories keeps shaky knowledge from out-ranking solid facts.
* **Recency** – an exponential freshness decay. When knowledge changes over
  time, newer memories can be preferred without deleting the old ones.

Two-phase retrieval is supported via an integrated inverted index:

1. **Candidate selection** – BM25-scored inverted index narrows the search space
   from all memories to just relevant candidates (O(1) per query term).
2. **Re-scoring** – Full TF-IDF cosine + keyword blend on the candidate set.

This dramatically reduces retrieval latency for large stores (1000+ memories)
while maintaining relevance quality.

Query caching (via :class:`~zeroh.optimization.cache.QueryCache`) avoids
redundant computation for repeated or similar queries in multi-turn contexts.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Set

from ..embeddings import TfidfVectorizer, Vector, cosine_similarity
from ..memory import MemoryStore
from ..memory.indexes import InvertedIndex
from ..models import Memory, RetrievalResult
from ..optimization.cache import QueryCache
from ..text import tokenize

# A predicate used to restrict retrieval to a subset of memories.
MemoryFilter = Callable[[Memory], bool]


class Retriever:
    """Builds and queries a TF-IDF index over a :class:`MemoryStore`.

    Now integrates an inverted index for fast candidate pre-filtering and
    a query cache for repeated-query optimization.

    Args:
        store: The backing memory store.
        keyword_weight: Blend factor in [0, 1]; the textual relevance is
            ``(1 - keyword_weight) * cosine + keyword_weight * keyword_overlap``.
        confidence_weight: How strongly a memory's ``confidence`` modulates its
            score, in [0, 1]. ``0`` (default) ignores confidence; ``1`` scales
            the score by confidence outright.
        recency_weight: How strongly freshness modulates the score, in [0, 1].
            Requires ``recency_half_life`` to take effect. ``0`` (default)
            ignores recency.
        recency_half_life: Age, in seconds, at which a memory's recency factor
            decays to ``0.5``. ``None`` (default) disables recency weighting.
        now: Callable returning the current Unix time (injectable for tests).
        enable_cache: Enable query result caching.
        cache_size: Maximum number of cached query results.
        cache_ttl: Time-to-live for cache entries in seconds.
        enable_index: Enable inverted index for candidate pre-filtering.
        index_threshold: Minimum number of memories before index is used
            (below this, full scan is more efficient).
    """

    def __init__(
        self,
        store: MemoryStore,
        keyword_weight: float = 0.25,
        *,
        confidence_weight: float = 0.0,
        recency_weight: float = 0.0,
        recency_half_life: Optional[float] = None,
        now: Callable[[], float] = time.time,
        enable_cache: bool = True,
        cache_size: int = 128,
        cache_ttl: float = 300.0,
        enable_index: bool = True,
        index_threshold: int = 50,
    ) -> None:
        if not 0.0 <= keyword_weight <= 1.0:
            raise ValueError("keyword_weight must be in [0, 1]")
        if not 0.0 <= confidence_weight <= 1.0:
            raise ValueError("confidence_weight must be in [0, 1]")
        if not 0.0 <= recency_weight <= 1.0:
            raise ValueError("recency_weight must be in [0, 1]")
        if recency_half_life is not None and recency_half_life <= 0:
            raise ValueError("recency_half_life must be positive when set")
        self.store = store
        self.keyword_weight = keyword_weight
        self.confidence_weight = confidence_weight
        self.recency_weight = recency_weight
        self.recency_half_life = recency_half_life
        self._now = now
        self._vectorizer = TfidfVectorizer()
        self._vectors: Dict[str, Vector] = {}
        self._tokens: Dict[str, set] = {}

        # Inverted index for fast candidate pre-filtering
        self.enable_index = enable_index
        self.index_threshold = index_threshold
        self._index = InvertedIndex(enable_bigrams=True)

        # Query cache for repeated-query optimization
        self.enable_cache = enable_cache
        self._cache = QueryCache(
            max_size=cache_size,
            ttl=cache_ttl,
            enable_prefix=True,
        )

        self.reindex()

    def reindex(self) -> None:
        """Rebuild the full index from the current active memories.

        TF-IDF weights depend on global document frequencies, so when the corpus
        changes we recompute document frequencies first, then every vector.
        Also rebuilds the inverted index and invalidates the query cache.
        """
        memories = self.store.all()
        self._vectorizer = TfidfVectorizer()
        for mem in memories:
            self._vectorizer.partial_fit(mem.content)
        self._vectors = {}
        self._tokens = {}
        for mem in memories:
            self._vectors[mem.id] = self._vectorizer.transform(mem.content)
            self._tokens[mem.id] = set(tokenize(mem.content))

        # Rebuild inverted index
        if self.enable_index:
            self._index.rebuild(memories)

        # Invalidate cache since corpus changed
        if self.enable_cache:
            self._cache.invalidate()

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        *,
        where: Optional[MemoryFilter] = None,
        source: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Return up to ``top_k`` memories ranked by blended relevance.

        Uses two-phase retrieval when the inverted index is enabled:
        1. Fast BM25 candidate selection via inverted index
        2. Full TF-IDF cosine + keyword scoring on candidates

        Results are cached for repeated queries.

        Args:
            query: The search text.
            top_k: Maximum number of results.
            min_score: Drop results scoring at or below this value.
            where: Optional predicate; only memories for which it returns truthy
                are considered. Lets callers scope context (e.g. by tag/metadata).
            source: Convenience filter restricting results to a single ``source``.
        """
        # Check cache first (only for unfiltered queries)
        cache_key = f"{query}::{top_k}::{source or ''}"
        if self.enable_cache and where is None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Determine candidate set
        candidates = self._get_candidates(query, top_k, source=source, where=where)

        # Score candidates
        query_vec = self._vectorizer.transform(query)
        query_tokens = set(tokenize(query))
        now = self._now() if self.recency_weight and self.recency_half_life else 0.0

        results: List[RetrievalResult] = []
        for mem in candidates:
            vec = self._vectors.get(mem.id)
            if vec is None:
                # Memory added after the last reindex; index it lazily.
                self._vectorizer.partial_fit(mem.content)
                vec = self._vectorizer.transform(mem.content)
                self._vectors[mem.id] = vec
                self._tokens[mem.id] = set(tokenize(mem.content))
                if self.enable_index:
                    self._index.add(mem)
            cosine = cosine_similarity(query_vec, vec)
            keyword = _keyword_overlap(query_tokens, self._tokens[mem.id])
            relevance = (1 - self.keyword_weight) * cosine + self.keyword_weight * keyword
            score = relevance * self._priority(mem, now)
            if score > min_score:
                results.append(RetrievalResult(memory=mem, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:top_k]

        # Cache the results
        if self.enable_cache and where is None:
            self._cache.put(cache_key, results)

        return results

    def _get_candidates(
        self,
        query: str,
        top_k: int,
        *,
        source: Optional[str],
        where: Optional[MemoryFilter],
    ) -> List[Memory]:
        """Get candidate memories using inverted index or full scan.

        For small stores (below index_threshold), a full scan is used.
        For larger stores, the inverted index pre-filters to a relevant
        candidate set, dramatically reducing scoring work.
        """
        all_memories = self.store.all()

        # Apply source and where filters
        filtered = all_memories
        if source is not None:
            filtered = [m for m in filtered if m.source == source]
        if where is not None:
            filtered = [m for m in filtered if where(m)]

        # Use inverted index for large stores
        if (self.enable_index and len(filtered) >= self.index_threshold
                and source is None and where is None):
            # Get candidate IDs from inverted index (expanded set for precision)
            candidate_ids = set(self._index.search(
                query, top_k=max(top_k * 4, 30)
            ))
            # Fall back to full scan if index returns too few candidates
            if len(candidate_ids) >= top_k:
                candidates = [m for m in filtered if m.id in candidate_ids]
                return candidates

        return filtered

    def _priority(self, mem: Memory, now: float) -> float:
        """Multiplicative confidence/recency factor in (0, 1].

        With both weights at ``0`` (the default) this is exactly ``1.0``, leaving
        the textual relevance score untouched. ``now`` is the query-time clock
        sample used for recency decay.
        """
        factor = 1.0
        if self.confidence_weight:
            conf = min(1.0, max(0.0, mem.confidence))
            factor *= (1 - self.confidence_weight) + self.confidence_weight * conf
        if self.recency_weight and self.recency_half_life:
            age = max(0.0, now - mem.created_at)
            decay = 0.5 ** (age / self.recency_half_life)
            factor *= (1 - self.recency_weight) + self.recency_weight * decay
        return factor

    @property
    def cache_stats(self) -> dict:
        """Return query cache statistics."""
        return self._cache.stats()


def _keyword_overlap(query_tokens: set, doc_tokens: set) -> float:
    """Fraction of query keywords present in the document (in [0, 1])."""
    if not query_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)
