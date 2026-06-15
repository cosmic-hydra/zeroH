"""BM25-enhanced inverted index for fast keyword-based memory lookup.

Instead of scanning all memories on every query (O(n) full scan), the inverted
index pre-maps tokens to the memory IDs that contain them, enabling O(1)
candidate set lookups per query term. Combined with the TF-IDF retriever, this
provides a two-phase retrieval strategy:

1. **Candidate selection** (this module): Fast set intersection to find memories
   that share query keywords, scored with BM25.
2. **Scoring** (retriever): TF-IDF cosine + keyword blend on the candidate set.

BM25 scoring provides better relevance ranking than simple hit counting by
incorporating term frequency saturation, document length normalization, and
inverse document frequency weighting.

Bigram indexing captures phrase-level patterns that single tokens miss,
improving precision for multi-word concepts (e.g. "machine learning" vs
"machine" + "learning" separately).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from ..models import Memory
from ..text import tokenize


class InvertedIndex:
    """BM25-scored inverted index with bigram support.

    Uses BM25 term weighting for relevance scoring during candidate selection,
    which accounts for term frequency saturation (diminishing returns of
    repeated terms), document length normalization, and inverse document
    frequency (rare terms are more informative).

    Bigram indexing captures adjacent word pairs, improving precision for
    phrase-like queries without requiring exact phrase matching.

    Args:
        min_token_length: Minimum token length to index (filters noise).
        enable_bigrams: Index adjacent token pairs for phrase matching.
        k1: BM25 term frequency saturation parameter (1.2-2.0 typical).
        b: BM25 length normalization parameter (0.75 typical).
    """

    def __init__(
        self,
        *,
        min_token_length: int = 2,
        enable_bigrams: bool = True,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.min_token_length = min_token_length
        self.enable_bigrams = enable_bigrams
        self.k1 = k1
        self.b = b
        # token → set of memory IDs containing that token
        self._index: Dict[str, Set[str]] = defaultdict(set)
        # memory_id → set of tokens (for efficient removal)
        self._memory_tokens: Dict[str, FrozenSet[str]] = {}
        # memory_id → token count (for BM25 length normalization)
        self._doc_lengths: Dict[str, int] = {}
        # Average document length (maintained incrementally)
        self._avg_doc_length: float = 0.0
        # Total number of documents
        self._n_docs: int = 0

    def add(self, memory: Memory) -> None:
        """Index a memory's content tokens and optional bigrams."""
        tokens = self._extract_tokens(memory.content)
        all_terms = set(tokens)

        if self.enable_bigrams:
            bigrams = self._extract_bigrams(memory.content)
            all_terms |= bigrams

        self._memory_tokens[memory.id] = frozenset(all_terms)
        # Store raw token count for BM25 length normalization
        self._doc_lengths[memory.id] = len(tokens)

        for token in all_terms:
            self._index[token].add(memory.id)

        # Update average document length
        self._n_docs += 1
        total_length = sum(self._doc_lengths.values())
        self._avg_doc_length = total_length / self._n_docs if self._n_docs else 0.0

    def remove(self, memory_id: str) -> None:
        """Remove a memory from the index."""
        tokens = self._memory_tokens.pop(memory_id, frozenset())
        self._doc_lengths.pop(memory_id, None)
        for token in tokens:
            self._index[token].discard(memory_id)
            if not self._index[token]:
                del self._index[token]

        # Update average document length
        self._n_docs = max(0, self._n_docs - 1)
        if self._n_docs > 0:
            total_length = sum(self._doc_lengths.values())
            self._avg_doc_length = total_length / self._n_docs
        else:
            self._avg_doc_length = 0.0

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
        use_bm25: bool = True,
    ) -> List[str]:
        """Find memory IDs matching query tokens, ranked by BM25 score.

        Args:
            query: The search query text.
            top_k: Maximum number of candidate IDs to return.
            min_overlap: Minimum number of query tokens a memory must contain.
            use_bm25: Use BM25 scoring instead of simple hit counting.

        Returns:
            Memory IDs sorted by BM25 score (or hit count) descending.
        """
        query_tokens = self._extract_tokens(query)
        query_terms = set(query_tokens)

        if self.enable_bigrams:
            query_terms |= self._extract_bigrams(query)

        if not query_terms:
            return []

        if use_bm25 and self._n_docs > 0:
            return self._bm25_search(query_terms, top_k=top_k, min_overlap=min_overlap)
        else:
            return self._hit_count_search(query_terms, top_k=top_k, min_overlap=min_overlap)

    def search_scored(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        min_overlap: int = 1,
    ) -> List[Tuple[str, float]]:
        """Find memory IDs with their BM25 scores.

        Returns:
            List of (memory_id, bm25_score) tuples, sorted by score descending.
        """
        query_tokens = self._extract_tokens(query)
        query_terms = set(query_tokens)

        if self.enable_bigrams:
            query_terms |= self._extract_bigrams(query)

        if not query_terms or self._n_docs == 0:
            return []

        scores = self._compute_bm25_scores(query_terms)

        # Filter by minimum overlap
        hit_counts: Dict[str, int] = defaultdict(int)
        for term in query_terms:
            for mid in self._index.get(term, set()):
                hit_counts[mid] += 1

        candidates = [
            (mid, score)
            for mid, score in scores.items()
            if hit_counts.get(mid, 0) >= min_overlap
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            candidates = candidates[:top_k]

        return candidates

    def idf(self, term: str) -> float:
        """Compute inverse document frequency for a term.

        Uses the standard BM25 IDF formula with smoothing to avoid negatives
        for very common terms.
        """
        if self._n_docs == 0:
            return 0.0
        df = len(self._index.get(term, set()))
        if df == 0:
            return 0.0
        # Smoothed IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        return math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def rebuild(self, memories: List[Memory]) -> None:
        """Clear and rebuild the index from a list of memories."""
        self._index.clear()
        self._memory_tokens.clear()
        self._doc_lengths.clear()
        self._n_docs = 0
        self._avg_doc_length = 0.0
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

    def stats(self) -> Dict[str, object]:
        """Return index statistics."""
        return {
            "vocabulary_size": self.vocabulary_size(),
            "memory_count": self.memory_count(),
            "total_postings": sum(len(s) for s in self._index.values()),
            "avg_doc_length": round(self._avg_doc_length, 1),
            "bigrams_enabled": self.enable_bigrams,
        }

    def _bm25_search(
        self,
        query_terms: set,
        *,
        top_k: Optional[int],
        min_overlap: int,
    ) -> List[str]:
        """Search using BM25 scoring."""
        scores = self._compute_bm25_scores(query_terms)

        # Filter by minimum overlap
        hit_counts: Dict[str, int] = defaultdict(int)
        for term in query_terms:
            for mid in self._index.get(term, set()):
                hit_counts[mid] += 1

        candidates = [
            (mid, score)
            for mid, score in scores.items()
            if hit_counts.get(mid, 0) >= min_overlap
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            candidates = candidates[:top_k]

        return [mid for mid, _ in candidates]

    def _hit_count_search(
        self,
        query_terms: set,
        *,
        top_k: Optional[int],
        min_overlap: int,
    ) -> List[str]:
        """Fallback search using simple hit counting."""
        hit_counts: Dict[str, int] = defaultdict(int)
        for token in query_terms:
            for memory_id in self._index.get(token, set()):
                hit_counts[memory_id] += 1

        candidates = [
            (mid, count) for mid, count in hit_counts.items()
            if count >= min_overlap
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            candidates = candidates[:top_k]

        return [mid for mid, _ in candidates]

    def _compute_bm25_scores(self, query_terms: set) -> Dict[str, float]:
        """Compute BM25 scores for all candidate documents."""
        scores: Dict[str, float] = defaultdict(float)
        avgdl = self._avg_doc_length if self._avg_doc_length > 0 else 1.0

        for term in query_terms:
            term_idf = self.idf(term)
            posting_list = self._index.get(term, set())

            for mid in posting_list:
                # Term frequency: count how many of the document's tokens match
                # Since we store frozensets, tf is 1 for unigrams (present/absent)
                # For a more precise TF we'd need token counts, but binary BM25
                # (tf=1 for present terms) works well for short documents.
                tf = 1.0
                doc_len = self._doc_lengths.get(mid, 1)

                # BM25 formula: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
                scores[mid] += term_idf * (numerator / denominator)

        return scores

    def _extract_tokens(self, text: str) -> FrozenSet[str]:
        """Extract indexable tokens from text."""
        tokens = tokenize(text, remove_stopwords=True)
        return frozenset(
            t for t in tokens if len(t) >= self.min_token_length
        )

    def _extract_bigrams(self, text: str) -> FrozenSet[str]:
        """Extract bigrams (adjacent token pairs) from text."""
        tokens = [
            t for t in tokenize(text, remove_stopwords=False)
            if len(t) >= self.min_token_length
        ]
        if len(tokens) < 2:
            return frozenset()
        bigrams = frozenset(
            f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)
        )
        return bigrams
