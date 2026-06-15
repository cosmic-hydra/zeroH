"""Retrieval over the memory store (the *R* in RAG).

Combines TF-IDF cosine similarity (semantic-ish) with exact keyword overlap, so
relevant grounding facts can be surfaced for a query. Retrieval is the
foundation of zeroH's anti-hallucination strategy: the agent only answers from
what it can actually retrieve.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..embeddings import TfidfVectorizer, Vector, cosine_similarity
from ..memory import MemoryStore
from ..models import Memory, RetrievalResult
from ..text import tokenize


class Retriever:
    """Builds and queries a TF-IDF index over a :class:`MemoryStore`.

    Args:
        store: The backing memory store.
        keyword_weight: Blend factor in [0, 1]; the final score is
            ``(1 - keyword_weight) * cosine + keyword_weight * keyword_overlap``.
    """

    def __init__(self, store: MemoryStore, keyword_weight: float = 0.25) -> None:
        if not 0.0 <= keyword_weight <= 1.0:
            raise ValueError("keyword_weight must be in [0, 1]")
        self.store = store
        self.keyword_weight = keyword_weight
        self._vectorizer = TfidfVectorizer()
        self._vectors: Dict[str, Vector] = {}
        self._tokens: Dict[str, set] = {}
        self.reindex()

    def reindex(self) -> None:
        """Rebuild the full index from the current active memories.

        TF-IDF weights depend on global document frequencies, so when the corpus
        changes we recompute document frequencies first, then every vector.
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

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """Return up to ``top_k`` memories ranked by blended relevance."""
        query_vec = self._vectorizer.transform(query)
        query_tokens = set(tokenize(query))
        results: List[RetrievalResult] = []
        for mem in self.store.all():
            vec = self._vectors.get(mem.id)
            if vec is None:
                # Memory added after the last reindex; index it lazily.
                self._vectorizer.partial_fit(mem.content)
                vec = self._vectorizer.transform(mem.content)
                self._vectors[mem.id] = vec
                self._tokens[mem.id] = set(tokenize(mem.content))
            cosine = cosine_similarity(query_vec, vec)
            keyword = _keyword_overlap(query_tokens, self._tokens[mem.id])
            score = (1 - self.keyword_weight) * cosine + self.keyword_weight * keyword
            if score > min_score:
                results.append(RetrievalResult(memory=mem, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


def _keyword_overlap(query_tokens: set, doc_tokens: set) -> float:
    """Fraction of query keywords present in the document (in [0, 1])."""
    if not query_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)
