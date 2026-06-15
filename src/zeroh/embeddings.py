"""A tiny, dependency-free TF-IDF vectorizer with cosine similarity.

This deliberately avoids heavyweight ML libraries (numpy, sentence-transformers,
faiss, ...) so zeroH runs anywhere, offline, with deterministic behavior. The
representation combines word tokens with character tri-grams, which gives robust
similarity even for short texts and minor typos.

Vectors are represented as sparse ``Dict[str, float]`` mappings (term -> weight).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List

from .text import char_ngrams, tokenize

Vector = Dict[str, float]


def _features(text: str) -> List[str]:
    """Combine content word tokens with character tri-grams."""
    feats = list(tokenize(text))
    feats.extend(f"#{g}" for g in char_ngrams(text, 3))
    return feats


class TfidfVectorizer:
    """Incremental TF-IDF vectorizer.

    The document-frequency table is built up via :meth:`fit` / :meth:`partial_fit`
    so the same vectorizer can grow alongside a live, append-only memory store.
    """

    def __init__(self) -> None:
        self._doc_freq: Counter = Counter()
        self._n_docs: int = 0

    @property
    def n_docs(self) -> int:
        return self._n_docs

    def partial_fit(self, text: str) -> None:
        """Update document-frequency statistics with a single new document."""
        unique = set(_features(text))
        for term in unique:
            self._doc_freq[term] += 1
        self._n_docs += 1

    def fit(self, corpus: Iterable[str]) -> "TfidfVectorizer":
        for text in corpus:
            self.partial_fit(text)
        return self

    def _idf(self, term: str) -> float:
        # Smoothed IDF; unseen terms get the maximum (most informative) weight.
        df = self._doc_freq.get(term, 0)
        return math.log((1 + self._n_docs) / (1 + df)) + 1.0

    def transform(self, text: str) -> Vector:
        """Convert text into an L2-normalized sparse TF-IDF vector."""
        counts = Counter(_features(text))
        if not counts:
            return {}
        total = sum(counts.values())
        vec: Vector = {}
        for term, count in counts.items():
            tf = count / total
            vec[term] = tf * self._idf(term)
        return _l2_normalize(vec)


def _l2_normalize(vec: Vector) -> Vector:
    norm = math.sqrt(sum(w * w for w in vec.values()))
    if norm == 0:
        return vec
    return {t: w / norm for t, w in vec.items()}


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity of two sparse vectors in [0, 1] for non-negative TF-IDF."""
    if not a or not b:
        return 0.0
    # Iterate over the smaller vector for efficiency.
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())
