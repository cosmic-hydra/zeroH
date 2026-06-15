"""Lightweight text utilities (tokenization, sentence splitting, n-grams).

Pure standard library so zeroH stays dependency-free.
"""
from __future__ import annotations

import re
from typing import List

_WORD_RE = re.compile(r"[a-z0-9]+")
# Split on sentence terminators while keeping it simple and robust.
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

# A small English stopword list. Removing these focuses similarity on the
# content-bearing terms of a sentence.
STOPWORDS = frozenset(
    """
    a an and are as at be but by for if in into is it no not of on or such
    that the their then there these they this to was will with from your you
    i we he she him her his our its do does did has have had can could would
    should about over under again more most some any all each other than
    """.split()
)

# Common negation words used by the verifier to detect contradictions.
NEGATION_WORDS = frozenset(
    """
    no not never none neither nor nobody nothing nowhere
    cannot cant won't wouldn't shouldn't couldn't didn't doesn't isn't
    aren't wasn't weren't hasn't haven't hadn't
    """.split()
)


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Lowercase, split into alphanumeric tokens, optionally drop stopwords."""
    tokens = _WORD_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def split_sentences(text: str) -> List[str]:
    """Split text into trimmed, non-empty sentences."""
    parts = _SENT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def char_ngrams(text: str, n: int = 3) -> List[str]:
    """Character n-grams of the normalized text (helps fuzzy/typo matching)."""
    cleaned = " ".join(_WORD_RE.findall(text.lower()))
    if len(cleaned) < n:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]


def estimate_tokens(text: str) -> int:
    """Estimate the token count of a text using a word/punctuation heuristic.

    This approximates BPE tokenization (roughly 1 token per 4 characters for
    English) without requiring any external tokenizer library. Conservative
    estimate ensures we stay within budgets.
    """
    if not text:
        return 0
    # Rough heuristic: ~4 chars per token for English, rounded up.
    return max(1, (len(text) + 3) // 4)


def has_negation(text: str) -> bool:
    """Return True if the text contains a negation word."""
    tokens = set(_WORD_RE.findall(text.lower()))
    return bool(tokens & NEGATION_WORDS)


def semantic_similarity_quick(a: str, b: str) -> float:
    """Fast token-overlap Jaccard similarity (no embeddings needed).

    Returns a value in [0, 1]. Useful for detecting near-duplicate memories
    without the overhead of full TF-IDF vectorization.
    """
    tokens_a = set(tokenize(a))
    tokens_b = set(tokenize(b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union else 0.0
