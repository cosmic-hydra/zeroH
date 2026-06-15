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
