"""Document ingestion: turn long text into well-sized, overlapping memories.

Feeding an LLM raw, unfocused context is one of the biggest drivers of low-
quality, hallucination-prone answers. Storing knowledge as small, sentence-aware,
slightly overlapping chunks dramatically improves retrieval precision (and thus
answer quality), because the retriever can surface exactly the relevant span
instead of a whole document.
"""
from __future__ import annotations

from typing import List

from ..text import split_sentences


def chunk_text(
    text: str,
    max_chars: int = 400,
    overlap_sentences: int = 1,
) -> List[str]:
    """Split ``text`` into sentence-aware chunks of up to ``max_chars``.

    Sentences are never broken mid-way. Consecutive chunks share
    ``overlap_sentences`` trailing sentences so that context spanning a chunk
    boundary is not lost during retrieval.

    Args:
        text: The source document.
        max_chars: Soft upper bound on chunk length (a single oversized sentence
            is still emitted as its own chunk).
        overlap_sentences: Number of sentences to repeat between adjacent chunks.

    Returns:
        A list of non-empty chunk strings.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be non-negative")

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        # +1 accounts for the joining space.
        addition = len(sentence) + (1 if current else 0)
        if current and current_len + addition > max_chars:
            chunks.append(" ".join(current))
            # Start the next chunk with the overlap tail of the previous one.
            current = current[len(current) - overlap_sentences :] if overlap_sentences else []
            current_len = len(" ".join(current))
            addition = len(sentence) + (1 if current else 0)
        current.append(sentence)
        current_len += addition

    if current:
        chunks.append(" ".join(current))
    return chunks
