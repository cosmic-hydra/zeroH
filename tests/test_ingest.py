"""Tests for document ingestion / chunking."""
import pytest

from zeroh.memory import MemoryStore, chunk_text
from zeroh.memory.ingest import chunk_text as chunk_text_direct


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_single_chunk():
    chunks = chunk_text("One sentence only.", max_chars=400)
    assert chunks == ["One sentence only."]


def test_long_text_splits_into_multiple_chunks():
    sentences = " ".join(f"Sentence number {i} has some content." for i in range(20))
    chunks = chunk_text(sentences, max_chars=80, overlap_sentences=0)
    assert len(chunks) > 1
    # No chunk should greatly exceed the soft limit (single sentences fit < 80).
    assert all(len(c) <= 120 for c in chunks)


def test_overlap_repeats_sentences():
    text = "Alpha one. Beta two. Gamma three. Delta four."
    chunks = chunk_text(text, max_chars=20, overlap_sentences=1)
    assert len(chunks) > 1
    # Consecutive chunks should share a boundary sentence.
    joined = " ".join(chunks)
    assert joined.count("Beta two.") >= 1


def test_invalid_params():
    with pytest.raises(ValueError):
        chunk_text("x", max_chars=0)
    with pytest.raises(ValueError):
        chunk_text("x", overlap_sentences=-1)


def test_chunk_text_exported_from_package():
    assert chunk_text is chunk_text_direct
