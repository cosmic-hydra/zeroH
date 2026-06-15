"""Tests for confidence-, recency- and filter-aware retrieval ranking."""
import time

import pytest

from zeroh.memory import MemoryStore
from zeroh.retrieval import Retriever


def test_confidence_weight_reorders_equally_relevant_memories():
    s = MemoryStore(":memory:")
    s.add_text("Project Apollo deadline is Friday.", source="rumor", confidence=0.2)
    s.add_text("Project Apollo deadline is Monday.", source="official", confidence=1.0)

    # Without confidence weighting the (tied) first-inserted memory wins.
    plain = Retriever(s, confidence_weight=0.0)
    # With full confidence weighting the high-confidence memory is preferred.
    weighted = Retriever(s, confidence_weight=1.0)

    assert "Monday" in weighted.search("Apollo deadline", top_k=1)[0].memory.content
    # Sanity: plain retriever ranks by text relevance only (both equally relevant).
    assert plain.search("Apollo deadline", top_k=2)  # both returned


def test_recency_weight_prefers_newer_memory():
    s = MemoryStore(":memory:")
    old = s.add_text("The office address is 1 Old Street.")
    new = s.add_text("The office address is 2 New Avenue.")
    # Force a clear age gap regardless of wall-clock timing.
    old.created_at = 0.0
    new.created_at = 1_000_000.0
    s.add(old)
    s.add(new)

    now = lambda: 1_000_000.0  # noqa: E731 - tiny clock stub
    r = Retriever(s, recency_weight=1.0, recency_half_life=100.0, now=now)
    top = r.search("office address", top_k=1)[0]
    assert "New Avenue" in top.memory.content


def test_default_ranking_is_unchanged_by_new_signals():
    s = MemoryStore(":memory:")
    s.add_text("The capital of France is Paris.")
    s.add_text("Bananas are yellow.")
    r = Retriever(s)  # all new weights default to 0 -> behaviour identical
    results = r.search("capital of France", top_k=2)
    assert "Paris" in results[0].memory.content


def test_source_filter_scopes_results():
    s = MemoryStore(":memory:")
    s.add_text("Paris is the capital of France.", source="geo")
    s.add_text("Paris Hilton is a celebrity.", source="gossip")
    r = Retriever(s)
    geo = r.search("Paris", top_k=5, source="geo")
    assert geo and all(res.memory.source == "geo" for res in geo)


def test_where_predicate_filters_results():
    s = MemoryStore(":memory:")
    s.add_text("Fact A about Paris.", source="a")
    s.add_text("Fact B about Paris.", source="b")
    r = Retriever(s)
    only_b = r.search("Paris", top_k=5, where=lambda m: m.source == "b")
    assert only_b and all(res.memory.source == "b" for res in only_b)


def test_invalid_weights_rejected():
    s = MemoryStore(":memory:")
    with pytest.raises(ValueError):
        Retriever(s, confidence_weight=2.0)
    with pytest.raises(ValueError):
        Retriever(s, recency_weight=-0.1)
    with pytest.raises(ValueError):
        Retriever(s, recency_half_life=0)
