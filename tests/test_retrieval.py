"""Tests for retrieval ranking."""
from zeroh.memory import MemoryStore
from zeroh.retrieval import Retriever


def _store():
    s = MemoryStore(":memory:")
    s.add_text("The capital of France is Paris.")
    s.add_text("The Eiffel Tower is located in Paris.")
    s.add_text("Bananas are a yellow fruit rich in potassium.")
    return s


def test_search_ranks_relevant_first():
    r = Retriever(_store())
    results = r.search("What is the capital of France?", top_k=3)
    assert results
    assert "capital of France" in results[0].memory.content


def test_search_top_k_limit():
    r = Retriever(_store())
    assert len(r.search("Paris", top_k=1)) == 1


def test_irrelevant_query_low_scores():
    r = Retriever(_store())
    results = r.search("quantum chromodynamics", top_k=3)
    assert all(res.score < 0.2 for res in results) or not results


def test_reindex_picks_up_new_memory():
    store = _store()
    r = Retriever(store)
    store.add_text("Mount Everest is the tallest mountain on Earth.")
    r.reindex()
    results = r.search("tallest mountain", top_k=1)
    assert "Everest" in results[0].memory.content
