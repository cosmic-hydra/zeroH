"""Tests for hallucination reduction, memory optimization, and token efficiency.

Covers the new features added in v0.4.0:
- Token-aware prompt budgeting (max_context_tokens)
- Adaptive retrieval (adaptive_top_k)
- Semantic deduplication (semantic_dedupe)
- Negation-aware contradiction detection
- Memory consolidation (consolidate)
- Improved prompt format with relevance scores
"""
import time

from zeroh import ZeroHPlugin, MemoryStore
from zeroh.text import estimate_tokens, has_negation, semantic_similarity_quick


# -- Token estimation --------------------------------------------------------

def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_short():
    assert estimate_tokens("hello") >= 1


def test_estimate_tokens_sentence():
    text = "The capital of France is Paris."
    tokens = estimate_tokens(text)
    # Should be roughly len/4, give or take
    assert 5 <= tokens <= 12


# -- Negation detection ------------------------------------------------------

def test_has_negation_positive():
    assert has_negation("Paris is not the capital of France.")
    assert has_negation("They never visited the museum.")
    assert has_negation("There is nothing wrong here.")
    assert has_negation("She won't come.")
    assert has_negation("It isn't true.")
    assert has_negation("I can't do that.")


def test_has_negation_negative():
    assert not has_negation("Paris is the capital of France.")
    assert not has_negation("They visited the museum.")


# -- Semantic similarity (Jaccard) -------------------------------------------

def test_semantic_similarity_identical():
    text = "The capital of France is Paris."
    assert semantic_similarity_quick(text, text) == 1.0


def test_semantic_similarity_different():
    a = "The capital of France is Paris."
    b = "Dogs are friendly animals."
    assert semantic_similarity_quick(a, b) < 0.2


def test_semantic_similarity_paraphrase():
    a = "Paris is the capital city of France."
    b = "France's capital city is Paris."
    # Paraphrases should have high overlap
    assert semantic_similarity_quick(a, b) >= 0.7


def test_semantic_similarity_empty():
    assert semantic_similarity_quick("", "hello") == 0.0
    assert semantic_similarity_quick("hello", "") == 0.0


# -- Semantic deduplication --------------------------------------------------

def test_semantic_dedupe_blocks_paraphrase():
    store = MemoryStore(":memory:")
    store.add_text("Paris is the capital of France.", source="kb")
    # Paraphrase should be rejected
    result = store.add_text(
        "France's capital is Paris.",
        source="kb",
        semantic_dedupe=True,
        similarity_threshold=0.7,
    )
    # Should return the existing memory (not create a new one)
    assert len(store) == 1


def test_semantic_dedupe_allows_different():
    store = MemoryStore(":memory:")
    store.add_text("Paris is the capital of France.", source="kb")
    store.add_text(
        "Berlin is the capital of Germany.",
        source="kb",
        semantic_dedupe=True,
    )
    assert len(store) == 2


def test_plugin_semantic_dedupe():
    zh = ZeroHPlugin(semantic_dedupe=True)
    zh.remember("The capital city of France is Paris.")
    zh.remember("The capital of France is Paris city.")
    # Near-duplicate should be rejected (high Jaccard overlap)
    assert len(zh.store) == 1


# -- Token budgeting ---------------------------------------------------------

def test_max_context_tokens_limits_context():
    zh = ZeroHPlugin(max_context_tokens=50)
    # Add many memories
    for i in range(20):
        zh.remember(f"Fact number {i}: this is a medium-length memory entry for testing purposes.")
    context = zh.build_context("facts about numbers")
    # Token budget should limit the number of items
    total_tokens = sum(estimate_tokens(r.memory.content) for r in context)
    assert total_tokens <= 50 + estimate_tokens(context[-1].memory.content) if context else True


def test_max_context_tokens_none_no_limit():
    zh = ZeroHPlugin(max_context_tokens=None)
    zh.remember("Paris is the capital of France.")
    zh.remember("Berlin is the capital of Germany.")
    zh.remember("Tokyo is the capital of Japan.")
    context = zh.build_context("capitals")
    # Without budget, all relevant memories should be returned
    assert len(context) >= 2


# -- Adaptive retrieval ------------------------------------------------------

def test_adaptive_top_k_narrow_query():
    zh = ZeroHPlugin(top_k=5, adaptive_top_k=True)
    # Short query -> fewer results
    k = zh._effective_top_k("Paris")
    assert k < 5


def test_adaptive_top_k_broad_query():
    zh = ZeroHPlugin(top_k=5, adaptive_top_k=True)
    # Long complex query -> more results
    k = zh._effective_top_k("what are the capitals of major European countries and their populations")
    assert k > 5


def test_adaptive_top_k_disabled():
    zh = ZeroHPlugin(top_k=5, adaptive_top_k=False)
    k = zh._effective_top_k("Paris")
    assert k == 5


# -- Negation-aware verification ---------------------------------------------

def test_negation_penalty_catches_contradiction():
    zh = ZeroHPlugin()
    zh.remember("Paris is the capital of France.")
    # Contradicting claim should NOT be supported due to negation penalty
    result = zh.verify("Paris is not the capital of France.")
    # The claim should have lower confidence because of negation mismatch
    assert result.confidence < 0.8 or result.abstained


def test_negation_matching_polarity_passes():
    zh = ZeroHPlugin()
    zh.remember("Paris is the capital of France.")
    result = zh.verify("Paris is the capital of France.")
    assert result.grounded


# -- Memory consolidation ----------------------------------------------------

def test_consolidate_removes_stale_low_confidence():
    zh = ZeroHPlugin()
    # Add a low-confidence memory with old timestamp
    old_time = time.time() - 60 * 24 * 3600  # 60 days ago
    from zeroh.models import Memory
    old_mem = Memory(
        content="Uncertain fact from long ago.",
        source="guess",
        confidence=0.2,
        created_at=old_time,
    )
    zh.store.add(old_mem)
    zh.retriever.reindex()

    # Also add a recent high-confidence memory
    zh.remember("Certain fact from today.", confidence=1.0)

    assert len(zh.store) == 2
    removed = zh.consolidate(max_age=30 * 24 * 3600, min_confidence=0.3)
    assert removed == 1
    assert len(zh.store) == 1


def test_consolidate_keeps_high_confidence():
    zh = ZeroHPlugin()
    from zeroh.models import Memory
    old_mem = Memory(
        content="Old but reliable fact.",
        source="kb",
        confidence=0.9,
        created_at=time.time() - 60 * 24 * 3600,
    )
    zh.store.add(old_mem)
    zh.retriever.reindex()

    removed = zh.consolidate(max_age=30 * 24 * 3600, min_confidence=0.3)
    assert removed == 0
    assert len(zh.store) == 1


# -- Improved prompt format --------------------------------------------------

def test_prompt_includes_relevance_scores():
    zh = ZeroHPlugin()
    zh.remember("Paris is the capital of France.")
    ctx = zh.build_context("capital of France")
    prompt = zh.build_prompt("capital of France", ctx)
    # Should include percentage score
    assert "%" in prompt
    assert "Context (relevance score in brackets):" in prompt


def test_prompt_includes_citation_instructions():
    zh = ZeroHPlugin()
    zh.remember("Paris is the capital of France.")
    ctx = zh.build_context("capital of France")
    prompt = zh.build_prompt("capital of France", ctx)
    assert "Cite context numbers" in prompt
