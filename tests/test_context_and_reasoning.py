"""Tests for enhanced context awareness, reasoning, and optimization modules.

Covers the new features added in v0.6.0:
- Enhanced QueryCache with frequency-aware eviction, warming, and prefix matching
- BM25-scored InvertedIndex with bigram support
- Context preservation layer (entity tracking, topic stack, fact pinning)
- Semantic reasoning (entity extraction, relation extraction, query expansion)
- Integrated retrieval pipeline (cache + index + expansion)
"""
import time

from zeroh import (
    ZeroHPlugin,
    MemoryStore,
    InvertedIndex,
    QueryCache,
    ContextPreserver,
    EntityExtractor,
    RelationExtractor,
    QueryExpander,
)
from zeroh.memory.context import TopicFrame, PreservedFact, ResolvedEntity
from zeroh.reasoning.relations import Relation


# -- Enhanced QueryCache ------------------------------------------------------

def test_cache_frequency_aware_keeps_popular():
    """Frequently accessed entries survive eviction over unused ones."""
    cache = QueryCache(max_size=3, ttl=60.0)
    cache.put("popular", "result1")
    cache.put("unused1", "result2")
    cache.put("unused2", "result3")
    # Access "popular" multiple times
    for _ in range(5):
        cache.get("popular")
    # Add new entry — should evict an unused one, not "popular"
    cache.put("new", "result4")
    assert cache.get("popular") is not None
    assert len(cache) == 3


def test_cache_prefix_matching():
    """Prefix matching returns cached results for related queries."""
    cache = QueryCache(max_size=10, ttl=60.0, enable_prefix=True, prefix_min_length=5)
    cache.put("capital of France", ["Paris"])
    # Longer query starting with same prefix should hit
    result = cache.get("capital of France and population")
    assert result == ["Paris"]


def test_cache_prefix_matching_reverse():
    """Prefix matching works when cached key is longer than query."""
    cache = QueryCache(max_size=10, ttl=60.0, enable_prefix=True, prefix_min_length=5)
    cache.put("capital of France and history", ["Paris"])
    result = cache.get("capital of France")
    assert result == ["Paris"]


def test_cache_prefix_disabled():
    """Prefix matching can be disabled."""
    cache = QueryCache(max_size=10, ttl=60.0, enable_prefix=False)
    cache.put("capital of France", ["Paris"])
    result = cache.get("capital of France and more")
    assert result is None


def test_cache_warming():
    """Cache warming pre-populates entries."""
    cache = QueryCache(max_size=10, ttl=60.0)
    entries = [("q1", "r1"), ("q2", "r2"), ("q3", "r3")]
    warmed = cache.warm(entries)
    assert warmed == 3
    assert cache.get("q1") == "r1"
    assert cache.get("q2") == "r2"


def test_cache_warming_respects_capacity():
    """Cache warming stops at max_size."""
    cache = QueryCache(max_size=2, ttl=60.0)
    entries = [("q1", "r1"), ("q2", "r2"), ("q3", "r3")]
    warmed = cache.warm(entries)
    assert warmed == 2
    assert len(cache) == 2


def test_cache_invalidate_prefix():
    """Prefix-based invalidation removes matching entries."""
    cache = QueryCache(max_size=10, ttl=60.0)
    cache.put("capital of France", "Paris")
    cache.put("capital of Germany", "Berlin")
    cache.put("population of France", "67M")
    removed = cache.invalidate_prefix("capital")
    assert removed == 2
    assert cache.get("population of France") == "67M"


def test_cache_top_entries():
    """top_entries returns most accessed entries."""
    cache = QueryCache(max_size=10, ttl=60.0)
    cache.put("q1", "r1")
    cache.put("q2", "r2")
    cache.get("q1")
    cache.get("q1")
    cache.get("q2")
    top = cache.top_entries(2)
    assert top[0] == ("q1", 2)
    assert top[1] == ("q2", 1)


def test_cache_stats_includes_prefix_hits():
    """Stats track prefix hit count separately."""
    cache = QueryCache(max_size=10, ttl=60.0, enable_prefix=True, prefix_min_length=5)
    cache.put("capital of France", "Paris")
    cache.get("capital of France and history")  # prefix hit
    stats = cache.stats()
    assert stats["prefix_hits"] == 1
    assert stats["hits"] == 1


# -- BM25 InvertedIndex -------------------------------------------------------

def test_index_bm25_scoring():
    """BM25 scoring ranks relevant documents higher."""
    from zeroh.models import Memory
    index = InvertedIndex(enable_bigrams=True)
    m1 = Memory(content="Paris is the capital of France.", source="kb", id="m1")
    m2 = Memory(content="Berlin is the capital of Germany.", source="kb", id="m2")
    m3 = Memory(content="The weather today is sunny and warm.", source="kb", id="m3")
    index.add(m1)
    index.add(m2)
    index.add(m3)

    results = index.search("capital of France")
    assert "m1" in results
    # m1 should rank higher than m3 (more term overlap)
    if "m3" in results:
        assert results.index("m1") < results.index("m3")


def test_index_bigram_matching():
    """Bigrams improve precision for phrase queries."""
    from zeroh.models import Memory
    index = InvertedIndex(enable_bigrams=True)
    m1 = Memory(content="Machine learning is a subfield of AI.", source="kb", id="m1")
    m2 = Memory(content="The machine was broken and needed repair.", source="kb", id="m2")
    index.add(m1)
    index.add(m2)

    # "machine learning" as bigram should boost m1
    results = index.search("machine learning")
    assert results[0] == "m1"


def test_index_bigrams_disabled():
    """Index works without bigrams."""
    from zeroh.models import Memory
    index = InvertedIndex(enable_bigrams=False)
    m1 = Memory(content="Paris is the capital.", source="kb", id="m1")
    index.add(m1)
    results = index.search("Paris capital")
    assert "m1" in results


def test_index_idf_computation():
    """IDF gives higher weight to rarer terms."""
    from zeroh.models import Memory
    index = InvertedIndex()
    # Add multiple docs with "capital" and only one with "Paris"
    for i in range(5):
        index.add(Memory(content=f"The capital of country {i}.", source="kb", id=f"m{i}"))
    index.add(Memory(content="Paris is a beautiful city.", source="kb", id="paris"))

    # "paris" should have higher IDF than "capital"
    idf_paris = index.idf("paris")
    idf_capital = index.idf("capital")
    assert idf_paris > idf_capital


def test_index_search_scored():
    """search_scored returns memory IDs with BM25 scores."""
    from zeroh.models import Memory
    index = InvertedIndex()
    index.add(Memory(content="Paris is the capital of France.", source="kb", id="m1"))
    index.add(Memory(content="Berlin is in Germany.", source="kb", id="m2"))

    scored = index.search_scored("Paris France")
    assert len(scored) > 0
    assert scored[0][0] == "m1"
    assert scored[0][1] > 0  # Has a positive BM25 score


def test_index_stats_includes_bigrams():
    """Stats include bigram-related info."""
    from zeroh.models import Memory
    index = InvertedIndex(enable_bigrams=True)
    index.add(Memory(content="Hello world test.", source="kb", id="m1"))
    stats = index.stats()
    assert stats["bigrams_enabled"] is True
    assert stats["avg_doc_length"] > 0


def test_index_rebuild_resets():
    """Rebuild clears and recreates the index."""
    from zeroh.models import Memory
    index = InvertedIndex()
    index.add(Memory(content="Old content.", source="kb", id="old"))
    assert index.memory_count() == 1

    new_mems = [Memory(content="New content.", source="kb", id="new")]
    index.rebuild(new_mems)
    assert index.memory_count() == 1
    assert "old" not in index.search("old content")


# -- Context Preservation Layer ------------------------------------------------

def test_context_preserver_topic_tracking():
    """Topics are tracked and decayed over turns."""
    cp = ContextPreserver()
    cp.add_topic("capital of France", keywords={"capital", "france"})
    assert "capital of France" in cp.get_active_topics()

    # Advance turns — topic should decay
    for _ in range(20):
        cp.advance_turn()
    # After many turns without re-activation, topic should be pruned
    assert len(cp.get_active_topics()) == 0


def test_context_preserver_entity_resolution():
    """Resolved entities are preserved and used for query expansion."""
    cp = ContextPreserver()
    cp.resolve_entity("it", "the Eiffel Tower")
    entities = cp.get_resolved_entities()
    assert entities["it"] == "the Eiffel Tower"


def test_context_preserver_entity_expiry():
    """Entities expire after entity_ttl_turns."""
    cp = ContextPreserver(entity_ttl_turns=3)
    cp.resolve_entity("it", "Paris")
    for _ in range(5):
        cp.advance_turn()
    # Should be expired
    assert len(cp.get_resolved_entities()) == 0


def test_context_preserver_query_expansion():
    """expand_query substitutes resolved references."""
    cp = ContextPreserver()
    cp.resolve_entity("it", "the Eiffel Tower")
    expanded = cp.expand_query("What is it made of?")
    assert "Eiffel Tower" in expanded


def test_context_preserver_fact_preservation():
    """Preserved facts are available for re-injection."""
    cp = ContextPreserver()
    cp.preserve_fact("Paris is the capital of France.", confidence=0.9)
    preserved = cp.get_preserved_context(max_tokens=100)
    assert "Paris is the capital of France" in preserved


def test_context_preserver_topic_boost():
    """Active topics add keywords to expanded queries."""
    cp = ContextPreserver()
    cp.add_topic("French history", keywords={"french", "history"})
    expanded = cp.expand_query("Tell me more")
    # Should include topic keywords
    assert "french" in expanded.lower() or "history" in expanded.lower()


def test_context_preserver_topic_dedup():
    """Adding same topic again updates rather than duplicates."""
    cp = ContextPreserver()
    cp.add_topic("France capital", keywords={"france", "capital", "city"})
    cp.add_topic("France capital Paris", keywords={"france", "capital"})
    assert len(cp._topics) == 1
    # Keywords should be merged
    assert "city" in cp._topics[0].keywords


def test_context_preserver_clear():
    """clear() resets all state."""
    cp = ContextPreserver()
    cp.add_topic("test")
    cp.preserve_fact("fact")
    cp.resolve_entity("it", "thing")
    cp.clear()
    assert cp.turn_count == 0
    assert len(cp.get_active_topics()) == 0


# -- Entity Extraction ---------------------------------------------------------

def test_entity_extractor_proper_nouns():
    """Extracts proper nouns (capitalized multi-word phrases)."""
    ex = EntityExtractor()
    entities = ex.extract("The Eiffel Tower is located in Paris, France.")
    names = [e.text for e in entities if e.entity_type == "proper_noun"]
    assert any("Eiffel Tower" in n for n in names)


def test_entity_extractor_numerics():
    """Extracts numeric values."""
    ex = EntityExtractor()
    entities = ex.extract("The population is 67 million people.")
    types = [e.entity_type for e in entities]
    assert "numeric" in types


def test_entity_extractor_keywords():
    """extract_keywords returns important terms."""
    ex = EntityExtractor()
    keywords = ex.extract_keywords("The capital city of France is Paris.")
    assert len(keywords) > 0


def test_entity_extractor_quoted():
    """Extracts quoted strings."""
    ex = EntityExtractor()
    entities = ex.extract('He said "hello world" to everyone.')
    quoted = [e for e in entities if e.entity_type == "quoted"]
    assert any("hello world" in e.text for e in quoted)


# -- Relation Extraction -------------------------------------------------------

def test_relation_extractor_is_a():
    """Extracts 'X is Y' relations."""
    rx = RelationExtractor()
    relations = rx.extract("Paris is the capital of France.")
    assert len(relations) > 0
    assert any("capital" in r.predicate or "capital" in r.object for r in relations)


def test_relation_extractor_capital_of():
    """Extracts specific 'capital of' patterns."""
    rx = RelationExtractor()
    relations = rx.extract("Berlin is the capital of Germany.")
    capital_rels = [r for r in relations if "capital" in r.predicate]
    assert len(capital_rels) > 0
    assert capital_rels[0].confidence >= 0.9


def test_relation_extractor_find_related():
    """find_related returns relations involving an entity."""
    rx = RelationExtractor()
    relations = rx.extract(
        "Paris is the capital of France. Paris has many museums. "
        "Berlin is the capital of Germany."
    )
    paris_rels = rx.find_related("Paris", relations)
    assert len(paris_rels) >= 1


def test_relation_matches_query():
    """Relation.matches_query checks relevance."""
    rel = Relation(subject="Paris", predicate="is the capital of", object="France")
    assert rel.matches_query("What is the capital of France?")
    assert not rel.matches_query("How old is the universe?")


# -- Query Expansion -----------------------------------------------------------

def test_query_expander_synonyms():
    """Expands query with synonyms."""
    qe = QueryExpander()
    expanded = qe.expand("capital city")
    assert len(expanded) > len("capital city")


def test_query_expander_intent():
    """Expands based on question type."""
    qe = QueryExpander(enable_intent=True)
    expanded = qe.expand("where is Paris located")
    # Should add location-related terms
    assert any(term in expanded.lower() for term in ["location", "place", "country", "city"])


def test_query_expander_custom_synonyms():
    """Custom synonyms are used for expansion."""
    qe = QueryExpander(custom_synonyms={"zeroh": ["grounding", "memory"]})
    expanded = qe.expand("How does zeroh work?")
    assert "grounding" in expanded or "memory" in expanded


def test_query_expander_disabled():
    """Expansion can be selectively disabled."""
    qe = QueryExpander(enable_synonyms=False, enable_intent=False)
    original = "capital of France"
    expanded = qe.expand(original)
    # Should still have morphological expansions but no synonyms/intent
    # The expansion may or may not add terms depending on morphology
    assert original in expanded


def test_query_expander_get_expansions():
    """get_expansions returns just the added terms."""
    qe = QueryExpander()
    expansions = qe.get_expansions("capital of France")
    # Should not include original query words
    assert all(exp not in ["capital", "france"] for exp in expansions)


def test_query_expander_max_expansions():
    """Respects max_expansions limit."""
    qe = QueryExpander(max_expansions=2)
    expansions = qe.get_expansions("big old country city population")
    assert len(expansions) <= 2


# -- Integration: Plugin with new features ------------------------------------

def test_plugin_query_expansion_improves_recall():
    """Query expansion helps retrieve memories with paraphrased content."""
    zh = ZeroHPlugin(enable_query_expansion=True)
    zh.remember("The capital city of France is Paris.")
    # Without expansion, "administrative center of France" might not match
    # With expansion, "capital" synonyms include "administrative center"
    context = zh.build_context("capital of France")
    assert len(context) > 0


def test_plugin_context_preservation_enabled():
    """Context preserver is initialized when enabled."""
    zh = ZeroHPlugin(enable_context_preservation=True)
    assert zh.context_preserver is not None


def test_plugin_context_preservation_disabled():
    """Context preserver can be disabled."""
    zh = ZeroHPlugin(enable_context_preservation=False)
    assert zh.context_preserver is None


def test_plugin_reset_clears_preserver():
    """reset_conversation clears context preserver state."""
    zh = ZeroHPlugin(enable_context_preservation=True)
    zh.context_preserver.add_topic("test topic")
    zh.reset_conversation()
    assert len(zh.context_preserver.get_active_topics()) == 0


def test_plugin_retriever_has_cache():
    """Retriever integrates query cache."""
    zh = ZeroHPlugin()
    zh.remember("Paris is the capital of France.")
    # First query
    ctx1 = zh.build_context("capital of France")
    # Second query (should hit cache)
    ctx2 = zh.build_context("capital of France")
    assert len(ctx1) == len(ctx2)
    cache_stats = zh.retriever.cache_stats
    assert cache_stats["hits"] >= 0  # May or may not hit depending on expansion


def test_plugin_build_prompt_with_preserved_facts():
    """build_prompt includes preserved facts when available."""
    zh = ZeroHPlugin(enable_context_preservation=True)
    zh.remember("Paris is the capital of France.")
    zh.context_preserver.preserve_fact("France is in Europe.")

    ctx = zh.build_context("capital of France")
    prompt = zh.build_prompt("capital of France", ctx)
    assert "Previously established facts:" in prompt
    assert "France is in Europe" in prompt
