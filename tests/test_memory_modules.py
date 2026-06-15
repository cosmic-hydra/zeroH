"""Tests for tiered memory, inverted index, and compression modules."""
import time

from zeroh.memory.tiered import TieredMemoryManager, AccessStats
from zeroh.memory.indexes import InvertedIndex
from zeroh.memory.compression import MemoryCompressor, CompressionReport
from zeroh.models import Memory


# -- TieredMemoryManager tests -----------------------------------------------

def _mem(mid: str, content: str = "test content") -> Memory:
    return Memory(content=content, source="test", id=mid)


def test_tiered_register_new_memory():
    mgr = TieredMemoryManager()
    tier = mgr.register(_mem("m1"))
    assert tier == "warm"
    assert mgr.tier_of("m1") == "warm"


def test_tiered_promote_on_access():
    mgr = TieredMemoryManager(promote_threshold=3)
    mgr.register(_mem("m1"))
    mgr.record_access("m1")
    mgr.record_access("m1")
    assert mgr.tier_of("m1") == "warm"
    mgr.record_access("m1")  # hits threshold
    assert mgr.tier_of("m1") == "hot"


def test_tiered_hot_capacity_eviction():
    mgr = TieredMemoryManager(hot_capacity=2, promote_threshold=1)
    for i in range(3):
        mgr.register(_mem(f"m{i}"))
        mgr.record_access(f"m{i}")
    # Should have evicted one from hot
    hot = mgr.get_hot_ids()
    assert len(hot) <= 2


def test_tiered_demote_stale():
    mgr = TieredMemoryManager(promote_threshold=1, demote_after=10.0)
    mgr.register(_mem("m1"))
    mgr.record_access("m1")
    assert mgr.tier_of("m1") == "hot"
    # Simulate time passing — memory becomes stale and cascades hot→warm→cold
    mgr._stats["m1"].last_accessed = time.time() - 20.0
    demoted = mgr.demote_stale()
    assert demoted >= 1
    # After full demotion pass it ends in cold (hot→warm then warm→cold)
    assert mgr.tier_of("m1") in ("warm", "cold")


def test_tiered_remove():
    mgr = TieredMemoryManager()
    mgr.register(_mem("m1"))
    mgr.remove("m1")
    assert mgr.tier_of("m1") == "cold"
    assert "m1" not in mgr._stats


def test_tiered_stats():
    mgr = TieredMemoryManager(promote_threshold=1)
    mgr.register(_mem("m1"))
    mgr.register(_mem("m2"))
    mgr.record_access("m1")
    stats = mgr.stats()
    assert stats["hot"] == 1
    assert stats["warm"] == 1


def test_tiered_warm_capacity_overflow():
    mgr = TieredMemoryManager(warm_capacity=2)
    for i in range(5):
        mgr.register(_mem(f"m{i}"))
    # Should move excess to cold
    assert len(mgr.get_warm_ids()) <= 2


# -- InvertedIndex tests -----------------------------------------------------

def test_index_add_and_search():
    idx = InvertedIndex()
    idx.add(Memory(content="Paris is the capital of France.", source="kb", id="m1"))
    idx.add(Memory(content="Berlin is the capital of Germany.", source="kb", id="m2"))
    results = idx.search("capital of France")
    assert "m1" in results
    # m1 should rank higher (more token overlap with query)
    assert results[0] == "m1"


def test_index_remove():
    idx = InvertedIndex()
    idx.add(Memory(content="Hello world.", source="test", id="m1"))
    assert idx.memory_count() == 1
    idx.remove("m1")
    assert idx.memory_count() == 0
    assert idx.search("hello") == []


def test_index_rebuild():
    idx = InvertedIndex()
    memories = [
        Memory(content="Cats are pets.", source="test", id="m1"),
        Memory(content="Dogs are pets.", source="test", id="m2"),
    ]
    idx.rebuild(memories)
    assert idx.memory_count() == 2
    results = idx.search("cats")
    assert "m1" in results


def test_index_min_overlap():
    idx = InvertedIndex()
    idx.add(Memory(content="Paris France capital city Europe.", source="kb", id="m1"))
    idx.add(Memory(content="Dogs cats pets animals.", source="kb", id="m2"))
    # With min_overlap=2, only m1 should match
    results = idx.search("Paris capital France", min_overlap=2)
    assert "m1" in results
    assert "m2" not in results


def test_index_top_k():
    idx = InvertedIndex()
    for i in range(10):
        idx.add(Memory(content=f"Memory about topic number {i}.", source="test", id=f"m{i}"))
    results = idx.search("memory topic number", top_k=3)
    assert len(results) <= 3


def test_index_stats():
    idx = InvertedIndex()
    idx.add(Memory(content="Hello world example.", source="test", id="m1"))
    stats = idx.stats()
    assert stats["memory_count"] == 1
    assert stats["vocabulary_size"] > 0


def test_index_token_frequency():
    idx = InvertedIndex()
    idx.add(Memory(content="Paris is great.", source="test", id="m1"))
    idx.add(Memory(content="Paris is beautiful.", source="test", id="m2"))
    assert idx.token_frequency("paris") == 2


# -- MemoryCompressor tests --------------------------------------------------

def test_compressor_analyze_no_redundancy():
    compressor = MemoryCompressor()
    memories = [
        Memory(content="Cats are independent animals.", source="test", id="m1"),
        Memory(content="The stock market crashed in 2008.", source="test", id="m2"),
    ]
    report = compressor.analyze(memories)
    assert report.total_memories == 2
    assert report.redundancy_ratio == 0.0


def test_compressor_analyze_with_redundancy():
    compressor = MemoryCompressor(similarity_threshold=0.7)
    memories = [
        Memory(content="The capital of France is Paris city.", source="test", id="m1"),
        Memory(content="Paris city is the capital of France.", source="test", id="m2"),
        Memory(content="Dogs are friendly animals.", source="test", id="m3"),
    ]
    report = compressor.analyze(memories)
    assert len(report.redundancy_groups) >= 1
    assert report.redundancy_ratio > 0


def test_compressor_merge_group():
    compressor = MemoryCompressor()
    memories = [
        Memory(content="Short.", source="test", id="m1"),
        Memory(content="This is a much longer memory with more detail.", source="test", id="m2"),
    ]
    merged = compressor.merge_group(memories)
    # Should pick the longest
    assert merged.content == "This is a much longer memory with more detail."
    assert merged.source == "compression"
    assert "m1" in merged.metadata["merged_from"]
    assert "m2" in merged.metadata["merged_from"]


def test_compressor_find_prefix_duplicates():
    compressor = MemoryCompressor()
    prefix = "A" * 60
    memories = [
        Memory(content=prefix + " ending one.", source="test", id="m1"),
        Memory(content=prefix + " ending two.", source="test", id="m2"),
        Memory(content="Completely different content here.", source="test", id="m3"),
    ]
    dups = compressor.find_prefix_duplicates(memories, min_prefix_length=50)
    assert len(dups) >= 1
    assert "m1" in dups[0][1]
    assert "m2" in dups[0][1]


def test_compressor_extract_unique_facts():
    compressor = MemoryCompressor()
    memories = [
        Memory(content="Paris is in France. The sky is blue.", source="test", id="m1"),
        Memory(content="Paris is in France. Water is wet.", source="test", id="m2"),
    ]
    facts = compressor.extract_unique_facts(memories)
    # "Paris is in France" should appear only once
    paris_count = sum(1 for f in facts if "Paris" in f)
    assert paris_count == 1
    assert len(facts) == 3  # Paris, sky, water
