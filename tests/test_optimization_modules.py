"""Tests for optimization package: cache, batch, and metrics modules."""
import time

from zeroh.optimization.cache import QueryCache
from zeroh.optimization.batch import BatchProcessor, BatchResult, ValidationIssue
from zeroh.optimization.metrics import MemoryMetrics, MemoryHealthReport
from zeroh.memory.store import MemoryStore
from zeroh.models import Memory


# -- QueryCache tests --------------------------------------------------------

def test_cache_put_and_get():
    cache = QueryCache(max_size=10, ttl=60.0)
    cache.put("query1", ["result1"])
    assert cache.get("query1") == ["result1"]


def test_cache_miss():
    cache = QueryCache(max_size=10, ttl=60.0)
    assert cache.get("nonexistent") is None


def test_cache_ttl_expiry():
    cache = QueryCache(max_size=10, ttl=0.01)
    cache.put("query1", ["result1"])
    time.sleep(0.02)
    assert cache.get("query1") is None


def test_cache_lru_eviction():
    cache = QueryCache(max_size=3, ttl=60.0)
    cache.put("q1", "r1")
    cache.put("q2", "r2")
    cache.put("q3", "r3")
    cache.put("q4", "r4")  # Should evict q1
    assert cache.get("q1") is None
    assert cache.get("q2") == "r2"


def test_cache_lru_access_refreshes():
    cache = QueryCache(max_size=3, ttl=60.0)
    cache.put("q1", "r1")
    cache.put("q2", "r2")
    cache.put("q3", "r3")
    cache.get("q1")  # Access q1 to make it recently used
    cache.put("q4", "r4")  # Should evict q2 (least recently used)
    assert cache.get("q1") == "r1"
    assert cache.get("q2") is None


def test_cache_invalidate_all():
    cache = QueryCache(max_size=10, ttl=60.0)
    cache.put("q1", "r1")
    cache.put("q2", "r2")
    removed = cache.invalidate()
    assert removed == 2
    assert len(cache) == 0


def test_cache_invalidate_key():
    cache = QueryCache(max_size=10, ttl=60.0)
    cache.put("q1", "r1")
    cache.put("q2", "r2")
    removed = cache.invalidate("q1")
    assert removed == 1
    assert cache.get("q1") is None
    assert cache.get("q2") == "r2"


def test_cache_stats():
    cache = QueryCache(max_size=10, ttl=60.0)
    cache.put("q1", "r1")
    cache.get("q1")  # hit
    cache.get("q2")  # miss
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_cache_contains():
    cache = QueryCache(max_size=10, ttl=60.0)
    cache.put("q1", "r1")
    assert "q1" in cache
    assert "q2" not in cache


# -- BatchProcessor tests ----------------------------------------------------

def _make_memories(contents):
    return [Memory(content=c, source="test", id=f"m{i}") for i, c in enumerate(contents)]


def test_batch_deduplicate():
    proc = BatchProcessor(dedupe_threshold=0.85)
    memories = _make_memories([
        "The capital of France is Paris.",
        "Paris is the capital of France.",  # near-duplicate
        "Berlin is the capital of Germany.",
    ])
    deduped = proc.deduplicate(memories)
    assert len(deduped) == 2  # One pair deduplicated


def test_batch_deduplicate_empty():
    proc = BatchProcessor()
    assert proc.deduplicate([]) == []


def test_batch_deduplicate_all_unique():
    proc = BatchProcessor()
    memories = _make_memories([
        "Cats are animals.",
        "The stock market is volatile.",
        "Mathematics involves numbers.",
    ])
    deduped = proc.deduplicate(memories)
    assert len(deduped) == 3


def test_batch_validate_too_short():
    proc = BatchProcessor(min_content_length=20)
    memories = [Memory(content="Hi.", source="test", id="m1")]
    issues = proc.validate(memories)
    assert any(i.issue_type == "too_short" for i in issues)


def test_batch_validate_too_long():
    proc = BatchProcessor(max_content_length=50)
    memories = [Memory(content="x" * 100, source="test", id="m1")]
    issues = proc.validate(memories)
    assert any(i.issue_type == "too_long" for i in issues)


def test_batch_validate_low_info():
    proc = BatchProcessor(min_info_tokens=5)
    memories = [Memory(content="hi there", source="test", id="m1")]
    issues = proc.validate(memories)
    assert any(i.issue_type == "low_info" for i in issues)


def test_batch_validate_zero_confidence():
    proc = BatchProcessor()
    memories = [Memory(content="Some valid content here.", source="test", id="m1", confidence=0.0)]
    issues = proc.validate(memories)
    assert any(i.issue_type == "zero_confidence" for i in issues)


def test_batch_filter_quality():
    proc = BatchProcessor()
    memories = _make_memories([
        "Very short.",
        "This is a sufficiently long memory with good content.",
        "Another good quality memory with useful information.",
    ])
    memories[0].confidence = 0.1
    memories[1].confidence = 0.9
    memories[2].confidence = 0.8
    filtered = proc.filter_quality(memories, min_confidence=0.5)
    assert len(filtered) == 2


def test_batch_filter_quality_with_budget():
    proc = BatchProcessor()
    memories = _make_memories([
        "Short memory number one for testing.",
        "Short memory number two for testing.",
        "Short memory number three for testing.",
    ])
    for m in memories:
        m.confidence = 1.0
    # Very small budget should only keep a few
    filtered = proc.filter_quality(memories, max_token_budget=20)
    assert len(filtered) < 3


def test_batch_chunk_for_parallel():
    proc = BatchProcessor()
    memories = _make_memories([f"Memory {i}." for i in range(10)])
    chunks = proc.chunk_for_parallel(memories, chunk_size=3)
    assert len(chunks) == 4  # 3 + 3 + 3 + 1
    assert len(chunks[0]) == 3
    assert len(chunks[-1]) == 1


# -- MemoryMetrics tests -----------------------------------------------------

def test_metrics_analyze_empty():
    store = MemoryStore(":memory:")
    metrics = MemoryMetrics(store)
    report = metrics.analyze()
    assert report.total_memories == 0
    assert report.health_score == 1.0


def test_metrics_analyze_with_data():
    store = MemoryStore(":memory:")
    store.add_text("Paris is the capital of France.", source="kb", confidence=0.9)
    store.add_text("Berlin is the capital of Germany.", source="kb", confidence=0.8)
    store.add_text("Low confidence guess.", source="guess", confidence=0.2)
    metrics = MemoryMetrics(store)
    report = metrics.analyze()
    assert report.active_memories == 3
    assert report.total_tokens > 0
    assert report.avg_tokens_per_memory > 0
    assert "kb" in report.sources


def test_metrics_token_budget_analysis():
    store = MemoryStore(":memory:")
    for i in range(5):
        store.add_text(f"Memory number {i} with some content.", source="test")
    metrics = MemoryMetrics(store)
    analysis = metrics.token_budget_analysis(query_budget=100)
    assert analysis["total_memories"] == 5
    assert analysis["avg_tokens_per_memory"] > 0


def test_metrics_source_analysis():
    store = MemoryStore(":memory:")
    store.add_text("Fact from KB.", source="kb", confidence=0.9)
    store.add_text("User said this.", source="user", confidence=1.0)
    metrics = MemoryMetrics(store)
    by_source = metrics.source_analysis()
    assert "kb" in by_source
    assert "user" in by_source
    assert by_source["kb"]["count"] == 1


def test_metrics_recommendations_on_low_confidence():
    store = MemoryStore(":memory:")
    for i in range(10):
        store.add_text(f"Low confidence fact {i}.", source="test", confidence=0.2)
    metrics = MemoryMetrics(store)
    report = metrics.analyze()
    categories = [r.category for r in report.recommendations]
    assert "consolidation" in categories


def test_metrics_health_score_degrades_with_issues():
    store = MemoryStore(":memory:")
    for i in range(10):
        store.add_text(f"Low confidence fact {i}.", source="test", confidence=0.1)
    metrics = MemoryMetrics(store)
    report = metrics.analyze()
    assert report.health_score < 0.8
