"""Tests for the memory store's introspection, portability and dedupe helpers."""
from zeroh.memory import MemoryStore
from zeroh.models import Memory


def _store():
    s = MemoryStore(":memory:")
    s.add_text("The capital of France is Paris.", source="kb")
    s.add_text("The Eiffel Tower is in Paris.", source="kb")
    s.add_text("Refunds within 30 days.", source="policy")
    return s


def test_stats_counts_active_inactive_and_sources():
    s = _store()
    m = s.add_text("temp", source="scratch")
    s.deactivate(m.id)
    stats = s.stats()
    assert stats["active"] == 3
    assert stats["inactive"] == 1
    assert stats["total"] == 4
    assert stats["by_source"]["kb"] == 2
    assert stats["by_source"]["policy"] == 1
    assert "scratch" not in stats["by_source"]  # inactive excluded


def test_by_source_and_sources():
    s = _store()
    kb = s.by_source("kb")
    assert {m.content for m in kb} == {
        "The capital of France is Paris.",
        "The Eiffel Tower is in Paris.",
    }
    assert set(s.sources()) == {"kb", "policy"}


def test_dedupe_skips_identical_active_content():
    s = MemoryStore(":memory:")
    a = s.add_text("same fact", source="x")
    b = s.add_text("same fact", source="y", dedupe=True)
    assert a.id == b.id  # returned existing instead of inserting a duplicate
    assert len(s) == 1


def test_find_by_content():
    s = _store()
    found = s.find_by_content("Refunds within 30 days.")
    assert found is not None and found.source == "policy"
    assert s.find_by_content("nonexistent") is None


def test_export_import_roundtrip():
    s = _store()
    dump = s.export_jsonl()
    assert dump.count("\n") == 2  # three lines -> two newlines

    restored = MemoryStore(":memory:")
    n = restored.import_jsonl(dump)
    assert n == 3
    assert len(restored) == 3
    contents = {m.content for m in restored.all()}
    assert "The capital of France is Paris." in contents


def test_import_is_idempotent_by_id():
    s = _store()
    dump = s.export_jsonl()
    target = MemoryStore(":memory:")
    target.import_jsonl(dump)
    target.import_jsonl(dump)  # second import must not duplicate
    assert len(target) == 3


def test_full_backup_preserves_inactive_and_supersede_state():
    # A faithful backup must not revive tombstoned/superseded memories as active.
    s = MemoryStore(":memory:")
    s.add_text("active fact", source="a")
    gone = s.add_text("forgotten fact", source="a")
    s.deactivate(gone.id)
    old = s.add_text("the sky is green", source="b")
    s.supersede(old.id, Memory(content="the sky is blue", source="b"))

    restored = MemoryStore(":memory:")
    restored.import_jsonl(s.export_jsonl(include_inactive=True))

    # Active set is preserved exactly (forgotten + superseded stay inactive)...
    assert {m.content for m in restored.all()} == {"active fact", "the sky is blue"}
    # ...while the tombstoned history is still retained for audit.
    everything = {m.content for m in restored.all(include_inactive=True)}
    assert "forgotten fact" in everything
    assert "the sky is green" in everything


def test_plain_memory_jsonl_still_imports_as_active():
    # Backward compatibility: records without active/supersedes load as active.
    s = MemoryStore(":memory:")
    s.import_jsonl('{"content": "legacy fact", "source": "old"}')
    assert [m.content for m in s.all()] == ["legacy fact"]
