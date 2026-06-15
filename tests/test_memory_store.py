"""Tests for the durable memory store."""
import os

from zeroh.memory import MemoryStore
from zeroh.models import Memory


def test_add_and_get():
    store = MemoryStore(":memory:")
    mem = store.add_text("hello world", source="test")
    assert store.get(mem.id).content == "hello world"
    assert len(store) == 1


def test_supersede_keeps_history():
    store = MemoryStore(":memory:")
    old = store.add_text("the sky is green")
    new = store.supersede(old.id, Memory(content="the sky is blue"))
    # Active set reflects the correction...
    active = [m.content for m in store.all()]
    assert active == ["the sky is blue"]
    # ...but the old memory is retained (tombstoned), not destroyed.
    all_including = [m.content for m in store.all(include_inactive=True)]
    assert "the sky is green" in all_including
    assert store.get(new.id).content == "the sky is blue"


def test_deactivate_excludes_from_active():
    store = MemoryStore(":memory:")
    m = store.add_text("temp fact")
    store.deactivate(m.id)
    assert len(store) == 0
    assert store.get(m.id) is not None  # still retrievable by id


def test_durability_across_reopen(tmp_path):
    db = os.path.join(tmp_path, "mem.db")
    store = MemoryStore(db)
    store.add_text("persisted fact", source="durable")
    store.close()

    reopened = MemoryStore(db)
    contents = [m.content for m in reopened.all()]
    assert "persisted fact" in contents
    reopened.close()
