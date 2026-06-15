"""Durable, append-only memory store backed by SQLite.

Design goals (directly addressing *data loss*):

* **Durability** – memories are persisted to SQLite with WAL journaling, so an
  agent never loses what it has learned across restarts/crashes.
* **Append-only history** – updates never destroy prior content; superseded
  memories are tombstoned (``active = 0``) so an audit trail is retained.
* **Deterministic & offline** – no external services required.

The store is intentionally storage-only: retrieval/ranking lives in
:mod:`zeroh.retrieval` so concerns stay separated.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Iterator, List, Optional

from ..models import Memory

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    source      TEXT NOT NULL,
    metadata    TEXT NOT NULL,
    created_at  REAL NOT NULL,
    confidence  REAL NOT NULL,
    active      INTEGER NOT NULL DEFAULT 1,
    supersedes  TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active);
"""


class MemoryStore:
    """A persistent collection of :class:`~zeroh.models.Memory` objects.

    Args:
        path: SQLite database path. Use ``":memory:"`` for an ephemeral store
            (handy for tests); use a file path for durable persistence.
    """

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        # check_same_thread=False + an explicit lock lets the store be shared
        # safely across threads (e.g. an agent serving concurrent requests).
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            try:
                self._conn.execute("PRAGMA journal_mode=WAL;")
            except sqlite3.OperationalError:
                # :memory: databases do not support WAL; that is fine.
                pass
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- writes ---------------------------------------------------------------
    def add(self, memory: Memory) -> Memory:
        """Persist a memory durably and return it."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO memories "
                "(id, content, source, metadata, created_at, confidence, active, supersedes) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, NULL)",
                (
                    memory.id,
                    memory.content,
                    memory.source,
                    json.dumps(memory.metadata),
                    memory.created_at,
                    float(memory.confidence),
                ),
            )
            self._conn.commit()
        return memory

    def add_text(
        self,
        content: str,
        source: str = "unknown",
        confidence: float = 1.0,
        *,
        dedupe: bool = False,
        semantic_dedupe: bool = False,
        similarity_threshold: float = 0.85,
        **metadata,
    ) -> Memory:
        """Convenience helper to create and store a memory from raw text.

        Args:
            dedupe: When ``True``, if an active memory with identical content
                already exists it is returned unchanged instead of inserting a
                duplicate. Keeps the store (and retrieval) free of redundant
                near-identical facts.
            semantic_dedupe: When ``True``, also rejects memories that are
                semantically near-duplicates (Jaccard token overlap above
                ``similarity_threshold``). This prevents context dilution from
                paraphrased duplicates, which waste LLM tokens and weaken
                retrieval precision.
            similarity_threshold: Jaccard similarity threshold for semantic
                deduplication (only used when ``semantic_dedupe=True``).
        """
        if dedupe:
            existing = self.find_by_content(content)
            if existing is not None:
                return existing
        if semantic_dedupe:
            existing = self.find_similar(content, threshold=similarity_threshold)
            if existing is not None:
                return existing
        mem = Memory(
            content=content,
            source=source,
            confidence=confidence,
            metadata=metadata,
        )
        return self.add(mem)

    def find_by_content(self, content: str) -> Optional[Memory]:
        """Return the oldest active memory whose content matches exactly."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE active = 1 AND content = ? "
                "ORDER BY created_at ASC LIMIT 1",
                (content,),
            ).fetchone()
        return _row_to_memory(row) if row else None

    def find_similar(
        self, content: str, threshold: float = 0.85
    ) -> Optional[Memory]:
        """Return an active memory semantically similar above ``threshold``.

        Uses fast Jaccard token overlap (no embedding model needed). This
        catches paraphrased duplicates that exact matching misses, preventing
        near-identical memories from diluting retrieval context.

        Note: performs a linear scan over active memories. For stores with
        thousands of entries, consider batching ingestion with periodic
        deduplication rather than checking on every add.
        """
        from ..text import tokenize

        # Pre-compute the input tokens once to avoid repeated work per row.
        input_tokens = set(tokenize(content))
        if not input_tokens:
            return None

        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE active = 1"
            ).fetchall()
        for row in rows:
            candidate_tokens = set(tokenize(row["content"]))
            if not candidate_tokens:
                continue
            intersection = len(input_tokens & candidate_tokens)
            union = len(input_tokens | candidate_tokens)
            if union and (intersection / union) >= threshold:
                return _row_to_memory(row)
        return None

    def supersede(self, old_id: str, new_memory: Memory) -> Memory:
        """Replace a memory with a corrected version without losing history.

        The old memory is tombstoned (``active = 0``) rather than deleted, and
        the new memory records which memory it supersedes.
        """
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET active = 0 WHERE id = ?", (old_id,)
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO memories "
                "(id, content, source, metadata, created_at, confidence, active, supersedes) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                (
                    new_memory.id,
                    new_memory.content,
                    new_memory.source,
                    json.dumps(new_memory.metadata),
                    new_memory.created_at,
                    float(new_memory.confidence),
                    old_id,
                ),
            )
            self._conn.commit()
        return new_memory

    def deactivate(self, memory_id: str) -> None:
        """Soft-delete a memory (kept for audit, excluded from retrieval)."""
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET active = 0 WHERE id = ?", (memory_id,)
            )
            self._conn.commit()

    # -- reads ----------------------------------------------------------------
    def get(self, memory_id: str) -> Optional[Memory]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return _row_to_memory(row) if row else None

    def all(self, include_inactive: bool = False) -> List[Memory]:
        query = "SELECT * FROM memories"
        if not include_inactive:
            query += " WHERE active = 1"
        query += " ORDER BY created_at ASC"
        with self._lock:
            rows = self._conn.execute(query).fetchall()
        return [_row_to_memory(r) for r in rows]

    def by_source(self, source: str, include_inactive: bool = False) -> List[Memory]:
        """Return memories originating from a given ``source``."""
        query = "SELECT * FROM memories WHERE source = ?"
        if not include_inactive:
            query += " AND active = 1"
        query += " ORDER BY created_at ASC"
        with self._lock:
            rows = self._conn.execute(query, (source,)).fetchall()
        return [_row_to_memory(r) for r in rows]

    def sources(self) -> List[str]:
        """Distinct sources currently represented among active memories."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT source FROM memories WHERE active = 1 "
                "ORDER BY source ASC"
            ).fetchall()
        return [r["source"] for r in rows]

    def stats(self) -> dict:
        """Summarize the store: active/inactive counts and per-source totals."""
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) AS c FROM memories"
            ).fetchone()["c"]
            rows = self._conn.execute(
                "SELECT source, COUNT(*) AS c FROM memories WHERE active = 1 "
                "GROUP BY source ORDER BY c DESC"
            ).fetchall()
        by_source = {r["source"]: int(r["c"]) for r in rows}
        # The active count is exactly the sum of the per-source group counts.
        active = sum(by_source.values())
        return {
            "active": active,
            "inactive": int(total) - active,
            "total": int(total),
            "by_source": by_source,
        }

    def export_jsonl(self, include_inactive: bool = False) -> str:
        """Serialize memories as newline-delimited JSON (one memory per line).

        A portable, human-readable backup format that round-trips through
        :meth:`import_jsonl`. Unlike :meth:`~zeroh.models.Memory.to_dict`, the
        export preserves each memory's ``active`` and ``supersedes`` state, so a
        full backup (``include_inactive=True``) restores the append-only audit
        trail faithfully rather than reviving tombstoned memories as active.
        """
        query = "SELECT * FROM memories"
        if not include_inactive:
            query += " WHERE active = 1"
        query += " ORDER BY created_at ASC"
        with self._lock:
            rows = self._conn.execute(query).fetchall()
        lines = [json.dumps(_row_to_export_dict(r), sort_keys=True) for r in rows]
        return "\n".join(lines)

    def import_jsonl(self, data: str) -> int:
        """Load memories from JSONL produced by :meth:`export_jsonl`.

        Returns the number of memories imported. Each record's ``active`` and
        ``supersedes`` state is restored as-stored (defaulting to an active,
        non-superseding memory when absent, so plain ``Memory.to_dict`` JSONL also
        loads). Existing ids are preserved, so re-importing is idempotent.
        """
        count = 0
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            self._restore(json.loads(line))
            count += 1
        return count

    def _restore(self, record: dict) -> None:
        """Insert a memory from an exported record, preserving lifecycle state."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO memories "
                "(id, content, source, metadata, created_at, confidence, active, supersedes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.get("id") or uuid.uuid4().hex,
                    record["content"],
                    record.get("source", "unknown"),
                    json.dumps(record.get("metadata", {}) or {}),
                    record.get("created_at", time.time()),
                    float(record.get("confidence", 1.0)),
                    int(record.get("active", 1)),
                    record.get("supersedes"),
                ),
            )
            self._conn.commit()

    def __iter__(self) -> Iterator[Memory]:
        return iter(self.all())

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM memories WHERE active = 1"
            ).fetchone()
        return int(row["c"])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        id=row["id"],
        content=row["content"],
        source=row["source"],
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        created_at=row["created_at"],
        confidence=row["confidence"],
    )


def _row_to_export_dict(row: sqlite3.Row) -> dict:
    """Serialize a full memory row, including its ``active``/``supersedes`` state."""
    return {
        "id": row["id"],
        "content": row["content"],
        "source": row["source"],
        "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
        "created_at": row["created_at"],
        "confidence": row["confidence"],
        "active": row["active"],
        "supersedes": row["supersedes"],
    }
