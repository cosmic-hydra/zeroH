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
        **metadata,
    ) -> Memory:
        """Convenience helper to create and store a memory from raw text."""
        mem = Memory(
            content=content,
            source=source,
            confidence=confidence,
            metadata=metadata,
        )
        return self.add(mem)

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
