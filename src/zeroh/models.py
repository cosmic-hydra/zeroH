"""Core data models used across zeroH.

These are intentionally simple, dependency-free dataclasses so they can be
serialized to/from SQLite and JSON without any third-party libraries.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Memory:
    """A single durable unit of knowledge stored by an agent.

    Attributes:
        content: The natural-language fact or note.
        source: Where the knowledge came from (user, tool, document, ...).
        metadata: Arbitrary structured metadata (tags, urls, timestamps, ...).
        id: Stable unique identifier.
        created_at: Unix timestamp when the memory was first stored.
        confidence: Caller-supplied trust score in [0, 1] for this memory.
    """

    content: str
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)
    created_at: float = field(default_factory=time.time)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        return cls(
            content=data["content"],
            source=data.get("source", "unknown"),
            metadata=data.get("metadata", {}) or {},
            id=data.get("id", _new_id()),
            created_at=data.get("created_at", time.time()),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class Citation:
    """A reference linking a piece of an answer back to a stored memory."""

    memory_id: str
    content: str
    source: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    """A memory returned by the retriever together with its relevance score."""

    memory: Memory
    score: float


@dataclass
class Claim:
    """A single atomic statement extracted from a candidate answer."""

    text: str
    supported: bool = False
    support_score: float = 0.0
    citations: List[Citation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "supported": self.supported,
            "support_score": self.support_score,
            "citations": [c.to_dict() for c in self.citations],
        }


@dataclass
class Answer:
    """The final, grounded response produced by the agent.

    `confidence` reflects how well the answer is grounded in stored memory.
    When `grounded` is False, the agent abstained rather than risk a
    hallucination.
    """

    text: str
    grounded: bool
    confidence: float
    claims: List[Claim] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    abstained: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "grounded": self.grounded,
            "confidence": self.confidence,
            "abstained": self.abstained,
            "claims": [c.to_dict() for c in self.claims],
            "citations": [c.to_dict() for c in self.citations],
        }
