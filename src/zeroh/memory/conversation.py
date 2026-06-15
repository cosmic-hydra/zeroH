"""Short-term conversation memory (the agent's working context).

Where :class:`~zeroh.memory.store.MemoryStore` is *long-term*, durable knowledge,
:class:`ConversationMemory` is the *short-term* context of the current dialogue:
a small, bounded, in-process ring buffer of the most recent turns.

It serves two jobs in the plug-in:

* **Continuity** – follow-up questions like "and where is *it*?" can be expanded
  with recent turns so retrieval still finds the right grounding facts.
* **Boundedness** – only the last ``max_turns`` turns are kept, so context never
  grows without limit and the agent stays focused on the live exchange.

It is intentionally ephemeral and dependency-free; nothing here is persisted, so
the durable store remains the single source of truth for grounded answers.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List


@dataclass
class Turn:
    """A single message in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, object]:
        return {"role": self.role, "content": self.content, "created_at": self.created_at}


class ConversationMemory:
    """A bounded buffer of the most recent conversation turns.

    Args:
        max_turns: Maximum number of turns to retain. Older turns are dropped as
            new ones arrive, keeping the working context small and relevant.
    """

    def __init__(self, max_turns: int = 8) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.max_turns = max_turns
        self._turns: Deque[Turn] = deque(maxlen=max_turns)

    def add(self, role: str, content: str) -> Turn:
        """Append a turn, evicting the oldest if the buffer is full."""
        turn = Turn(role=role, content=content)
        self._turns.append(turn)
        return turn

    def add_user(self, content: str) -> Turn:
        return self.add("user", content)

    def add_assistant(self, content: str) -> Turn:
        return self.add("assistant", content)

    def recent(self, n: int = 0) -> List[Turn]:
        """Return the last ``n`` turns (all of them when ``n`` <= 0)."""
        turns = list(self._turns)
        if n and n > 0:
            return turns[-n:]
        return turns

    def query_context(self, n: int = 3) -> str:
        """Build a retrieval cue from the last ``n`` *user* turns.

        Concatenating the recent user turns lets a terse follow-up
        ("and the population?") still retrieve grounding facts about the entity
        established earlier in the conversation.
        """
        users = [t.content for t in self._turns if t.role == "user"]
        return " ".join(users[-n:]) if n > 0 else " ".join(users)

    def transcript(self, n: int = 0) -> str:
        """Render the last ``n`` turns as a simple ``Role: text`` transcript."""
        turns = self.recent(n)
        return "\n".join(f"{t.role.capitalize()}: {t.content}" for t in turns)

    def clear(self) -> None:
        """Forget the entire short-term conversation."""
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)

    def __iter__(self):
        return iter(list(self._turns))
