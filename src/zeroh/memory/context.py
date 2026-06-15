"""Context preservation layer to prevent data loss between retrieval exchanges.

In multi-turn conversations, context can be lost between exchanges due to:
1. Ring buffer eviction in ConversationMemory (older turns dropped)
2. Retrieval shifting focus away from earlier conversation topics
3. No mechanism to carry forward important resolved entities/facts

The :class:`ContextPreserver` addresses these gaps by:

* **Tracking resolved entities**: When the LLM resolves an ambiguous reference
  (e.g., "it" → "the Eiffel Tower"), the resolved form is preserved for future
  retrieval cues.
* **Maintaining a topic stack**: Active conversation topics are tracked so that
  follow-up questions can still retrieve relevant context even when the query
  itself is terse.
* **Preserving key facts**: High-confidence facts surfaced during the exchange
  are pinned and re-injected into future contexts.
* **Cross-turn coreference**: Pronouns and references in new queries are expanded
  with resolved antecedents from earlier turns.
"""
from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

from ..text import tokenize


@dataclass
class ResolvedEntity:
    """An entity resolved from a conversational reference."""

    surface_form: str  # The original mention (e.g., "it", "that city")
    resolved_form: str  # The resolved entity (e.g., "Paris")
    turn_index: int  # When it was resolved
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)


@dataclass
class TopicFrame:
    """A topic frame tracking an active conversation subject."""

    topic: str  # The main topic (e.g., "capital of France")
    keywords: Set[str]  # Key terms for retrieval boosting
    first_mentioned: int  # Turn when first mentioned
    last_active: int  # Turn when last referenced
    relevance: float = 1.0  # Decays over time
    created_at: float = field(default_factory=time.time)


@dataclass
class PreservedFact:
    """A high-confidence fact preserved from a previous exchange."""

    content: str
    source_turn: int  # Which turn surfaced this fact
    memory_id: Optional[str] = None  # Link back to the memory store
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)


class ContextPreserver:
    """Preserves conversational context between retrieval exchanges.

    Tracks resolved entities, active topics, and key facts across turns to
    prevent information loss that would otherwise degrade multi-turn
    conversation quality.

    Args:
        max_entities: Maximum resolved entities to track.
        max_topics: Maximum active topic frames.
        max_facts: Maximum preserved facts.
        topic_decay: Relevance decay factor per turn for inactive topics.
        entity_ttl_turns: How many turns a resolved entity remains active.
    """

    def __init__(
        self,
        *,
        max_entities: int = 20,
        max_topics: int = 8,
        max_facts: int = 15,
        topic_decay: float = 0.85,
        entity_ttl_turns: int = 10,
    ) -> None:
        self.max_entities = max_entities
        self.max_topics = max_topics
        self.max_facts = max_facts
        self.topic_decay = topic_decay
        self.entity_ttl_turns = entity_ttl_turns

        self._entities: Deque[ResolvedEntity] = deque(maxlen=max_entities)
        self._topics: List[TopicFrame] = []
        self._facts: Deque[PreservedFact] = deque(maxlen=max_facts)
        self._turn_counter: int = 0

    @property
    def turn_count(self) -> int:
        """Current turn counter."""
        return self._turn_counter

    def advance_turn(self) -> None:
        """Signal a new conversational turn, decaying topic relevance."""
        self._turn_counter += 1
        # Decay inactive topics
        for topic in self._topics:
            if topic.last_active < self._turn_counter:
                topic.relevance *= self.topic_decay
        # Prune very low relevance topics
        self._topics = [
            t for t in self._topics if t.relevance > 0.1
        ]

    def resolve_entity(
        self,
        surface_form: str,
        resolved_form: str,
        confidence: float = 1.0,
    ) -> ResolvedEntity:
        """Record a resolved entity reference.

        E.g., resolve_entity("it", "the Eiffel Tower") records that
        "it" in this conversation refers to "the Eiffel Tower".
        """
        entity = ResolvedEntity(
            surface_form=surface_form.lower(),
            resolved_form=resolved_form,
            turn_index=self._turn_counter,
            confidence=confidence,
        )
        self._entities.append(entity)
        return entity

    def add_topic(self, topic: str, keywords: Optional[Set[str]] = None) -> TopicFrame:
        """Register an active conversation topic.

        Topics boost retrieval for subsequent queries even when the query
        itself doesn't mention the topic explicitly.
        """
        if keywords is None:
            keywords = set(tokenize(topic))

        # Check if this topic already exists (update if so)
        for existing in self._topics:
            overlap = existing.keywords & keywords
            if len(overlap) >= len(keywords) * 0.6:
                existing.last_active = self._turn_counter
                existing.relevance = 1.0
                existing.keywords |= keywords
                return existing

        frame = TopicFrame(
            topic=topic,
            keywords=keywords,
            first_mentioned=self._turn_counter,
            last_active=self._turn_counter,
        )
        self._topics.append(frame)

        # Enforce max topics by removing lowest relevance
        if len(self._topics) > self.max_topics:
            self._topics.sort(key=lambda t: t.relevance, reverse=True)
            self._topics = self._topics[:self.max_topics]

        return frame

    def preserve_fact(
        self,
        content: str,
        memory_id: Optional[str] = None,
        confidence: float = 1.0,
    ) -> PreservedFact:
        """Pin a high-confidence fact for re-injection in future contexts."""
        fact = PreservedFact(
            content=content,
            source_turn=self._turn_counter,
            memory_id=memory_id,
            confidence=confidence,
        )
        self._facts.append(fact)
        return fact

    def expand_query(self, query: str) -> str:
        """Expand a query with resolved entities and active topic context.

        Replaces pronoun-like references with resolved forms and appends
        active topic keywords to improve retrieval breadth.

        Args:
            query: The raw user query.

        Returns:
            Expanded query with additional context for retrieval.
        """
        expanded = query

        # Resolve entity references (pronoun substitution)
        expanded = self._substitute_references(expanded)

        # Append active topic keywords for retrieval boosting
        topic_boost = self._get_topic_boost()
        if topic_boost:
            expanded = f"{expanded} {topic_boost}"

        return expanded

    def get_preserved_context(self, max_tokens: int = 200) -> str:
        """Build a preserved context string from pinned facts.

        Returns facts that should be re-injected into the prompt to prevent
        information loss between turns.
        """
        if not self._facts:
            return ""

        from ..text import estimate_tokens

        lines = []
        tokens_used = 0
        # Most recent facts first
        for fact in reversed(self._facts):
            fact_tokens = estimate_tokens(fact.content)
            if tokens_used + fact_tokens > max_tokens:
                break
            lines.append(fact.content)
            tokens_used += fact_tokens

        return "\n".join(reversed(lines))

    def get_active_topics(self) -> List[str]:
        """Return currently active topic strings."""
        return [t.topic for t in self._topics if t.relevance > 0.3]

    def get_topic_keywords(self) -> Set[str]:
        """Return all keywords from active topics for retrieval boosting."""
        keywords: Set[str] = set()
        for topic in self._topics:
            if topic.relevance > 0.3:
                keywords |= topic.keywords
        return keywords

    def get_resolved_entities(self) -> Dict[str, str]:
        """Return active entity resolutions as {surface_form: resolved_form}."""
        cutoff = self._turn_counter - self.entity_ttl_turns
        resolutions: Dict[str, str] = {}
        for entity in self._entities:
            if entity.turn_index >= cutoff:
                resolutions[entity.surface_form] = entity.resolved_form
        return resolutions

    def clear(self) -> None:
        """Reset all preserved context."""
        self._entities.clear()
        self._topics.clear()
        self._facts.clear()
        self._turn_counter = 0

    def stats(self) -> Dict[str, object]:
        """Return preservation layer statistics."""
        return {
            "turn_count": self._turn_counter,
            "active_entities": len([
                e for e in self._entities
                if e.turn_index >= self._turn_counter - self.entity_ttl_turns
            ]),
            "active_topics": len([t for t in self._topics if t.relevance > 0.3]),
            "preserved_facts": len(self._facts),
        }

    def _substitute_references(self, query: str) -> str:
        """Replace pronoun/reference tokens with resolved entity forms."""
        cutoff = self._turn_counter - self.entity_ttl_turns
        active_entities = {
            e.surface_form: e.resolved_form
            for e in self._entities
            if e.turn_index >= cutoff
        }

        if not active_entities:
            return query

        # Simple pronoun patterns that likely reference recent entities
        pronouns = {"it", "this", "that", "they", "them", "its", "their"}
        words = query.split()
        result_words = []
        for word in words:
            clean = word.lower().strip(".,!?;:")
            if clean in active_entities:
                result_words.append(active_entities[clean])
            elif clean in pronouns and active_entities:
                # Use the most recent entity resolution
                most_recent = max(
                    (e for e in self._entities if e.turn_index >= cutoff),
                    key=lambda e: e.turn_index,
                )
                result_words.append(most_recent.resolved_form)
            else:
                result_words.append(word)
        return " ".join(result_words)

    def _get_topic_boost(self) -> str:
        """Build a keyword boost string from active topics."""
        active = [t for t in self._topics if t.relevance > 0.5]
        if not active:
            return ""
        # Take top keywords from most relevant active topics
        boost_words: List[str] = []
        for topic in sorted(active, key=lambda t: t.relevance, reverse=True)[:3]:
            boost_words.extend(list(topic.keywords)[:3])
        return " ".join(boost_words[:6])
