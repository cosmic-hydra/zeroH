"""Lightweight entity extraction using pattern-based recognition.

Extracts named entities (proper nouns, capitalized phrases), numeric values,
and key concepts from text without requiring ML models or external dependencies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from ..text import tokenize, STOPWORDS


@dataclass
class Entity:
    """An extracted entity from text."""

    text: str
    entity_type: str  # "proper_noun", "concept", "numeric", "quoted"
    start: int = 0  # Character offset in source text
    end: int = 0
    frequency: int = 1

    @property
    def normalized(self) -> str:
        """Lowercase normalized form for deduplication."""
        return self.text.lower().strip()


# Patterns for entity recognition
_PROPER_NOUN_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
)
_NUMERIC_RE = re.compile(
    r"\b(\d+(?:\.\d+)?(?:\s*(?:km|kg|m|cm|mm|miles|feet|percent|%|"
    r"million|billion|thousand|hundred|year|years|days|hours))?)\b"
)
_QUOTED_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')
_CONCEPT_RE = re.compile(
    r"\b([A-Z][A-Z]+(?:\s+[A-Z]+)*)\b"  # All-caps acronyms/abbreviations
)


class EntityExtractor:
    """Extract entities and key concepts from text.

    Uses pattern matching to identify:
    - Proper nouns (capitalized multi-word phrases)
    - Numeric values with units
    - Quoted strings
    - Acronyms and abbreviations
    - Key content words (high information density)

    Args:
        min_entity_length: Minimum character length for extracted entities.
        max_entities: Maximum number of entities to return per text.
    """

    def __init__(
        self,
        *,
        min_entity_length: int = 2,
        max_entities: int = 20,
    ) -> None:
        self.min_entity_length = min_entity_length
        self.max_entities = max_entities

    def extract(self, text: str) -> List[Entity]:
        """Extract all entities from text.

        Returns entities sorted by likely importance (proper nouns first,
        then concepts, then numerics).
        """
        entities: List[Entity] = []

        # Proper nouns (capitalized phrases)
        entities.extend(self._extract_proper_nouns(text))

        # Numeric values
        entities.extend(self._extract_numerics(text))

        # Quoted strings
        entities.extend(self._extract_quoted(text))

        # Acronyms
        entities.extend(self._extract_acronyms(text))

        # Key content words
        entities.extend(self._extract_key_concepts(text))

        # Deduplicate
        entities = self._deduplicate(entities)

        return entities[:self.max_entities]

    def extract_keywords(self, text: str) -> List[str]:
        """Extract just the key terms (simplified for retrieval boosting).

        Returns a flat list of important terms that characterize the text.
        """
        entities = self.extract(text)
        keywords = []
        for entity in entities:
            if entity.entity_type in ("proper_noun", "concept", "quoted"):
                keywords.append(entity.text)
        # Also add high-IDF content words
        content_words = [
            t for t in tokenize(text, remove_stopwords=True)
            if len(t) >= 3
        ]
        keywords.extend(content_words[:5])
        return keywords

    def _extract_proper_nouns(self, text: str) -> List[Entity]:
        """Extract capitalized multi-word phrases."""
        entities = []
        # Skip the first word of sentences (might just be sentence-initial caps)
        sentences = re.split(r'[.!?]\s+', text)
        for sent in sentences:
            words = sent.split()
            if not words:
                continue
            # Find proper nouns (skip first word of sentence)
            for match in _PROPER_NOUN_RE.finditer(sent):
                noun = match.group(1)
                # Skip if it's just the first word and single word
                if match.start() == 0 and " " not in noun:
                    continue
                if len(noun) >= self.min_entity_length:
                    entities.append(Entity(
                        text=noun,
                        entity_type="proper_noun",
                        start=match.start(),
                        end=match.end(),
                    ))
        return entities

    def _extract_numerics(self, text: str) -> List[Entity]:
        """Extract numeric values with optional units."""
        entities = []
        for match in _NUMERIC_RE.finditer(text):
            value = match.group(1).strip()
            if len(value) >= self.min_entity_length:
                entities.append(Entity(
                    text=value,
                    entity_type="numeric",
                    start=match.start(),
                    end=match.end(),
                ))
        return entities

    def _extract_quoted(self, text: str) -> List[Entity]:
        """Extract quoted strings."""
        entities = []
        for match in _QUOTED_RE.finditer(text):
            quoted = match.group(1) or match.group(2)
            if quoted and len(quoted) >= self.min_entity_length:
                entities.append(Entity(
                    text=quoted,
                    entity_type="quoted",
                    start=match.start(),
                    end=match.end(),
                ))
        return entities

    def _extract_acronyms(self, text: str) -> List[Entity]:
        """Extract all-caps abbreviations/acronyms."""
        entities = []
        for match in _CONCEPT_RE.finditer(text):
            acronym = match.group(1)
            if len(acronym) >= 2 and acronym not in ("I", "A"):
                entities.append(Entity(
                    text=acronym,
                    entity_type="concept",
                    start=match.start(),
                    end=match.end(),
                ))
        return entities

    def _extract_key_concepts(self, text: str) -> List[Entity]:
        """Extract high-information content words as concepts."""
        tokens = tokenize(text, remove_stopwords=True)
        # Count frequency of content words
        freq: Dict[str, int] = {}
        for t in tokens:
            if len(t) >= 4:  # Longer words tend to be more informative
                freq[t] = freq.get(t, 0) + 1

        entities = []
        for word, count in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]:
            entities.append(Entity(
                text=word,
                entity_type="concept",
                frequency=count,
            ))
        return entities

    def _deduplicate(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities, keeping the first occurrence."""
        seen: Set[str] = set()
        unique: List[Entity] = []
        for entity in entities:
            key = entity.normalized
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        return unique
