"""Lightweight relation extraction using pattern-based heuristics.

Identifies simple subject-predicate-object triples from text, enabling
basic knowledge graph-style reasoning over stored memories.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..text import tokenize, STOPWORDS


@dataclass
class Relation:
    """A subject-predicate-object triple extracted from text."""

    subject: str
    predicate: str
    object: str
    confidence: float = 0.8
    source_text: str = ""

    @property
    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def matches_query(self, query: str) -> bool:
        """Check if this relation is relevant to a query."""
        query_tokens = set(tokenize(query, remove_stopwords=True))
        relation_tokens = set(tokenize(
            f"{self.subject} {self.predicate} {self.object}",
            remove_stopwords=True,
        ))
        overlap = query_tokens & relation_tokens
        return len(overlap) >= 1


# Patterns for common relation types
_IS_A_RE = re.compile(
    r"(.+?)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(.+?)(?:\.|,|$)",
    re.IGNORECASE,
)
_HAS_RE = re.compile(
    r"(.+?)\s+(?:has|have|had)\s+(.+?)(?:\.|,|$)",
    re.IGNORECASE,
)
_VERB_RE = re.compile(
    r"(.+?)\s+(is|are|was|were|has|have|had|can|will|does|did)\s+(.+?)(?:\.|,|$)",
    re.IGNORECASE,
)
_CAPITAL_OF_RE = re.compile(
    r"(.+?)\s+is\s+the\s+capital\s+of\s+(.+?)(?:\.|,|$)",
    re.IGNORECASE,
)
_LOCATED_IN_RE = re.compile(
    r"(.+?)\s+(?:is|are)\s+(?:located\s+)?in\s+(.+?)(?:\.|,|$)",
    re.IGNORECASE,
)


class RelationExtractor:
    """Extract subject-predicate-object relations from text.

    Uses pattern-based heuristics to identify common relation types:
    - "X is Y" (classification/identity)
    - "X has Y" (possession/attribute)
    - "X is the capital of Y" (specific relations)
    - "X is located in Y" (spatial relations)

    These relations enable query-time reasoning: if the user asks about X,
    we can also retrieve facts about related entities Y.

    Args:
        min_confidence: Minimum confidence threshold for extracted relations.
    """

    def __init__(self, *, min_confidence: float = 0.5) -> None:
        self.min_confidence = min_confidence

    def extract(self, text: str) -> List[Relation]:
        """Extract all relations from text.

        Returns relations sorted by confidence (highest first).
        """
        relations: List[Relation] = []

        # Split into sentences for processing
        sentences = re.split(r'[.!?]\s+', text.strip())

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            relations.extend(self._extract_from_sentence(sent))

        # Deduplicate and filter
        relations = self._deduplicate(relations)
        relations = [r for r in relations if r.confidence >= self.min_confidence]
        relations.sort(key=lambda r: r.confidence, reverse=True)

        return relations

    def find_related(self, entity: str, relations: List[Relation]) -> List[Relation]:
        """Find all relations involving a given entity.

        Useful for expanding context: given an entity, find all known
        facts about it from extracted relations.
        """
        entity_lower = entity.lower()
        results = []
        for rel in relations:
            if (entity_lower in rel.subject.lower() or
                    entity_lower in rel.object.lower()):
                results.append(rel)
        return results

    def _extract_from_sentence(self, sentence: str) -> List[Relation]:
        """Extract relations from a single sentence."""
        relations: List[Relation] = []

        # High-confidence specific patterns
        for match in _CAPITAL_OF_RE.finditer(sentence):
            relations.append(Relation(
                subject=match.group(1).strip(),
                predicate="is the capital of",
                object=match.group(2).strip(),
                confidence=0.95,
                source_text=sentence,
            ))

        for match in _LOCATED_IN_RE.finditer(sentence):
            relations.append(Relation(
                subject=match.group(1).strip(),
                predicate="is located in",
                object=match.group(2).strip(),
                confidence=0.85,
                source_text=sentence,
            ))

        # General "X is Y" pattern (lower confidence)
        if not relations:
            for match in _IS_A_RE.finditer(sentence):
                subj = match.group(1).strip()
                obj = match.group(2).strip()
                # Filter out very short or stopword-only subjects/objects
                if (len(subj) >= 2 and len(obj) >= 2 and
                        subj.lower() not in STOPWORDS and
                        obj.lower() not in STOPWORDS):
                    relations.append(Relation(
                        subject=subj,
                        predicate="is",
                        object=obj,
                        confidence=0.7,
                        source_text=sentence,
                    ))
                    break  # Only take first match per sentence

        # "X has Y" pattern
        for match in _HAS_RE.finditer(sentence):
            subj = match.group(1).strip()
            obj = match.group(2).strip()
            if (len(subj) >= 2 and len(obj) >= 2 and
                    subj.lower() not in STOPWORDS):
                relations.append(Relation(
                    subject=subj,
                    predicate="has",
                    object=obj,
                    confidence=0.7,
                    source_text=sentence,
                ))
                break

        return relations

    def _deduplicate(self, relations: List[Relation]) -> List[Relation]:
        """Remove duplicate relations, keeping highest confidence."""
        seen: dict = {}
        for rel in relations:
            key = (rel.subject.lower(), rel.predicate.lower(), rel.object.lower())
            if key not in seen or rel.confidence > seen[key].confidence:
                seen[key] = rel
        return list(seen.values())
