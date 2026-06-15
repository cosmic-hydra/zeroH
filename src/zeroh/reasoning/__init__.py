"""Semantic reasoning for improved context awareness and query understanding.

This module provides lightweight, stdlib-only semantic analysis that enhances
zeroH's retrieval and grounding capabilities:

* :class:`EntityExtractor` — Extracts key entities and concepts from text
  using pattern-based recognition (proper nouns, capitalized phrases, numbers).
* :class:`RelationExtractor` — Identifies simple subject-predicate-object
  relations for knowledge graph-style reasoning.
* :class:`QueryExpander` — Expands queries with synonyms, related terms, and
  inferred context to improve retrieval recall.
"""
from .entities import EntityExtractor, Entity
from .relations import RelationExtractor, Relation
from .query_expansion import QueryExpander

__all__ = [
    "EntityExtractor",
    "Entity",
    "RelationExtractor",
    "Relation",
    "QueryExpander",
]
