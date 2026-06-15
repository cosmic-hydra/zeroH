"""Query expansion for improved retrieval recall.

Expands user queries with synonyms, related terms, and inferred context
to improve retrieval recall without sacrificing precision. Uses lightweight
heuristics (no external APIs or ML models required).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from ..text import tokenize, STOPWORDS


# Lightweight synonym/related-term mappings for common query patterns.
# These capture common paraphrases that would otherwise miss during retrieval.
_SYNONYM_MAP: Dict[str, List[str]] = {
    "capital": ["capital city", "administrative center"],
    "country": ["nation", "state"],
    "city": ["town", "urban"],
    "population": ["inhabitants", "people", "residents"],
    "president": ["leader", "head of state"],
    "language": ["tongue", "dialect"],
    "largest": ["biggest", "most populous"],
    "smallest": ["tiniest", "least"],
    "located": ["situated", "found"],
    "created": ["made", "built", "founded"],
    "invented": ["created", "designed", "developed"],
    "died": ["passed away", "deceased"],
    "born": ["birthplace", "native"],
    "cost": ["price", "expense", "value"],
    "fast": ["quick", "rapid", "speedy"],
    "old": ["ancient", "historic", "aged"],
    "new": ["recent", "modern", "latest"],
    "big": ["large", "huge", "massive"],
    "small": ["tiny", "little", "compact"],
}

# Question-word to intent mapping for targeted expansion
_QUESTION_INTENTS: Dict[str, List[str]] = {
    "what": [],
    "who": ["person", "name"],
    "where": ["location", "place", "country", "city"],
    "when": ["date", "year", "time"],
    "how many": ["number", "count", "total"],
    "how much": ["amount", "cost", "price"],
    "how": ["method", "way", "process"],
    "why": ["reason", "cause", "because"],
}


class QueryExpander:
    """Expand queries with synonyms and related terms for better recall.

    Uses lightweight heuristics to generate expanded query terms that
    capture paraphrases, related concepts, and implicit intents.

    Args:
        max_expansions: Maximum number of expansion terms to add.
        enable_synonyms: Use synonym mapping for expansion.
        enable_intent: Use question-type intent expansion.
        custom_synonyms: Additional synonym mappings to merge.
    """

    def __init__(
        self,
        *,
        max_expansions: int = 5,
        enable_synonyms: bool = True,
        enable_intent: bool = True,
        custom_synonyms: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.max_expansions = max_expansions
        self.enable_synonyms = enable_synonyms
        self.enable_intent = enable_intent
        self._synonyms = dict(_SYNONYM_MAP)
        if custom_synonyms:
            self._synonyms.update(custom_synonyms)

    def expand(self, query: str) -> str:
        """Expand a query with additional related terms.

        The original query is preserved; expansions are appended to broaden
        retrieval without overriding user intent.

        Returns:
            Expanded query string.
        """
        original_tokens = set(tokenize(query, remove_stopwords=True))
        expansions: List[str] = []

        # Synonym expansion
        if self.enable_synonyms:
            expansions.extend(self._synonym_expand(original_tokens))

        # Intent-based expansion
        if self.enable_intent:
            expansions.extend(self._intent_expand(query))

        # Morphological expansion (simple stemming variants)
        expansions.extend(self._morphological_expand(original_tokens))

        # Deduplicate and limit
        seen = set(original_tokens)
        unique_expansions: List[str] = []
        for exp in expansions:
            exp_lower = exp.lower()
            if exp_lower not in seen and exp_lower not in STOPWORDS:
                seen.add(exp_lower)
                unique_expansions.append(exp)

        unique_expansions = unique_expansions[:self.max_expansions]

        if unique_expansions:
            return f"{query} {' '.join(unique_expansions)}"
        return query

    def get_expansions(self, query: str) -> List[str]:
        """Return just the expansion terms without modifying the query.

        Useful for debugging or selective application of expansions.
        """
        original_tokens = set(tokenize(query, remove_stopwords=True))
        expansions: List[str] = []

        if self.enable_synonyms:
            expansions.extend(self._synonym_expand(original_tokens))
        if self.enable_intent:
            expansions.extend(self._intent_expand(query))
        expansions.extend(self._morphological_expand(original_tokens))

        # Deduplicate
        seen = set(original_tokens)
        unique: List[str] = []
        for exp in expansions:
            exp_lower = exp.lower()
            if exp_lower not in seen and exp_lower not in STOPWORDS:
                seen.add(exp_lower)
                unique.append(exp)

        return unique[:self.max_expansions]

    def add_synonyms(self, word: str, synonyms: List[str]) -> None:
        """Add custom synonym mappings for domain-specific terms."""
        existing = self._synonyms.get(word, [])
        self._synonyms[word] = list(set(existing + synonyms))

    def _synonym_expand(self, tokens: Set[str]) -> List[str]:
        """Find synonyms for query tokens."""
        expansions: List[str] = []
        for token in tokens:
            if token in self._synonyms:
                for synonym in self._synonyms[token]:
                    if synonym.lower() not in tokens:
                        expansions.append(synonym)
        return expansions

    def _intent_expand(self, query: str) -> List[str]:
        """Expand based on question type/intent."""
        query_lower = query.lower().strip()
        expansions: List[str] = []

        for q_word, intent_terms in _QUESTION_INTENTS.items():
            if query_lower.startswith(q_word):
                expansions.extend(intent_terms)
                break

        return expansions

    def _morphological_expand(self, tokens: Set[str]) -> List[str]:
        """Simple morphological variations (plural/singular, verb forms)."""
        expansions: List[str] = []
        for token in tokens:
            if len(token) < 4:
                continue
            # Simple plural/singular
            if token.endswith("s") and len(token) > 4:
                expansions.append(token[:-1])  # cities -> city (imperfect)
            elif token.endswith("ies") and len(token) > 5:
                expansions.append(token[:-3] + "y")  # countries -> country
            elif not token.endswith("s"):
                expansions.append(token + "s")  # country -> countrys (imperfect but useful for matching)
            # -ing/-ed variants
            if token.endswith("ing") and len(token) > 5:
                expansions.append(token[:-3])  # building -> build
                expansions.append(token[:-3] + "e")  # making -> make
            elif token.endswith("ed") and len(token) > 4:
                expansions.append(token[:-2])  # created -> creat (partial, but helps)
                expansions.append(token[:-1])  # created -> create (if -e ending)
        return expansions
