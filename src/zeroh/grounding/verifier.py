"""Grounding and claim verification.

This module is the heart of zeroH's hallucination defense. Given a candidate
answer, the :class:`Verifier` decomposes it into atomic claims and checks each
one against retrieved memories. Unsupported claims are flagged, so the agent can
abstain or strip them instead of confidently stating something it cannot back up.
"""
from __future__ import annotations

from typing import List

from ..models import Citation, Claim
from ..retrieval import Retriever
from ..text import split_sentences, tokenize


class Verifier:
    """Verifies that statements are supported by stored memories.

    Verification is deliberately **strict and coverage-based**: a claim is only
    considered grounded when a single retrieved memory covers (almost) all of the
    claim's content words. This biases the system toward *abstention over
    fabrication* — e.g. the claim "the capital of France is Berlin" is rejected
    against a memory saying "Paris" because the salient token ``berlin`` is not
    covered, even though most other words overlap.

    Args:
        retriever: Retriever over the agent's memory store.
        support_threshold: Minimum coverage in [0, 1] (fraction of the claim's
            content words found in a single supporting memory) for the claim to
            be considered grounded.
        top_k: Number of candidate memories to consider per claim.
    """

    def __init__(
        self,
        retriever: Retriever,
        support_threshold: float = 0.7,
        top_k: int = 3,
    ) -> None:
        self.retriever = retriever
        self.support_threshold = support_threshold
        self.top_k = top_k

    def verify_statement(self, statement: str) -> Claim:
        """Verify a single statement against memory, returning a scored claim."""
        results = self.retriever.search(statement, top_k=self.top_k)
        claim = Claim(text=statement)
        stmt_tokens = set(tokenize(statement))
        if not results or not stmt_tokens:
            return claim

        best_score = 0.0
        for res in results:
            # Coverage is the fraction of the claim's content words contained in
            # THIS memory. A high-coverage memory entails the claim; a memory
            # that merely shares the topic (but omits a salient token) does not.
            coverage = _entailment_overlap(stmt_tokens, res.memory.content)
            # Require the memory to actually be relevant (retrieved with signal)
            # as well as high-coverage, so unrelated text can't accidentally
            # "cover" a short claim.
            support = coverage if res.score > 0 else 0.0
            best_score = max(best_score, support)
            claim.citations.append(
                Citation(
                    memory_id=res.memory.id,
                    content=res.memory.content,
                    source=res.memory.source,
                    score=round(support, 4),
                )
            )

        # Keep only the most relevant citations, strongest first.
        claim.citations.sort(key=lambda c: c.score, reverse=True)
        claim.support_score = round(best_score, 4)
        claim.supported = best_score >= self.support_threshold
        # Drop citations that did not meaningfully contribute.
        claim.citations = [c for c in claim.citations if c.score > 0.0][: self.top_k]
        return claim

    def verify_text(self, text: str) -> List[Claim]:
        """Split text into sentence-level claims and verify each independently."""
        return [self.verify_statement(s) for s in split_sentences(text)]


def _entailment_overlap(statement_tokens: set, memory_content: str) -> float:
    """Fraction of the statement's content words found in the memory text.

    This is a cheap proxy for "does this memory entail the statement": if the
    memory contains most of the statement's meaningful words, it likely supports
    it. Returns a value in [0, 1].
    """
    if not statement_tokens:
        return 0.0
    memory_tokens = set(tokenize(memory_content))
    return len(statement_tokens & memory_tokens) / len(statement_tokens)
