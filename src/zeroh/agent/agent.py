"""The grounded agent: zeroH's high-level entry point.

:class:`GroundedAgent` composes the memory store, retriever, verifier and
hallucination detector into a single API that an application can use to:

* **remember** facts durably (no data loss),
* **recall** relevant facts for a query, and
* **answer** questions *only* from what it can ground in memory — either by
  composing an extractive answer from stored facts, or by acting as a guardrail
  that verifies/strips an externally-generated (e.g. LLM) candidate answer.

The guiding rule is *abstention over fabrication*: when memory cannot support an
answer, the agent says so instead of guessing.
"""
from __future__ import annotations

from typing import List, Optional

from ..grounding import Verifier
from ..hallucination import HallucinationDetector
from ..memory import MemoryStore
from ..models import Answer, Citation, Claim, Memory, RetrievalResult
from ..retrieval import Retriever

ABSTAIN_MESSAGE = (
    "I don't have enough grounded information in my memory to answer that "
    "reliably."
)


class GroundedAgent:
    """A memory-grounded agent that minimizes hallucination and data loss.

    Args:
        store: Optional pre-existing memory store. If omitted, an in-memory
            store is created. Pass ``MemoryStore("path.db")`` for durability.
        support_threshold: Per-claim grounding threshold (see :class:`Verifier`).
        abstain_threshold: Minimum answer confidence required to answer at all;
            below this the agent abstains.
    """

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        support_threshold: float = 0.7,
        abstain_threshold: float = 0.2,
    ) -> None:
        self.store = store if store is not None else MemoryStore(":memory:")
        self.retriever = Retriever(self.store)
        self.verifier = Verifier(self.retriever, support_threshold=support_threshold)
        self.detector = HallucinationDetector(self.verifier)
        self.abstain_threshold = abstain_threshold

    # -- memory management ----------------------------------------------------
    def remember(
        self,
        content: str,
        source: str = "user",
        confidence: float = 1.0,
        **metadata,
    ) -> Memory:
        """Durably store a new fact and refresh the retrieval index."""
        mem = self.store.add_text(
            content, source=source, confidence=confidence, **metadata
        )
        self.retriever.reindex()
        return mem

    def remember_many(self, contents: List[str], source: str = "user") -> List[Memory]:
        """Store several facts at once (re-indexing only once for efficiency)."""
        mems = [self.store.add_text(c, source=source) for c in contents]
        self.retriever.reindex()
        return mems

    def correct(self, old_id: str, new_content: str, source: str = "user") -> Memory:
        """Supersede an outdated memory with a corrected one (history retained)."""
        new_mem = Memory(content=new_content, source=source)
        self.store.supersede(old_id, new_mem)
        self.retriever.reindex()
        return new_mem

    def recall(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve the most relevant stored memories for a query."""
        return self.retriever.search(query, top_k=top_k)

    # -- answering ------------------------------------------------------------
    def answer(
        self,
        query: str,
        candidate: Optional[str] = None,
        top_k: int = 5,
    ) -> Answer:
        """Produce a grounded :class:`Answer` for ``query``.

        If ``candidate`` is provided (e.g. from an LLM), it is treated as a draft
        to be verified and, if necessary, stripped of unsupported claims. If no
        candidate is given, the agent composes an extractive answer purely from
        retrieved memories.
        """
        if candidate is not None:
            return self._verify_candidate(candidate)
        return self._extractive_answer(query, top_k=top_k)

    # -- internals ------------------------------------------------------------
    def _extractive_answer(self, query: str, top_k: int) -> Answer:
        results = self.recall(query, top_k=top_k)
        if not results or results[0].score < self.abstain_threshold:
            return Answer(
                text=ABSTAIN_MESSAGE,
                grounded=False,
                confidence=round(results[0].score, 4) if results else 0.0,
                abstained=True,
            )

        used = [r for r in results if r.score >= self.abstain_threshold]
        citations = [
            Citation(
                memory_id=r.memory.id,
                content=r.memory.content,
                source=r.memory.source,
                score=round(r.score, 4),
            )
            for r in used
        ]
        # Compose the answer from the supporting facts themselves, so every
        # sentence is, by construction, backed by a citation.
        text = " ".join(r.memory.content for r in used)
        confidence = round(results[0].score, 4)
        return Answer(
            text=text,
            grounded=True,
            confidence=confidence,
            citations=citations,
        )

    def _verify_candidate(self, candidate: str) -> Answer:
        report = self.detector.analyze(candidate)
        supported: List[Claim] = report.supported_claims
        all_citations = [c for claim in supported for c in claim.citations]

        if not supported:
            return Answer(
                text=ABSTAIN_MESSAGE,
                grounded=False,
                confidence=round(1.0 - report.risk, 4),
                claims=report.claims,
                abstained=True,
            )

        confidence = round(1.0 - report.risk, 4)
        if confidence < self.abstain_threshold:
            return Answer(
                text=ABSTAIN_MESSAGE,
                grounded=False,
                confidence=confidence,
                claims=report.claims,
                abstained=True,
            )

        text = " ".join(c.text for c in supported)
        return Answer(
            text=text,
            grounded=True,
            confidence=confidence,
            claims=report.claims,
            citations=all_citations,
        )
