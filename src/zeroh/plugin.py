"""ZeroHPlugin — the bring-your-own-LLM plug-in.

This is zeroH's headline interface. Rather than being an agent that generates
text itself, zeroH **wraps the LLM you already use** (cloud via API key, or a
local model) and surrounds every call with a durable, grounded memory layer:

1. **Retrieve** the most relevant stored memories for the query.
2. **Augment** the prompt with that context plus strict grounding instructions.
3. **Generate** using *your* LLM.
4. **Verify** the output claim-by-claim against memory; strip or re-ask on
   unsupported claims, and **abstain** rather than fabricate.

The verify → re-ask → abstain loop is what drives the residual hallucination
rate toward near-zero: any sentence the model produces that cannot be grounded
in memory is removed before it ever reaches the user.

Usage::

    from zeroh import ZeroHPlugin
    from zeroh.llm import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM(model="gpt-4o-mini", api_key="...")
    zh = ZeroHPlugin(llm)
    zh.ingest(long_document, source="handbook")
    result = zh.complete("What is our refund policy?")
    print(result.text, result.confidence, result.citations)
"""
from __future__ import annotations

from typing import List, Optional

from .grounding import Verifier
from .hallucination import HallucinationDetector
from .llm import LLM
from .memory import MemoryStore, chunk_text
from .models import Answer, Citation, Memory, RetrievalResult
from .retrieval import Retriever

ABSTAIN_MESSAGE = (
    "I don't have enough grounded information in my memory to answer that "
    "reliably."
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant. Answer ONLY using the facts in the provided "
    "context. If the context does not contain the answer, reply exactly with "
    "'I don't know.' Do not use outside knowledge. Do not speculate. Prefer "
    "short, factual sentences so each statement can be verified."
)


class ZeroHPlugin:
    """Grounded-memory plug-in that wraps any user-supplied :class:`LLM`.

    Args:
        llm: The user's model (see :mod:`zeroh.llm`). May be ``None`` to use the
            plug-in purely as a memory/verification layer (no generation).
        store: Optional pre-existing :class:`MemoryStore`. Defaults to an
            in-memory store; pass ``MemoryStore("path.db")`` for durability.
        top_k: Number of memories to retrieve as context per query.
        support_threshold: Per-claim grounding threshold (see :class:`Verifier`).
        min_context_score: Minimum retrieval score for a memory to be injected
            as context; also the bar below which the plug-in abstains outright.
        max_retries: Number of times to re-ask the LLM with corrective feedback
            when its draft contains unsupported claims.
        system_prompt: System instruction sent to the LLM.
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        store: Optional[MemoryStore] = None,
        *,
        top_k: int = 5,
        support_threshold: float = 0.7,
        min_context_score: float = 0.1,
        max_retries: int = 1,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm
        self.store = store if store is not None else MemoryStore(":memory:")
        self.retriever = Retriever(self.store)
        self.verifier = Verifier(self.retriever, support_threshold=support_threshold)
        self.detector = HallucinationDetector(self.verifier)
        self.top_k = top_k
        self.min_context_score = min_context_score
        self.max_retries = max_retries
        self.system_prompt = system_prompt

    # -- memory management ----------------------------------------------------
    def remember(
        self,
        content: str,
        source: str = "user",
        confidence: float = 1.0,
        **metadata,
    ) -> Memory:
        """Durably store a single fact and refresh the retrieval index."""
        mem = self.store.add_text(
            content, source=source, confidence=confidence, **metadata
        )
        self.retriever.reindex()
        return mem

    def remember_many(self, contents: List[str], source: str = "user") -> List[Memory]:
        """Store several facts at once, re-indexing only once."""
        mems = [self.store.add_text(c, source=source) for c in contents]
        self.retriever.reindex()
        return mems

    def ingest(
        self,
        document: str,
        source: str = "document",
        *,
        max_chars: int = 400,
        overlap_sentences: int = 1,
        **metadata,
    ) -> List[Memory]:
        """Chunk a long document and store each chunk as a memory.

        This is the recommended way to load substantial knowledge: sentence-aware
        overlapping chunks give the retriever precise, well-scoped context, which
        is the single biggest lever on downstream answer quality.
        """
        chunks = chunk_text(
            document, max_chars=max_chars, overlap_sentences=overlap_sentences
        )
        mems = [
            self.store.add_text(chunk, source=source, **metadata) for chunk in chunks
        ]
        self.retriever.reindex()
        return mems

    def correct(self, old_id: str, new_content: str, source: str = "user") -> Memory:
        """Supersede an outdated memory; the old value is retained for audit."""
        new_mem = Memory(content=new_content, source=source)
        self.store.supersede(old_id, new_mem)
        self.retriever.reindex()
        return new_mem

    def recall(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Retrieve the most relevant stored memories for ``query``."""
        return self.retriever.search(query, top_k=top_k or self.top_k)

    # -- prompt augmentation --------------------------------------------------
    def build_context(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Return the memories (above ``min_context_score``) used as context."""
        results = self.recall(query, top_k=top_k)
        return [r for r in results if r.score >= self.min_context_score]

    def build_prompt(self, query: str, context: List[RetrievalResult]) -> str:
        """Render the retrieval-augmented prompt sent to the LLM."""
        lines = ["Context:"]
        for i, res in enumerate(context, 1):
            lines.append(f"[{i}] {res.memory.content}")
        lines.append("")
        lines.append(f"Question: {query}")
        lines.append(
            "Answer using only the context above. If it is not answerable from "
            "the context, say 'I don't know.'"
        )
        return "\n".join(lines)

    # -- generation + verification -------------------------------------------
    def complete(self, query: str, top_k: Optional[int] = None) -> Answer:
        """Run the full grounded pipeline and return a verified :class:`Answer`.

        Requires an LLM. For LLM-free extractive answers, use
        :meth:`answer_from_memory`.
        """
        if self.llm is None:
            raise ValueError(
                "No LLM configured. Pass an `llm` to ZeroHPlugin, or use "
                "answer_from_memory()/verify() for LLM-free operation."
            )

        context = self.build_context(query, top_k=top_k)
        if not context:
            # Nothing relevant in memory -> never let the model guess.
            return Answer(
                text=ABSTAIN_MESSAGE, grounded=False, confidence=0.0, abstained=True
            )

        feedback = ""
        last_report = None
        for _ in range(self.max_retries + 1):
            prompt = self.build_prompt(query, context)
            if feedback:
                prompt += "\n\n" + feedback
            draft = self.llm.complete(prompt, system=self.system_prompt)
            last_report = self.detector.analyze(draft)
            if not last_report.unsupported_claims:
                break
            # Re-ask with targeted feedback about the unsupported sentences.
            bad = "; ".join(c.text for c in last_report.unsupported_claims)
            feedback = (
                "Your previous answer included statements not supported by the "
                f"context: {bad}. Remove anything not grounded in the context."
            )

        return self._answer_from_report(last_report)

    def verify(self, candidate: str) -> Answer:
        """Verify an externally-produced answer against memory (guardrail mode).

        Use this to fact-check text from *any* source — including an LLM you call
        yourself — without zeroH doing the generation.
        """
        report = self.detector.analyze(candidate)
        return self._answer_from_report(report)

    def answer_from_memory(self, query: str, top_k: Optional[int] = None) -> Answer:
        """Compose an extractive answer purely from memory (no LLM needed)."""
        context = self.build_context(query, top_k=top_k)
        if not context:
            return Answer(
                text=ABSTAIN_MESSAGE, grounded=False, confidence=0.0, abstained=True
            )
        citations = [
            Citation(
                memory_id=r.memory.id,
                content=r.memory.content,
                source=r.memory.source,
                score=round(r.score, 4),
            )
            for r in context
        ]
        text = " ".join(r.memory.content for r in context)
        return Answer(
            text=text,
            grounded=True,
            confidence=round(context[0].score, 4),
            citations=citations,
        )

    # -- internals ------------------------------------------------------------
    def _answer_from_report(self, report) -> Answer:
        supported = report.supported_claims
        confidence = round(1.0 - report.risk, 4)
        if not supported:
            return Answer(
                text=ABSTAIN_MESSAGE,
                grounded=False,
                confidence=confidence,
                claims=report.claims,
                abstained=True,
            )
        citations = [c for claim in supported for c in claim.citations]
        text = " ".join(c.text for c in supported)
        return Answer(
            text=text,
            grounded=True,
            confidence=confidence,
            claims=report.claims,
            citations=citations,
        )
