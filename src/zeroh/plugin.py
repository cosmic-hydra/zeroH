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

Beyond single-shot completion the plug-in also offers multi-turn :meth:`chat`,
which threads a bounded short-term :class:`~zeroh.memory.ConversationMemory`
through retrieval so follow-up questions stay grounded and in-context.

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

from typing import Callable, List, Optional

from .grounding import Verifier
from .hallucination import HallucinationDetector
from .llm import LLM
from .memory import ConversationMemory, MemoryStore, chunk_text
from .models import Answer, Citation, Memory, RetrievalResult
from .retrieval import MemoryFilter, Retriever
from .text import estimate_tokens

ABSTAIN_MESSAGE = (
    "I don't have enough grounded information in my memory to answer that "
    "reliably."
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant. Answer ONLY using the facts in the provided "
    "context. If the context does not contain the answer, reply exactly with "
    "'I don't know.' Do not use outside knowledge. Do not speculate. Prefer "
    "short, factual sentences so each statement can be verified independently."
)

# Observability callback: receives an event name and a payload dict.
EventHook = Callable[[str, dict], None]


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
        abstain_message: Text returned when the plug-in declines to answer.
        keyword_weight: Retrieval blend of keyword overlap vs. TF-IDF cosine.
        confidence_weight: How strongly memory confidence influences ranking
            (see :class:`Retriever`). ``0`` (default) ignores it.
        recency_weight: How strongly memory freshness influences ranking. Needs
            ``recency_half_life`` to take effect.
        recency_half_life: Age (seconds) at which a memory's recency factor
            halves. ``None`` (default) disables recency weighting.
        conversation: Short-term memory for :meth:`chat`. Pass an int for the
            number of turns to keep, a :class:`ConversationMemory`, or ``None``
            to create a default 8-turn buffer.
        on_event: Optional observability hook called at each pipeline stage with
            ``(event_name, payload)``. Exceptions raised by the hook are ignored.
        max_context_tokens: Token budget for context injected into the prompt.
            When retrieval results exceed this budget, lower-scoring memories
            are dropped. ``None`` (default) means no budget enforcement.
        adaptive_top_k: When ``True`` (default ``False``), automatically expands
            ``top_k`` for broad queries (many content words) and shrinks it for
            narrow queries, improving both recall and token efficiency.
        semantic_dedupe: When ``True`` (default ``False``), ingestion and
            ``remember()`` reject near-duplicate memories automatically,
            preventing context dilution from paraphrased facts.
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
        abstain_message: str = ABSTAIN_MESSAGE,
        keyword_weight: float = 0.25,
        confidence_weight: float = 0.0,
        recency_weight: float = 0.0,
        recency_half_life: Optional[float] = None,
        conversation=None,
        on_event: Optional[EventHook] = None,
        max_context_tokens: Optional[int] = None,
        adaptive_top_k: bool = False,
        semantic_dedupe: bool = False,
    ) -> None:
        self.llm = llm
        self.store = store if store is not None else MemoryStore(":memory:")
        self.retriever = Retriever(
            self.store,
            keyword_weight=keyword_weight,
            confidence_weight=confidence_weight,
            recency_weight=recency_weight,
            recency_half_life=recency_half_life,
        )
        self.verifier = Verifier(self.retriever, support_threshold=support_threshold)
        self.detector = HallucinationDetector(self.verifier)
        self.top_k = top_k
        self.min_context_score = min_context_score
        self.max_retries = max_retries
        self.system_prompt = system_prompt
        self.abstain_message = abstain_message
        self.on_event = on_event
        self.max_context_tokens = max_context_tokens
        self.adaptive_top_k = adaptive_top_k
        self.semantic_dedupe = semantic_dedupe
        if isinstance(conversation, ConversationMemory):
            self.conversation = conversation
        elif isinstance(conversation, int):
            self.conversation = ConversationMemory(max_turns=conversation)
        else:
            self.conversation = ConversationMemory()

    # -- memory management ----------------------------------------------------
    def remember(
        self,
        content: str,
        source: str = "user",
        confidence: float = 1.0,
        *,
        dedupe: bool = False,
        **metadata,
    ) -> Memory:
        """Durably store a single fact and refresh the retrieval index."""
        mem = self.store.add_text(
            content, source=source, confidence=confidence, dedupe=dedupe,
            semantic_dedupe=self.semantic_dedupe, **metadata
        )
        self.retriever.reindex()
        self._emit("remember", memory_id=mem.id, content=content, source=source)
        return mem

    def remember_many(self, contents: List[str], source: str = "user") -> List[Memory]:
        """Store several facts at once, re-indexing only once."""
        mems = [self.store.add_text(c, source=source) for c in contents]
        self.retriever.reindex()
        self._emit("remember_many", count=len(mems), source=source)
        return mems

    def ingest(
        self,
        document: str,
        source: str = "document",
        *,
        max_chars: int = 400,
        overlap_sentences: int = 1,
        dedupe: bool = False,
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
            self.store.add_text(
                chunk, source=source, dedupe=dedupe,
                semantic_dedupe=self.semantic_dedupe, **metadata
            )
            for chunk in chunks
        ]
        self.retriever.reindex()
        self._emit("ingest", source=source, chunks=len(mems))
        return mems

    def correct(self, old_id: str, new_content: str, source: str = "user") -> Memory:
        """Supersede an outdated memory; the old value is retained for audit."""
        new_mem = Memory(content=new_content, source=source)
        self.store.supersede(old_id, new_mem)
        self.retriever.reindex()
        self._emit("correct", old_id=old_id, new_id=new_mem.id)
        return new_mem

    def forget(self, memory_id: str) -> None:
        """Soft-delete a memory (kept for audit, excluded from retrieval)."""
        self.store.deactivate(memory_id)
        self.retriever.reindex()
        self._emit("forget", memory_id=memory_id)

    def recall(
        self,
        query: str,
        top_k: Optional[int] = None,
        *,
        source: Optional[str] = None,
        where: Optional[MemoryFilter] = None,
    ) -> List[RetrievalResult]:
        """Retrieve the most relevant stored memories for ``query``."""
        k = self.top_k if top_k is None else top_k
        return self.retriever.search(query, top_k=k, source=source, where=where)

    # -- introspection & portability -----------------------------------------
    def stats(self) -> dict:
        """Return memory statistics (active/inactive counts, per-source totals)."""
        return self.store.stats()

    def export(self, include_inactive: bool = False) -> str:
        """Export memories as JSONL for backup/transfer."""
        return self.store.export_jsonl(include_inactive=include_inactive)

    def load(self, data: str) -> int:
        """Import memories from JSONL (see :meth:`export`) and reindex."""
        count = self.store.import_jsonl(data)
        self.retriever.reindex()
        self._emit("load", count=count)
        return count

    def reset_conversation(self) -> None:
        """Clear the short-term conversation context used by :meth:`chat`."""
        self.conversation.clear()

    def consolidate(
        self,
        *,
        max_age: Optional[float] = None,
        min_confidence: float = 0.3,
    ) -> int:
        """Deactivate stale, low-confidence memories to keep the store lean.

        This is a maintenance operation that reduces noise in retrieval by
        removing memories that are both old AND low-confidence. It prevents
        memory bloat from degrading retrieval precision over time.

        Args:
            max_age: Maximum age in seconds; memories older than this AND below
                ``min_confidence`` are deactivated. ``None`` uses 30 days.
            min_confidence: Confidence threshold; only memories below this value
                are considered for deactivation (regardless of age).

        Returns:
            Number of memories deactivated.
        """
        import time as _time

        if max_age is None:
            max_age = 30 * 24 * 3600  # 30 days
        cutoff = _time.time() - max_age
        count = 0
        for mem in self.store.all():
            if mem.confidence < min_confidence and mem.created_at < cutoff:
                self.store.deactivate(mem.id)
                count += 1
        if count:
            self.retriever.reindex()
            self._emit("consolidate", deactivated=count)
        return count

    # -- prompt augmentation --------------------------------------------------
    def build_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        *,
        source: Optional[str] = None,
        where: Optional[MemoryFilter] = None,
    ) -> List[RetrievalResult]:
        """Return the memories (above ``min_context_score``) used as context.

        When ``adaptive_top_k`` is enabled, the effective top_k scales with
        query complexity (more content words → more memories retrieved). Token
        budgeting (``max_context_tokens``) then trims the bottom of the list to
        stay within the LLM's context window budget.
        """
        k = self._effective_top_k(query) if top_k is None else top_k
        results = self.recall(query, top_k=k, source=source, where=where)
        context = [r for r in results if r.score >= self.min_context_score]

        # Token budgeting: drop lowest-scoring memories that exceed the budget.
        if self.max_context_tokens and context:
            context = self._apply_token_budget(context)

        self._emit("retrieve", query=query, context_size=len(context))
        return context

    def _effective_top_k(self, query: str) -> int:
        """Compute adaptive top_k based on query specificity.

        Narrow queries (few content words) need fewer memories; broad queries
        benefit from more context. This saves tokens on simple lookups while
        improving recall on complex questions.
        """
        if not self.adaptive_top_k:
            return self.top_k
        from .text import tokenize
        content_words = len(tokenize(query))
        if content_words <= 2:
            # Very specific query — fewer memories needed
            return max(2, self.top_k - 2)
        elif content_words >= 6:
            # Broad/complex query — retrieve more for coverage
            return min(self.top_k + 3, 15)
        return self.top_k

    def _apply_token_budget(
        self, context: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Trim context to fit within ``max_context_tokens``.

        Keeps the highest-scoring memories and drops from the bottom of the
        ranked list. This is the primary token optimization: it prevents large
        memory stores from blowing up the prompt size.
        """
        budget = self.max_context_tokens
        total = 0
        kept: List[RetrievalResult] = []
        for res in context:
            tokens = estimate_tokens(res.memory.content)
            if total + tokens > budget and kept:
                break
            total += tokens
            kept.append(res)
        return kept

    def build_prompt(
        self,
        query: str,
        context: List[RetrievalResult],
        *,
        history: Optional[str] = None,
    ) -> str:
        """Render the retrieval-augmented prompt sent to the LLM.

        The prompt format is optimized for grounding: it numbers each context
        item (enabling citation), includes relevance scores to signal confidence
        to the model, and ends with explicit constraints that minimize
        hallucination by encouraging the model to stick to provided facts.
        """
        lines: List[str] = []
        if history:
            lines.append("Recent conversation:")
            lines.append(history)
            lines.append("")
        lines.append("Context (relevance score in brackets):")
        for i, res in enumerate(context, 1):
            score_pct = int(res.score * 100)
            lines.append(f"[{i}] ({score_pct}%) {res.memory.content}")
        lines.append("")
        lines.append(f"Question: {query}")
        lines.append(
            "Answer using only the context above. Cite context numbers in "
            "brackets like [1]. If no context answers the question, say "
            "'I don't know.' Do not add information beyond what is stated above."
        )
        return "\n".join(lines)

    # -- generation + verification -------------------------------------------
    def complete(
        self,
        query: str,
        top_k: Optional[int] = None,
        *,
        source: Optional[str] = None,
        where: Optional[MemoryFilter] = None,
    ) -> Answer:
        """Run the full grounded pipeline and return a verified :class:`Answer`.

        Requires an LLM. For LLM-free extractive answers, use
        :meth:`answer_from_memory`.
        """
        self._require_llm()
        context = self.build_context(query, top_k=top_k, source=source, where=where)
        if not context:
            return self._abstain()
        return self._generate(query, context)

    def chat(
        self,
        message: str,
        top_k: Optional[int] = None,
        *,
        source: Optional[str] = None,
        where: Optional[MemoryFilter] = None,
    ) -> Answer:
        """Multi-turn grounded completion with short-term conversation memory.

        The current ``message`` plus recent user turns are used to retrieve
        context, so terse follow-ups ("and its population?") stay grounded. Every
        answer is still verified claim-by-claim; the plug-in abstains rather than
        fabricate. The exchange is recorded in :attr:`conversation`.
        """
        self._require_llm()
        # Capture the transcript of *prior* turns before recording this message,
        # so the prompt's conversation history doesn't duplicate the question.
        history = self.conversation.transcript(n=4) or None
        self.conversation.add_user(message)
        # Blend recent user turns into the retrieval cue for follow-up continuity,
        # but keep the actual question (and the prompt) focused on this message.
        cue = self.conversation.query_context(n=3) or message
        context = self.build_context(cue, top_k=top_k, source=source, where=where)
        if not context:
            answer = self._abstain()
        else:
            answer = self._generate(message, context, history=history)
        self.conversation.add_assistant(answer.text)
        return answer

    def verify(self, candidate: str) -> Answer:
        """Verify an externally-produced answer against memory (guardrail mode).

        Use this to fact-check text from *any* source — including an LLM you call
        yourself — without zeroH doing the generation.
        """
        report = self.detector.analyze(candidate)
        return self._answer_from_report(report, raw_draft=candidate)

    def answer_from_memory(
        self,
        query: str,
        top_k: Optional[int] = None,
        *,
        source: Optional[str] = None,
        where: Optional[MemoryFilter] = None,
    ) -> Answer:
        """Compose an extractive answer purely from memory (no LLM needed)."""
        context = self.build_context(query, top_k=top_k, source=source, where=where)
        if not context:
            return self._abstain()
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
        answer = Answer(
            text=text,
            grounded=True,
            confidence=round(context[0].score, 4),
            citations=citations,
        )
        self._emit("answer", grounded=True, abstained=False, source="memory")
        return answer

    # -- internals ------------------------------------------------------------
    def _generate(
        self,
        query: str,
        context: List[RetrievalResult],
        *,
        history: Optional[str] = None,
    ) -> Answer:
        """Draft with the LLM, verify, re-ask on unsupported claims, finalize."""
        feedback = ""
        last_report = None
        draft = ""
        attempt = 0
        for attempt in range(self.max_retries + 1):
            prompt = self.build_prompt(query, context, history=history)
            if feedback:
                prompt += "\n\n" + feedback
            draft = self.llm.complete(prompt, system=self.system_prompt)
            last_report = self.detector.analyze(draft)
            self._emit(
                "draft",
                attempt=attempt,
                draft=draft,
                unsupported=len(last_report.unsupported_claims),
            )
            if not last_report.unsupported_claims:
                break
            # Re-ask with targeted feedback about the unsupported sentences.
            bad = "; ".join(c.text for c in last_report.unsupported_claims)
            feedback = (
                "Your previous answer included statements not supported by the "
                f"context: {bad}. Remove anything not grounded in the context."
            )

        # `attempt` is the index of the final LLM call == number of re-asks issued.
        return self._answer_from_report(last_report, retries=attempt, raw_draft=draft)

    def _abstain(self) -> Answer:
        self._emit("abstain")
        return Answer(
            text=self.abstain_message, grounded=False, confidence=0.0, abstained=True
        )

    def _require_llm(self) -> None:
        if self.llm is None:
            raise ValueError(
                "No LLM configured. Pass an `llm` to ZeroHPlugin, or use "
                "answer_from_memory()/verify() for LLM-free operation."
            )

    def _answer_from_report(
        self, report, *, retries: int = 0, raw_draft: Optional[str] = None
    ) -> Answer:
        supported = report.supported_claims
        confidence = round(1.0 - report.risk, 4)
        if not supported:
            self._emit("abstain", reason="no_supported_claims")
            return Answer(
                text=self.abstain_message,
                grounded=False,
                confidence=confidence,
                claims=report.claims,
                abstained=True,
                retries=retries,
                raw_draft=raw_draft,
            )
        citations = [c for claim in supported for c in claim.citations]
        text = " ".join(c.text for c in supported)
        self._emit("answer", grounded=True, abstained=False, confidence=confidence)
        return Answer(
            text=text,
            grounded=True,
            confidence=confidence,
            claims=report.claims,
            citations=citations,
            retries=retries,
            raw_draft=raw_draft,
        )

    def _emit(self, event: str, **payload) -> None:
        """Invoke the observability hook, never letting it break the pipeline."""
        if self.on_event is None:
            return
        try:
            self.on_event(event, payload)
        except Exception:  # noqa: BLE001 - observability must not crash callers
            pass
