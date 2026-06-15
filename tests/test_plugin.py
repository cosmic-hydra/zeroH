"""End-to-end tests for the ZeroHPlugin (bring-your-own-LLM)."""
import pytest

from zeroh import ZeroHPlugin
from zeroh.llm import CallableLLM
from zeroh.plugin import ABSTAIN_MESSAGE


def _kb_plugin(llm=None):
    zh = ZeroHPlugin(llm)
    zh.remember_many(
        [
            "The capital of France is Paris.",
            "The Eiffel Tower is located in Paris.",
            "Python was created by Guido van Rossum.",
        ],
        source="kb",
    )
    return zh


def test_complete_requires_llm():
    zh = _kb_plugin()  # no LLM
    with pytest.raises(ValueError):
        zh.complete("capital of France")


def test_complete_strips_unsupported_claim():
    # A model that returns one true and one fabricated sentence.
    llm = CallableLLM(
        lambda p, s: "The capital of France is Paris. The capital of France is Berlin."
    )
    zh = _kb_plugin(llm)
    res = zh.complete("What is the capital of France?")
    assert "Paris" in res.text
    assert "Berlin" not in res.text
    assert res.grounded
    assert res.citations


def test_complete_abstains_when_no_context():
    llm = CallableLLM(lambda p, s: "Anything at all.")
    zh = _kb_plugin(llm)
    res = zh.complete("Who won the 2049 World Series?")
    assert res.abstained
    assert res.text == ABSTAIN_MESSAGE


def test_complete_retries_then_abstains_on_full_hallucination():
    # A model that always fabricates -> plugin must abstain, never pass it on.
    llm = CallableLLM(lambda p, s: "The capital of France is Berlin.")
    zh = _kb_plugin(llm)
    res = zh.complete("What is the capital of France?")
    assert res.abstained
    assert "Berlin" not in res.text


def test_verify_guardrail_without_generation():
    zh = _kb_plugin()
    assert zh.verify("The capital of France is Berlin.").abstained
    good = zh.verify("The capital of France is Paris.")
    assert good.grounded and "Paris" in good.text


def test_answer_from_memory_is_llm_free():
    zh = _kb_plugin()
    res = zh.answer_from_memory("capital of France")
    assert res.grounded
    assert "Paris" in res.text


def test_ingest_chunks_document_into_memories():
    zh = ZeroHPlugin()
    doc = " ".join(f"Fact {i} states detail number {i} clearly." for i in range(30))
    mems = zh.ingest(doc, source="doc", max_chars=80, overlap_sentences=0)
    assert len(mems) > 1
    assert len(zh.store) == len(mems)


def test_build_prompt_includes_context_and_question():
    zh = _kb_plugin()
    ctx = zh.build_context("capital of France")
    prompt = zh.build_prompt("capital of France", ctx)
    assert "Context:" in prompt
    assert "Question: capital of France" in prompt


def test_correct_supersedes_without_data_loss():
    zh = ZeroHPlugin()
    mem = zh.remember("The sky is green.")
    zh.correct(mem.id, "The sky is blue.")
    res = zh.answer_from_memory("What color is the sky?")
    assert "blue" in res.text
    history = [m.content for m in zh.store.all(include_inactive=True)]
    assert "The sky is green." in history
