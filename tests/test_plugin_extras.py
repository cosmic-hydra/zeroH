"""Tests for the enhanced ZeroHPlugin: chat, observability, introspection."""
import pytest

from zeroh import ConversationMemory, ZeroHPlugin
from zeroh.llm import CallableLLM


def _kb(llm=None, **kwargs):
    zh = ZeroHPlugin(llm, **kwargs)
    zh.remember_many(
        [
            "The capital of France is Paris.",
            "The Eiffel Tower is located in Paris.",
            "The Eiffel Tower was completed in 1889.",
        ],
        source="kb",
    )
    return zh


def test_chat_requires_llm():
    zh = _kb()
    with pytest.raises(ValueError):
        zh.chat("hello")


def test_chat_records_turns_and_grounds_answer():
    llm = CallableLLM(lambda p, s: "The capital of France is Paris.")
    zh = _kb(llm)
    ans = zh.chat("What is the capital of France?")
    assert ans.grounded and "Paris" in ans.text
    # One user + one assistant turn recorded.
    assert len(zh.conversation) == 2
    assert zh.conversation.recent()[0].role == "user"
    assert zh.conversation.recent()[1].role == "assistant"


def test_chat_followup_uses_conversation_context_for_retrieval():
    # The model only ever returns a fact about the Eiffel Tower's completion.
    llm = CallableLLM(lambda p, s: "The Eiffel Tower was completed in 1889.")
    zh = _kb(llm)
    zh.chat("Tell me about the Eiffel Tower.")
    # A terse follow-up with no entity of its own should still retrieve context
    # via the prior user turn, so the answer stays grounded (not abstained).
    ans = zh.chat("When was it completed?")
    assert ans.grounded
    assert "1889" in ans.text


def test_chat_abstains_when_no_context_but_still_records_turn():
    llm = CallableLLM(lambda p, s: "Anything.")
    zh = _kb(llm)
    ans = zh.chat("Who won the 2049 World Series?")
    assert ans.abstained
    assert len(zh.conversation) == 2  # user + assistant(abstain) still recorded


def test_reset_conversation():
    llm = CallableLLM(lambda p, s: "The capital of France is Paris.")
    zh = _kb(llm)
    zh.chat("capital of France?")
    assert len(zh.conversation) > 0
    zh.reset_conversation()
    assert len(zh.conversation) == 0


def test_custom_conversation_buffer_size():
    llm = CallableLLM(lambda p, s: "The capital of France is Paris.")
    zh = _kb(llm, conversation=2)
    assert isinstance(zh.conversation, ConversationMemory)
    assert zh.conversation.max_turns == 2


def test_answer_reports_retries_and_raw_draft():
    # First draft contains a hallucination; the model "fixes" it on re-ask.
    drafts = iter(
        [
            "The capital of France is Paris. The capital of France is Berlin.",
            "The capital of France is Paris.",
        ]
    )
    llm = CallableLLM(lambda p, s: next(drafts))
    zh = _kb(llm, max_retries=1)
    ans = zh.complete("What is the capital of France?")
    assert ans.grounded
    assert ans.retries == 1
    assert ans.raw_draft == "The capital of France is Paris."
    assert "Berlin" not in ans.text


def test_on_event_observability_hook_fires():
    events = []
    llm = CallableLLM(lambda p, s: "The capital of France is Paris.")
    zh = _kb(llm, on_event=lambda name, payload: events.append(name))
    zh.complete("capital of France?")
    assert "retrieve" in events
    assert "draft" in events
    assert "answer" in events


def test_on_event_hook_errors_are_swallowed():
    def boom(name, payload):
        raise RuntimeError("observer crashed")

    llm = CallableLLM(lambda p, s: "The capital of France is Paris.")
    zh = _kb(llm, on_event=boom)
    # Must not raise despite the misbehaving hook.
    ans = zh.complete("capital of France?")
    assert ans.grounded


def test_custom_abstain_message():
    zh = _kb(abstain_message="No clue.")
    ans = zh.answer_from_memory("Who won the 2049 World Series?")
    assert ans.abstained and ans.text == "No clue."


def test_stats_and_forget():
    zh = _kb()
    assert zh.stats()["active"] == 3
    mem = zh.remember("Temporary note.", source="scratch")
    assert zh.stats()["active"] == 4
    zh.forget(mem.id)
    assert zh.stats()["active"] == 3
    # Forgotten memory no longer retrievable.
    assert all("Temporary" not in r.memory.content for r in zh.recall("Temporary"))


def test_export_and_load_between_plugins():
    zh = _kb()
    dump = zh.export()
    other = ZeroHPlugin()
    assert other.load(dump) == 3
    assert other.answer_from_memory("capital of France").grounded


def test_scoped_retrieval_by_source():
    zh = ZeroHPlugin()
    zh.remember("Paris is the capital of France.", source="geo")
    zh.remember("Paris Hilton is a celebrity.", source="gossip")
    res = zh.answer_from_memory("Paris", source="geo")
    assert "capital" in res.text
    assert "celebrity" not in res.text


def test_remember_dedupe():
    zh = ZeroHPlugin()
    zh.remember("A unique fact.", dedupe=True)
    zh.remember("A unique fact.", dedupe=True)
    assert zh.stats()["active"] == 1


def test_recall_top_k_zero_returns_empty():
    # An explicit top_k=0 must mean "no results", not silently fall back to default.
    zh = _kb()
    assert zh.recall("capital of France", top_k=0) == []
    assert zh.answer_from_memory("capital of France", top_k=0).abstained
