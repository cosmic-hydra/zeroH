"""Tests for short-term conversation memory."""
import pytest

from zeroh.memory import ConversationMemory, Turn


def test_add_and_recent():
    conv = ConversationMemory(max_turns=5)
    conv.add_user("hello")
    conv.add_assistant("hi there")
    assert len(conv) == 2
    assert [t.role for t in conv.recent()] == ["user", "assistant"]
    assert conv.recent(1)[0].content == "hi there"


def test_bounded_buffer_evicts_oldest():
    conv = ConversationMemory(max_turns=2)
    conv.add_user("one")
    conv.add_user("two")
    conv.add_user("three")
    contents = [t.content for t in conv.recent()]
    assert contents == ["two", "three"]
    assert len(conv) == 2


def test_query_context_uses_recent_user_turns():
    conv = ConversationMemory()
    conv.add_user("Tell me about the Eiffel Tower.")
    conv.add_assistant("It is in Paris.")
    conv.add_user("When was it built?")
    cue = conv.query_context(n=2)
    assert "Eiffel Tower" in cue
    assert "When was it built?" in cue
    # Assistant turns are excluded from the retrieval cue.
    assert "It is in Paris" not in cue


def test_transcript_and_clear():
    conv = ConversationMemory()
    conv.add_user("q")
    conv.add_assistant("a")
    transcript = conv.transcript()
    assert "User: q" in transcript
    assert "Assistant: a" in transcript
    conv.clear()
    assert len(conv) == 0


def test_invalid_max_turns():
    with pytest.raises(ValueError):
        ConversationMemory(max_turns=0)


def test_turn_serializes():
    t = Turn(role="user", content="hi")
    d = t.to_dict()
    assert d["role"] == "user" and d["content"] == "hi" and "created_at" in d
