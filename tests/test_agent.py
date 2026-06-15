"""End-to-end tests for the grounded agent."""
from zeroh import GroundedAgent
from zeroh.agent import ABSTAIN_MESSAGE


def _agent():
    agent = GroundedAgent()
    agent.remember_many(
        [
            "The capital of France is Paris.",
            "The Eiffel Tower is located in Paris.",
            "Python is a programming language created by Guido van Rossum.",
        ],
        source="facts",
    )
    return agent


def test_extractive_answer_is_grounded_with_citations():
    agent = _agent()
    ans = agent.answer("What is the capital of France?")
    assert ans.grounded
    assert not ans.abstained
    assert "Paris" in ans.text
    assert ans.citations
    assert ans.confidence > 0


def test_agent_abstains_on_unknown_query():
    agent = _agent()
    ans = agent.answer("Who won the 2049 chess world championship?")
    assert ans.abstained
    assert ans.text == ABSTAIN_MESSAGE


def test_guardrail_rejects_hallucinated_candidate():
    agent = _agent()
    ans = agent.answer("capital", candidate="The capital of France is Berlin.")
    assert ans.abstained
    assert not ans.grounded


def test_guardrail_accepts_supported_candidate():
    agent = _agent()
    ans = agent.answer("capital", candidate="The capital of France is Paris.")
    assert ans.grounded
    assert "Paris" in ans.text


def test_guardrail_strips_unsupported_sentence():
    agent = _agent()
    candidate = "The capital of France is Paris. The capital of France is Berlin."
    ans = agent.answer("capital", candidate=candidate)
    assert "Paris" in ans.text
    assert "Berlin" not in ans.text


def test_correct_supersedes_without_data_loss():
    agent = GroundedAgent()
    mem = agent.remember("The sky is green.")
    agent.correct(mem.id, "The sky is blue.")
    ans = agent.answer("What color is the sky?")
    assert "blue" in ans.text
    # History retained in the underlying store.
    all_contents = [m.content for m in agent.store.all(include_inactive=True)]
    assert "The sky is green." in all_contents


def test_answer_serializes_to_dict():
    agent = _agent()
    ans = agent.answer("What is the capital of France?")
    d = ans.to_dict()
    assert set(["text", "grounded", "confidence", "claims", "citations"]).issubset(d)
