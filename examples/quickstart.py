"""Runnable demo of zeroH's grounded, anti-hallucination memory.

Run with:  python examples/quickstart.py
"""
from zeroh import GroundedAgent


def main() -> None:
    # Use an in-memory store here; pass MemoryStore("agent.db") for durability.
    agent = GroundedAgent()

    # 1. Teach the agent some facts. These are stored durably (no data loss).
    agent.remember_many(
        [
            "The capital of France is Paris.",
            "The Eiffel Tower is located in Paris.",
            "Python is a programming language created by Guido van Rossum.",
            "Water boils at 100 degrees Celsius at sea level.",
        ],
        source="knowledge-base",
    )

    print("=" * 70)
    print("1) Grounded extractive answer (composed only from memory)")
    ans = agent.answer("What is the capital of France?")
    print(f"   answer    : {ans.text}")
    print(f"   grounded  : {ans.grounded}   confidence: {ans.confidence}")
    print(f"   citations : {[c.source for c in ans.citations]}")

    print("=" * 70)
    print("2) Abstain instead of hallucinate when memory has no answer")
    ans = agent.answer("Who won the 2049 chess world championship?")
    print(f"   answer    : {ans.text}")
    print(f"   abstained : {ans.abstained}")

    print("=" * 70)
    print("3) Guardrail mode: verify an LLM draft and reject hallucinations")
    draft = "The capital of France is Berlin."
    ans = agent.answer("capital of France", candidate=draft)
    print(f"   draft     : {draft}")
    print(f"   result    : {ans.text}")
    print(f"   abstained : {ans.abstained}")

    print("=" * 70)
    print("4) Guardrail mode: strip only the unsupported sentence")
    draft = "The capital of France is Paris. The Eiffel Tower is in Rome."
    ans = agent.answer("France", candidate=draft)
    print(f"   draft     : {draft}")
    print(f"   kept      : {ans.text}")

    print("=" * 70)
    print("5) Correct a fact without losing history")
    mem = agent.remember("The sky is green.")
    agent.correct(mem.id, "The sky is blue.")
    ans = agent.answer("What color is the sky?")
    print(f"   answer    : {ans.text}")
    history = [m.content for m in agent.store.all(include_inactive=True)]
    print(f"   retained  : {'The sky is green.' in history} (old value kept for audit)")


if __name__ == "__main__":
    main()
