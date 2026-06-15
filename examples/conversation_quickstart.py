"""Demo of zeroH's multi-turn grounded chat (short-term + long-term memory).

`ZeroHPlugin.chat()` keeps a bounded short-term conversation memory on top of the
durable store, so follow-up questions stay grounded and in-context — while every
answer is still verified claim-by-claim and abstains rather than fabricate.

This demo uses a tiny offline "LLM" (a plain function) so it runs with no network
or API key. Swap it for a real provider in one line:

    from zeroh.llm import OpenAICompatibleLLM, OllamaLLM
    llm = OpenAICompatibleLLM(model="gpt-4o-mini", api_key="sk-...")   # cloud
    llm = OllamaLLM(model="llama3")                                    # local

Run with:  python examples/conversation_quickstart.py
"""
from zeroh import ZeroHPlugin
from zeroh.llm import CallableLLM
from zeroh.text import tokenize  # content-word tokenizer (drops stopwords)


def fake_local_llm(prompt: str, system: str = "") -> str:
    """A deterministic stand-in for a real model.

    zeroH hands the model a prompt containing the retrieved ``Context:`` lines and
    the current ``Question:``. This fake "answers" by returning the context line
    that best overlaps the question, or declines with "I don't know." when nothing
    matches — so zeroH's verify/abstain behaviour is exercised offline. A real
    model would reason over the same context instead.
    """
    question, context = "", []
    for line in prompt.splitlines():
        if line.startswith("Question:"):
            question = line[len("Question:"):]
        elif line.startswith("[") and "] " in line:
            context.append(line.split("] ", 1)[1])

    q = set(tokenize(question))
    best, best_overlap = None, 0
    for line in context:
        overlap = len(q & set(tokenize(line)))
        if overlap > best_overlap:
            best, best_overlap = line, overlap
    return best if best else "I don't know."


def main() -> None:
    llm = CallableLLM(fake_local_llm)
    zh = ZeroHPlugin(llm, conversation=6)  # keep the last 6 turns of context

    # Store discrete facts so retrieval can surface the single relevant one per
    # turn (use ingest() for whole documents, which auto-chunks).
    zh.remember_many(
        [
            "The Eiffel Tower is located in Paris, France.",
            "The Eiffel Tower was completed in 1889.",
            "The Eiffel Tower is 330 metres tall.",
            "The capital of France is Paris.",
        ],
        source="handbook",
    )

    print("=" * 70)
    print("Multi-turn grounded chat — terse follow-ups resolve via conversation")
    print("memory, and each answer is grounded in a specific stored fact.")
    for message in [
        "Tell me about the Eiffel Tower.",
        "When was it completed?",   # "it" -> Eiffel Tower (from the prior turn)
        "How tall is it?",
    ]:
        ans = zh.chat(message)
        grounded_in = ans.citations[0].content if ans.citations else "(none)"
        print(f"\n  you    : {message}")
        print(f"  zeroH  : {ans.text}")
        print(f"  ↳ grounded in: {grounded_in}")

    print("\n" + "=" * 70)
    print("Abstains instead of guessing when memory has no answer")
    ans = zh.chat("Who designed the building next door?")
    print(f"  zeroH  : {ans.text}")
    print(f"  ↳ abstained = {ans.abstained}")

    print("=" * 70)
    print(f"Short-term turns retained : {len(zh.conversation)}")
    print(f"Long-term memory stats    : {zh.stats()}")


if __name__ == "__main__":
    main()
