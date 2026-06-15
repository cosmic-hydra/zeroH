"""Demo of zeroH as a plug-in around *your own* LLM.

zeroH never generates text itself — it wraps the LLM you already use (cloud via
API key, or a local model) with a grounded memory layer and verifies the output.

This demo uses a fake local "LLM" (a plain function) so it runs offline. Swap it
for a real provider in one line:

    from zeroh.llm import OpenAICompatibleLLM, OllamaLLM
    llm = OpenAICompatibleLLM(model="gpt-4o-mini", api_key="sk-...")   # cloud
    llm = OllamaLLM(model="llama3")                                    # local

Run with:  python examples/plugin_quickstart.py
"""
from zeroh import ZeroHPlugin
from zeroh.llm import CallableLLM


def fake_local_llm(prompt: str, system: str = "") -> str:
    """Stand-in for a real model.

    It returns one grounded sentence and one fabricated sentence, so we can see
    zeroH strip the hallucination.
    """
    return (
        "The capital of France is Paris. "
        "The capital of France is also secretly Berlin."
    )


def main() -> None:
    llm = CallableLLM(fake_local_llm)
    zh = ZeroHPlugin(llm)

    # Enhanced memory: ingest a whole document; it is chunked automatically.
    handbook = (
        "The capital of France is Paris. France is a country in Western Europe. "
        "The Eiffel Tower is located in Paris and was completed in 1889. "
        "Our refund policy allows returns within 30 days of purchase."
    )
    chunks = zh.ingest(handbook, source="handbook")

    print("=" * 70)
    print(f"1) Ingested handbook into {len(chunks)} memory chunk(s)")

    print("=" * 70)
    print("2) complete(): retrieve -> augment -> YOUR llm -> verify -> answer")
    res = zh.complete("What is the capital of France?")
    print(f"   model said : {fake_local_llm('')!r}")
    print(f"   zeroH kept : {res.text}")
    print(f"   grounded   : {res.grounded}   confidence: {res.confidence}")
    print(f"   citations  : {[c.source for c in res.citations]}")
    print("   -> the fabricated 'Berlin' sentence was stripped before output")

    print("=" * 70)
    print("3) Abstain instead of hallucinating when memory lacks the answer")
    res = zh.complete("Who won the 2049 Formula 1 championship?")
    print(f"   answer     : {res.text}")
    print(f"   abstained  : {res.abstained}")

    print("=" * 70)
    print("4) Guardrail mode: verify text from ANY source (no generation)")
    guard = ZeroHPlugin()  # no LLM at all
    guard.remember("Our refund policy allows returns within 30 days.")
    checked = guard.verify("Refunds are accepted for a full year.")
    print(f"   claim      : 'Refunds are accepted for a full year.'")
    print(f"   abstained  : {checked.abstained} (unsupported -> rejected)")


if __name__ == "__main__":
    main()
