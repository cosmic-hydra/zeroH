# zeroH — Zero Hallucination for AI agents

**zeroH** is a durable, grounded **memory layer** for AI agents. It enhances an
agent's memory and is built to **stop hallucinations and data loss** by making
every answer traceable to something the agent actually knows.

It is **pure Python (standard library only)** for the core — no heavyweight ML
dependencies, no network calls, fully deterministic — so it runs anywhere and is
easy to embed in any agent stack.

## Why zeroH?

Large language models hallucinate: they state things confidently that are not
backed by any source, and they forget facts between sessions. zeroH addresses
both problems directly:

| Problem | How zeroH addresses it |
| --- | --- |
| **Data loss** | A durable SQLite-backed memory store with WAL journaling. Corrections *supersede* old facts instead of deleting them, so history is never lost. |
| **Hallucination** | Every candidate claim is verified against retrieved memory. Unsupported claims are flagged and stripped. |
| **Confident guessing** | The agent *abstains* ("I don't know") when memory can't support an answer — **abstention over fabrication**. |
| **Untraceable answers** | Every grounded answer comes with **citations** back to the source memories and a **confidence** score. |

## How it works

```
remember()                         answer()
    |                                  |
    v                                  v
MemoryStore --> Retriever --> Verifier --> HallucinationDetector --> Answer
 (durable)      (TF-IDF +      (coverage-     (per-claim risk,        (text +
                 keyword)       based          abstain if              citations +
                                entailment)    unsupported)            confidence)
```

1. **MemoryStore** — durable, append-only SQLite storage (`zeroh.memory`).
2. **Retriever** — TF-IDF cosine similarity blended with keyword overlap
   (`zeroh.retrieval`).
3. **Verifier** — strict, coverage-based claim verification: a claim is grounded
   only when a single memory covers (almost) all of its content words
   (`zeroh.grounding`).
4. **HallucinationDetector** — turns per-claim verification into an overall risk
   score and can strip unsupported sentences (`zeroh.hallucination`).
5. **GroundedAgent** — the high-level API that ties it all together
   (`zeroh.agent`).

## Install

```bash
pip install -e ".[dev]"   # editable install with test extras
```

## Quick start

```python
from zeroh import GroundedAgent, MemoryStore

# In-memory by default; pass MemoryStore("agent.db") for durable persistence.
agent = GroundedAgent()

# Teach it facts (stored durably).
agent.remember("The capital of France is Paris.", source="atlas")

# Grounded answer, composed only from memory, with citations + confidence.
ans = agent.answer("What is the capital of France?")
print(ans.text)         # "The capital of France is Paris."
print(ans.confidence)   # e.g. 0.69
print(ans.citations)    # [Citation(source="atlas", ...)]

# Abstains instead of hallucinating when it doesn't know.
print(agent.answer("Who won the 2049 World Cup?").abstained)  # True
```

### Guardrail mode (verify an LLM's draft)

Use zeroH to fact-check answers produced by *any* LLM against your trusted
memory. Unsupported sentences are rejected or stripped:

```python
draft = "The capital of France is Berlin."        # a hallucination
checked = agent.answer("capital of France", candidate=draft)
print(checked.abstained)   # True — the unsupported claim was rejected

draft = "The capital of France is Paris. The Eiffel Tower is in Rome."
checked = agent.answer("France", candidate=draft)
print(checked.text)        # "The capital of France is Paris."  (Rome stripped)
```

### Correct facts without losing history

```python
mem = agent.remember("The sky is green.")
agent.correct(mem.id, "The sky is blue.")     # supersedes, keeps the old value
print(agent.answer("What color is the sky?").text)   # "The sky is blue."
```

## Run the demo & tests

```bash
python examples/quickstart.py   # end-to-end walkthrough
pytest                          # run the test suite
```

## Project layout

```
src/zeroh/
|- models.py            # Memory, Citation, Claim, Answer dataclasses
|- text.py              # tokenization / sentence splitting (stdlib)
|- embeddings.py        # dependency-free TF-IDF + cosine similarity
|- memory/              # durable SQLite-backed MemoryStore
|- retrieval/           # Retriever (RAG)
|- grounding/           # Verifier (claim verification)
|- hallucination/       # HallucinationDetector + risk report
\- agent/               # GroundedAgent (high-level API)
```

## Design notes & limitations

zeroH's verification is intentionally **lexical and strict** rather than
semantic. This keeps it dependency-free, deterministic and offline, and biases
strongly toward abstaining when unsure — exactly what you want for a
"zero hallucination" guarantee. The trade-off is that paraphrases or synonyms
may not be recognized as supporting a claim. For production use you can raise
recall by feeding richer memories, or by swapping in a stronger embedding model
behind the same `Retriever`/`Verifier` interfaces.

## License

MIT
