# zeroH — Zero Hallucination for AI agents

**zeroH is a plug-in, not an agent.** Bring your own LLM — a **cloud model via
API key** (OpenAI, Azure, Groq, Together, OpenRouter, …) or a **local model**
(Ollama, LM Studio, vLLM, llama.cpp) — and zeroH wraps it with a durable,
grounded **memory layer** that stops hallucination and data loss.

zeroH never generates text itself. It surrounds *your* model with retrieval,
grounding and verification so every answer is traceable to something the agent
actually knows. The core is **pure Python (standard library only)** — no
heavyweight ML dependencies, no vendor lock-in, fully deterministic and offline.

## What it does

```
                         ┌──────────────── zeroH plug-in ────────────────┐
  query ───────────────► │ retrieve ─► augment prompt ─► [YOUR LLM] ─►    │
                         │                                  verify ─► strip│ ──► grounded answer
  remember()/ingest() ─► │ durable memory store                /abstain   │      (+ citations,
                         └───────────────────────────────────────────────┘       confidence)
```

1. **Retrieve** the most relevant memories for the query (RAG).
2. **Augment** the prompt with that context + strict grounding instructions.
3. **Generate** using *your* LLM (cloud or local).
4. **Verify** the output claim-by-claim against memory; **strip** unsupported
   sentences, **re-ask** the model, and **abstain** rather than fabricate.

That verify → re-ask → abstain loop is what pushes the residual hallucination
rate toward near-zero: nothing the model says reaches the user unless it can be
grounded in memory.

## Why it cuts hallucination and data loss

| Problem | How zeroH addresses it |
| --- | --- |
| **Data loss** | Durable SQLite store (WAL). Corrections *supersede* facts instead of deleting them — history is never lost. |
| **Hallucination** | Every claim is verified against retrieved memory; unsupported claims are stripped or the model is re-asked. |
| **Confident guessing** | The plug-in **abstains** ("I don't know") when memory can't support an answer — *abstention over fabrication*. |
| **Poor‑quality I/O** | Documents are **ingested as sentence-aware, overlapping chunks**, so retrieval surfaces precise context — the single biggest lever on answer quality. |
| **Untraceable answers** | Every grounded answer ships with **citations** and a **confidence** score. |

## Install

```bash
pip install -e ".[dev]"
```

## Bring your own LLM

```python
from zeroh.llm import OpenAICompatibleLLM, OllamaLLM, CallableLLM

# Cloud, via your API key (OpenAI / Azure / Groq / Together / OpenRouter / …)
llm = OpenAICompatibleLLM(model="gpt-4o-mini", api_key="sk-...")

# Local — no key needed (Ollama / LM Studio / vLLM / llama.cpp server)
llm = OllamaLLM(model="llama3")
llm = OpenAICompatibleLLM(model="llama3", base_url="http://localhost:11434/v1")

# Or wrap any client/function you already have:
llm = CallableLLM(lambda prompt, system: my_client.chat(system, prompt))
```

A single `OpenAICompatibleLLM` adapter speaks the OpenAI `/chat/completions`
wire format, which both cloud providers and local servers expose — so the same
code targets either by changing `base_url`.

## Quick start

```python
from zeroh import ZeroHPlugin
from zeroh.llm import OpenAICompatibleLLM

llm = OpenAICompatibleLLM(model="gpt-4o-mini", api_key="sk-...")
zh = ZeroHPlugin(llm)

# Enhanced memory: ingest a whole document (auto-chunked) or add discrete facts.
zh.ingest(open("handbook.md").read(), source="handbook")
zh.remember("Refunds are accepted within 30 days.", source="policy")

# Grounded completion: retrieve → augment → your LLM → verify.
res = zh.complete("What is our refund policy?")
print(res.text)         # only grounded sentences
print(res.confidence)   # 0..1
print(res.citations)    # sources backing the answer

# Abstains instead of hallucinating when memory has no answer.
print(zh.complete("Who won the 2049 World Cup?").abstained)  # True
```

### Multi-turn chat (short-term + long-term memory)

`chat()` adds a bounded **short-term conversation memory** on top of the durable
store, so terse follow-ups stay grounded and in-context — without ever relaxing
verification:

```python
zh.ingest(open("handbook.md").read(), source="handbook")

zh.chat("Tell me about the Eiffel Tower.")
zh.chat("When was it completed?")   # "it" is resolved from the prior turn
zh.reset_conversation()             # clear the short-term context when done
```

The recent user turns are folded into retrieval so the right grounding facts are
still found; every answer is then verified claim-by-claim exactly as `complete()`.

### Guardrail mode (no generation)

Already have an answer from somewhere — an LLM you call yourself, a cache, a
human draft? Fact-check it against memory without zeroH generating anything:

```python
zh = ZeroHPlugin()                       # no LLM required
zh.remember("The capital of France is Paris.")

zh.verify("The capital of France is Berlin.").abstained   # True  (rejected)
zh.verify("The capital of France is Paris.").grounded     # True  (kept)
```

### Correct facts without losing history

```python
mem = zh.remember("The sky is green.")
zh.correct(mem.id, "The sky is blue.")    # supersedes; old value kept for audit
```

### Tuning memory & retrieval

Retrieval can fold two extra signals into ranking — both **off by default**, so
existing behaviour is unchanged until you opt in:

```python
zh = ZeroHPlugin(
    llm,
    confidence_weight=0.5,      # trust high-confidence memories more
    recency_weight=0.5,         # prefer fresher knowledge ...
    recency_half_life=86_400,   # ... decaying to half-weight after a day
)

# Scope retrieval to a source, or any metadata predicate:
zh.complete("refund window?", source="handbook")
zh.recall("policy", where=lambda m: m.metadata.get("team") == "support")

# Inspect, back up and restore memory:
zh.stats()                      # {'active': 12, 'inactive': 1, 'by_source': {...}}
dump = zh.export()              # JSONL backup (round-trips via zh.load(dump))
zh.forget(mem.id)               # soft-delete (kept for audit, dropped from recall)

# Observe every pipeline stage (retrieve / draft / answer / abstain):
zh = ZeroHPlugin(llm, on_event=lambda event, data: print(event, data))
```

Wrap any provider in `RetryLLM` for resilience against flaky networks or rate
limits:

```python
from zeroh.llm import RetryLLM, OpenAICompatibleLLM
llm = RetryLLM(OpenAICompatibleLLM(model="gpt-4o-mini", api_key="sk-..."), max_retries=3)
```

## Command-line interface

Installing zeroH also provides a `zeroh` command (and `python -m zeroh`) for
driving a durable store straight from the shell:

```bash
zeroh remember "The capital of France is Paris." --source kb
zeroh ingest handbook.md --source handbook          # or: cat doc | zeroh ingest -
zeroh recall "capital of France"
zeroh ask "What is the capital of France?"           # extractive, no LLM needed
zeroh verify "The capital of France is Berlin."      # guardrail fact-check
zeroh stats
zeroh export > backup.jsonl                          # zeroh import < backup.jsonl
```

The store defaults to `$ZEROH_DB` or `./zeroh.db`; pass `--db` to override and
`--json` to any command for machine-readable output.

## Public API

| Symbol | Purpose |
| --- | --- |
| `ZeroHPlugin` | The plug-in. Wraps your LLM with grounded memory + verification. Adds `chat()`, `stats()`, `export()`/`load()`, `forget()`. |
| `zeroh.llm.LLM` | Base interface — implement `complete()` for any model. |
| `OpenAICompatibleLLM`, `OllamaLLM`, `CallableLLM`, `EchoLLM` | Ready-made providers. |
| `RetryLLM` | Wrap any provider with exponential-backoff retries. |
| `MemoryStore`, `chunk_text` | Durable storage (+ `stats`, `export_jsonl`/`import_jsonl`, dedupe) + chunking. |
| `ConversationMemory` | Bounded short-term memory powering multi-turn `chat()`. |
| `Retriever`, `Verifier`, `HallucinationDetector` | Composable building blocks (retrieval supports confidence/recency/filters). |
| `GroundedAgent` | Legacy convenience wrapper (no-LLM extractive answers). |
| `zeroh` CLI | `remember`, `ingest`, `recall`, `ask`, `verify`, `stats`, `export`/`import`. |

## Run the demos & tests

```bash
python examples/plugin_quickstart.py        # plug-in around your own LLM
python examples/conversation_quickstart.py  # multi-turn grounded chat
python examples/quickstart.py               # LLM-free memory/guardrail demo
zeroh ask "..."                             # the command-line interface
pytest                                      # full test suite
```

## Project layout

```
src/zeroh/
|- plugin.py            # ZeroHPlugin — the bring-your-own-LLM plug-in (+ chat)
|- cli.py / __main__.py # the `zeroh` command-line interface
|- llm/                 # LLM interface + cloud/local providers + RetryLLM
|- memory/              # durable MemoryStore + ingestion + ConversationMemory
|- retrieval/           # Retriever (RAG; confidence/recency/filter aware)
|- grounding/           # Verifier (claim verification)
|- hallucination/       # HallucinationDetector + risk report
|- agent/               # GroundedAgent (legacy convenience)
|- embeddings.py        # dependency-free TF-IDF + cosine similarity
|- text.py              # tokenization / sentence splitting
\- models.py            # Memory, Citation, Claim, Answer dataclasses
```

## Design notes & honest limitations

zeroH's verification is intentionally **lexical and strict** rather than
semantic: a claim is grounded only when a single memory covers (almost) all of
its content words. This keeps the plug-in dependency-free, deterministic and
offline, and biases strongly toward abstaining when unsure — which is exactly
how the residual hallucination rate is driven down to a fraction of a percent.
The trade-off is recall: paraphrases or synonyms may not be recognized as
support. Raise recall by ingesting richer memories, or by swapping a stronger
embedding model behind the same `Retriever` / `Verifier` interfaces.

Quality/accuracy numbers depend on your model, your stored knowledge and your
thresholds — treat the headline figures as a design target achieved through
strict grounding and abstention, not a guarantee for every workload.

## License

MIT
