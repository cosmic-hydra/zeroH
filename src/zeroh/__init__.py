"""zeroH — Zero Hallucination for AI agents.

zeroH is a **plug-in**, not an agent. Bring your own LLM (cloud via API key, or
a local model) and zeroH wraps it with a durable, grounded memory layer that:

* **Remembers** facts and documents durably across restarts (no data loss).
* **Augments** every prompt with the most relevant retrieved context (RAG).
* **Grounds** each answer in stored memory with explicit citations.
* **Verifies** the model's output claim-by-claim and **abstains** instead of
  hallucinating — driving the residual hallucination rate toward near-zero.
* **Remembers the conversation** too: a bounded short-term memory keeps
  multi-turn :meth:`~zeroh.plugin.ZeroHPlugin.chat` grounded and in-context.

Quick start (bring your own LLM)::

    from zeroh import ZeroHPlugin
    from zeroh.llm import OpenAICompatibleLLM, OllamaLLM, CallableLLM

    # Cloud (API key) ...
    llm = OpenAICompatibleLLM(model="gpt-4o-mini", api_key="sk-...")
    # ... or local (Ollama / LM Studio / vLLM) ...
    # llm = OllamaLLM(model="llama3")
    # ... or wrap any function you already have:
    # llm = CallableLLM(lambda prompt, system: my_client.chat(system, prompt))

    zh = ZeroHPlugin(llm)
    zh.ingest(open("handbook.md").read(), source="handbook")
    result = zh.complete("What is our refund policy?")
    print(result.text, result.confidence, result.citations)

LLM-free guardrail mode (verify text from any source)::

    zh = ZeroHPlugin()                       # no LLM needed
    zh.remember("The capital of France is Paris.")
    print(zh.verify("The capital of France is Berlin.").abstained)  # True

A command-line interface is also available::

    zeroh remember "The capital of France is Paris." --source kb
    zeroh ask "What is the capital of France?"
"""
from .grounding import Verifier
from .hallucination import HallucinationDetector, HallucinationReport
from .llm import CallableLLM, EchoLLM, LLM, OllamaLLM, OpenAICompatibleLLM, RetryLLM
from .memory import (
    CompressionReport,
    ContextPreserver,
    ConversationMemory,
    InvertedIndex,
    MemoryCompressor,
    MemoryStore,
    RedundancyGroup,
    TieredMemoryManager,
    Turn,
    chunk_text,
)
from .models import Answer, Citation, Claim, Memory, RetrievalResult
from .optimization import BatchProcessor, MemoryMetrics, QueryCache
from .plugin import ABSTAIN_MESSAGE, ZeroHPlugin
from .reasoning import EntityExtractor, QueryExpander, RelationExtractor
from .retrieval import MemoryFilter, Retriever

# `GroundedAgent` remains available as a convenience/no-LLM fallback, but the
# plug-in (`ZeroHPlugin`) is the primary, recommended interface.
from .agent import GroundedAgent

__version__ = "0.6.0"

__all__ = [
    # primary plug-in API
    "ZeroHPlugin",
    "ABSTAIN_MESSAGE",
    # bring-your-own-LLM providers
    "LLM",
    "CallableLLM",
    "OpenAICompatibleLLM",
    "OllamaLLM",
    "EchoLLM",
    "RetryLLM",
    # memory + components
    "MemoryStore",
    "ConversationMemory",
    "Turn",
    "chunk_text",
    "TieredMemoryManager",
    "InvertedIndex",
    "MemoryCompressor",
    "CompressionReport",
    "RedundancyGroup",
    "ContextPreserver",
    "Retriever",
    "MemoryFilter",
    "Verifier",
    "HallucinationDetector",
    "HallucinationReport",
    # optimization
    "QueryCache",
    "BatchProcessor",
    "MemoryMetrics",
    # reasoning
    "EntityExtractor",
    "RelationExtractor",
    "QueryExpander",
    # data models
    "Memory",
    "Citation",
    "Claim",
    "Answer",
    "RetrievalResult",
    # legacy convenience
    "GroundedAgent",
    "__version__",
]
