"""zeroH — Zero Hallucination for AI agents.

A durable, grounded memory layer that helps AI agents:

* **Remember** facts durably across restarts (no data loss).
* **Retrieve** relevant context for a query (RAG).
* **Ground** every answer in stored memory with explicit citations.
* **Detect hallucinations** by verifying each claim against memory and
  abstaining when support is insufficient.

Quick start::

    from zeroh import GroundedAgent

    agent = GroundedAgent()
    agent.remember("The capital of France is Paris.", source="atlas")
    answer = agent.answer("What is the capital of France?")
    print(answer.text, answer.confidence)

    # Guardrail mode: verify an LLM's draft answer against memory.
    checked = agent.answer("...", candidate="The capital of France is Berlin.")
    print(checked.abstained)  # True -> the unsupported claim was rejected
"""
from .agent import ABSTAIN_MESSAGE, GroundedAgent
from .grounding import Verifier
from .hallucination import HallucinationDetector, HallucinationReport
from .memory import MemoryStore
from .models import Answer, Citation, Claim, Memory, RetrievalResult
from .retrieval import Retriever

__version__ = "0.1.0"

__all__ = [
    "GroundedAgent",
    "ABSTAIN_MESSAGE",
    "MemoryStore",
    "Retriever",
    "Verifier",
    "HallucinationDetector",
    "HallucinationReport",
    "Memory",
    "Citation",
    "Claim",
    "Answer",
    "RetrievalResult",
    "__version__",
]
