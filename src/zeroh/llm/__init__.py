"""Bring-your-own-LLM providers for the zeroH plug-in.

zeroH does not ship a model. Plug in your own via one of these adapters, or
implement :class:`LLM` / wrap a callable with :class:`CallableLLM`. Wrap any
provider in :class:`RetryLLM` to add resilience against flaky networks.
"""
from .base import LLM, CallableLLM
from .providers import EchoLLM, LLMError, OllamaLLM, OpenAICompatibleLLM, RetryLLM

__all__ = [
    "LLM",
    "CallableLLM",
    "OpenAICompatibleLLM",
    "OllamaLLM",
    "EchoLLM",
    "RetryLLM",
    "LLMError",
]
