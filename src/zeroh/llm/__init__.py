"""Bring-your-own-LLM providers for the zeroH plug-in.

zeroH does not ship a model. Plug in your own via one of these adapters, or
implement :class:`LLM` / wrap a callable with :class:`CallableLLM`.
"""
from .base import LLM, CallableLLM
from .providers import EchoLLM, LLMError, OllamaLLM, OpenAICompatibleLLM

__all__ = [
    "LLM",
    "CallableLLM",
    "OpenAICompatibleLLM",
    "OllamaLLM",
    "EchoLLM",
    "LLMError",
]
