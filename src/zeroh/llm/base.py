"""LLM provider interface.

zeroH is a *plug-in*, not an agent: it never ships its own model. Instead users
bring their own LLM — a cloud model via API key (OpenAI, Azure, Anthropic-compat,
...), a local model (Ollama, LM Studio, vLLM, llama.cpp server), or any callable
— by implementing this tiny interface.

The contract is intentionally minimal: one method, :meth:`LLM.complete`, that
maps a prompt (plus optional system instruction) to a string completion. This
keeps zeroH model-agnostic and dependency-free.
"""
from __future__ import annotations

import abc
from typing import Callable, Optional


class LLM(abc.ABC):
    """Abstract base class for any text-generating model zeroH plugs into.

    Implement :meth:`complete` to wrap your model. zeroH supplies a fully
    prepared prompt (including any retrieved grounding context and grounding
    instructions); the implementation only needs to run inference and return the
    raw text.
    """

    @abc.abstractmethod
    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Return the model's text completion for ``prompt``.

        Args:
            prompt: The user/content prompt (already augmented by zeroH).
            system: Optional system instruction.
            temperature: Sampling temperature. zeroH defaults to ``0.0`` because
                deterministic, low-temperature decoding reduces hallucination.
            max_tokens: Optional cap on generated tokens.
        """
        raise NotImplementedError


class CallableLLM(LLM):
    """Adapt any callable into an :class:`LLM`.

    The wrapped function is called as ``fn(prompt, system)`` if it accepts two
    positional arguments, otherwise as ``fn(prompt)``. This is the fastest way to
    plug in an existing client (e.g. a lambda calling your own SDK).

    Example::

        llm = CallableLLM(lambda prompt, system: my_client.chat(system, prompt))
    """

    def __init__(self, fn: Callable[..., str]) -> None:
        if not callable(fn):
            raise TypeError("CallableLLM requires a callable")
        self._fn = fn

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            return str(self._fn(prompt, system))
        except TypeError:
            # The callable only accepts the prompt.
            return str(self._fn(prompt))
