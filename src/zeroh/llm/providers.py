"""Concrete LLM providers for cloud and local models.

All real network providers are implemented against the **OpenAI-compatible**
``/chat/completions`` API using only the Python standard library (``urllib``),
so zeroH stays dependency-free. That single wire format covers an enormous range
of deployments:

* **Cloud:** OpenAI, Azure OpenAI, Together, Groq, Fireworks, OpenRouter, and
  any other OpenAI-compatible endpoint — just pass your API key and ``base_url``.
* **Local:** Ollama (``http://localhost:11434/v1``), LM Studio
  (``http://localhost:1234/v1``), vLLM, llama.cpp server — usually no key needed.

For non-OpenAI-shaped clients, wrap your own SDK with
:class:`zeroh.llm.CallableLLM` instead.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

from .base import LLM


class LLMError(RuntimeError):
    """Raised when a provider call fails (network, auth, or API error)."""


class EchoLLM(LLM):
    """A trivial offline LLM that echoes its prompt.

    Useful for tests and for exercising zeroH's grounding pipeline without any
    real model or network access.
    """

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        return f"{self.prefix}{prompt}"


class OpenAICompatibleLLM(LLM):
    """Call any OpenAI-compatible ``/chat/completions`` endpoint.

    Works with cloud providers (via an API key) and local servers (Ollama, LM
    Studio, vLLM, ...) using only the standard library.

    Args:
        model: Model name (e.g. ``"gpt-4o-mini"``, ``"llama3"``).
        api_key: API key for cloud providers. Optional for local servers.
        base_url: API root. Defaults to OpenAI. For Ollama use
            ``"http://localhost:11434/v1"``; for LM Studio
            ``"http://localhost:1234/v1"``.
        timeout: Per-request timeout in seconds.
        default_max_tokens: Fallback ``max_tokens`` when the caller omits one.
        extra_headers: Optional additional HTTP headers (e.g. for Azure).
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        *,
        timeout: float = 60.0,
        default_max_tokens: Optional[int] = None,
        extra_headers: Optional[dict] = None,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_max_tokens = default_max_tokens
        self.extra_headers = dict(extra_headers or {})

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        if tokens is not None:
            payload["max_tokens"] = tokens

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        headers.update(self.extra_headers)

        url = f"{self.base_url}/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode("utf-8", "replace") if exc.fp else str(exc)
            raise LLMError(f"LLM request failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise LLMError(f"Could not reach LLM endpoint {url}: {exc.reason}") from exc

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover
            raise LLMError(f"Unexpected LLM response shape: {body!r}") from exc


def OllamaLLM(  # noqa: N802 - factory styled as a class for ergonomics
    model: str,
    base_url: str = "http://localhost:11434/v1",
    **kwargs,
) -> OpenAICompatibleLLM:
    """Convenience factory for a local `Ollama <https://ollama.com>`_ model."""
    return OpenAICompatibleLLM(model=model, base_url=base_url, **kwargs)
