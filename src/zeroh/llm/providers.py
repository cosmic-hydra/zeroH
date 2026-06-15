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
:class:`zeroh.llm.CallableLLM` instead. Any provider can be made resilient to
flaky networks or rate limits by wrapping it in :class:`RetryLLM`.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Callable, Optional, Tuple, TypeVar

from .base import LLM

# HTTP status codes worth retrying: rate limiting + transient server errors.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

_T = TypeVar("_T")


def _retry_call(
    call: Callable[[], _T],
    *,
    max_retries: int,
    backoff: float,
    sleep: Callable[[float], None],
    catch: Tuple[type, ...] = (Exception,),
) -> _T:
    """Call ``call`` with exponential backoff, shared by the retrying providers.

    Retries while the raised exception is an instance of ``catch``, is not
    explicitly marked permanent (``exc.transient is False``), and attempts
    remain; otherwise the exception propagates. Delays follow
    ``backoff * 2**attempt``.
    """
    for attempt in range(max_retries + 1):
        try:
            return call()
        except catch as exc:  # type: ignore[misc]
            if getattr(exc, "transient", True) is False or attempt >= max_retries:
                raise
            sleep(backoff * (2 ** attempt))
    raise AssertionError("unreachable")  # pragma: no cover - loop returns or raises


class LLMError(RuntimeError):
    """Raised when a provider call fails (network, auth, or API error).

    Args:
        message: Human-readable description.
        transient: ``True`` for failures that may succeed on retry (timeouts,
            rate limits, 5xx), ``False`` for permanent ones (auth, bad request).
    """

    def __init__(self, message: str, *, transient: bool = False) -> None:
        super().__init__(message)
        self.transient = transient


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
        max_retries: Number of times to retry on transient failures (timeouts,
            rate limits, 5xx) with exponential backoff. ``0`` disables retries.
        backoff: Base delay in seconds for the exponential backoff.
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
        max_retries: int = 2,
        backoff: float = 0.5,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_max_tokens = default_max_tokens
        self.extra_headers = dict(extra_headers or {})
        self.max_retries = max_retries
        self.backoff = backoff
        # Injectable for tests so backoff doesn't slow the suite down.
        self._sleep = time.sleep

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
        data = json.dumps(payload).encode("utf-8")

        body = _retry_call(
            lambda: self._post(url, data, headers),
            max_retries=self.max_retries,
            backoff=self.backoff,
            sleep=self._sleep,
            catch=(LLMError,),
        )

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover
            raise LLMError(f"Unexpected LLM response shape: {body!r}") from exc

    def _post(self, url: str, data: bytes, headers: dict) -> dict:
        """Perform a single POST and return the parsed JSON body.

        Network failures, rate limits and 5xx are raised as *transient*
        :class:`LLMError`; auth/4xx errors are raised as permanent.
        """
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode("utf-8", "replace") if exc.fp else str(exc)
            raise LLMError(
                f"LLM request failed ({exc.code}): {detail}",
                transient=exc.code in _RETRYABLE_STATUS,
            ) from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise LLMError(
                f"Could not reach LLM endpoint {url}: {exc.reason}", transient=True
            ) from exc


class RetryLLM(LLM):
    """Wrap any :class:`LLM` with exponential-backoff retries.

    Handy for adding resilience to a :class:`CallableLLM` around your own SDK, or
    to layer extra retries on top of any provider::

        llm = RetryLLM(CallableLLM(my_client.chat), max_retries=3)

    Args:
        inner: The wrapped LLM to call.
        max_retries: Maximum retry attempts after the first call.
        backoff: Base delay in seconds for exponential backoff.
        retry_on: Exception types that should trigger a retry.
    """

    def __init__(
        self,
        inner: LLM,
        *,
        max_retries: int = 2,
        backoff: float = 0.5,
        retry_on: tuple = (Exception,),
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.inner = inner
        self.max_retries = max_retries
        self.backoff = backoff
        self.retry_on = retry_on
        self._sleep = time.sleep  # injectable for tests

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        return _retry_call(
            lambda: self.inner.complete(
                prompt, system, temperature=temperature, max_tokens=max_tokens
            ),
            max_retries=self.max_retries,
            backoff=self.backoff,
            sleep=self._sleep,
            catch=self.retry_on,
        )


def OllamaLLM(  # noqa: N802 - factory styled as a class for ergonomics
    model: str,
    base_url: str = "http://localhost:11434/v1",
    **kwargs,
) -> OpenAICompatibleLLM:
    """Convenience factory for a local `Ollama <https://ollama.com>`_ model."""
    return OpenAICompatibleLLM(model=model, base_url=base_url, **kwargs)
