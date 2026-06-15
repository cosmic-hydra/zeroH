"""Tests for LLM retry/backoff resilience."""
import pytest

from zeroh.llm import CallableLLM, OpenAICompatibleLLM, RetryLLM
from zeroh.llm.providers import LLMError


def test_retryllm_retries_then_succeeds():
    calls = {"n": 0}

    def flaky(prompt, system=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    llm = RetryLLM(CallableLLM(flaky), max_retries=3)
    llm._sleep = lambda *_: None  # no real backoff in tests
    assert llm.complete("hi") == "ok"
    assert calls["n"] == 3


def test_retryllm_gives_up_after_max_retries():
    def always_fail(prompt, system=None):
        raise RuntimeError("nope")

    llm = RetryLLM(CallableLLM(always_fail), max_retries=2)
    llm._sleep = lambda *_: None
    with pytest.raises(RuntimeError):
        llm.complete("hi")


def test_retryllm_does_not_retry_permanent_errors():
    calls = {"n": 0}

    def permanent(prompt, system=None):
        calls["n"] += 1
        raise LLMError("bad auth", transient=False)

    llm = RetryLLM(CallableLLM(permanent), max_retries=5)
    llm._sleep = lambda *_: None
    with pytest.raises(LLMError):
        llm.complete("hi")
    assert calls["n"] == 1  # no retries for permanent failures


def test_retryllm_rejects_negative_retries():
    with pytest.raises(ValueError):
        RetryLLM(CallableLLM(lambda p, s=None: "x"), max_retries=-1)


def test_openai_provider_retries_transient_then_succeeds():
    llm = OpenAICompatibleLLM(model="m", api_key="k", max_retries=2)
    llm._sleep = lambda *_: None
    attempts = {"n": 0}

    def fake_post(url, data, headers):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise LLMError("503 unavailable", transient=True)
        return {"choices": [{"message": {"content": "hello"}}]}

    llm._post = fake_post  # type: ignore[assignment]
    assert llm.complete("hi") == "hello"
    assert attempts["n"] == 2


def test_openai_provider_does_not_retry_permanent_error():
    llm = OpenAICompatibleLLM(model="m", api_key="k", max_retries=3)
    llm._sleep = lambda *_: None
    attempts = {"n": 0}

    def fake_post(url, data, headers):
        attempts["n"] += 1
        raise LLMError("401 unauthorized", transient=False)

    llm._post = fake_post  # type: ignore[assignment]
    with pytest.raises(LLMError):
        llm.complete("hi")
    assert attempts["n"] == 1


def test_llmerror_has_transient_flag():
    assert LLMError("x", transient=True).transient is True
    assert LLMError("y").transient is False
