"""Tests for LLM provider adapters."""
import pytest

from zeroh.llm import CallableLLM, EchoLLM, LLM, OllamaLLM, OpenAICompatibleLLM


def test_echo_llm_returns_prompt():
    llm = EchoLLM()
    assert llm.complete("hello") == "hello"
    assert isinstance(llm, LLM)


def test_echo_llm_prefix():
    assert EchoLLM(prefix=">> ").complete("hi") == ">> hi"


def test_callable_llm_two_args():
    llm = CallableLLM(lambda prompt, system: f"{system}|{prompt}")
    assert llm.complete("q", system="s") == "s|q"


def test_callable_llm_one_arg():
    llm = CallableLLM(lambda prompt: prompt.upper())
    assert llm.complete("abc") == "ABC"


def test_callable_llm_rejects_non_callable():
    with pytest.raises(TypeError):
        CallableLLM(42)


def test_openai_compatible_requires_model():
    with pytest.raises(ValueError):
        OpenAICompatibleLLM(model="")


def test_ollama_factory_targets_local_endpoint():
    llm = OllamaLLM(model="llama3")
    assert isinstance(llm, OpenAICompatibleLLM)
    assert "11434" in llm.base_url
    assert llm.model == "llama3"


def test_openai_compatible_strips_trailing_slash():
    llm = OpenAICompatibleLLM(model="m", base_url="http://x/v1/")
    assert llm.base_url == "http://x/v1"
