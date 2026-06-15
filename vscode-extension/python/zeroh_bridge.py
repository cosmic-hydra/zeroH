#!/usr/bin/env python3
"""JSON bridge between the zeroH VS Code extension and the ``zeroh`` package.

The extension invokes this script once per command. A single JSON request is
read from ``stdin`` (or from the first CLI argument) and a single JSON response
is written to ``stdout``. All grounded-memory work is delegated to the installed
``zeroh`` Python package, so this file stays a thin, dependency-free adapter.

Request shape::

    {
      "command": "remember" | "ingest" | "recall" | "verify"
                 | "answer" | "complete" | "ping",
      "db": "/abs/path/to/memory.db",   # durable SQLite store (optional)
      ...command-specific fields...,
      "llm": {                          # only used by "complete"
        "provider": "openai" | "ollama",
        "model": "gpt-4o-mini",
        "apiKey": "sk-...",
        "baseUrl": "http://localhost:11434/v1"
      }
    }

Response shape::

    {"ok": true, "result": {...}}
    {"ok": false, "error": "human readable message"}
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional


def _error(message: str, *, code: str = "error") -> Dict[str, Any]:
    return {"ok": False, "error": message, "code": code}


def _build_llm(cfg: Optional[Dict[str, Any]]):
    """Construct an ``LLM`` from the extension's settings, or ``None``."""
    if not cfg:
        return None
    from zeroh.llm import OllamaLLM, OpenAICompatibleLLM

    provider = (cfg.get("provider") or "openai").lower()
    model = cfg.get("model") or "gpt-4o-mini"
    api_key = cfg.get("apiKey") or None
    base_url = cfg.get("baseUrl") or None

    if provider == "ollama":
        kwargs: Dict[str, Any] = {"model": model}
        if base_url:
            kwargs["base_url"] = base_url
        return OllamaLLM(**kwargs)

    # Default: any OpenAI-compatible endpoint (cloud or local).
    kwargs = {"model": model}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAICompatibleLLM(**kwargs)


def _plugin(req: Dict[str, Any], *, with_llm: bool = False):
    from zeroh import ZeroHPlugin
    from zeroh.memory import MemoryStore

    db = req.get("db") or ":memory:"
    store = MemoryStore(db)
    llm = _build_llm(req.get("llm")) if with_llm else None
    return ZeroHPlugin(llm, store=store)


def handle(req: Dict[str, Any]) -> Dict[str, Any]:
    command = (req.get("command") or "").strip()

    if command == "ping":
        from zeroh import __version__

        return {"ok": True, "result": {"version": __version__}}

    if command == "remember":
        content = (req.get("content") or "").strip()
        if not content:
            return _error("No content provided to remember.")
        zh = _plugin(req)
        mem = zh.remember(content, source=req.get("source") or "user")
        return {"ok": True, "result": {"id": mem.id, "content": mem.content}}

    if command == "ingest":
        document = req.get("document") or ""
        if not document.strip():
            return _error("No document text provided to ingest.")
        zh = _plugin(req)
        mems = zh.ingest(document, source=req.get("source") or "document")
        return {"ok": True, "result": {"chunks": len(mems)}}

    if command == "recall":
        query = (req.get("query") or "").strip()
        if not query:
            return _error("No query provided to recall.")
        zh = _plugin(req)
        results = zh.recall(query, top_k=req.get("topK"))
        return {
            "ok": True,
            "result": {
                "results": [
                    {
                        "content": r.memory.content,
                        "source": r.memory.source,
                        "score": round(r.score, 4),
                    }
                    for r in results
                ]
            },
        }

    if command == "verify":
        candidate = (req.get("text") or "").strip()
        if not candidate:
            return _error("No text provided to verify.")
        zh = _plugin(req)
        return {"ok": True, "result": zh.verify(candidate).to_dict()}

    if command == "answer":
        query = (req.get("query") or "").strip()
        if not query:
            return _error("No query provided.")
        zh = _plugin(req)
        return {
            "ok": True,
            "result": zh.answer_from_memory(query, top_k=req.get("topK")).to_dict(),
        }

    if command == "complete":
        query = (req.get("query") or "").strip()
        if not query:
            return _error("No query provided.")
        zh = _plugin(req, with_llm=True)
        if zh.llm is None:
            return _error(
                "No LLM configured. Set 'zeroh.llm.*' in settings, or use the "
                "'Answer From Memory' command for LLM-free operation.",
                code="no-llm",
            )
        return {
            "ok": True,
            "result": zh.complete(query, top_k=req.get("topK")).to_dict(),
        }

    return _error(f"Unknown command: {command!r}", code="unknown-command")


def _read_request() -> Dict[str, Any]:
    raw = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    if not raw or not raw.strip():
        raise ValueError("Empty request.")
    return json.loads(raw)


def main() -> int:
    try:
        req = _read_request()
    except Exception as exc:  # noqa: BLE001 - report parse errors to the caller
        json.dump(_error(f"Invalid request: {exc}", code="bad-request"), sys.stdout)
        return 2

    try:
        response = handle(req)
    except ModuleNotFoundError as exc:
        response = _error(
            "The 'zeroh' Python package is not installed for the configured "
            f"interpreter ({exc.name!r} missing). Install it with "
            "'pip install zeroh' and set 'zeroh.pythonPath' if needed.",
            code="missing-package",
        )
    except Exception as exc:  # noqa: BLE001 - surface any runtime error as JSON
        response = _error(str(exc))

    json.dump(response, sys.stdout)
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
