"""Command-line interface for zeroH.

A thin shell over :class:`~zeroh.plugin.ZeroHPlugin` so the grounded-memory layer
is usable straight from a terminal — handy for scripting, quick experiments, and
inspecting a durable store::

    zeroh remember "The capital of France is Paris." --source kb
    zeroh ingest handbook.md --source handbook
    zeroh recall "capital of France"
    zeroh ask "What is the capital of France?"     # extractive, no LLM needed
    zeroh verify "The capital of France is Berlin." # guardrail fact-check
    zeroh stats
    zeroh export > backup.jsonl

The store defaults to ``$ZEROH_DB`` or ``./zeroh.db`` so memories persist between
invocations; pass ``--db`` to override (use ``:memory:`` for an ephemeral run).
Add ``--json`` to any command for machine-readable output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __version__
from .memory import MemoryStore
from .plugin import ZeroHPlugin

DEFAULT_DB = os.environ.get("ZEROH_DB", "zeroh.db")


def _plugin(db: str) -> ZeroHPlugin:
    return ZeroHPlugin(store=MemoryStore(db))


def _print(data, as_json: bool, human: str) -> None:
    if as_json:
        json.dump(data, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        print(human)


def _cmd_remember(args: argparse.Namespace) -> int:
    zh = _plugin(args.db)
    mem = zh.remember(
        args.text, source=args.source, confidence=args.confidence, dedupe=args.dedupe
    )
    _print(
        {"id": mem.id, "content": mem.content, "source": mem.source},
        args.json,
        f"remembered [{mem.source}] {mem.id[:8]}: {mem.content}",
    )
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    text = sys.stdin.read() if args.path in (None, "-") else _read_file(args.path)
    if not text.strip():
        _fail("Nothing to ingest (empty input).", args.json)
        return 1
    zh = _plugin(args.db)
    mems = zh.ingest(
        text,
        source=args.source,
        max_chars=args.max_chars,
        overlap_sentences=args.overlap,
        dedupe=args.dedupe,
    )
    _print(
        {"chunks": len(mems), "source": args.source},
        args.json,
        f"ingested {len(mems)} chunk(s) from "
        f"{'stdin' if args.path in (None, '-') else args.path} into [{args.source}]",
    )
    return 0


def _cmd_recall(args: argparse.Namespace) -> int:
    zh = _plugin(args.db)
    results = zh.recall(args.query, top_k=args.top_k, source=args.source)
    payload = [
        {"score": round(r.score, 4), "source": r.memory.source, "content": r.memory.content}
        for r in results
    ]
    human = "\n".join(
        f"{r.score:6.3f}  [{r.memory.source}] {r.memory.content}" for r in results
    ) or "(no matches)"
    _print(payload, args.json, human)
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    zh = _plugin(args.db)
    ans = zh.answer_from_memory(args.query, top_k=args.top_k, source=args.source)
    _print(ans.to_dict(), args.json, _format_answer(ans))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    zh = _plugin(args.db)
    ans = zh.verify(args.text)
    _print(ans.to_dict(), args.json, _format_answer(ans))
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    zh = _plugin(args.db)
    stats = zh.stats()
    by_source = "\n".join(f"  {s}: {n}" for s, n in stats["by_source"].items())
    human = (
        f"active: {stats['active']}  inactive: {stats['inactive']}  "
        f"total: {stats['total']}\nby source:\n{by_source or '  (none)'}"
    )
    _print(stats, args.json, human)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    store = MemoryStore(args.db)
    sys.stdout.write(store.export_jsonl(include_inactive=args.all))
    sys.stdout.write("\n")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    data = sys.stdin.read() if args.path in (None, "-") else _read_file(args.path)
    zh = _plugin(args.db)
    count = zh.load(data)
    _print({"imported": count}, args.json, f"imported {count} memories")
    return 0


def _format_answer(ans) -> str:
    head = ans.text
    tag = "abstained" if ans.abstained else f"grounded (confidence {ans.confidence})"
    sources = ", ".join(sorted({c.source for c in ans.citations}))
    foot = f"\n  -> {tag}" + (f"; sources: {sources}" if sources else "")
    return head + foot


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _fail(message: str, as_json: bool) -> None:
    if as_json:
        json.dump({"error": message}, sys.stdout)
        sys.stdout.write("\n")
    else:
        print(f"error: {message}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zeroh",
        description="Grounded, anti-hallucination memory for AI agents.",
    )
    parser.add_argument("--version", action="version", version=f"zeroh {__version__}")
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"SQLite store path (default: {DEFAULT_DB}; use ':memory:' for ephemeral).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("remember", help="Store a single fact.")
    p.add_argument("text")
    p.add_argument("--source", default="user")
    p.add_argument("--confidence", type=float, default=1.0)
    p.add_argument("--dedupe", action="store_true", help="Skip if content exists.")
    p.set_defaults(func=_cmd_remember)

    p = sub.add_parser("ingest", help="Chunk and store a document (file or stdin).")
    p.add_argument("path", nargs="?", default="-", help="File path, or '-' for stdin.")
    p.add_argument("--source", default="document")
    p.add_argument("--max-chars", dest="max_chars", type=int, default=400)
    p.add_argument("--overlap", type=int, default=1)
    p.add_argument("--dedupe", action="store_true")
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("recall", help="Retrieve relevant memories for a query.")
    p.add_argument("query")
    p.add_argument("--top-k", dest="top_k", type=int, default=5)
    p.add_argument("--source", default=None, help="Restrict to one source.")
    p.set_defaults(func=_cmd_recall)

    p = sub.add_parser("ask", help="Extractive grounded answer (no LLM required).")
    p.add_argument("query")
    p.add_argument("--top-k", dest="top_k", type=int, default=5)
    p.add_argument("--source", default=None)
    p.set_defaults(func=_cmd_ask)

    p = sub.add_parser("verify", help="Fact-check a statement against memory.")
    p.add_argument("text")
    p.set_defaults(func=_cmd_verify)

    p = sub.add_parser("stats", help="Show memory statistics.")
    p.set_defaults(func=_cmd_stats)

    p = sub.add_parser("export", help="Export memories as JSONL to stdout.")
    p.add_argument("--all", action="store_true", help="Include inactive memories.")
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("import", help="Import memories from JSONL (file or stdin).")
    p.add_argument("path", nargs="?", default="-")
    p.set_defaults(func=_cmd_import)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        _fail(f"file not found: {exc.filename}", getattr(args, "json", False))
        return 1
    except Exception as exc:  # noqa: BLE001 - surface CLI errors cleanly
        _fail(str(exc), getattr(args, "json", False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
