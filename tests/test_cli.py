"""Tests for the zeroh command-line interface."""
import json
import os

import pytest

from zeroh.cli import main


@pytest.fixture()
def db(tmp_path):
    return os.path.join(tmp_path, "cli.db")


def _run(capsys, *args):
    code = main(list(args))
    out = capsys.readouterr().out
    return code, out


def test_remember_and_recall(capsys, db):
    code, out = _run(capsys, "--db", db, "remember", "The capital of France is Paris.")
    assert code == 0 and "remembered" in out

    code, out = _run(capsys, "--db", db, "recall", "capital of France")
    assert code == 0 and "Paris" in out


def test_json_output(capsys, db):
    _run(capsys, "--db", db, "remember", "Water boils at 100 C.", "--source", "sci")
    code, out = _run(capsys, "--db", db, "--json", "stats")
    assert code == 0
    data = json.loads(out)
    assert data["active"] == 1
    assert data["by_source"]["sci"] == 1


def test_ask_extractive_answer(capsys, db):
    _run(capsys, "--db", db, "remember", "The capital of France is Paris.")
    code, out = _run(capsys, "--db", db, "ask", "What is the capital of France?")
    assert code == 0
    assert "Paris" in out
    assert "grounded" in out


def test_ask_abstains_when_unknown(capsys, db):
    _run(capsys, "--db", db, "remember", "The capital of France is Paris.")
    code, out = _run(capsys, "--db", db, "ask", "Who won the 2049 World Series?")
    assert code == 0
    assert "abstained" in out


def test_verify_rejects_hallucination(capsys, db):
    _run(capsys, "--db", db, "remember", "The capital of France is Paris.")
    code, out = _run(capsys, "--db", db, "verify", "The capital of France is Berlin.")
    assert code == 0
    assert "abstained" in out


def test_ingest_from_file(capsys, db, tmp_path):
    doc = os.path.join(tmp_path, "doc.txt")
    with open(doc, "w", encoding="utf-8") as fh:
        fh.write("Refunds are accepted within 30 days. Shipping takes five days.")
    code, out = _run(capsys, "--db", db, "ingest", doc, "--source", "policy")
    assert code == 0 and "ingested" in out

    code, out = _run(capsys, "--db", db, "recall", "refunds")
    assert "30 days" in out


def test_export_import_roundtrip(capsys, db, tmp_path):
    _run(capsys, "--db", db, "remember", "Fact one.", "--source", "s")
    _run(capsys, "--db", db, "remember", "Fact two.", "--source", "s")
    _, dump = _run(capsys, "--db", db, "export")
    assert dump.strip()

    dumpfile = os.path.join(tmp_path, "dump.jsonl")
    with open(dumpfile, "w", encoding="utf-8") as fh:
        fh.write(dump)

    db2 = os.path.join(tmp_path, "cli2.db")
    code, out = _run(capsys, "--db", db2, "import", dumpfile)
    assert code == 0
    code, out = _run(capsys, "--db", db2, "--json", "stats")
    assert json.loads(out)["active"] == 2


def test_dedupe_flag(capsys, db):
    _run(capsys, "--db", db, "remember", "Same.", "--dedupe")
    _run(capsys, "--db", db, "remember", "Same.", "--dedupe")
    _, out = _run(capsys, "--db", db, "--json", "stats")
    assert json.loads(out)["active"] == 1


def test_missing_file_reports_error(capsys, db):
    code, out = _run(capsys, "--db", db, "ingest", "/nonexistent/path.txt")
    assert code == 1
