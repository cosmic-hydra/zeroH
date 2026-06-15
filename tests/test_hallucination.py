"""Tests for grounding verification and hallucination detection."""
from zeroh.grounding import Verifier
from zeroh.hallucination import HallucinationDetector
from zeroh.memory import MemoryStore
from zeroh.retrieval import Retriever


def _detector():
    store = MemoryStore(":memory:")
    store.add_text("The capital of France is Paris.")
    store.add_text("Water boils at 100 degrees Celsius at sea level.")
    retriever = Retriever(store)
    verifier = Verifier(retriever)
    return HallucinationDetector(verifier)


def test_supported_claim_is_grounded():
    det = _detector()
    report = det.analyze("The capital of France is Paris.")
    assert not report.is_hallucinating
    assert report.grounded_ratio == 1.0
    assert report.supported_claims[0].citations


def test_contradicting_claim_is_flagged():
    det = _detector()
    report = det.analyze("The capital of France is Berlin.")
    assert report.is_hallucinating
    assert report.unsupported_claims


def test_unknown_claim_is_flagged():
    det = _detector()
    report = det.analyze("The moon is made of green cheese.")
    assert report.is_hallucinating
    assert report.risk > 0.5


def test_filter_supported_strips_hallucinations():
    det = _detector()
    text = "The capital of France is Paris. The capital of France is Berlin."
    filtered = det.filter_supported(text)
    assert "Paris" in filtered
    assert "Berlin" not in filtered


def test_mixed_text_partial_grounding():
    det = _detector()
    text = "Water boils at 100 degrees Celsius at sea level. Dragons live in caves."
    report = det.analyze(text)
    assert 0.0 < report.grounded_ratio < 1.0
