"""Hallucination detection built on top of grounding verification.

The :class:`HallucinationDetector` turns per-claim verification into an overall
risk assessment for a candidate answer, and can flag or strip the specific
sentences that are not backed by memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..grounding import Verifier
from ..models import Claim


@dataclass
class HallucinationReport:
    """Summary of how well a candidate answer is grounded in memory."""

    risk: float                       # Overall hallucination risk in [0, 1].
    grounded_ratio: float             # Fraction of claims that were supported.
    claims: List[Claim] = field(default_factory=list)

    @property
    def unsupported_claims(self) -> List[Claim]:
        return [c for c in self.claims if not c.supported]

    @property
    def supported_claims(self) -> List[Claim]:
        return [c for c in self.claims if c.supported]

    @property
    def is_hallucinating(self) -> bool:
        """True when at least one claim lacks support in memory."""
        return bool(self.unsupported_claims)


class HallucinationDetector:
    """Assesses whether a candidate answer contains unsupported claims.

    Args:
        verifier: The grounding verifier to use.
        risk_threshold: Risk above which an answer should be treated as a likely
            hallucination by callers.
    """

    def __init__(self, verifier: Verifier, risk_threshold: float = 0.5) -> None:
        self.verifier = verifier
        self.risk_threshold = risk_threshold

    def analyze(self, candidate_answer: str) -> HallucinationReport:
        """Verify every claim in ``candidate_answer`` and score overall risk."""
        claims = self.verifier.verify_text(candidate_answer)
        if not claims:
            # Nothing verifiable was said -> maximally risky (e.g. empty/garbled).
            return HallucinationReport(risk=1.0, grounded_ratio=0.0, claims=[])

        supported = sum(1 for c in claims if c.supported)
        grounded_ratio = supported / len(claims)
        # Risk is driven both by how many claims are unsupported and by how weak
        # the support is for the ones that did pass.
        avg_support = sum(c.support_score for c in claims) / len(claims)
        risk = round(1.0 - (0.6 * grounded_ratio + 0.4 * avg_support), 4)
        return HallucinationReport(
            risk=max(0.0, min(1.0, risk)),
            grounded_ratio=round(grounded_ratio, 4),
            claims=claims,
        )

    def filter_supported(self, candidate_answer: str) -> str:
        """Return only the sentences of ``candidate_answer`` backed by memory."""
        report = self.analyze(candidate_answer)
        return " ".join(c.text for c in report.supported_claims)
