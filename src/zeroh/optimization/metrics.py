"""Memory usage metrics and optimization recommendations.

This module analyzes the state of a memory store and provides actionable
metrics and recommendations for improving performance, reducing redundancy,
and optimizing token usage.

Use this as a diagnostic tool: run periodically to identify opportunities
for consolidation, deduplication, or configuration tuning.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..memory.store import MemoryStore
from ..models import Memory
from ..text import estimate_tokens, semantic_similarity_quick, tokenize


@dataclass
class Recommendation:
    """An actionable optimization recommendation."""

    category: str  # "deduplication", "consolidation", "config", "quality"
    priority: str  # "high", "medium", "low"
    title: str
    description: str
    action: str  # What to do about it
    impact: str  # Expected benefit


@dataclass
class MemoryHealthReport:
    """Comprehensive health report for a memory store."""

    total_memories: int = 0
    active_memories: int = 0
    inactive_memories: int = 0
    total_tokens: int = 0
    avg_tokens_per_memory: float = 0.0
    total_characters: int = 0
    sources: Dict[str, int] = field(default_factory=dict)
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    age_distribution: Dict[str, int] = field(default_factory=dict)
    estimated_redundancy: float = 0.0
    recommendations: List[Recommendation] = field(default_factory=list)

    @property
    def health_score(self) -> float:
        """Overall health score from 0 (poor) to 1 (excellent).

        Considers redundancy, confidence, and content quality.
        """
        score = 1.0
        # Penalize high redundancy
        score -= self.estimated_redundancy * 0.4
        # Penalize if too many low-confidence memories
        low_conf = self.confidence_distribution.get("low", 0)
        if self.active_memories > 0:
            score -= (low_conf / self.active_memories) * 0.3
        # Penalize extreme token counts (only if we have memories)
        if self.active_memories > 0:
            if self.avg_tokens_per_memory > 200:
                score -= 0.1
            elif self.avg_tokens_per_memory < 5:
                score -= 0.2
        return max(0.0, min(1.0, score))


class MemoryMetrics:
    """Analyzes memory store health and provides optimization recommendations.

    Args:
        store: The MemoryStore to analyze.
        redundancy_sample_size: Number of memories to sample for redundancy
            estimation (full pairwise comparison is O(n²)).
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        redundancy_sample_size: int = 100,
    ) -> None:
        self.store = store
        self.redundancy_sample_size = redundancy_sample_size

    def analyze(self) -> MemoryHealthReport:
        """Run a full analysis of the memory store.

        Returns a MemoryHealthReport with metrics and recommendations.
        """
        report = MemoryHealthReport()
        all_memories = self.store.all()
        stats = self.store.stats()

        report.active_memories = stats.get("active", 0)
        report.inactive_memories = stats.get("inactive", 0)
        report.total_memories = report.active_memories + report.inactive_memories
        report.sources = stats.get("by_source", stats.get("sources", {}))

        if not all_memories:
            return report

        # Token and character analysis
        total_tokens = 0
        total_chars = 0
        for mem in all_memories:
            tokens = estimate_tokens(mem.content)
            total_tokens += tokens
            total_chars += len(mem.content)

        report.total_tokens = total_tokens
        report.total_characters = total_chars
        report.avg_tokens_per_memory = total_tokens / len(all_memories)

        # Confidence distribution
        report.confidence_distribution = self._confidence_distribution(all_memories)

        # Age distribution
        report.age_distribution = self._age_distribution(all_memories)

        # Redundancy estimation (sample-based for efficiency)
        report.estimated_redundancy = self._estimate_redundancy(all_memories)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report, all_memories)

        return report

    def token_budget_analysis(
        self, query_budget: int = 500
    ) -> Dict[str, Any]:
        """Analyze how well memories fit within a typical query token budget.

        Args:
            query_budget: Typical max_context_tokens budget to simulate.

        Returns:
            Analysis of how many memories fit, what's wasted, etc.
        """
        all_memories = self.store.all()
        if not all_memories:
            return {"fits_all": True, "total_tokens": 0}

        token_counts = [estimate_tokens(m.content) for m in all_memories]
        total = sum(token_counts)
        avg = total / len(token_counts)

        # How many average memories fit in budget?
        fits_in_budget = int(query_budget / avg) if avg > 0 else 0

        return {
            "total_memories": len(all_memories),
            "total_tokens": total,
            "avg_tokens_per_memory": round(avg, 1),
            "max_tokens": max(token_counts),
            "min_tokens": min(token_counts),
            "memories_fitting_budget": fits_in_budget,
            "budget_utilization": min(1.0, fits_in_budget / len(all_memories))
            if all_memories else 1.0,
        }

    def source_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Analyze memory characteristics per source.

        Helps identify which sources produce high or low quality memories.
        """
        all_memories = self.store.all()
        by_source: Dict[str, List[Memory]] = {}
        for mem in all_memories:
            by_source.setdefault(mem.source, []).append(mem)

        result = {}
        for source, memories in by_source.items():
            tokens = [estimate_tokens(m.content) for m in memories]
            confidences = [m.confidence for m in memories]
            result[source] = {
                "count": len(memories),
                "avg_tokens": round(sum(tokens) / len(tokens), 1),
                "avg_confidence": round(sum(confidences) / len(confidences), 3),
                "total_tokens": sum(tokens),
            }
        return result

    def _confidence_distribution(
        self, memories: List[Memory]
    ) -> Dict[str, int]:
        """Bucket memories by confidence level."""
        dist = {"high": 0, "medium": 0, "low": 0}
        for mem in memories:
            if mem.confidence >= 0.8:
                dist["high"] += 1
            elif mem.confidence >= 0.5:
                dist["medium"] += 1
            else:
                dist["low"] += 1
        return dist

    def _age_distribution(self, memories: List[Memory]) -> Dict[str, int]:
        """Bucket memories by age."""
        now = time.time()
        dist = {"recent": 0, "moderate": 0, "old": 0}
        for mem in memories:
            age_days = (now - mem.created_at) / 86400
            if age_days < 7:
                dist["recent"] += 1
            elif age_days < 30:
                dist["moderate"] += 1
            else:
                dist["old"] += 1
        return dist

    def _estimate_redundancy(self, memories: List[Memory]) -> float:
        """Estimate redundancy ratio using sampling."""
        if len(memories) < 2:
            return 0.0

        # Sample for efficiency
        sample = memories[:self.redundancy_sample_size]
        redundant = 0
        total_comparisons = 0

        for i in range(len(sample)):
            for j in range(i + 1, min(i + 10, len(sample))):
                sim = semantic_similarity_quick(
                    sample[i].content, sample[j].content
                )
                total_comparisons += 1
                if sim >= 0.75:
                    redundant += 1

        if total_comparisons == 0:
            return 0.0
        return redundant / total_comparisons

    def _generate_recommendations(
        self,
        report: MemoryHealthReport,
        memories: List[Memory],
    ) -> List[Recommendation]:
        """Generate actionable recommendations based on metrics."""
        recs: List[Recommendation] = []

        # High redundancy
        if report.estimated_redundancy > 0.2:
            recs.append(Recommendation(
                category="deduplication",
                priority="high",
                title="High memory redundancy detected",
                description=f"Estimated {report.estimated_redundancy:.0%} redundancy "
                    "among sampled memories.",
                action="Run MemoryCompressor.analyze() to identify groups, then "
                    "merge or deactivate duplicates.",
                impact="Improved retrieval precision and reduced token usage.",
            ))

        # Low confidence memories
        low_conf = report.confidence_distribution.get("low", 0)
        if report.active_memories > 0 and low_conf / report.active_memories > 0.3:
            recs.append(Recommendation(
                category="consolidation",
                priority="medium",
                title="Many low-confidence memories",
                description=f"{low_conf} memories ({low_conf/report.active_memories:.0%}) "
                    "have confidence below 0.5.",
                action="Use plugin.consolidate() to prune stale low-confidence "
                    "memories, or review and upgrade their confidence.",
                impact="Better grounding quality and less noise in retrieval.",
            ))

        # Very large memories
        oversized = sum(
            1 for m in memories if estimate_tokens(m.content) > 200
        )
        if oversized > 0:
            recs.append(Recommendation(
                category="quality",
                priority="medium",
                title="Oversized memories detected",
                description=f"{oversized} memories exceed 200 tokens each.",
                action="Consider re-ingesting with smaller chunk_text(max_chars) "
                    "for better retrieval precision.",
                impact="More precise retrieval with better token budget utilization.",
            ))

        # Old memories dominating
        old = report.age_distribution.get("old", 0)
        if report.active_memories > 0 and old / report.active_memories > 0.8:
            recs.append(Recommendation(
                category="consolidation",
                priority="low",
                title="Store dominated by old memories",
                description=f"{old}/{report.active_memories} memories are over 30 days old.",
                action="Review old memories for relevance. Enable recency_weight "
                    "in the Retriever to down-rank stale content.",
                impact="Fresher context provided to the LLM.",
            ))

        # Token budget efficiency
        if report.avg_tokens_per_memory > 150:
            recs.append(Recommendation(
                category="config",
                priority="medium",
                title="High average token count per memory",
                description=f"Average {report.avg_tokens_per_memory:.0f} tokens/memory "
                    "limits how many memories fit in context.",
                action="Use smaller max_chars in chunk_text() during ingestion, "
                    "or set max_context_tokens to enforce a budget.",
                impact="More diverse context within the same token budget.",
            ))

        return recs
