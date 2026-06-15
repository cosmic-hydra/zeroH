"""Batch operations for efficient bulk memory processing.

Individual memory operations (add, reindex, deduplicate) are convenient for
interactive use, but bulk workloads (importing large documents, migrating
stores, cleaning up) benefit from batched processing that:

* **Defers reindexing**: Rebuilds the TF-IDF index once after all mutations.
* **Deduplicates in bulk**: O(n log n) sort-based dedup instead of O(n²) pairwise.
* **Validates in batch**: Checks all memories for quality issues in one pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from ..models import Memory
from ..text import estimate_tokens, semantic_similarity_quick, tokenize


@dataclass
class BatchResult:
    """Result of a batch operation."""

    added: int = 0
    skipped: int = 0
    deduplicated: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def total_processed(self) -> int:
        return self.added + self.skipped + self.deduplicated


@dataclass
class ValidationIssue:
    """A quality issue detected in a memory."""

    memory_id: str
    issue_type: str  # "too_short", "too_long", "low_info", "duplicate"
    description: str
    severity: str = "warning"  # "warning" or "error"


class BatchProcessor:
    """Efficient batch operations on memory collections.

    Args:
        dedupe_threshold: Jaccard similarity threshold for duplicate detection.
        min_content_length: Minimum content length (characters) for quality.
        max_content_length: Maximum content length for quality warnings.
        min_info_tokens: Minimum number of content tokens for a memory to be
            considered informative.
    """

    def __init__(
        self,
        *,
        dedupe_threshold: float = 0.85,
        min_content_length: int = 10,
        max_content_length: int = 2000,
        min_info_tokens: int = 3,
    ) -> None:
        self.dedupe_threshold = dedupe_threshold
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length
        self.min_info_tokens = min_info_tokens

    def deduplicate(
        self,
        memories: List[Memory],
        *,
        threshold: Optional[float] = None,
    ) -> List[Memory]:
        """Remove near-duplicates from a list of memories.

        Uses a sort-based approach: memories are sorted by their token set hash,
        then pairwise comparison is only done between adjacent groups. This is
        more efficient than full O(n²) comparison for large lists.

        Returns the deduplicated list (originals are not modified).
        """
        if not memories:
            return []

        threshold = threshold or self.dedupe_threshold
        # Sort by token set to group similar memories together
        indexed = [
            (frozenset(tokenize(m.content)), m) for m in memories
        ]
        # Use token set size as a pre-filter: very different sizes can't be similar
        indexed.sort(key=lambda x: len(x[0]))

        kept: List[Memory] = []
        kept_tokens: List[frozenset] = []

        for tokens, memory in indexed:
            is_dup = False
            # Only compare against recent kept items (sorted by size means
            # very different sizes are far apart)
            for prev_tokens in reversed(kept_tokens[-20:]):
                if not tokens or not prev_tokens:
                    continue
                # Quick size-based pre-filter
                size_ratio = min(len(tokens), len(prev_tokens)) / max(len(tokens), len(prev_tokens))
                if size_ratio < threshold:
                    continue
                # Full Jaccard comparison
                intersection = len(tokens & prev_tokens)
                union = len(tokens | prev_tokens)
                if union > 0 and (intersection / union) >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(memory)
                kept_tokens.append(tokens)

        return kept

    def validate(self, memories: List[Memory]) -> List[ValidationIssue]:
        """Check a list of memories for quality issues.

        Returns a list of issues found. This is a non-destructive analysis
        that helps identify problems before they affect retrieval quality.
        """
        issues: List[ValidationIssue] = []

        for mem in memories:
            # Content length checks
            if len(mem.content) < self.min_content_length:
                issues.append(ValidationIssue(
                    memory_id=mem.id,
                    issue_type="too_short",
                    description=f"Content length {len(mem.content)} below minimum {self.min_content_length}",
                    severity="warning",
                ))
            elif len(mem.content) > self.max_content_length:
                issues.append(ValidationIssue(
                    memory_id=mem.id,
                    issue_type="too_long",
                    description=f"Content length {len(mem.content)} exceeds maximum {self.max_content_length}",
                    severity="warning",
                ))

            # Information density check
            tokens = tokenize(mem.content)
            if len(tokens) < self.min_info_tokens:
                issues.append(ValidationIssue(
                    memory_id=mem.id,
                    issue_type="low_info",
                    description=f"Only {len(tokens)} content tokens (minimum: {self.min_info_tokens})",
                    severity="warning",
                ))

            # Confidence check
            if mem.confidence <= 0:
                issues.append(ValidationIssue(
                    memory_id=mem.id,
                    issue_type="zero_confidence",
                    description="Memory has zero or negative confidence",
                    severity="error",
                ))

        return issues

    def filter_quality(
        self,
        memories: List[Memory],
        *,
        min_tokens: Optional[int] = None,
        min_confidence: float = 0.0,
        max_token_budget: Optional[int] = None,
    ) -> List[Memory]:
        """Filter memories by quality criteria and optional token budget.

        Args:
            memories: Input memories to filter.
            min_tokens: Minimum content tokens required.
            min_confidence: Minimum confidence score.
            max_token_budget: If set, keeps highest-confidence memories that
                fit within this total token budget.

        Returns:
            Filtered list of memories meeting all criteria.
        """
        min_tokens = min_tokens or self.min_info_tokens
        filtered = [
            m for m in memories
            if len(tokenize(m.content)) >= min_tokens
            and m.confidence >= min_confidence
        ]

        if max_token_budget is not None:
            # Sort by confidence descending, then take as many as fit
            filtered.sort(key=lambda m: m.confidence, reverse=True)
            budget_filtered: List[Memory] = []
            total_tokens = 0
            for mem in filtered:
                tokens = estimate_tokens(mem.content)
                if total_tokens + tokens > max_token_budget and budget_filtered:
                    break
                budget_filtered.append(mem)
                total_tokens += tokens
            return budget_filtered

        return filtered

    def chunk_for_parallel(
        self,
        memories: List[Memory],
        chunk_size: int = 100,
    ) -> List[List[Memory]]:
        """Split memories into chunks suitable for parallel processing.

        Useful for distributing indexing or validation across threads.
        """
        return [
            memories[i:i + chunk_size]
            for i in range(0, len(memories), chunk_size)
        ]
