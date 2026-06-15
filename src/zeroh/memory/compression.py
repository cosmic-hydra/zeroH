"""Memory content compression and deduplication utilities.

For large memory stores, storage and retrieval efficiency degrades as redundant
or overlapping content accumulates. This module provides utilities for:

* **Prefix deduplication**: Identifies memories sharing long common prefixes and
  stores them more compactly in memory (reduces working set size).
* **Content summarization**: Merges multiple related memories into a single
  summary memory, reducing the number of items the retriever must score.
* **Redundancy detection**: Finds groups of memories with high pairwise overlap.

These utilities are designed to be run as periodic maintenance operations (not
on every query) and respect the append-only audit trail by creating new summary
memories rather than modifying existing ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..models import Memory
from ..text import semantic_similarity_quick, split_sentences, tokenize


@dataclass
class RedundancyGroup:
    """A group of memories with high pairwise similarity."""

    representative_id: str
    member_ids: List[str]
    similarity: float
    content_preview: str = ""


@dataclass
class CompressionReport:
    """Results from a compression analysis."""

    total_memories: int = 0
    redundancy_groups: List[RedundancyGroup] = field(default_factory=list)
    estimated_savings: int = 0  # Characters saveable
    unique_memories: int = 0

    @property
    def redundancy_ratio(self) -> float:
        """Fraction of memories that are redundant (in a group with > 1 member)."""
        if self.total_memories == 0:
            return 0.0
        redundant = sum(
            len(g.member_ids) for g in self.redundancy_groups
        )
        return redundant / self.total_memories


class MemoryCompressor:
    """Analyzes and compresses memory stores for efficiency.

    Args:
        similarity_threshold: Jaccard similarity above which memories are
            considered redundant (default 0.75).
        min_group_size: Minimum group size to report as redundant (default 2).
    """

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.75,
        min_group_size: int = 2,
    ) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        self.similarity_threshold = similarity_threshold
        self.min_group_size = min_group_size

    def analyze(self, memories: List[Memory]) -> CompressionReport:
        """Identify redundancy groups and estimate compression potential.

        This is a maintenance operation with O(n²) pairwise comparison cost.
        For large stores, consider running on a sample or by source.
        """
        report = CompressionReport(total_memories=len(memories))
        if len(memories) < 2:
            report.unique_memories = len(memories)
            return report

        # Build groups using union-find approach
        assigned: Set[str] = set()
        groups: List[RedundancyGroup] = []

        for i, mem_a in enumerate(memories):
            if mem_a.id in assigned:
                continue
            group_members = [mem_a.id]
            for j in range(i + 1, len(memories)):
                mem_b = memories[j]
                if mem_b.id in assigned:
                    continue
                sim = semantic_similarity_quick(mem_a.content, mem_b.content)
                if sim >= self.similarity_threshold:
                    group_members.append(mem_b.id)
                    assigned.add(mem_b.id)

            if len(group_members) >= self.min_group_size:
                assigned.add(mem_a.id)
                groups.append(RedundancyGroup(
                    representative_id=mem_a.id,
                    member_ids=group_members[1:],
                    similarity=self.similarity_threshold,
                    content_preview=mem_a.content[:80],
                ))
                # Estimate savings: all members except representative
                savings = sum(
                    len(memories[idx].content)
                    for idx, m in enumerate(memories)
                    if m.id in group_members[1:]
                )
                report.estimated_savings += savings

        report.redundancy_groups = groups
        report.unique_memories = len(memories) - sum(
            len(g.member_ids) for g in groups
        )
        return report

    def merge_group(
        self,
        memories: List[Memory],
        *,
        source: str = "compression",
    ) -> Memory:
        """Merge a group of similar memories into a single representative.

        Picks the longest memory as the representative content (most complete).
        The returned Memory is new and should be added to the store while
        deactivating the originals (caller's responsibility).
        """
        if not memories:
            raise ValueError("Cannot merge empty group")
        # Pick the longest as most informative
        representative = max(memories, key=lambda m: len(m.content))
        return Memory(
            content=representative.content,
            source=source,
            confidence=max(m.confidence for m in memories),
            metadata={
                "merged_from": [m.id for m in memories],
                "original_source": representative.source,
            },
        )

    def find_prefix_duplicates(
        self,
        memories: List[Memory],
        *,
        min_prefix_length: int = 50,
    ) -> List[Tuple[str, List[str]]]:
        """Find memories sharing long common prefixes.

        Returns list of (prefix, [memory_ids]) tuples where the shared prefix
        is at least ``min_prefix_length`` characters. These indicate copied or
        templated content that could be consolidated.
        """
        prefix_groups: Dict[str, List[str]] = {}
        for mem in memories:
            # Use first min_prefix_length chars as grouping key
            if len(mem.content) >= min_prefix_length:
                prefix = mem.content[:min_prefix_length]
                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []
                prefix_groups[prefix].append(mem.id)

        return [
            (prefix, ids) for prefix, ids in prefix_groups.items()
            if len(ids) >= 2
        ]

    def extract_unique_facts(self, memories: List[Memory]) -> List[str]:
        """Extract the set of unique factual sentences across all memories.

        Useful for consolidating a group of overlapping memories into a minimal
        set of distinct facts (deduplicating at sentence level).
        """
        seen_tokens: Set[frozenset] = set()
        unique_facts: List[str] = []

        for mem in memories:
            for sentence in split_sentences(mem.content):
                tokens = frozenset(tokenize(sentence))
                if tokens and tokens not in seen_tokens:
                    seen_tokens.add(tokens)
                    unique_facts.append(sentence)

        return unique_facts
