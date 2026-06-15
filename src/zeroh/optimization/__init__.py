"""Optimization utilities for zeroH memory and retrieval.

This package provides performance-oriented tools that complement the core
memory and retrieval modules:

* :mod:`~zeroh.optimization.cache` — LRU query result caching
* :mod:`~zeroh.optimization.batch` — Efficient bulk memory operations
* :mod:`~zeroh.optimization.metrics` — Memory usage analysis and recommendations
"""
from .batch import BatchProcessor
from .cache import QueryCache
from .metrics import MemoryMetrics

__all__ = ["QueryCache", "BatchProcessor", "MemoryMetrics"]
