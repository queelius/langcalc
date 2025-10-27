"""
Data structures for efficient language modeling.

This module provides suffix arrays, incremental data structures,
and utilities for corpus indexing.
"""

from langcalc.data.suffix_array import SuffixArray

__all__ = [
    "SuffixArray",
]

# Incremental suffix array will be available after migration
try:
    from langcalc.data.incremental import IncrementalSuffixArray
    __all__.append("IncrementalSuffixArray")
except ImportError:
    pass
