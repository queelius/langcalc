"""
Suffix array implementation for efficient n-gram operations.

This module provides a clean, efficient suffix array data structure
for fast pattern matching and n-gram retrieval.
"""

from typing import List, Tuple, Optional, Iterator
import numpy as np
from bisect import bisect_left, bisect_right


class SuffixArray:
    """
    Efficient suffix array for n-gram operations.

    Supports:
    - Fast longest suffix matching
    - Pattern search in O(m log n) time
    - Range queries for all occurrences
    - Memory-efficient storage
    """

    def __init__(self, tokens: List[int]):
        """
        Build a suffix array from token sequence.

        Args:
            tokens: List of token ids
        """
        self.tokens = tokens
        self.n = len(tokens)
        self.suffix_array = self._build_suffix_array()
        self.lcp_array = self._build_lcp_array()

    def _build_suffix_array(self) -> np.ndarray:
        """Build suffix array using efficient sorting."""
        # Create list of (suffix, original_index) tuples
        suffixes = [(self.tokens[i:], i) for i in range(self.n)]

        # Sort suffixes lexicographically
        suffixes.sort(key=lambda x: x[0])

        # Extract the indices
        return np.array([idx for _, idx in suffixes], dtype=np.int32)

    def _build_lcp_array(self) -> np.ndarray:
        """
        Build Longest Common Prefix array.

        LCP[i] = length of longest common prefix between
                 suffix_array[i] and suffix_array[i-1]
        """
        lcp = np.zeros(self.n, dtype=np.int32)
        rank = np.zeros(self.n, dtype=np.int32)

        # Build rank array (inverse of suffix array)
        for i in range(self.n):
            rank[self.suffix_array[i]] = i

        k = 0
        for i in range(self.n):
            if rank[i] == self.n - 1:
                k = 0
                continue

            j = self.suffix_array[rank[i] + 1]

            # Count matching prefix
            while i + k < self.n and j + k < self.n and \
                  self.tokens[i + k] == self.tokens[j + k]:
                k += 1

            lcp[rank[i]] = k

            if k > 0:
                k -= 1

        return lcp

    def find_longest_suffix(self, query: List[int]) -> Tuple[int, int]:
        """
        Find the longest suffix of query that appears in the text.

        Args:
            query: Query token sequence

        Returns:
            (position, length) of the longest matching suffix
        """
        best_pos = -1
        best_len = 0

        # Try suffixes of increasing length
        for suffix_len in range(1, len(query) + 1):
            suffix = query[-suffix_len:]
            positions = self.search(suffix)

            if positions:
                best_pos = positions[0]
                best_len = suffix_len
            else:
                break

        return best_pos, best_len

    def search(self, pattern: List[int]) -> List[int]:
        """
        Search for all occurrences of pattern.

        Args:
            pattern: Pattern to search for

        Returns:
            List of starting positions where pattern occurs
        """
        if not pattern:
            return []

        # Binary search for the range of suffixes starting with pattern
        left = self._binary_search_left(pattern)
        right = self._binary_search_right(pattern)

        if left >= right:
            return []

        # Return all positions in the range
        return [self.suffix_array[i] for i in range(left, right)]

    def _binary_search_left(self, pattern: List[int]) -> int:
        """Find leftmost position where pattern could be inserted."""
        left, right = 0, self.n

        while left < right:
            mid = (left + right) // 2
            suffix_start = self.suffix_array[mid]

            # Compare pattern with suffix
            if self._compare_pattern_at(pattern, suffix_start) > 0:
                left = mid + 1
            else:
                right = mid

        return left

    def _binary_search_right(self, pattern: List[int]) -> int:
        """Find rightmost position where pattern could be inserted."""
        left, right = 0, self.n

        while left < right:
            mid = (left + right) // 2
            suffix_start = self.suffix_array[mid]

            # Compare pattern with suffix
            if self._compare_pattern_at(pattern, suffix_start, prefix_only=True) >= 0:
                left = mid + 1
            else:
                right = mid

        return left

    def _compare_pattern_at(self, pattern: List[int], pos: int,
                           prefix_only: bool = False) -> int:
        """
        Compare pattern with suffix starting at pos.

        Returns:
            -1 if pattern < suffix
            0 if pattern == suffix (or prefix match if prefix_only)
            1 if pattern > suffix
        """
        pattern_len = len(pattern)
        suffix_len = self.n - pos

        for i in range(min(pattern_len, suffix_len)):
            if pattern[i] < self.tokens[pos + i]:
                return -1
            elif pattern[i] > self.tokens[pos + i]:
                return 1

        if prefix_only and pattern_len <= suffix_len:
            return -1  # Consider pattern as smaller for right boundary

        if pattern_len < suffix_len:
            return -1
        elif pattern_len > suffix_len:
            return 1
        else:
            return 0

    def get_context(self, position: int, window: int = 10) -> Tuple[List[int], List[int]]:
        """
        Get context around a position.

        Args:
            position: Position in the text
            window: Context window size

        Returns:
            (before, after) token sequences
        """
        start = max(0, position - window)
        end = min(self.n, position + window)

        before = self.tokens[start:position]
        after = self.tokens[position:end]

        return before, after

    def ngrams(self, n: int) -> Iterator[Tuple[Tuple[int, ...], int]]:
        """
        Iterate over all n-grams with their frequencies.

        Args:
            n: Size of n-grams

        Yields:
            (ngram, frequency) tuples
        """
        ngram_counts = {}

        for i in range(self.n - n + 1):
            ngram = tuple(self.tokens[i:i + n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        for ngram, count in sorted(ngram_counts.items(), key=lambda x: -x[1]):
            yield ngram, count

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"SuffixArray(n={self.n})"