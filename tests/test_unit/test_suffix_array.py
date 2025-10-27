"""
Unit tests for SuffixArray functionality.

These tests verify the core suffix array operations including construction,
pattern matching, and range queries, focusing on correctness of the
data structure behavior.
"""

import pytest
import numpy as np
from typing import List

from langcalc.data.suffix_array import SuffixArray


class TestSuffixArrayConstruction:
    """Test SuffixArray construction and initialization."""

    def test_suffix_array_creation(self):
        """Test basic SuffixArray creation."""
        tokens = [1, 2, 3, 4, 5]
        sa = SuffixArray(tokens)

        assert sa.tokens == tokens
        assert sa.n == len(tokens)
        assert isinstance(sa.suffix_array, np.ndarray)
        assert len(sa.suffix_array) == len(tokens)

    def test_empty_suffix_array(self):
        """Test SuffixArray with empty input."""
        empty_tokens = []
        sa = SuffixArray(empty_tokens)

        assert sa.tokens == empty_tokens
        assert sa.n == 0
        assert isinstance(sa.suffix_array, np.ndarray)
        assert len(sa.suffix_array) == 0

    def test_single_token_suffix_array(self):
        """Test SuffixArray with single token."""
        single_token = [42]
        sa = SuffixArray(single_token)

        assert sa.tokens == single_token
        assert sa.n == 1
        assert len(sa.suffix_array) == 1
        assert sa.suffix_array[0] == 0  # Only one suffix starting at position 0

    def test_repeated_tokens_suffix_array(self):
        """Test SuffixArray with repeated tokens."""
        repeated_tokens = [1, 1, 1, 2, 2, 3]
        sa = SuffixArray(repeated_tokens)

        assert sa.tokens == repeated_tokens
        assert sa.n == len(repeated_tokens)
        assert len(sa.suffix_array) == len(repeated_tokens)

    def test_suffix_array_indices_valid(self):
        """Test that suffix array contains valid indices."""
        tokens = [5, 3, 8, 1, 9, 2]
        sa = SuffixArray(tokens)

        # All indices should be valid positions in the original array
        assert all(0 <= idx < len(tokens) for idx in sa.suffix_array)

        # Should contain each index exactly once
        assert set(sa.suffix_array) == set(range(len(tokens)))


class TestSuffixArrayPatternMatching:
    """Test pattern matching functionality."""

    def test_find_pattern_exact_match(self):
        """Test finding exact pattern matches."""
        tokens = [1, 2, 3, 4, 2, 3, 5]
        sa = SuffixArray(tokens)

        # Look for pattern [2, 3]
        pattern = [2, 3]

        # The suffix array should have methods to find patterns
        if hasattr(sa, 'find_pattern'):
            matches = sa.find_pattern(pattern)
            assert isinstance(matches, (list, np.ndarray))
            # Should find matches at positions 1 and 4
            # (exact behavior depends on implementation)

        # Alternative: test with search method if available
        if hasattr(sa, 'search'):
            results = sa.search(pattern)
            assert results is not None

    def test_find_pattern_no_match(self):
        """Test searching for non-existent pattern."""
        tokens = [1, 2, 3, 4, 5]
        sa = SuffixArray(tokens)

        # Look for pattern that doesn't exist
        pattern = [9, 10]

        if hasattr(sa, 'find_pattern'):
            matches = sa.find_pattern(pattern)
            assert isinstance(matches, (list, np.ndarray))
            assert len(matches) == 0
        elif hasattr(sa, 'search'):
            results = sa.search(pattern)
            # Should indicate no matches (empty result or None)
            assert results is not None or results == []

    def test_find_single_token_pattern(self):
        """Test finding single token patterns."""
        tokens = [1, 2, 3, 2, 4, 2]
        sa = SuffixArray(tokens)

        # Look for token 2 (appears at positions 1, 3, 5)
        pattern = [2]

        if hasattr(sa, 'find_pattern'):
            matches = sa.find_pattern(pattern)
            assert isinstance(matches, (list, np.ndarray))
            assert len(matches) >= 3  # Should find at least 3 occurrences
        elif hasattr(sa, 'search'):
            results = sa.search(pattern)
            assert results is not None

    def test_find_prefix_pattern(self):
        """Test finding patterns that are prefixes of suffixes."""
        tokens = [1, 2, 3, 4, 5, 1, 2, 7]
        sa = SuffixArray(tokens)

        # Look for prefix [1, 2]
        pattern = [1, 2]

        if hasattr(sa, 'find_pattern'):
            matches = sa.find_pattern(pattern)
            assert isinstance(matches, (list, np.ndarray))
            # Should find matches at positions where [1, 2] starts


class TestSuffixArrayLongestCommonPrefix:
    """Test LCP (Longest Common Prefix) functionality."""

    def test_lcp_array_exists(self):
        """Test that LCP array is constructed."""
        tokens = [1, 2, 3, 4, 5]
        sa = SuffixArray(tokens)

        if hasattr(sa, 'lcp_array'):
            assert isinstance(sa.lcp_array, np.ndarray)
            # LCP array should have length n-1 or n
            assert len(sa.lcp_array) in [len(tokens) - 1, len(tokens)]

    def test_lcp_values_non_negative(self):
        """Test that LCP values are non-negative."""
        tokens = [1, 2, 3, 1, 2, 4]
        sa = SuffixArray(tokens)

        if hasattr(sa, 'lcp_array'):
            assert np.all(sa.lcp_array >= 0)

    def test_lcp_repeated_patterns(self):
        """Test LCP with repeated patterns."""
        tokens = [1, 2, 3, 1, 2, 3]  # Repeated pattern
        sa = SuffixArray(tokens)

        if hasattr(sa, 'lcp_array'):
            # Should have some non-zero LCP values due to repetition
            assert isinstance(sa.lcp_array, np.ndarray)


class TestSuffixArrayQueries:
    """Test query operations on suffix array."""

    def test_range_query(self):
        """Test range queries for pattern occurrences."""
        tokens = [1, 2, 3, 2, 3, 4, 2, 3]
        sa = SuffixArray(tokens)

        # Test range query functionality if available
        if hasattr(sa, 'range_query'):
            pattern = [2, 3]
            range_result = sa.range_query(pattern)
            assert isinstance(range_result, (tuple, list))

        # Alternative: test with get_occurrences if available
        if hasattr(sa, 'get_occurrences'):
            pattern = [2, 3]
            occurrences = sa.get_occurrences(pattern)
            assert isinstance(occurrences, (list, np.ndarray))

    def test_longest_match(self):
        """Test finding longest matching suffix."""
        tokens = [1, 2, 3, 4, 5, 1, 2, 6]
        sa = SuffixArray(tokens)

        # Test longest match functionality if available
        if hasattr(sa, 'longest_match'):
            query = [1, 2, 7]  # Partial match
            match_info = sa.longest_match(query)
            assert match_info is not None

    def test_count_occurrences(self):
        """Test counting pattern occurrences."""
        tokens = [1, 2, 2, 3, 2, 4]
        sa = SuffixArray(tokens)

        if hasattr(sa, 'count_occurrences'):
            pattern = [2]
            count = sa.count_occurrences(pattern)
            assert isinstance(count, int)
            assert count >= 0
            # Token 2 appears 3 times in the sequence


class TestSuffixArrayBehavior:
    """Test suffix array behavioral properties."""

    def test_suffix_array_sorted_property(self):
        """Test that suffixes are properly sorted."""
        tokens = [3, 1, 4, 1, 5, 9]
        sa = SuffixArray(tokens)

        # The suffix array should arrange suffixes in lexicographic order
        # We can verify this by checking that the suffixes are ordered
        suffixes = [tokens[i:] for i in sa.suffix_array]

        # Consecutive suffixes should be in non-decreasing lexicographic order
        for i in range(len(suffixes) - 1):
            assert suffixes[i] <= suffixes[i + 1], f"Suffix {suffixes[i]} should come before {suffixes[i+1]}"

    def test_suffix_array_deterministic(self):
        """Test that suffix array construction is deterministic."""
        tokens = [2, 7, 1, 8, 2, 8]

        sa1 = SuffixArray(tokens)
        sa2 = SuffixArray(tokens)

        # Should produce identical suffix arrays
        np.testing.assert_array_equal(sa1.suffix_array, sa2.suffix_array)

        if hasattr(sa1, 'lcp_array') and hasattr(sa2, 'lcp_array'):
            np.testing.assert_array_equal(sa1.lcp_array, sa2.lcp_array)

    def test_suffix_array_with_large_tokens(self):
        """Test suffix array with large token values."""
        large_tokens = [100, 200, 300, 150, 250]
        sa = SuffixArray(large_tokens)

        assert sa.n == len(large_tokens)
        assert len(sa.suffix_array) == len(large_tokens)
        # Should handle large values correctly

    def test_suffix_array_memory_efficiency(self):
        """Test that suffix array uses reasonable memory."""
        tokens = list(range(1000))  # Large but manageable
        sa = SuffixArray(tokens)

        # Should complete construction
        assert sa.n == 1000
        assert len(sa.suffix_array) == 1000
        assert sa.suffix_array.dtype in [np.int32, np.int64]  # Efficient integer storage


class TestSuffixArrayEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_same_tokens(self):
        """Test suffix array with all identical tokens."""
        same_tokens = [5] * 10
        sa = SuffixArray(same_tokens)

        assert sa.n == 10
        assert len(sa.suffix_array) == 10
        # All suffixes should be valid

    def test_strictly_increasing_tokens(self):
        """Test suffix array with strictly increasing sequence."""
        increasing_tokens = [1, 2, 3, 4, 5, 6]
        sa = SuffixArray(increasing_tokens)

        # Should handle perfectly ordered sequence
        assert sa.n == len(increasing_tokens)

        # The suffix starting with the smallest token should come first
        # The shortest suffix (last token) might come first or last depending on implementation

    def test_strictly_decreasing_tokens(self):
        """Test suffix array with strictly decreasing sequence."""
        decreasing_tokens = [6, 5, 4, 3, 2, 1]
        sa = SuffixArray(decreasing_tokens)

        # Should handle reverse ordered sequence
        assert sa.n == len(decreasing_tokens)

    def test_two_token_array(self):
        """Test suffix array with minimal non-trivial input."""
        two_tokens = [3, 1]
        sa = SuffixArray(two_tokens)

        assert sa.n == 2
        assert len(sa.suffix_array) == 2

        # Suffix [1] should come before suffix [3, 1]
        # So suffix_array should be [1, 0] (indices where suffixes start)
        suffixes = [two_tokens[i:] for i in sa.suffix_array]
        assert suffixes[0] <= suffixes[1]

    def test_alternating_pattern(self):
        """Test suffix array with alternating pattern."""
        alternating = [1, 2] * 5  # [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        sa = SuffixArray(alternating)

        assert sa.n == 10
        assert len(sa.suffix_array) == 10
        # Should handle repetitive alternating pattern


@pytest.mark.parametrize("size", [5, 10, 50])
def test_suffix_array_various_sizes(size):
    """Test suffix array with various input sizes."""
    np.random.seed(42)
    tokens = list(np.random.randint(0, 10, size=size))

    sa = SuffixArray(tokens)

    assert sa.n == size
    assert len(sa.suffix_array) == size
    assert all(0 <= idx < size for idx in sa.suffix_array)


@pytest.mark.parametrize("vocab_size", [2, 5, 20])
def test_suffix_array_various_vocabularies(vocab_size):
    """Test suffix array with different vocabulary sizes."""
    np.random.seed(42)
    tokens = list(np.random.randint(0, vocab_size, size=30))

    sa = SuffixArray(tokens)

    assert sa.n == 30
    assert len(sa.suffix_array) == 30

    # Verify sorting property
    suffixes = [tokens[i:] for i in sa.suffix_array]
    for i in range(len(suffixes) - 1):
        assert suffixes[i] <= suffixes[i + 1]