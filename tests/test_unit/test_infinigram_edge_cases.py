#!/usr/bin/env python3
"""
Edge case and error handling tests for InfinigramModel.

These tests cover boundary conditions, error cases, and unusual inputs
that the main test suite doesn't address.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add langcalc to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langcalc.models import InfinigramModel


class TestInfinigramEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_corpus(self):
        """Test initialization with empty corpus."""
        # This may or may not be valid - test the actual behavior
        with pytest.raises((ValueError, IndexError)):
            model = InfinigramModel([])

    def test_single_token_corpus(self):
        """Test corpus with only one token."""
        corpus = [42]
        model = InfinigramModel(corpus)

        # Should still work
        logprobs = model.logprobs([42, 43], context=[])
        assert len(logprobs) == 2

    def test_very_large_context(self):
        """Test with context much larger than corpus."""
        corpus = [1, 2, 3, 4, 5]
        model = InfinigramModel(corpus)

        # Context longer than entire corpus
        huge_context = list(range(1000))
        logprobs = model.logprobs([1, 2], context=huge_context)

        assert len(logprobs) == 2
        assert all(np.isfinite(lp) or lp == -np.inf for lp in logprobs)

    def test_negative_smoothing(self):
        """Test that negative smoothing is handled appropriately."""
        corpus = [1, 2, 3]

        # Should either raise error or clamp to valid range
        with pytest.raises((ValueError, AssertionError)):
            model = InfinigramModel(corpus, smoothing=-0.1)

    def test_zero_smoothing(self):
        """Test with zero smoothing (may cause numerical issues)."""
        corpus = [1, 2, 3]
        model = InfinigramModel(corpus, smoothing=0.0)

        # Should still return valid logprobs
        logprobs = model.logprobs([99], context=[1])
        assert isinstance(logprobs[0], (float, np.floating))
        # With zero smoothing, unseen token should have -inf logprob
        assert logprobs[0] == -np.inf

    def test_very_high_smoothing(self):
        """Test with very high smoothing (approaches uniform distribution)."""
        corpus = [1, 2, 3, 4, 5]
        model = InfinigramModel(corpus, smoothing=1000.0)

        # All tokens should have similar probabilities
        logprobs = model.logprobs(list(range(256)), context=[1])

        # Standard deviation should be low (near-uniform)
        assert np.std(logprobs) < 1.0

    def test_token_values_outside_byte_range(self):
        """Test that tokens outside 0-255 range are handled."""
        # This tests the adapter's validation
        corpus = [1, 2, 3]
        model = InfinigramModel(corpus)

        # Token 256 is outside byte range
        # Should either handle gracefully or raise clear error
        try:
            logprobs = model.logprobs([256], context=[1])
            # If it doesn't raise, should return valid logprob
            assert isinstance(logprobs[0], (float, np.floating))
        except (ValueError, IndexError) as e:
            # Acceptable to raise error for out-of-range tokens
            assert "256" in str(e) or "range" in str(e).lower()

    def test_sample_zero_max_tokens(self):
        """Test sampling with max_tokens=0."""
        corpus = [1, 2, 3]
        model = InfinigramModel(corpus)

        samples = model.sample(max_tokens=0)
        assert len(samples) == 0

    def test_sample_negative_max_tokens(self):
        """Test sampling with negative max_tokens."""
        corpus = [1, 2, 3]
        model = InfinigramModel(corpus)

        # Should either return empty list or raise error
        samples = model.sample(max_tokens=-5)
        assert len(samples) == 0

    def test_score_single_token_sequence(self):
        """Test scoring a single-token sequence."""
        corpus = [1, 2, 3, 4, 5]
        model = InfinigramModel(corpus)

        score = model.score([1])
        # Should be valid (single token has no context)
        assert isinstance(score, (float, np.floating))
        assert score <= 0.0

    def test_confidence_empty_context(self):
        """Test confidence with empty context."""
        corpus = [1, 2, 3]
        model = InfinigramModel(corpus)

        confidence = model.confidence([])
        assert isinstance(confidence, (float, np.floating))
        assert 0.0 <= confidence <= 1.0

    def test_longest_suffix_empty_context(self):
        """Test longest suffix with empty context."""
        corpus = [1, 2, 3]
        model = InfinigramModel(corpus)

        pos, length = model.longest_suffix([])
        assert isinstance(pos, (int, np.integer))
        assert isinstance(length, (int, np.integer))

    def test_longest_suffix_no_match(self):
        """Test longest suffix when no match exists."""
        corpus = [1, 2, 3]
        model = InfinigramModel(corpus)

        # Context that doesn't exist in corpus
        pos, length = model.longest_suffix([99, 98, 97])

        # Length should be 0 or very small
        assert length >= 0

    def test_update_with_empty_tokens(self):
        """Test update with empty token list."""
        corpus = [1, 2, 3]
        model = InfinigramModel(corpus)

        initial_size = len(model.infinigram.corpus)
        model.update([])

        # Should not crash, corpus size should be unchanged
        assert len(model.infinigram.corpus) == initial_size

    def test_logprobs_empty_token_list(self):
        """Test logprobs with empty token list."""
        corpus = [1, 2, 3]
        model = InfinigramModel(corpus)

        logprobs = model.logprobs([], context=[1])
        assert len(logprobs) == 0

    def test_byte_sequence_with_null_bytes(self):
        """Test handling of null bytes in corpus."""
        # Text with null bytes
        text = "hello\x00world"
        corpus = list(text.encode('utf-8'))

        model = InfinigramModel(corpus)

        # Should handle null bytes correctly
        logprobs = model.logprobs([0], context=[])  # 0 is null byte
        assert isinstance(logprobs[0], (float, np.floating))

    def test_max_length_zero(self):
        """Test with max_length set to 0."""
        corpus = [1, 2, 3, 4, 5]

        # max_length=0 means no context used
        model = InfinigramModel(corpus, max_length=0)

        # Should still work (unigram model essentially)
        logprobs = model.logprobs([1, 2], context=[1, 2, 3])
        assert len(logprobs) == 2

    def test_max_length_larger_than_corpus(self):
        """Test max_length larger than corpus size."""
        corpus = [1, 2, 3]
        model = InfinigramModel(corpus, max_length=1000)

        # Should work, just never use more context than available
        logprobs = model.logprobs([1], context=[1, 2])
        assert isinstance(logprobs[0], (float, np.floating))

    def test_min_count_higher_than_any_pattern(self):
        """Test min_count that filters out all patterns."""
        corpus = [1, 2, 3, 4, 5]  # All tokens appear once
        model = InfinigramModel(corpus, min_count=10)

        # Should fall back to smoothing for all predictions
        logprobs = model.logprobs([1, 2, 3], context=[])

        # All should have similar (smoothed) probabilities
        assert np.std(logprobs) < 1.0


class TestInfinigramNumericalStability:
    """Test numerical stability and precision."""

    def test_very_long_sequence_score(self):
        """Test scoring very long sequences doesn't overflow."""
        corpus = list(range(100))
        model = InfinigramModel(corpus)

        # Very long sequence
        sequence = list(range(50)) * 10  # 500 tokens
        score = model.score(sequence)

        # Should be finite (very negative but not -inf)
        assert np.isfinite(score)

    def test_extreme_temperature(self):
        """Test extreme temperature values."""
        corpus = [1, 2, 3, 4, 5]
        model = InfinigramModel(corpus)

        # Very low temperature (near-deterministic)
        samples = model.sample(temperature=0.001, max_tokens=5)
        assert len(samples) == 5

        # Very high temperature (near-uniform)
        samples = model.sample(temperature=100.0, max_tokens=5)
        assert len(samples) == 5

    def test_logprobs_do_not_overflow(self):
        """Test that logprobs computation doesn't overflow with large vocab."""
        corpus = list(range(256))  # Full byte range
        model = InfinigramModel(corpus)

        # Get logprobs for all 256 possible bytes
        logprobs = model.logprobs(list(range(256)), context=[1, 2, 3])

        # Should all be finite or -inf
        assert all(np.isfinite(lp) or lp == -np.inf for lp in logprobs)
        assert not np.any(np.isnan(logprobs))
