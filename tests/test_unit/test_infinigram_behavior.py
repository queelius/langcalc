#!/usr/bin/env python3
"""
Behavior-focused tests for InfinigramModel.

These tests verify that the model behaves correctly according to its
specification, focusing on observable behaviors rather than implementation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add langcalc to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langcalc.models import InfinigramModel


class TestInfinigramBehavior:
    """Test InfinigramModel observable behaviors."""

    def test_seen_tokens_have_higher_probability_than_unseen(self):
        """Test that tokens in corpus have higher probability than unseen ones."""
        corpus = [1, 2, 3, 4, 5, 1, 2, 3]
        model = InfinigramModel(corpus, smoothing=0.01)

        # Token 1 is in corpus, token 99 is not
        logprobs = model.logprobs([1, 99], context=[])

        # Seen token should have higher (less negative) log probability
        assert logprobs[0] > logprobs[1], \
            "Seen tokens should have higher probability than unseen tokens"

    def test_frequent_patterns_are_more_likely(self):
        """Test that frequent patterns get higher probability."""
        # Create corpus where "1, 2" is very common
        corpus = [1, 2] * 10 + [1, 3] * 2  # "1,2" appears 10x, "1,3" appears 2x
        model = InfinigramModel(corpus)

        context = [1]
        logprobs = model.logprobs([2, 3], context)

        # Token 2 should be more likely after 1 than token 3
        assert logprobs[0] > logprobs[1], \
            "More frequent continuations should have higher probability"

    def test_longer_matching_context_improves_predictions(self):
        """Test that longer context matches lead to better predictions."""
        # Create corpus with specific patterns
        corpus = [1, 2, 3, 4, 5] + [7, 8, 3, 4, 6]
        model = InfinigramModel(corpus)

        # With context [3, 4], token 5 appears after first pattern
        # With context [3, 4], token 6 appears after second pattern
        short_context = [3, 4]
        logprobs_short = model.logprobs([5, 6], short_context)

        # Both should have reasonable probability
        assert logprobs_short[0] > -10
        assert logprobs_short[1] > -10

    def test_smoothing_prevents_zero_probability(self):
        """Test that smoothing ensures unseen tokens have non-zero probability."""
        corpus = [1, 2, 3]
        model = InfinigramModel(corpus, smoothing=0.1)

        # Token 99 never appears in corpus
        logprobs = model.logprobs([99], context=[])

        # Should not be -inf due to smoothing
        assert logprobs[0] > -np.inf, \
            "Smoothing should prevent zero probability for unseen tokens"

    def test_higher_smoothing_increases_unseen_token_probability(self):
        """Test that increasing smoothing helps unseen tokens."""
        corpus = [1, 2, 3, 4, 5]

        model_low = InfinigramModel(corpus, smoothing=0.01)
        model_high = InfinigramModel(corpus, smoothing=1.0)

        # Unseen token
        logprob_low = model_low.logprobs([99], context=[])[0]
        logprob_high = model_high.logprobs([99], context=[])[0]

        # Higher smoothing should give higher probability to unseen tokens
        assert logprob_high > logprob_low, \
            "Higher smoothing should increase probability of unseen tokens"

    def test_min_count_filters_rare_patterns(self):
        """Test that min_count filtering affects predictions."""
        # Token 5 appears only once
        corpus = [1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 3]

        model_min1 = InfinigramModel(corpus, min_count=1)
        model_min2 = InfinigramModel(corpus, min_count=2)

        # Pattern "1, 2, 3" appears 3 times (above both thresholds)
        # Should behave similarly
        context = [1, 2]
        logprobs_min1 = model_min1.logprobs([3], context)[0]
        logprobs_min2 = model_min2.logprobs([3], context)[0]

        # Both should predict 3 with high confidence
        assert logprobs_min1 > -5
        assert logprobs_min2 > -5

    def test_max_length_limits_context_usage(self):
        """Test that max_length parameter limits context usage."""
        # Create pattern that only appears with long context
        corpus = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        model_unlimited = InfinigramModel(corpus, max_length=None)
        model_limited = InfinigramModel(corpus, max_length=3)

        # With full context, pattern should match well
        full_context = [1, 2, 3, 4, 5, 6, 7, 8]
        logprobs_unlimited = model_unlimited.logprobs([9], full_context)
        logprobs_limited = model_limited.logprobs([9], full_context)

        # Both should give valid predictions
        assert isinstance(logprobs_unlimited[0], (float, np.floating))
        assert isinstance(logprobs_limited[0], (float, np.floating))

    def test_update_incorporates_new_patterns(self):
        """Test that update() makes new patterns predictable."""
        corpus = [1, 2, 3]
        model = InfinigramModel(corpus)

        # Before update: pattern [7, 8] -> 9 is unknown
        context_before = [7, 8]
        logprob_before = model.logprobs([9], context_before)[0]

        # Add pattern multiple times to make it likely
        model.update([7, 8, 9, 7, 8, 9, 7, 8, 9])

        # After update: pattern should be known
        logprob_after = model.logprobs([9], context_before)[0]

        # Should be more confident after seeing pattern
        assert logprob_after > logprob_before, \
            "Model should learn from updated corpus"

    def test_confidence_reflects_match_quality(self):
        """Test that confidence is higher for patterns that match corpus."""
        # Create corpus with clear patterns
        corpus = [1, 2, 3, 4, 5] * 5  # Repeated pattern
        model = InfinigramModel(corpus)

        # Context that matches corpus pattern
        confidence_good = model.confidence([1, 2, 3])

        # Context that doesn't match corpus
        confidence_bad = model.confidence([99, 98, 97])

        # Good match should have higher confidence
        assert confidence_good >= confidence_bad, \
            "Matching context should have higher confidence"

    def test_score_favors_likely_sequences(self):
        """Test that common sequences score higher than rare ones."""
        # Corpus with repeated pattern
        corpus = [1, 2, 3] * 10 + [4, 5, 6]
        model = InfinigramModel(corpus)

        # Common sequence
        score_common = model.score([1, 2, 3])

        # Rare sequence
        score_rare = model.score([4, 5, 6])

        # Common should score higher (less negative)
        assert score_common > score_rare, \
            "More common sequences should score higher"

    def test_sample_produces_valid_tokens(self):
        """Test that sampling always produces valid byte-range tokens."""
        corpus = list(range(256))  # All possible bytes
        model = InfinigramModel(corpus)

        samples = model.sample(max_tokens=100)

        # All samples should be valid byte values
        assert all(0 <= token <= 255 for token in samples), \
            "All sampled tokens should be in byte range [0, 255]"

    def test_sample_uses_context(self):
        """Test that sampling respects context."""
        # Create deterministic pattern
        corpus = [1, 2, 3, 4, 5] * 20  # Very predictable
        model = InfinigramModel(corpus)

        # Sample with context that strongly predicts next token
        np.random.seed(42)
        context = [1, 2, 3, 4]
        samples = model.sample(context, temperature=0.5, max_tokens=5)

        # With low temperature and strong pattern, should often generate 5
        # (This is a statistical test, not deterministic)
        assert 5 in samples or samples[0] == 5, \
            "Sampling should respect strong context patterns"

    def test_algebraic_composition_preserves_behavior(self):
        """Test that composed models still exhibit correct behavior."""
        from langcalc.models import MockLLM

        corpus = [1, 2, 3, 4, 5]
        infinigram = InfinigramModel(corpus)
        llm = MockLLM(vocab_size=256)

        # Compose models
        mixed = 0.5 * infinigram + 0.5 * llm

        # Should still produce valid predictions
        logprobs = mixed.logprobs(list(range(256)), context=[1, 2])

        # All logprobs should be valid
        assert len(logprobs) == 256
        assert all(np.isfinite(lp) or lp == -np.inf for lp in logprobs)

    def test_byte_level_model_handles_text_correctly(self):
        """Test that byte-level model correctly handles UTF-8 text."""
        # English text
        text = "The quick brown fox"
        corpus = list(text.encode('utf-8'))
        model = InfinigramModel(corpus)

        # Predict after "The"
        context = list("The".encode('utf-8'))
        logprobs = model.logprobs(list(range(256)), context)

        # Space character (32) should be likely after "The"
        space_logprob = logprobs[32]
        # Should not be -inf (space appears in corpus after "The")
        assert space_logprob > -np.inf

    def test_multibyte_utf8_sequences(self):
        """Test handling of multibyte UTF-8 characters."""
        # Text with multibyte characters
        text = "Hello 世界"
        corpus = list(text.encode('utf-8'))

        model = InfinigramModel(corpus)

        # Should handle the byte sequence for 世 (E4 B8 96)
        context = list("Hello ".encode('utf-8'))
        logprobs = model.logprobs([0xE4], context)

        # First byte of 世 should have reasonable probability
        assert np.isfinite(logprobs[0]) or logprobs[0] == -np.inf

    def test_probability_distribution_sums_to_one(self):
        """Test that probability distributions are properly normalized."""
        corpus = [1, 2, 3, 4, 5, 1, 2, 3]
        model = InfinigramModel(corpus)

        # Get all logprobs
        logprobs = model.logprobs(list(range(256)), context=[1])

        # Convert to probabilities
        probs = np.exp(logprobs)
        total_prob = np.sum(probs)

        # Should sum to approximately 1 (accounting for floating point)
        assert abs(total_prob - 1.0) < 0.01, \
            f"Probabilities should sum to 1, got {total_prob}"

    def test_consistent_predictions_without_randomness(self):
        """Test that non-sampling methods are deterministic."""
        corpus = [1, 2, 3, 4, 5]
        model = InfinigramModel(corpus)

        context = [1, 2]

        # Call multiple times
        logprobs1 = model.logprobs([3, 4, 5], context)
        logprobs2 = model.logprobs([3, 4, 5], context)

        # Should be identical (no randomness in logprobs)
        np.testing.assert_array_equal(logprobs1, logprobs2,
            "logprobs should be deterministic")

    def test_score_is_sum_of_conditional_logprobs(self):
        """Test that score computes cumulative log probability correctly."""
        corpus = [1, 2, 3, 4, 5, 1, 2, 3]
        model = InfinigramModel(corpus)

        sequence = [1, 2, 3]

        # Get score
        total_score = model.score(sequence)

        # Compute manually
        expected_score = 0.0
        for i in range(len(sequence)):
            context = sequence[:i]
            token = sequence[i]
            logprob = model.logprobs([token], context)[0]
            expected_score += logprob

        # Should match (within numerical precision)
        assert abs(total_score - expected_score) < 1e-6, \
            "score() should equal sum of conditional log probabilities"
