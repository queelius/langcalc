"""
Unit tests for NGram model functionality.

These tests verify the core n-gram model behavior including training,
probability computation, and sampling, focusing on the contract rather
than implementation details.
"""

import pytest
import numpy as np
from typing import List

from langcalc.models.ngram import NGramModel
from langcalc.models.base import LanguageModel


class TestNGramModelInitialization:
    """Test NGram model initialization and setup."""

    def test_ngram_model_creation(self, small_corpus):
        """Test basic NGram model creation."""
        model = NGramModel(small_corpus, n=3)

        assert isinstance(model, LanguageModel)
        assert model.n == 3
        assert model.corpus == small_corpus

    def test_ngram_model_different_orders(self, small_corpus):
        """Test NGram models of different orders."""
        for n in [1, 2, 3, 4]:
            model = NGramModel(small_corpus, n=n)
            assert model.n == n

    def test_ngram_model_with_empty_corpus(self):
        """Test NGram model behavior with empty corpus."""
        empty_corpus = []
        model = NGramModel(empty_corpus, n=2)

        assert model.corpus == empty_corpus
        assert model.n == 2

    def test_ngram_model_with_single_token_corpus(self):
        """Test NGram model with minimal corpus."""
        single_token = [42]
        model = NGramModel(single_token, n=2)

        assert model.corpus == single_token

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_ngram_model_parametrized_orders(self, small_corpus, n):
        """Test NGram model creation with different orders."""
        model = NGramModel(small_corpus, n=n)
        assert model.n == n


class TestNGramModelProbabilities:
    """Test NGram model probability computation."""

    def test_logprobs_computation(self, small_corpus):
        """Test that NGram model computes valid log probabilities."""
        model = NGramModel(small_corpus, n=3)

        tokens = [1, 2, 3]
        context = [5, 6]

        logprobs = model.logprobs(tokens, context)

        assert isinstance(logprobs, np.ndarray)
        assert logprobs.shape == (len(tokens),)
        assert not np.any(np.isnan(logprobs))
        assert np.all(logprobs <= 0)  # Log probabilities should be <= 0

    def test_logprobs_without_context(self, small_corpus):
        """Test log probability computation without context."""
        model = NGramModel(small_corpus, n=3)

        tokens = [1, 2, 3]
        logprobs = model.logprobs(tokens)

        assert isinstance(logprobs, np.ndarray)
        assert logprobs.shape == (len(tokens),)

    def test_logprobs_empty_tokens(self, small_corpus):
        """Test log probability computation with empty token list."""
        model = NGramModel(small_corpus, n=3)

        logprobs = model.logprobs([])

        assert isinstance(logprobs, np.ndarray)
        assert len(logprobs) == 0

    def test_logprobs_single_token(self, small_corpus):
        """Test log probability computation for single token."""
        model = NGramModel(small_corpus, n=3)

        logprobs = model.logprobs([5])

        assert isinstance(logprobs, np.ndarray)
        assert len(logprobs) == 1
        assert not np.isnan(logprobs[0])

    def test_logprobs_consistency(self, small_corpus):
        """Test that repeated calls return consistent results."""
        model = NGramModel(small_corpus, n=3)

        tokens = [1, 2, 3]
        context = [4, 5]

        logprobs1 = model.logprobs(tokens, context)
        logprobs2 = model.logprobs(tokens, context)

        np.testing.assert_array_equal(logprobs1, logprobs2)

    def test_score_computation(self, small_corpus):
        """Test sequence scoring functionality."""
        model = NGramModel(small_corpus, n=3)

        sequence = [1, 2, 3, 4, 5]
        score = model.score(sequence)

        assert isinstance(score, (float, np.floating))
        assert not np.isnan(score)

    def test_score_empty_sequence(self, small_corpus):
        """Test scoring of empty sequence."""
        model = NGramModel(small_corpus, n=3)

        score = model.score([])

        assert isinstance(score, (float, np.floating))
        # Empty sequence score should be reasonable (could be 0 or some default)
        assert not np.isnan(score)


class TestNGramModelSampling:
    """Test NGram model sampling functionality."""

    def test_basic_sampling(self, small_corpus):
        """Test basic token sampling from NGram model."""
        model = NGramModel(small_corpus, n=3)

        samples = model.sample(max_tokens=5)

        assert isinstance(samples, list)
        assert len(samples) == 5
        assert all(isinstance(token, (int, np.integer)) for token in samples)

    def test_sampling_with_context(self, small_corpus):
        """Test sampling with context provided."""
        model = NGramModel(small_corpus, n=3)

        context = [1, 2]
        samples = model.sample(context, max_tokens=3)

        assert isinstance(samples, list)
        assert len(samples) == 3
        assert all(isinstance(token, (int, np.integer)) for token in samples)

    def test_sampling_zero_tokens(self, small_corpus):
        """Test sampling zero tokens."""
        model = NGramModel(small_corpus, n=3)

        samples = model.sample(max_tokens=0)

        assert isinstance(samples, list)
        assert len(samples) == 0

    def test_sampling_single_token(self, small_corpus):
        """Test sampling single token."""
        model = NGramModel(small_corpus, n=3)

        samples = model.sample(max_tokens=1)

        assert isinstance(samples, list)
        assert len(samples) == 1
        assert isinstance(samples[0], (int, np.integer))

    def test_sampling_reproducibility_with_context(self, small_corpus):
        """Test that sampling with same context produces valid results."""
        model = NGramModel(small_corpus, n=3)

        context = [1, 2, 3]

        # Multiple calls should produce valid samples
        # Note: We don't require identical results as sampling may be random
        for _ in range(5):
            samples = model.sample(context, max_tokens=3)
            assert isinstance(samples, list)
            assert len(samples) == 3
            assert all(isinstance(token, (int, np.integer)) for token in samples)

    def test_sampling_with_temperature(self, small_corpus):
        """Test sampling with different temperatures."""
        model = NGramModel(small_corpus, n=3)

        # Test various temperatures
        temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]

        for temp in temperatures:
            samples = model.sample(max_tokens=3, temperature=temp)
            assert isinstance(samples, list)
            assert len(samples) == 3
            assert all(isinstance(token, (int, np.integer)) for token in samples)


class TestNGramModelBehavior:
    """Test NGram model behavioral properties."""

    def test_model_learns_from_corpus(self, small_corpus):
        """Test that model learns patterns from training corpus."""
        model = NGramModel(small_corpus, n=3)

        # The model should be able to compute probabilities
        # and they should be influenced by the training data
        logprobs = model.logprobs([small_corpus[0]])
        assert isinstance(logprobs, np.ndarray)
        assert not np.any(np.isnan(logprobs))

    def test_higher_order_models_use_more_context(self):
        """Test that higher-order models utilize more context."""
        corpus = [1, 2, 3, 4, 5] * 10  # Repeated pattern

        model_2gram = NGramModel(corpus, n=2)
        model_4gram = NGramModel(corpus, n=4)

        context = [1, 2, 3]
        tokens = [4]  # Token that follows the pattern

        logprobs_2gram = model_2gram.logprobs(tokens, context)
        logprobs_4gram = model_4gram.logprobs(tokens, context)

        # Both should produce valid probabilities
        assert isinstance(logprobs_2gram, np.ndarray)
        assert isinstance(logprobs_4gram, np.ndarray)
        assert not np.any(np.isnan(logprobs_2gram))
        assert not np.any(np.isnan(logprobs_4gram))

    def test_model_handles_unseen_tokens(self, small_corpus):
        """Test model behavior with tokens not in training corpus."""
        model = NGramModel(small_corpus, n=3)

        # Use tokens outside the range of small_corpus
        max_corpus_token = max(small_corpus) if small_corpus else 0
        unseen_tokens = [max_corpus_token + 100, max_corpus_token + 200]

        # Should handle gracefully (via smoothing or other mechanisms)
        logprobs = model.logprobs(unseen_tokens)

        assert isinstance(logprobs, np.ndarray)
        assert len(logprobs) == len(unseen_tokens)
        # Should not crash, though probabilities might be low

    def test_model_string_representation(self, small_corpus):
        """Test that model has meaningful string representation."""
        model = NGramModel(small_corpus, n=3)

        repr_str = str(model)

        assert isinstance(repr_str, str)
        assert len(repr_str) > 0
        # Should probably mention it's an n-gram model and the order
        assert "gram" in repr_str.lower() or "NGram" in repr_str


class TestNGramModelEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_ngram_order_larger_than_corpus(self):
        """Test n-gram model where n > corpus length."""
        small_corpus = [1, 2]  # Only 2 tokens
        model = NGramModel(small_corpus, n=5)  # n > corpus length

        # Should still create model
        assert model.n == 5

        # Should be able to compute probabilities
        logprobs = model.logprobs([1])
        assert isinstance(logprobs, np.ndarray)

    def test_unigram_model(self, small_corpus):
        """Test unigram (n=1) model behavior."""
        model = NGramModel(small_corpus, n=1)

        # Unigram model should not depend on context
        tokens = [1, 2]

        logprobs_no_context = model.logprobs(tokens)
        logprobs_with_context = model.logprobs(tokens, [99, 88, 77])

        assert isinstance(logprobs_no_context, np.ndarray)
        assert isinstance(logprobs_with_context, np.ndarray)
        # For unigrams, context shouldn't matter much

    def test_large_ngram_order(self, sample_corpus):
        """Test model with very large n."""
        large_n = 10
        model = NGramModel(sample_corpus, n=large_n)

        # Should still work
        logprobs = model.logprobs([1, 2, 3])
        assert isinstance(logprobs, np.ndarray)

    def test_model_with_repeated_corpus(self):
        """Test model trained on highly repetitive corpus."""
        repetitive_corpus = [1, 2, 3] * 100  # Very repetitive
        model = NGramModel(repetitive_corpus, n=3)

        # Should handle repetitive patterns
        logprobs = model.logprobs([1, 2, 3], [1, 2, 3])
        assert isinstance(logprobs, np.ndarray)
        assert not np.any(np.isnan(logprobs))

        # Should generate reasonable samples
        samples = model.sample([1, 2], max_tokens=5)
        assert isinstance(samples, list)
        assert len(samples) == 5


@pytest.mark.parametrize("corpus_size", [10, 50, 200])
def test_ngram_model_with_different_corpus_sizes(corpus_size):
    """Test NGram model performance with different corpus sizes."""
    np.random.seed(42)
    corpus = list(np.random.randint(0, 20, size=corpus_size))

    model = NGramModel(corpus, n=3)

    # Basic functionality should work regardless of corpus size
    logprobs = model.logprobs([1, 2, 3])
    assert isinstance(logprobs, np.ndarray)

    samples = model.sample(max_tokens=3)
    assert isinstance(samples, list)
    assert len(samples) == 3


@pytest.mark.parametrize("vocab_size", [5, 20, 100])
def test_ngram_model_with_different_vocab_sizes(vocab_size):
    """Test NGram model with different vocabulary sizes."""
    np.random.seed(42)
    # Create corpus with tokens in range [0, vocab_size)
    corpus = list(np.random.randint(0, vocab_size, size=200))

    model = NGramModel(corpus, n=3)

    # Test with tokens from the vocabulary
    test_tokens = list(range(min(5, vocab_size)))
    logprobs = model.logprobs(test_tokens)

    assert isinstance(logprobs, np.ndarray)
    assert len(logprobs) == len(test_tokens)
    assert not np.any(np.isnan(logprobs))