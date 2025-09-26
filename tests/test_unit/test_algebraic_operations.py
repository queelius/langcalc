"""
Unit tests for algebraic operations on language models.

These tests verify the core algebraic functionality of the model composition
framework, focusing on behavior rather than implementation details.
"""

import pytest
import numpy as np
from typing import List

from ngram_projections.models.base import LanguageModel
from ngram_projections.models.ngram import NGramModel
from ngram_projections.models.llm import MockLLM
from ngram_projections.models.mixture import MixtureModel


class TestBasicAlgebraicOperations:
    """Test basic algebraic operations between language models."""

    def test_model_addition_creates_mixture(self, ngram_model, mock_llm):
        """Test that adding models creates a mixture model."""
        mixture = ngram_model + mock_llm

        assert hasattr(mixture, 'models')
        assert len(mixture.models) == 2
        assert isinstance(mixture, MixtureModel)

    def test_weighted_model_creation(self, ngram_model, mock_llm):
        """Test creating weighted model combinations."""
        weighted = 0.3 * ngram_model + 0.7 * mock_llm

        assert isinstance(weighted, MixtureModel)
        # Weights should be normalized
        total_weight = sum(weighted.weights)
        assert abs(total_weight - 1.0) < 1e-6

    def test_model_composition_operator(self, ngram_model, mock_llm):
        """Test sequential composition using >> operator."""
        composed = ngram_model >> mock_llm

        # Should create some form of composed model
        assert composed is not None
        assert hasattr(composed, 'logprobs')
        assert hasattr(composed, 'sample')

    def test_model_fallback_operator(self, ngram_model, mock_llm):
        """Test fallback using | operator."""
        fallback = ngram_model | mock_llm

        # Should create a fallback model
        assert fallback is not None
        assert hasattr(fallback, 'logprobs')
        assert hasattr(fallback, 'sample')

    def test_mixture_logprobs_computation(self, ngram_model, mock_llm, test_tokens, test_context):
        """Test that mixture models compute valid log probabilities."""
        mixture = ngram_model + mock_llm

        logprobs = mixture.logprobs(test_tokens, test_context)

        assert isinstance(logprobs, np.ndarray)
        assert logprobs.shape == (len(test_tokens),)
        assert not np.any(np.isnan(logprobs))
        assert np.all(logprobs <= 0)  # Log probabilities should be <= 0

    def test_mixture_sampling(self, ngram_model, mock_llm, test_context):
        """Test that mixture models can generate samples."""
        mixture = ngram_model + mock_llm
        max_tokens = 5

        samples = mixture.sample(test_context, max_tokens=max_tokens)

        assert isinstance(samples, list)
        assert len(samples) == max_tokens
        assert all(isinstance(token, int) for token in samples)

    def test_weighted_model_logprobs(self, ngram_model, mock_llm, test_tokens, test_context):
        """Test weighted model log probability computation."""
        weighted = 0.6 * ngram_model + 0.4 * mock_llm

        logprobs = weighted.logprobs(test_tokens, test_context)

        assert isinstance(logprobs, np.ndarray)
        assert logprobs.shape == (len(test_tokens),)
        assert not np.any(np.isnan(logprobs))

    def test_model_scoring(self, ngram_model, test_context, test_tokens):
        """Test model scoring functionality."""
        sequence = test_context + test_tokens

        score = ngram_model.score(sequence)

        assert isinstance(score, (int, float, np.number))
        assert not np.isnan(score)


class TestAlgebraicProperties:
    """Test algebraic properties of model operations."""

    def test_mixture_weights_normalize(self, sample_corpus):
        """Test that mixture weights are properly normalized."""
        model_a = MockLLM(vocab_size=50, seed=1, name="A")
        model_b = MockLLM(vocab_size=50, seed=2, name="B")

        # Test various weight combinations
        test_weights = [(2.0, 3.0), (0.1, 0.9), (10, 5)]

        for w1, w2 in test_weights:
            mixture = w1 * model_a + w2 * model_b
            total_weight = sum(mixture.weights)
            assert abs(total_weight - 1.0) < 1e-6, f"Weights {mixture.weights} don't sum to 1"

    def test_addition_commutativity_behavior(self, sample_corpus):
        """Test that addition creates similar mixture structures regardless of order."""
        model_a = MockLLM(vocab_size=50, seed=1, name="A")
        model_b = MockLLM(vocab_size=50, seed=2, name="B")

        mixture_1 = model_a + model_b
        mixture_2 = model_b + model_a

        # Both should be mixture models with same number of components
        assert isinstance(mixture_1, MixtureModel)
        assert isinstance(mixture_2, MixtureModel)
        assert len(mixture_1.models) == len(mixture_2.models)

    def test_fallback_chaining(self, sample_corpus):
        """Test that fallback operations can be chained."""
        model_a = MockLLM(vocab_size=50, seed=1, name="A")
        model_b = MockLLM(vocab_size=50, seed=2, name="B")
        model_c = MockLLM(vocab_size=50, seed=3, name="C")

        chained_fallback = model_a | model_b | model_c

        # Should create a valid fallback model
        assert chained_fallback is not None
        assert hasattr(chained_fallback, 'logprobs')

    def test_complex_algebraic_expression(self, sample_corpus):
        """Test complex algebraic expressions with multiple operations."""
        n3 = NGramModel(sample_corpus, n=3)
        n4 = NGramModel(sample_corpus, n=4)
        llm = MockLLM(vocab_size=100, seed=42)

        # Complex expression: (0.3*n3 + 0.4*n4 + 0.3*llm) | n3
        complex_model = (0.3 * n3 + 0.4 * n4 + 0.3 * llm) | n3

        assert complex_model is not None

        # Test that it can compute logprobs
        context = [1, 2, 3, 4, 5]
        tokens = [10, 11, 12]

        logprobs = complex_model.logprobs(tokens, context)
        assert isinstance(logprobs, np.ndarray)
        assert not np.any(np.isnan(logprobs))

    def test_model_string_representation(self, ngram_model, mock_llm):
        """Test that models have meaningful string representations."""
        mixture = ngram_model + mock_llm
        weighted = 0.3 * ngram_model + 0.7 * mock_llm

        # Should have string representations (exact format may vary)
        assert str(mixture) is not None
        assert str(weighted) is not None
        assert len(str(mixture)) > 0
        assert len(str(weighted)) > 0


class TestModelInterfaces:
    """Test that all models conform to the LanguageModel interface."""

    def test_base_language_model_interface(self, test_language_model):
        """Test that base LanguageModel interface works correctly."""
        assert isinstance(test_language_model, LanguageModel)
        assert hasattr(test_language_model, 'logprobs')
        assert hasattr(test_language_model, 'sample')

    def test_ngram_model_interface(self, ngram_model):
        """Test that NGramModel implements LanguageModel interface."""
        assert isinstance(ngram_model, LanguageModel)

        # Test interface methods exist and work
        logprobs = ngram_model.logprobs([1, 2, 3])
        assert isinstance(logprobs, np.ndarray)

        samples = ngram_model.sample(max_tokens=3)
        assert isinstance(samples, list)

    def test_mock_llm_interface(self, mock_llm):
        """Test that MockLLM implements LanguageModel interface."""
        assert isinstance(mock_llm, LanguageModel)

        # Test interface methods exist and work
        logprobs = mock_llm.logprobs([1, 2, 3])
        assert isinstance(logprobs, np.ndarray)

        samples = mock_llm.sample(max_tokens=3)
        assert isinstance(samples, list)

    def test_mixture_model_interface(self, ngram_model, mock_llm):
        """Test that MixtureModel implements LanguageModel interface."""
        mixture = ngram_model + mock_llm
        assert isinstance(mixture, LanguageModel)

        # Test interface methods exist and work
        logprobs = mixture.logprobs([1, 2, 3])
        assert isinstance(logprobs, np.ndarray)

        samples = mixture.sample(max_tokens=3)
        assert isinstance(samples, list)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_token_list(self, ngram_model):
        """Test behavior with empty token lists."""
        logprobs = ngram_model.logprobs([])
        assert isinstance(logprobs, np.ndarray)
        assert len(logprobs) == 0

    def test_zero_max_tokens_sampling(self, ngram_model):
        """Test sampling with zero max tokens."""
        samples = ngram_model.sample(max_tokens=0)
        assert isinstance(samples, list)
        assert len(samples) == 0

    def test_large_context(self, ngram_model):
        """Test behavior with very large contexts."""
        large_context = list(range(1000))

        # Should not crash
        logprobs = ngram_model.logprobs([1, 2, 3], large_context)
        assert isinstance(logprobs, np.ndarray)

    def test_single_model_mixture(self, ngram_model):
        """Test mixture behavior with a single model."""
        # This might create a degenerate case or just return the model
        single_mixture = 1.0 * ngram_model

        # Should still behave like a language model
        assert hasattr(single_mixture, 'logprobs')
        assert hasattr(single_mixture, 'sample')


@pytest.mark.parametrize("weight", [0.1, 0.5, 0.9])
def test_mixture_weights_parametrized(ngram_model, mock_llm, weight):
    """Test mixture models with different weight configurations."""
    mixture = weight * ngram_model + (1 - weight) * mock_llm

    # Should create valid mixture
    assert isinstance(mixture, MixtureModel)

    # Test computation works
    logprobs = mixture.logprobs([1, 2, 3])
    assert isinstance(logprobs, np.ndarray)
    assert not np.any(np.isnan(logprobs))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_ngram_orders(sample_corpus, n):
    """Test n-gram models of different orders."""
    ngram = NGramModel(sample_corpus, n=n)

    assert isinstance(ngram, LanguageModel)

    # Test basic functionality
    logprobs = ngram.logprobs([1, 2, 3])
    assert isinstance(logprobs, np.ndarray)

    samples = ngram.sample(max_tokens=3)
    assert isinstance(samples, list)