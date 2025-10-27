"""
Integration tests for model composition workflows.

These tests verify that different components work together correctly
in realistic usage scenarios, focusing on end-to-end behavior.
"""

import pytest
import numpy as np
from typing import List, Dict

from langcalc.models.base import LanguageModel
from langcalc.models.ngram import NGramModel
from langcalc.models.llm import MockLLM
from langcalc.projections.recency import RecencyProjection
from langcalc.projections.semantic import SemanticProjection


class TestBasicCompositionWorkflows:
    """Test basic model composition workflows."""

    def test_simple_mixture_workflow(self, sample_corpus):
        """Test complete workflow with simple mixture model."""
        # Create components
        ngram = NGramModel(sample_corpus, n=3)
        llm = MockLLM(vocab_size=100, seed=42)

        # Create mixture
        mixture = 0.6 * ngram + 0.4 * llm

        # Test complete evaluation workflow
        context = [1, 2, 3, 4]
        tokens = [10, 20, 30]

        # Compute probabilities
        logprobs = mixture.logprobs(tokens, context)
        assert isinstance(logprobs, np.ndarray)
        assert len(logprobs) == len(tokens)
        assert not np.any(np.isnan(logprobs))

        # Generate samples
        samples = mixture.sample(context, max_tokens=5)
        assert isinstance(samples, list)
        assert len(samples) == 5

        # Score complete sequences
        full_sequence = context + tokens
        score = mixture.score(full_sequence)
        assert isinstance(score, (float, np.floating))
        assert not np.isnan(score)

    def test_projected_model_workflow(self, sample_corpus):
        """Test workflow with projected models."""
        # Create base model
        ngram = NGramModel(sample_corpus, n=4)

        # Create projection
        recency = RecencyProjection(corpus=sample_corpus, max_context=15)

        # Apply projection
        projected_model = ngram @ recency

        # Test with long context that should trigger projection
        long_context = list(range(100))
        tokens = [5, 6, 7]

        # Should work despite long context
        logprobs = projected_model.logprobs(tokens, long_context)
        assert isinstance(logprobs, np.ndarray)
        assert len(logprobs) == len(tokens)

        # Sampling should also work
        samples = projected_model.sample(long_context, max_tokens=3)
        assert isinstance(samples, list)
        assert len(samples) == 3

    def test_complex_composition_workflow(self, sample_corpus):
        """Test complex composition with multiple operations."""
        # Create multiple components
        n3 = NGramModel(sample_corpus, n=3)
        n4 = NGramModel(sample_corpus, n=4)
        llm = MockLLM(vocab_size=100, seed=123)
        recency = RecencyProjection(corpus=sample_corpus, max_context=20)

        # Create complex composition: (projected n-gram + weighted mixture) | fallback
        base_mixture = 0.3 * (n3 @ recency) + 0.4 * n4 + 0.3 * llm
        final_model = base_mixture | n3

        # Test the complex model
        context = list(range(50))  # Long context
        test_tokens = [15, 25, 35]

        # Should compute probabilities
        logprobs = final_model.logprobs(test_tokens, context)
        assert isinstance(logprobs, np.ndarray)
        assert len(logprobs) == len(test_tokens)
        assert not np.any(np.isnan(logprobs))

        # Should generate samples
        samples = final_model.sample(context, max_tokens=5)
        assert isinstance(samples, list)
        assert len(samples) == 5

    def test_fallback_chain_workflow(self, sample_corpus):
        """Test fallback chain behavior in realistic scenario."""
        # Create models with different capabilities
        primary = NGramModel(sample_corpus, n=5)  # High-order, might fail on short contexts
        secondary = NGramModel(sample_corpus, n=3)  # Medium-order fallback
        tertiary = MockLLM(vocab_size=100, seed=42)  # Always available fallback

        # Chain fallbacks
        fallback_chain = primary | secondary | tertiary

        # Test with various contexts
        test_contexts = [
            [],  # Empty context
            [1],  # Very short context
            [1, 2, 3],  # Medium context
            list(range(20))  # Long context
        ]

        for context in test_contexts:
            # Should always produce valid results due to fallback chain
            logprobs = fallback_chain.logprobs([10, 20], context)
            assert isinstance(logprobs, np.ndarray)
            assert len(logprobs) == 2
            assert not np.any(np.isnan(logprobs))

            # Should always be able to sample
            samples = fallback_chain.sample(context, max_tokens=3)
            assert isinstance(samples, list)
            assert len(samples) == 3


class TestProjectionIntegration:
    """Test integration of projections with various models."""

    def test_multiple_projections_composition(self, sample_corpus):
        """Test applying multiple projections in sequence."""
        # Create base model
        ngram = NGramModel(sample_corpus, n=3)

        # Create projections
        recency = RecencyProjection(corpus=sample_corpus, max_context=30)
        semantic = SemanticProjection(embedding_dim=16)

        # Apply projections sequentially
        step1 = ngram @ recency
        # Note: Second projection might not be supported by all implementations
        # but we test the interface
        if hasattr(step1, '__matmul__'):
            step2 = step1 @ semantic
            projected_model = step2
        else:
            projected_model = step1

        # Test with very long context
        very_long_context = list(range(200))
        tokens = [1, 2, 3]

        logprobs = projected_model.logprobs(tokens, very_long_context)
        assert isinstance(logprobs, np.ndarray)
        assert len(logprobs) == len(tokens)

    def test_projection_with_mixture_models(self, sample_corpus):
        """Test projections applied to mixture models."""
        # Create mixture
        ngram1 = NGramModel(sample_corpus, n=3)
        ngram2 = NGramModel(sample_corpus, n=4)
        mixture = 0.5 * ngram1 + 0.5 * ngram2

        # Apply projection to mixture
        recency = RecencyProjection(corpus=sample_corpus, max_context=25)

        if hasattr(mixture, '__matmul__'):
            projected_mixture = mixture @ recency

            # Test with long context
            long_context = list(range(100))
            logprobs = projected_mixture.logprobs([5, 6, 7], long_context)
            assert isinstance(logprobs, np.ndarray)

    def test_projection_impact_on_performance(self, sample_corpus):
        """Test that projections have measurable impact on context handling."""
        # Create model and projection
        ngram = NGramModel(sample_corpus, n=4)
        recency = RecencyProjection(corpus=sample_corpus, max_context=10)

        # Test without projection
        long_context = list(range(100))
        tokens = [1, 2, 3]

        logprobs_original = ngram.logprobs(tokens, long_context)

        # Test with projection
        projected = ngram @ recency
        logprobs_projected = projected.logprobs(tokens, long_context)

        # Both should be valid, but may differ due to context truncation
        assert isinstance(logprobs_original, np.ndarray)
        assert isinstance(logprobs_projected, np.ndarray)
        assert len(logprobs_original) == len(logprobs_projected)


class TestDataFlowIntegration:
    """Test data flow through complex model compositions."""

    def test_consistent_vocabulary_handling(self, sample_corpus):
        """Test that vocabulary is handled consistently across compositions."""
        # Create models with same vocabulary scope
        vocab_size = 50
        corpus_subset = [token % vocab_size for token in sample_corpus]

        ngram = NGramModel(corpus_subset, n=3)
        llm = MockLLM(vocab_size=vocab_size, seed=42)

        # Create mixture
        mixture = 0.7 * ngram + 0.3 * llm

        # Test with tokens within vocabulary
        valid_tokens = [10, 20, 30]  # Within vocab_size
        logprobs = mixture.logprobs(valid_tokens)

        assert isinstance(logprobs, np.ndarray)
        assert len(logprobs) == len(valid_tokens)

        # Test sampling - should produce valid tokens
        samples = mixture.sample(max_tokens=10)
        assert all(isinstance(token, (int, np.integer)) for token in samples)

    def test_error_propagation_in_compositions(self, sample_corpus):
        """Test that errors are handled gracefully in compositions."""
        # Create models
        ngram = NGramModel(sample_corpus, n=3)
        llm = MockLLM(vocab_size=100, seed=42)

        # Create composition
        composition = 0.5 * ngram + 0.5 * llm

        # Test with potentially problematic inputs
        edge_cases = [
            [],  # Empty tokens
            [999999],  # Very large token ID
            [-1],  # Negative token (if not handled)
        ]

        for tokens in edge_cases:
            try:
                logprobs = composition.logprobs(tokens)
                # If no error, result should still be valid
                if logprobs is not None:
                    assert isinstance(logprobs, np.ndarray)
                    assert len(logprobs) == len(tokens)
            except (ValueError, IndexError, TypeError):
                # Acceptable to raise errors for invalid inputs
                pass

    def test_memory_efficiency_in_large_compositions(self, sample_corpus):
        """Test memory efficiency with large model compositions."""
        # Create multiple models
        models = []
        models.append(NGramModel(sample_corpus, n=2))
        models.append(NGramModel(sample_corpus, n=3))
        models.extend([MockLLM(vocab_size=100, seed=i) for i in range(3)])

        # Create large mixture using proper model algebra
        weight = 1.0 / len(models)
        mixture = None
        for model in models:
            weighted_model = weight * model
            if mixture is None:
                mixture = weighted_model
            else:
                mixture = mixture + weighted_model

        # Should still function efficiently
        test_tokens = [1, 2, 3, 4, 5]
        logprobs = mixture.logprobs(test_tokens)

        assert isinstance(logprobs, np.ndarray)
        assert len(logprobs) == len(test_tokens)

    def test_deterministic_behavior_in_compositions(self, sample_corpus):
        """Test that compositions behave deterministically."""
        # Create identical compositions
        ngram1 = NGramModel(sample_corpus, n=3)
        ngram2 = NGramModel(sample_corpus, n=3)
        llm1 = MockLLM(vocab_size=100, seed=42)
        llm2 = MockLLM(vocab_size=100, seed=42)

        mixture1 = 0.6 * ngram1 + 0.4 * llm1
        mixture2 = 0.6 * ngram2 + 0.4 * llm2

        # Should produce identical results
        context = [1, 2, 3]
        tokens = [10, 20, 30]

        logprobs1 = mixture1.logprobs(tokens, context)
        logprobs2 = mixture2.logprobs(tokens, context)

        np.testing.assert_array_almost_equal(logprobs1, logprobs2)


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""

    def test_factual_grounding_scenario(self, factual_sentences):
        """Test a realistic factual grounding scenario."""
        # Convert sentences to token sequences (simplified)
        factual_corpus = []
        for sentence in factual_sentences:
            # Simple tokenization: hash words to integers
            tokens = [hash(word) % 1000 for word in sentence.lower().split()]
            factual_corpus.extend(tokens)

        # Create factual n-gram model
        factual_ngram = NGramModel(factual_corpus, n=3)

        # Create general LLM
        general_llm = MockLLM(vocab_size=1000, seed=42)

        # Create grounded model: mostly LLM with factual constraints
        grounded_model = 0.95 * general_llm + 0.05 * factual_ngram

        # Test prediction
        test_context = [hash("the") % 1000, hash("capital") % 1000, hash("of") % 1000]
        logprobs = grounded_model.logprobs([hash("france") % 1000], test_context)

        assert isinstance(logprobs, np.ndarray)
        assert len(logprobs) == 1
        assert not np.isnan(logprobs[0])

    def test_domain_adaptation_scenario(self, sample_corpus):
        """Test domain adaptation through model composition."""
        # General model (trained on broad corpus)
        general_model = NGramModel(sample_corpus, n=3)

        # Domain-specific model (trained on specific patterns)
        domain_corpus = [1, 2, 3, 4] * 50  # Specific pattern
        domain_model = NGramModel(domain_corpus, n=3)

        # Adaptive mixture: more domain-specific for domain patterns
        domain_weight = 0.7
        general_weight = 1.0 - domain_weight
        adaptive_model = domain_weight * domain_model + general_weight * general_model

        # Test on domain-specific context
        domain_context = [1, 2]
        logprobs = adaptive_model.logprobs([3, 4], domain_context)

        assert isinstance(logprobs, np.ndarray)
        assert len(logprobs) == 2

        # Test on general context
        general_context = [99, 88]
        logprobs_general = adaptive_model.logprobs([77, 66], general_context)

        assert isinstance(logprobs_general, np.ndarray)
        assert len(logprobs_general) == 2

    def test_multi_scale_modeling_scenario(self, sample_corpus):
        """Test multi-scale modeling with different n-gram orders."""
        # Create models of different scales
        unigram = NGramModel(sample_corpus, n=1)  # Local statistics
        trigram = NGramModel(sample_corpus, n=3)  # Medium-range dependencies
        fivegram = NGramModel(sample_corpus, n=5)  # Long-range dependencies

        # Combine with recency projection for long contexts
        recency = RecencyProjection(corpus=sample_corpus, max_context=20)
        long_range_model = fivegram @ recency

        # Create hierarchical model
        multi_scale = 0.2 * unigram + 0.5 * trigram + 0.3 * long_range_model

        # Test with various context lengths
        contexts = [
            [1, 2],  # Short context
            [1, 2, 3, 4, 5, 6],  # Medium context
            list(range(50))  # Long context
        ]

        for context in contexts:
            logprobs = multi_scale.logprobs([10, 11], context)
            assert isinstance(logprobs, np.ndarray)
            assert len(logprobs) == 2
            assert not np.any(np.isnan(logprobs))


@pytest.mark.parametrize("mixture_weight", [0.1, 0.5, 0.9])
def test_parametrized_mixture_integration(sample_corpus, mixture_weight):
    """Test mixture integration with various weight settings."""
    ngram = NGramModel(sample_corpus, n=3)
    llm = MockLLM(vocab_size=100, seed=42)

    mixture = mixture_weight * ngram + (1 - mixture_weight) * llm

    # Should work for all weight combinations
    logprobs = mixture.logprobs([1, 2, 3])
    assert isinstance(logprobs, np.ndarray)
    assert len(logprobs) == 3
    assert not np.any(np.isnan(logprobs))


@pytest.mark.integration
@pytest.mark.slow
def test_large_scale_composition(sample_corpus):
    """Test large-scale model composition (marked as slow test)."""
    # Create many models for large composition
    models = []
    for i in range(10):
        models.append(NGramModel(sample_corpus, n=2 + (i % 3)))
        models.append(MockLLM(vocab_size=100, seed=i))

    # Create uniform mixture using proper model algebra
    weight = 1.0 / len(models)
    large_mixture = None
    for model in models:
        weighted_model = weight * model
        if large_mixture is None:
            large_mixture = weighted_model
        else:
            large_mixture = large_mixture + weighted_model

    # Should still function
    logprobs = large_mixture.logprobs([1, 2, 3])
    assert isinstance(logprobs, np.ndarray)
    assert len(logprobs) == 3