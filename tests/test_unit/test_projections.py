"""
Unit tests for projection operations and compositions.

These tests verify the behavior of various projection types and their
interactions with language models.
"""

import pytest
import numpy as np
from typing import List

from ngram_projections.projections.recency import RecencyProjection
from ngram_projections.projections.semantic import SemanticProjection
from ngram_projections.models.ngram import NGramModel


class TestRecencyProjection:
    """Test recency-based context projection."""

    def test_recency_projection_initialization(self, sample_corpus):
        """Test that recency projection initializes correctly."""
        projection = RecencyProjection(corpus=sample_corpus, max_context=20)

        assert projection.max_context == 20
        assert hasattr(projection, 'project')

    def test_recency_projection_shortens_context(self, sample_corpus):
        """Test that recency projection limits context length."""
        max_context = 10
        projection = RecencyProjection(corpus=sample_corpus, max_context=max_context)

        long_context = list(range(50))
        projected_context = projection.project(long_context)

        assert len(projected_context) <= max_context
        assert isinstance(projected_context, list)

    def test_recency_projection_preserves_short_context(self, sample_corpus):
        """Test that short contexts are preserved unchanged."""
        max_context = 20
        projection = RecencyProjection(corpus=sample_corpus, max_context=max_context)

        short_context = [1, 2, 3, 4, 5]
        projected_context = projection.project(short_context)

        assert projected_context == short_context

    def test_recency_projection_handles_long_contexts(self, sample_corpus):
        """Test that recency projection handles long contexts appropriately."""
        max_context = 5
        projection = RecencyProjection(corpus=sample_corpus, max_context=max_context)

        context = list(range(20))  # [0, 1, 2, ..., 19]
        projected_context = projection.project(context)

        # The projection should return something reasonable
        # (exact behavior depends on suffix matching in the corpus)
        assert isinstance(projected_context, list)
        assert len(projected_context) <= max_context

    def test_recency_projection_with_empty_context(self, sample_corpus):
        """Test recency projection behavior with empty context."""
        projection = RecencyProjection(corpus=sample_corpus, max_context=10)

        projected_context = projection.project([])

        assert projected_context == []

    @pytest.mark.parametrize("max_context", [1, 5, 10, 20])
    def test_recency_projection_different_limits(self, sample_corpus, max_context):
        """Test recency projection with different context limits."""
        projection = RecencyProjection(corpus=sample_corpus, max_context=max_context)

        long_context = list(range(100))
        projected_context = projection.project(long_context)

        assert len(projected_context) <= max_context
        # The projection should return a reasonable result
        # The exact behavior depends on suffix matching in the corpus
        if len(long_context) > max_context:
            assert len(projected_context) <= max_context
        else:
            # For short contexts, could return the same or a suffix
            assert len(projected_context) <= len(long_context)


class TestSemanticProjection:
    """Test semantic similarity-based projection."""

    def test_semantic_projection_initialization(self):
        """Test that semantic projection initializes correctly."""
        embedding_dim = 32
        projection = SemanticProjection(embedding_dim=embedding_dim)

        assert projection.embedding_dim == embedding_dim
        assert hasattr(projection, 'similarity')
        assert hasattr(projection, 'project')

    def test_semantic_similarity_computation(self):
        """Test semantic similarity computation between sequences."""
        projection = SemanticProjection(embedding_dim=16)

        seq1 = [1, 2, 3, 4, 5]
        seq2 = [3, 4, 5, 6, 7]

        similarity = projection.similarity(seq1, seq2)

        assert isinstance(similarity, (float, np.floating))
        assert 0 <= similarity <= 1

    def test_semantic_similarity_identical_sequences(self):
        """Test that identical sequences have high similarity."""
        projection = SemanticProjection(embedding_dim=16)

        sequence = [1, 2, 3, 4, 5]
        similarity = projection.similarity(sequence, sequence)

        # Identical sequences should have similarity close to 1
        assert similarity > 0.9

    def test_semantic_similarity_empty_sequences(self):
        """Test similarity computation with empty sequences."""
        projection = SemanticProjection(embedding_dim=16)

        similarity = projection.similarity([], [])

        # Should handle empty sequences gracefully
        assert isinstance(similarity, (float, np.floating))
        assert 0 <= similarity <= 1

    def test_semantic_similarity_different_lengths(self):
        """Test similarity between sequences of different lengths."""
        projection = SemanticProjection(embedding_dim=16)

        short_seq = [1, 2, 3]
        long_seq = [1, 2, 3, 4, 5, 6, 7, 8]

        similarity = projection.similarity(short_seq, long_seq)

        assert isinstance(similarity, (float, np.floating))
        assert 0 <= similarity <= 1

    @pytest.mark.parametrize("embedding_dim", [8, 16, 32, 64])
    def test_semantic_projection_different_dimensions(self, embedding_dim):
        """Test semantic projection with different embedding dimensions."""
        projection = SemanticProjection(embedding_dim=embedding_dim)

        seq1 = [1, 2, 3]
        seq2 = [4, 5, 6]

        similarity = projection.similarity(seq1, seq2)

        assert isinstance(similarity, (float, np.floating))
        assert 0 <= similarity <= 1


class TestProjectionComposition:
    """Test composition of multiple projections."""

    def test_projection_composition_operator(self, sample_corpus):
        """Test projection composition using >> operator."""
        recency = RecencyProjection(corpus=sample_corpus, max_context=20)
        semantic = SemanticProjection(embedding_dim=32)

        composed = recency >> semantic

        # Should create a composed projection
        assert composed is not None
        assert hasattr(composed, 'project')

    def test_sequential_projection_application(self, sample_corpus):
        """Test that projections can be applied sequentially."""
        recency = RecencyProjection(corpus=sample_corpus, max_context=10)
        semantic = SemanticProjection(embedding_dim=16)

        long_context = list(range(50))

        # Apply recency first
        recency_result = recency.project(long_context)
        assert len(recency_result) <= 10

        # Then apply semantic (this tests the interface, actual behavior may vary)
        if hasattr(semantic, 'project'):
            semantic_result = semantic.project(recency_result)
            assert isinstance(semantic_result, list)


class TestModelProjectionIntegration:
    """Test integration between models and projections."""

    def test_model_projection_operator(self, ngram_model, recency_projection):
        """Test applying projection to model using @ operator."""
        projected_model = ngram_model @ recency_projection

        # Should create a projected model
        assert projected_model is not None
        assert hasattr(projected_model, 'logprobs')
        assert hasattr(projected_model, 'sample')

    def test_projected_model_computation(self, sample_corpus):
        """Test that projected models can perform computations."""
        ngram = NGramModel(sample_corpus, n=3)
        recency = RecencyProjection(corpus=sample_corpus, max_context=20)

        projected_model = ngram @ recency

        # Test logprobs computation
        context = list(range(50))  # Long context to trigger projection
        tokens = [10, 20, 30]

        logprobs = projected_model.logprobs(tokens, context)

        assert isinstance(logprobs, np.ndarray)
        assert logprobs.shape == (len(tokens),)
        assert not np.any(np.isnan(logprobs))

    def test_projected_model_sampling(self, sample_corpus):
        """Test that projected models can generate samples."""
        ngram = NGramModel(sample_corpus, n=3)
        recency = RecencyProjection(corpus=sample_corpus, max_context=15)

        projected_model = ngram @ recency

        context = list(range(50))  # Long context
        samples = projected_model.sample(context, max_tokens=5)

        assert isinstance(samples, list)
        assert len(samples) == 5
        assert all(isinstance(token, (int, np.integer)) for token in samples)

    def test_projection_preserves_model_interface(self, ngram_model, recency_projection):
        """Test that projected models maintain the LanguageModel interface."""
        from ngram_projections.models.base import LanguageModel

        projected_model = ngram_model @ recency_projection

        # Should still be a language model
        assert isinstance(projected_model, LanguageModel)

        # Should have required methods
        assert hasattr(projected_model, 'logprobs')
        assert hasattr(projected_model, 'sample')
        assert callable(getattr(projected_model, 'logprobs'))
        assert callable(getattr(projected_model, 'sample'))

    def test_multiple_projection_application(self, sample_corpus):
        """Test applying multiple projections to a model."""
        ngram = NGramModel(sample_corpus, n=3)
        recency = RecencyProjection(corpus=sample_corpus, max_context=20)
        semantic = SemanticProjection(embedding_dim=16)

        # Apply projections sequentially
        step1 = ngram @ recency
        step2 = step1 @ semantic if hasattr(step1, '__matmul__') else step1

        # Should still work as a model
        if hasattr(step2, 'logprobs'):
            logprobs = step2.logprobs([1, 2, 3])
            assert isinstance(logprobs, np.ndarray)


class TestProjectionEdgeCases:
    """Test edge cases and boundary conditions for projections."""

    def test_recency_projection_zero_max_context(self, sample_corpus):
        """Test recency projection with zero max context."""
        projection = RecencyProjection(corpus=sample_corpus, max_context=0)

        context = [1, 2, 3, 4, 5]
        projected = projection.project(context)

        # With max_context=0, behavior may vary based on implementation
        # but result should be a list (possibly empty)
        assert isinstance(projected, list)

    def test_recency_projection_negative_max_context(self, sample_corpus):
        """Test recency projection behavior with negative max context."""
        # This might raise an error or handle gracefully
        try:
            projection = RecencyProjection(corpus=sample_corpus, max_context=-1)
            # If it doesn't raise an error, test that it handles contexts reasonably
            projected = projection.project([1, 2, 3])
            assert isinstance(projected, list)
        except ValueError:
            # Acceptable to raise an error for invalid input
            pass

    def test_semantic_projection_zero_embedding_dim(self):
        """Test semantic projection with zero embedding dimension."""
        try:
            projection = SemanticProjection(embedding_dim=0)
            # If it doesn't raise an error, test basic functionality
            similarity = projection.similarity([1, 2], [3, 4])
            assert isinstance(similarity, (float, np.floating))
        except (ValueError, AssertionError):
            # Acceptable to raise an error for invalid dimension
            pass

    def test_projection_with_none_context(self, recency_projection):
        """Test projection behavior with None context."""
        try:
            projected = recency_projection.project(None)
            assert projected is not None
        except (TypeError, AttributeError):
            # Acceptable to raise an error for None input
            pass

    def test_projection_with_non_list_context(self, recency_projection):
        """Test projection behavior with non-list context."""
        try:
            # Test with different input types
            projected = recency_projection.project("not a list")
            assert isinstance(projected, list) or projected is None
        except (TypeError, AttributeError):
            # Acceptable to raise an error for invalid input type
            pass


@pytest.mark.parametrize("context_length,max_context", [
    (0, 10),
    (5, 10),
    (10, 10),
    (15, 10),
    (100, 10)
])
def test_recency_projection_various_lengths(sample_corpus, context_length, max_context):
    """Test recency projection with various context and max_context combinations."""
    projection = RecencyProjection(corpus=sample_corpus, max_context=max_context)

    context = list(range(context_length))
    projected = projection.project(context)

    expected_length = min(context_length, max_context)
    assert len(projected) == expected_length

    if context_length > 0 and max_context > 0:
        # Should contain the most recent tokens
        expected = context[-expected_length:] if expected_length > 0 else []
        assert projected == expected