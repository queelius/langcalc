#!/usr/bin/env python3
"""
Tests for InfinigramModel - LangCalc adapter for the external infinigram package.

These tests focus on the adapter functionality and integration with LangCalc's
LanguageModel interface. The underlying infinigram implementation is tested
in the infinigram package itself.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add langcalc to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langcalc.models import InfinigramModel, MockLLM


class TestInfinigramModel:
    """Test InfinigramModel adapter for LangCalc."""

    def test_infinigram_initialization(self):
        """Test basic initialization."""
        corpus = [1, 2, 3, 4, 2, 3, 5]
        model = InfinigramModel(corpus)

        assert model is not None
        assert model.vocab_size == 256  # Byte-level vocabulary
        assert model.max_length is None

    def test_infinigram_with_max_length(self):
        """Test initialization with max_length."""
        corpus = [1, 2, 3, 4, 2, 3, 5]
        model = InfinigramModel(corpus, max_length=5)

        assert model.max_length == 5

    def test_infinigram_with_parameters(self):
        """Test initialization with various parameters."""
        corpus = [1, 2, 3, 4, 5]
        model = InfinigramModel(corpus, max_length=10, min_count=2, smoothing=0.1)

        assert isinstance(model, InfinigramModel)
        assert model.max_length == 10
        assert model.min_count == 2
        assert model.smoothing == 0.1

    def test_logprobs_interface(self):
        """Test logprobs method conforms to LanguageModel interface."""
        text = "the cat sat on the mat"
        corpus = list(text.encode('utf-8'))
        model = InfinigramModel(corpus)

        context = list("the cat".encode('utf-8'))
        tokens = list(range(256))

        logprobs = model.logprobs(tokens, context)

        # Check return type and shape
        assert isinstance(logprobs, np.ndarray)
        assert len(logprobs) == 256
        assert logprobs.dtype in [np.float64, np.float32, np.float16]

        # Check all logprobs are valid (finite or -inf)
        assert all(np.isfinite(lp) or lp == -np.inf for lp in logprobs)

    def test_logprobs_no_context(self):
        """Test logprobs with no context."""
        corpus = [1, 2, 3, 4, 2, 3, 5]
        model = InfinigramModel(corpus)

        logprobs = model.logprobs([1, 2, 3, 4, 5], context=None)

        assert isinstance(logprobs, np.ndarray)
        assert len(logprobs) == 5

    def test_sample_interface(self):
        """Test sample method conforms to LanguageModel interface."""
        text = "the cat sat on the mat"
        corpus = list(text.encode('utf-8'))
        model = InfinigramModel(corpus)

        context = list("the".encode('utf-8'))
        samples = model.sample(context, max_tokens=10)

        # Check return type
        assert isinstance(samples, list)
        assert len(samples) == 10
        assert all(isinstance(token, (int, np.integer)) for token in samples)

    def test_sample_temperature(self):
        """Test sampling with different temperatures."""
        text = "the cat sat on the mat. the dog sat on the rug."
        corpus = list(text.encode('utf-8'))
        model = InfinigramModel(corpus)

        context = list("the".encode('utf-8'))

        # Lower temperature should be more deterministic
        samples1 = model.sample(context, temperature=0.1, max_tokens=5)
        samples2 = model.sample(context, temperature=2.0, max_tokens=5)

        assert len(samples1) == 5
        assert len(samples2) == 5

    def test_score_interface(self):
        """Test score method conforms to LanguageModel interface."""
        text = "the cat sat on the mat"
        corpus = list(text.encode('utf-8'))
        model = InfinigramModel(corpus)

        sequence = list("cat".encode('utf-8'))
        score = model.score(sequence)

        # Check return type
        assert isinstance(score, (float, np.floating))
        # Score should be negative (log probability)
        assert score <= 0.0

    def test_score_empty_sequence(self):
        """Test scoring an empty sequence."""
        corpus = [1, 2, 3, 4, 5]
        model = InfinigramModel(corpus)

        score = model.score([])
        assert score == 0.0

    def test_algebraic_composition_addition(self):
        """Test model can be composed with addition."""
        text = "hello world"
        corpus = list(text.encode('utf-8'))
        infinigram = InfinigramModel(corpus)
        llm = MockLLM(vocab_size=256)

        # Test weighted addition
        mixed = 0.7 * llm + 0.3 * infinigram

        context = list("he".encode('utf-8'))
        logprobs = mixed.logprobs(list(range(256)), context)

        assert len(logprobs) == 256
        assert all(np.isfinite(lp) or lp == -np.inf for lp in logprobs)

    def test_algebraic_composition_fallback(self):
        """Test model can be composed with fallback operator."""
        text = "hello world"
        corpus = list(text.encode('utf-8'))
        infinigram = InfinigramModel(corpus)
        llm = MockLLM(vocab_size=256)

        # Test fallback composition
        fallback_model = llm | infinigram

        context = list("he".encode('utf-8'))
        logprobs = fallback_model.logprobs(list(range(256)), context)

        assert len(logprobs) == 256

    def test_confidence_passthrough(self):
        """Test confidence method is passed through to underlying model."""
        text = "the cat sat on the mat"
        corpus = list(text.encode('utf-8'))
        model = InfinigramModel(corpus)

        context = list("the cat".encode('utf-8'))
        confidence = model.confidence(context)

        # Check return type
        assert isinstance(confidence, (float, np.floating))
        # Confidence should be between 0 and 1
        assert 0.0 <= confidence <= 1.0

    def test_longest_suffix_passthrough(self):
        """Test longest_suffix method is passed through."""
        corpus = [1, 2, 3, 4, 2, 3, 5]
        model = InfinigramModel(corpus)

        context = [2, 3]
        pos, length = model.longest_suffix(context)

        # Check return type (accept both int and numpy integers)
        assert isinstance(pos, (int, np.integer))
        assert isinstance(length, (int, np.integer))
        assert length >= 0

    def test_update_passthrough(self):
        """Test update method is passed through."""
        corpus = [1, 2, 3, 4, 5]
        model = InfinigramModel(corpus)

        initial_size = len(model.infinigram.corpus)

        # Add new tokens
        model.update([6, 7, 8])

        # Check corpus was updated
        assert len(model.infinigram.corpus) == initial_size + 3
        assert model.infinigram.corpus[-3:] == [6, 7, 8]

    def test_smoothing_parameter(self):
        """Test smoothing parameter is used."""
        corpus = [1, 2, 3, 4, 5]

        # Create with different smoothing values
        model1 = InfinigramModel(corpus, smoothing=0.01)
        model2 = InfinigramModel(corpus, smoothing=1.0)

        assert model1.smoothing == 0.01
        assert model2.smoothing == 1.0

        # With higher smoothing, unseen tokens should have higher probability
        context = [1]
        logprobs1 = model1.logprobs([99], context)  # Unseen token
        logprobs2 = model2.logprobs([99], context)

        # Higher smoothing should give higher (less negative) logprob
        assert logprobs2[0] > logprobs1[0]

    def test_repr(self):
        """Test string representation."""
        corpus = [1, 2, 3, 4, 5]
        model = InfinigramModel(corpus, max_length=10)

        repr_str = repr(model)
        assert "InfinigramModel" in repr_str
        assert "corpus_size=5" in repr_str
        assert "max_length=10" in repr_str

    def test_byte_level_text_encoding(self):
        """Test working with UTF-8 encoded text."""
        # Multi-byte UTF-8 characters
        text = "Hello 世界! 🌍"
        corpus = list(text.encode('utf-8'))

        model = InfinigramModel(corpus)

        # Should handle multi-byte sequences correctly
        context = list("Hello".encode('utf-8'))
        logprobs = model.logprobs(list(range(256)), context)

        assert len(logprobs) == 256
        assert all(np.isfinite(lp) or lp == -np.inf for lp in logprobs)


class TestInfinigramIntegration:
    """Test InfinigramModel integration with LangCalc framework."""

    def test_mixture_with_ngram(self):
        """Test mixing InfinigramModel with NGramModel."""
        from langcalc.models import NGramModel

        text = "the cat sat on the mat"
        corpus = list(text.encode('utf-8'))

        infinigram = InfinigramModel(corpus)
        ngram = NGramModel(corpus, n=3)

        # Mix the models
        mixed = 0.5 * infinigram + 0.5 * ngram

        context = list("the".encode('utf-8'))
        logprobs = mixed.logprobs(list(range(256)), context)

        assert len(logprobs) == 256

    def test_sequential_composition(self):
        """Test sequential composition with >> operator."""
        text = "hello world"
        corpus = list(text.encode('utf-8'))

        infinigram = InfinigramModel(corpus)
        llm = MockLLM(vocab_size=256)

        # Sequential composition
        sequential = infinigram >> llm

        # Should be able to sample
        samples = sequential.sample(max_tokens=5)
        assert len(samples) == 5
