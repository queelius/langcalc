"""
Shared pytest fixtures for the ngram-projections test suite.

This module provides common test fixtures and utilities used across
all test modules. Fixtures follow the principle of testing behavior
rather than implementation details.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Any

# Use new langcalc package imports
from langcalc.models.base import LanguageModel
from langcalc.models.ngram import NGramModel
from langcalc.models.llm import MockLLM
from langcalc.projections.recency import RecencyProjection
from langcalc.projections.semantic import SemanticProjection


@pytest.fixture(scope="session")
def random_seed():
    """Fixed random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_vocab() -> List[str]:
    """Sample vocabulary for testing."""
    return ["the", "cat", "sat", "on", "mat", "dog", "ran", "fast", "big", "small"]


@pytest.fixture
def sample_tokens(sample_vocab) -> List[int]:
    """Sample token IDs based on vocabulary."""
    vocab_size = len(sample_vocab)
    return list(range(vocab_size))


@pytest.fixture
def sample_corpus(random_seed) -> List[int]:
    """Generate a deterministic sample corpus for testing."""
    np.random.seed(random_seed)
    return list(np.random.randint(0, 100, size=1000))


@pytest.fixture
def small_corpus() -> List[int]:
    """Small corpus for quick tests."""
    return [1, 2, 3, 4, 5, 1, 2, 6, 7, 8, 1, 2, 3, 9, 10]


@pytest.fixture
def factual_sentences() -> List[str]:
    """Sample factual sentences for grounding tests."""
    return [
        "The capital of France is Paris",
        "Water boils at 100 degrees Celsius",
        "Einstein developed the theory of relativity",
        "The Earth orbits around the Sun",
        "Darwin proposed evolution by natural selection"
    ]


@pytest.fixture
def test_context() -> List[int]:
    """Standard test context for model evaluation."""
    return [1, 2, 3]


@pytest.fixture
def test_tokens() -> List[int]:
    """Standard test tokens for scoring."""
    return [10, 20, 30]


@pytest.fixture
def ngram_model(sample_corpus) -> NGramModel:
    """Create a trained n-gram model."""
    return NGramModel(sample_corpus, n=3)


@pytest.fixture
def mock_llm() -> MockLLM:
    """Create a mock LLM for testing."""
    return MockLLM(vocab_size=100, seed=42, name="TestLLM")


@pytest.fixture
def recency_projection(sample_corpus) -> RecencyProjection:
    """Create a recency projection."""
    return RecencyProjection(corpus=sample_corpus, max_context=20)


@pytest.fixture
def semantic_projection() -> SemanticProjection:
    """Create a semantic projection."""
    return SemanticProjection(embedding_dim=32)


@pytest.fixture
def model_configs() -> Dict[str, Dict[str, Any]]:
    """Standard model configurations for testing."""
    return {
        "small_ngram": {"n": 2, "vocab_size": 50},
        "medium_ngram": {"n": 3, "vocab_size": 100},
        "large_ngram": {"n": 4, "vocab_size": 100},
        "mock_llm": {"vocab_size": 100, "seed": 1}
    }


@pytest.fixture
def test_weights() -> List[float]:
    """Standard weight configurations for mixture testing."""
    return [0.3, 0.4, 0.3]


class TestLanguageModel(LanguageModel):
    """Minimal test implementation of LanguageModel for testing base functionality."""

    def __init__(self, name: str = "test_model"):
        self.name = name
        self.vocab_size = 10

    def logprobs(self, tokens: List[int], context: List[int] = None) -> np.ndarray:
        """Return deterministic log probabilities for testing."""
        return np.array([-1.0] * len(tokens))

    def sample(self, context: List[int] = None, temperature: float = 1.0,
               max_tokens: int = 5) -> List[int]:
        """Return deterministic samples for testing."""
        return list(range(max_tokens))

    def score(self, sequence: List[int]) -> float:
        """Return deterministic score for testing."""
        return -len(sequence) * 0.5  # Simple scoring based on sequence length

    def __repr__(self) -> str:
        return f"TestLanguageModel({self.name})"


@pytest.fixture
def test_language_model() -> TestLanguageModel:
    """Create a minimal test language model."""
    return TestLanguageModel()


# Helper functions for test assertions
def assert_valid_logprobs(logprobs: np.ndarray, expected_length: int = None):
    """Assert that log probabilities are valid."""
    assert isinstance(logprobs, np.ndarray)
    assert not np.any(np.isnan(logprobs))
    assert not np.any(np.isinf(logprobs))
    assert np.all(logprobs <= 0)  # Log probs should be <= 0
    if expected_length:
        assert len(logprobs) == expected_length


def assert_valid_samples(samples: List[int], expected_length: int = None):
    """Assert that samples are valid."""
    assert isinstance(samples, list)
    assert all(isinstance(token, int) for token in samples)
    if expected_length:
        assert len(samples) == expected_length


def assert_valid_probabilities(probs: Dict[str, float]):
    """Assert that probability distribution is valid."""
    assert isinstance(probs, dict)
    assert len(probs) > 0
    total_prob = sum(probs.values())
    assert abs(total_prob - 1.0) < 1e-6, f"Probabilities don't sum to 1: {total_prob}"
    assert all(0 <= p <= 1 for p in probs.values())


# Parametrized fixtures for comprehensive testing
@pytest.fixture(params=[2, 3, 4])
def ngram_orders(request):
    """Test different n-gram orders."""
    return request.param


@pytest.fixture(params=[0.1, 0.3, 0.5, 0.7, 0.9])
def mixture_weights(request):
    """Test different mixture weights."""
    return request.param


@pytest.fixture(params=[10, 50, 100])
def vocab_sizes(request):
    """Test different vocabulary sizes."""
    return request.param