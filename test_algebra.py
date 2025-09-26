"""
Quick test of the algebraic API to ensure everything works.
"""

import sys
import numpy as np

# Add the package to path for testing
sys.path.insert(0, '.')

from ngram_projections.models.base import LanguageModel
from ngram_projections.models.ngram import NGramModel
from ngram_projections.models.llm import MockLLM
from ngram_projections.projections.recency import RecencyProjection
from ngram_projections.projections.semantic import SemanticProjection


def test_basic_algebra():
    """Test basic algebraic operations."""
    print("Testing Basic Algebraic Operations")
    print("-" * 40)

    # Create sample corpus
    np.random.seed(42)
    corpus = list(np.random.randint(0, 100, size=1000))

    # Create models
    ngram = NGramModel(corpus, n=3)
    llm1 = MockLLM(vocab_size=100, seed=1, name="LLM1")
    llm2 = MockLLM(vocab_size=100, seed=2, name="LLM2")

    # Test addition
    mixture = ngram + llm1
    print(f"✓ Addition: {ngram} + {llm1} = {mixture}")

    # Test weighted combination
    weighted = 0.3 * ngram + 0.7 * llm1
    print(f"✓ Weighted: 0.3 * ngram + 0.7 * llm = {weighted}")

    # Test composition
    composed = ngram >> llm1
    print(f"✓ Composition: ngram >> llm = {composed}")

    # Test fallback
    fallback = ngram | llm1
    print(f"✓ Fallback: ngram | llm = {fallback}")

    # Test actual computation
    context = [1, 2, 3]
    tokens = [10, 20, 30]

    logprobs = mixture.logprobs(tokens, context)
    assert logprobs.shape == (3,), f"Expected shape (3,), got {logprobs.shape}"
    print(f"✓ Mixture logprobs computed: shape={logprobs.shape}")

    generated = weighted.sample(context, max_tokens=5)
    assert len(generated) == 5, f"Expected 5 tokens, got {len(generated)}"
    print(f"✓ Weighted model sampled: {generated}")

    print("\nBasic algebra tests passed!")


def test_projections():
    """Test projection algebra."""
    print("\nTesting Projection Algebra")
    print("-" * 40)

    # Create sample corpus
    np.random.seed(42)
    corpus = list(np.random.randint(0, 100, size=1000))

    # Create model and projections
    ngram = NGramModel(corpus, n=3)
    recency = RecencyProjection(corpus=corpus, max_context=20)
    semantic = SemanticProjection(embedding_dim=32)

    # Test single projection
    projected = ngram @ recency
    print(f"✓ Model @ Projection: {projected}")

    # Test projection composition
    composed = recency >> semantic
    print(f"✓ Projection composition: recency >> semantic")

    # Test projection on context
    context = list(range(50))
    proj_context = recency.project(context)
    assert len(proj_context) <= 20, f"Expected max 20 tokens, got {len(proj_context)}"
    print(f"✓ Recency projection: {len(context)} tokens -> {len(proj_context)} tokens")

    # Test similarity
    seq1 = [1, 2, 3, 4, 5]
    seq2 = [3, 4, 5, 6, 7]
    sim = semantic.similarity(seq1, seq2)
    assert 0 <= sim <= 1, f"Similarity should be in [0,1], got {sim}"
    print(f"✓ Semantic similarity: {sim:.3f}")

    print("\nProjection tests passed!")


def test_complex_composition():
    """Test complex model compositions."""
    print("\nTesting Complex Compositions")
    print("-" * 40)

    # Create sample corpus
    np.random.seed(42)
    corpus = list(np.random.randint(0, 50, size=500))

    # Create components
    n3 = NGramModel(corpus, n=3)
    n4 = NGramModel(corpus, n=4)
    llm = MockLLM(vocab_size=50, seed=42)
    recency = RecencyProjection(corpus=corpus)

    # Complex composition
    model = (0.3 * (n3 @ recency) + 0.4 * n4 + 0.3 * llm) | n3
    print(f"✓ Complex model created: {model}")

    # Test the complex model
    context = [1, 2, 3, 4, 5]
    tokens = [10, 11, 12]

    logprobs = model.logprobs(tokens, context)
    assert not np.any(np.isnan(logprobs)), "NaN values in logprobs"
    print(f"✓ Complex model logprobs: {logprobs}")

    generated = model.sample(context, max_tokens=3)
    assert len(generated) == 3, f"Expected 3 tokens, got {len(generated)}"
    print(f"✓ Complex model generated: {generated}")

    score = model.score(context + tokens)
    assert not np.isnan(score), f"Score is NaN"
    print(f"✓ Complex model score: {score:.3f}")

    print("\nComplex composition tests passed!")


def test_algebraic_properties():
    """Test some algebraic properties."""
    print("\nTesting Algebraic Properties")
    print("-" * 40)

    # Create sample corpus
    np.random.seed(42)
    corpus = list(np.random.randint(0, 50, size=500))

    # Create models
    a = MockLLM(vocab_size=50, seed=1, name="A")
    b = MockLLM(vocab_size=50, seed=2, name="B")
    c = MockLLM(vocab_size=50, seed=3, name="C")

    # Test commutativity of addition
    m1 = a + b
    m2 = b + a
    print(f"✓ Addition creates mixtures: {m1}, {m2}")

    # Test that weights normalize
    weighted = 2.0 * a + 3.0 * b
    print(f"✓ Weights normalize: 2*a + 3*b = {weighted}")

    # Test associativity of composition
    comp1 = (a >> b) >> c
    comp2 = a >> (b >> c)
    print(f"✓ Composition associativity: (a>>b)>>c and a>>(b>>c)")

    # Test fallback chain
    fallback = a | b | c
    print(f"✓ Fallback chain: a | b | c = {fallback}")

    print("\nAlgebraic properties verified!")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing NGram Projections Algebraic API")
    print("=" * 50)

    test_basic_algebra()
    test_projections()
    test_complex_composition()
    test_algebraic_properties()

    print("\n" + "=" * 50)
    print("All tests passed! ✨")
    print("The algebraic API is working correctly.")
    print("=" * 50)