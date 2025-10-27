"""
Demonstration of the elegant algebraic API for composing language models.

This example shows how complex model compositions become trivial to express
using our algebraic operators and functional combinators.
"""

import sys
import numpy as np
from typing import List

# Import our algebraic language model framework using langcalc package
from langcalc.models.base import LanguageModel
from langcalc.models.ngram import NGramModel
from langcalc.models.llm import MockLLM, HuggingFaceModel
from langcalc.models.mixture import MixtureModel

from langcalc.projections.base import IdentityProjection
from langcalc.projections.recency import RecencyProjection
from langcalc.projections.edit_distance import EditDistanceProjection
from langcalc.projections.semantic import SemanticProjection, AttentionProjection

from langcalc.algebra.combinators import (
    compose, ensemble, cascade, choose, adapt, memoize
)


def create_sample_corpus() -> List[int]:
    """Create a sample corpus for demonstration."""
    # In practice, this would be real tokenized text
    np.random.seed(42)
    return list(np.random.randint(0, 1000, size=10000))


def example_1_basic_algebra():
    """
    Example 1: Basic algebraic operations.

    Shows how models can be combined using mathematical operators.
    """
    print("=" * 60)
    print("Example 1: Basic Algebraic Operations")
    print("=" * 60)

    corpus = create_sample_corpus()

    # Create base models
    ngram_model = NGramModel(corpus, n=3)
    llm1 = MockLLM(vocab_size=1000, seed=1, name="LLM1")
    llm2 = MockLLM(vocab_size=1000, seed=2, name="LLM2")

    # 1. Addition creates equal-weight mixture
    mixture = ngram_model + llm1
    print(f"\n1. Addition: {ngram_model} + {llm1} = {mixture}")

    # 2. Weighted combination
    weighted = 0.3 * ngram_model + 0.7 * llm1
    print(f"\n2. Weighted: 0.3 * {ngram_model} + 0.7 * {llm1} = {weighted}")

    # 3. Multiple weighted models
    complex_mixture = 0.2 * ngram_model + 0.5 * llm1 + 0.3 * llm2
    print(f"\n3. Complex: 0.2 * ngram + 0.5 * llm1 + 0.3 * llm2 = {complex_mixture}")

    # 4. Sequential composition
    sequential = ngram_model >> llm1 >> llm2
    print(f"\n4. Sequential: {ngram_model} >> {llm1} >> {llm2} = {sequential}")

    # 5. Fallback/Union
    fallback = ngram_model | llm1
    print(f"\n5. Fallback: {ngram_model} | {llm1} = {fallback}")

    # Test the composed model
    context = [1, 2, 3, 4, 5]
    tokens = [10, 20, 30]
    logprobs = complex_mixture.logprobs(tokens, context)
    print(f"\n6. Testing complex mixture on tokens {tokens}: {logprobs}")


def example_2_projections():
    """
    Example 2: Projection algebra.

    Shows how projections can be composed and applied to models.
    """
    print("\n" + "=" * 60)
    print("Example 2: Projection Algebra")
    print("=" * 60)

    corpus = create_sample_corpus()

    # Create projections
    recency = RecencyProjection(corpus=corpus)
    edit_dist = EditDistanceProjection(max_distance=2)
    semantic = SemanticProjection(embedding_dim=64)
    attention = AttentionProjection(top_k=5)

    # Create base model
    ngram = NGramModel(corpus, n=4)

    # 1. Apply single projection
    ngram_recency = ngram @ recency
    print(f"\n1. Model with projection: {ngram} @ {recency} = {ngram_recency}")

    # 2. Compose projections
    composed_proj = recency >> semantic
    print(f"\n2. Composed projection: {recency} >> {semantic} = {composed_proj}")

    # 3. Apply composed projection
    ngram_composed = ngram @ composed_proj
    print(f"\n3. Model with composed projection: {ngram_composed}")

    # 4. Union of projections
    union_proj = recency | attention
    ngram_union = ngram @ union_proj
    print(f"\n4. Model with union projection: {ngram_union}")

    # 5. Intersection of projections
    intersection_proj = recency & attention
    ngram_intersection = ngram @ intersection_proj
    print(f"\n5. Model with intersection projection: {ngram_intersection}")

    # Test projection effects
    context = list(range(100, 110))
    projected = recency.project(context)
    print(f"\n6. Recency projection of {context[:5]}... = {projected}")


def example_3_functional_combinators():
    """
    Example 3: Functional combinators.

    Shows higher-order functions for sophisticated model composition.
    """
    print("\n" + "=" * 60)
    print("Example 3: Functional Combinators")
    print("=" * 60)

    corpus = create_sample_corpus()

    # Create models
    ngram = NGramModel(corpus, n=3)
    fast_llm = MockLLM(vocab_size=1000, seed=1, name="FastLLM")
    accurate_llm = MockLLM(vocab_size=1000, seed=2, name="AccurateLLM")

    # 1. Compose multiple models (right-to-left like math)
    composed = compose(ngram, fast_llm, accurate_llm)
    print(f"\n1. Composed: compose(ngram, fast, accurate) = {composed}")

    # 2. Ensemble with custom aggregation
    models = [ngram, fast_llm, accurate_llm]

    # Mean ensemble
    mean_ensemble = ensemble(models)
    print(f"\n2. Mean ensemble: {mean_ensemble}")

    # Max ensemble
    max_ensemble = ensemble(models, aggregator=lambda x: np.max(x, axis=0))
    print(f"\n3. Max ensemble: {max_ensemble}")

    # 3. Cascade with confidence thresholds
    cascaded = cascade([fast_llm, accurate_llm, ngram],
                      confidence_thresholds=[-5.0, -10.0, -np.inf])
    print(f"\n4. Cascaded: {cascaded}")

    # 4. Conditional model selection
    def is_short_context(context):
        return context is None or len(context) < 5

    conditional = choose(is_short_context,
                        if_true=ngram,
                        if_false=accurate_llm)
    print(f"\n5. Conditional: use ngram for short context, LLM otherwise")

    # 5. Memoized model for efficiency
    memoized = memoize(accurate_llm, cache_size=100)
    print(f"\n6. Memoized: {memoized}")


def example_4_complex_composition():
    """
    Example 4: Complex real-world composition.

    Shows how to build a sophisticated hybrid model using the algebra.
    """
    print("\n" + "=" * 60)
    print("Example 4: Complex Real-World Composition")
    print("=" * 60)

    corpus = create_sample_corpus()

    # Base models
    ngram_3 = NGramModel(corpus, n=3, smoothing='kneser_ney')
    ngram_5 = NGramModel(corpus, n=5, smoothing='laplace')
    fast_llm = MockLLM(vocab_size=1000, seed=1, name="FastLLM")
    accurate_llm = MockLLM(vocab_size=1000, seed=2, name="PowerfulLLM")

    # Projections
    recency = RecencyProjection(corpus=corpus, max_context=50)
    semantic = SemanticProjection(embedding_dim=128)
    attention = AttentionProjection(top_k=10)

    # Build a sophisticated hybrid model:
    # 1. N-gram models with different projections
    ngram_recency = ngram_3 @ recency
    ngram_semantic = ngram_5 @ semantic

    # 2. Weighted n-gram mixture
    ngram_mix = 0.6 * ngram_recency + 0.4 * ngram_semantic

    # 3. LLM with attention projection
    llm_attended = accurate_llm @ attention

    # 4. Cascade: try fast model first, fall back to accurate
    llm_cascade = fast_llm | llm_attended

    # 5. Final hybrid: weighted combination of n-gram and LLM components
    hybrid = 0.3 * ngram_mix + 0.7 * llm_cascade

    # 6. Add memoization for efficiency
    efficient_hybrid = memoize(hybrid)

    print(f"\nFinal hybrid model composition:")
    print(f"  hybrid = memoize(")
    print(f"    0.3 * (0.6 * (ngram_3 @ recency) + 0.4 * (ngram_5 @ semantic))")
    print(f"    + 0.7 * (fast_llm | (accurate_llm @ attention))")
    print(f"  )")

    # Test the hybrid model
    context = list(range(10))
    tokens_to_score = [100, 200, 300]

    print(f"\nTesting hybrid model:")
    print(f"  Context: {context}")
    print(f"  Tokens to score: {tokens_to_score}")

    logprobs = efficient_hybrid.logprobs(tokens_to_score, context)
    print(f"  Log probabilities: {logprobs}")

    generated = efficient_hybrid.sample(context, temperature=0.8, max_tokens=5)
    print(f"  Generated tokens: {generated}")


def example_5_mathematical_beauty():
    """
    Example 5: Mathematical beauty of the API.

    Shows how the algebra enables concise, expressive model definitions.
    """
    print("\n" + "=" * 60)
    print("Example 5: Mathematical Beauty of the API")
    print("=" * 60)

    corpus = create_sample_corpus()

    # Models
    n3 = NGramModel(corpus, n=3)
    n5 = NGramModel(corpus, n=5)
    llm = MockLLM(name="LLM")

    # Projections
    r = RecencyProjection(corpus=corpus)  # recency
    s = SemanticProjection()              # semantic
    e = EditDistanceProjection()          # edit distance

    print("\nConcise algebraic expressions:")
    print("=" * 40)

    # Example 1: Simple mixture
    model1 = n3 + llm
    print(f"n3 + llm                    # Simple mixture")

    # Example 2: Weighted with projection
    model2 = 0.3 * (n3 @ r) + 0.7 * llm
    print(f"0.3 * (n3 @ r) + 0.7 * llm  # Weighted with projection")

    # Example 3: Complex projection algebra
    model3 = n5 @ (r >> s)
    print(f"n5 @ (r >> s)               # Composed projections")

    # Example 4: Parallel projections with union
    model4 = n3 @ (r | s | e)
    print(f"n3 @ (r | s | e)            # Union of projections")

    # Example 5: Sequential composition
    model5 = n3 >> n5 >> llm
    print(f"n3 >> n5 >> llm             # Sequential composition")

    # Example 6: Fallback chain
    model6 = n3 | n5 | llm
    print(f"n3 | n5 | llm               # Fallback chain")

    # Example 7: Everything combined
    model7 = ((0.3 * (n3 @ r) + 0.7 * (n5 @ s)) >> llm) | n3
    print(f"((0.3*(n3@r) + 0.7*(n5@s)) >> llm) | n3  # Complex hybrid")

    print("\n" + "=" * 40)
    print("The algebra makes complex compositions trivial to express!")


if __name__ == "__main__":
    # Run all examples
    example_1_basic_algebra()
    example_2_projections()
    example_3_functional_combinators()
    example_4_complex_composition()
    example_5_mathematical_beauty()

    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)