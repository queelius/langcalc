#!/usr/bin/env python3
"""
Elegant Examples of the Algebraic Language Model Framework

This module showcases the power and elegance of the algebraic API
through practical examples that demonstrate its composability,
mathematical properties, and real-world applications.
"""

from typing import Dict, List
import numpy as np

from model_algebra import (
    AlgebraicModel, AlgebraicModelWrapper,
    LongestSuffixTransform, MaxKWordsTransform, RecencyWeightTransform,
    ModelBuilder
)
from algebra_integration import (
    SuffixArrayModel, AdaptiveSuffixModel,
    RecencyBiasedModel, CacheModel, FocusTransform
)
from suffix_array_demo import SuffixArray
from lightweight_grounding import MockLLM, WikipediaNGram, LightweightNGramModel


# ============================================================================
# Example 1: Simple Yet Powerful Compositions
# ============================================================================

def example_basic_algebra():
    """
    Demonstrate how simple algebraic operations create sophisticated models.
    """
    print("Example 1: Basic Algebraic Elegance")
    print("-" * 50)

    # Create base models
    llm = AlgebraicModelWrapper(MockLLM("gpt"))
    wiki = AlgebraicModelWrapper(WikipediaNGram())

    # Simple addition creates a mixture
    grounded = llm + 0.1 * wiki
    print("llm + 0.1*wiki creates a 90/10 mixture")

    # Union takes maximum (optimistic)
    optimistic = llm | wiki
    print("llm | wiki takes max probability (optimistic)")

    # Intersection takes minimum (conservative)
    conservative = llm & wiki
    print("llm & wiki takes min probability (conservative)")

    # Temperature control
    sharp = llm ** 0.5      # Sharpen (more confident)
    smooth = llm ** 2.0     # Smooth (less confident)
    print("llm**0.5 sharpens, llm**2.0 smooths distribution")

    # Chaining operations
    final = ((llm + wiki) ** 0.8).top_k(10).threshold(0.01)
    print("Chain: ((llm + wiki) ** 0.8).top_k(10).threshold(0.01)")

    context = ["The", "theory", "of"]
    result = final.predict(context, 5)
    print(f"Result: {list(result.keys())[:3]}")
    print()


# ============================================================================
# Example 2: Context Transformations Pipeline
# ============================================================================

def example_context_pipeline():
    """
    Show how context transformations create adaptive models.
    """
    print("Example 2: Context Transformation Pipeline")
    print("-" * 50)

    corpus = "The quick brown fox jumps over the lazy dog. " * 10
    sa = SuffixArray(corpus)

    # Create base model
    model = SuffixArrayModel(sa)

    # Apply transformations using << operator
    # Each transformation modifies how context is processed

    # Find longest matching suffix in corpus
    suffix_aware = model << LongestSuffixTransform(sa, max_length=10)

    # Limit to recent context
    recent_focused = model << MaxKWordsTransform(5)

    # Apply recency weighting
    recency_weighted = model << RecencyWeightTransform(0.9)

    # Compose transformations (applied left to right)
    pipeline = model << (MaxKWordsTransform(10) | LongestSuffixTransform(sa))

    print("Transformations:")
    print("  model << LongestSuffixTransform    # Use longest match")
    print("  model << MaxKWordsTransform(5)     # Limit to 5 words")
    print("  model << RecencyWeightTransform    # Weight by recency")
    print("  model << (t1 | t2)                 # Compose transforms")
    print()


# ============================================================================
# Example 3: Adaptive Weighting Based on Context
# ============================================================================

def example_adaptive_weighting():
    """
    Create models that adapt their behavior based on context properties.
    """
    print("Example 3: Adaptive Context-Aware Models")
    print("-" * 50)

    # Create specialized models
    technical = AlgebraicModelWrapper(MockLLM("technical"))
    casual = AlgebraicModelWrapper(MockLLM("casual"))
    formal = AlgebraicModelWrapper(MockLLM("formal"))

    class SmartMixer(AlgebraicModel):
        """Intelligently mix models based on context analysis."""

        def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
            text = " ".join(context).lower()

            # Analyze context
            has_technical = any(word in text for word in
                              ['algorithm', 'function', 'neural', 'data'])
            has_formal = any(word in text for word in
                           ['therefore', 'however', 'moreover', 'whereas'])
            has_casual = any(word in text for word in
                           ['hey', 'yeah', 'cool', 'awesome'])

            # Compute adaptive weights
            weights = {
                'technical': 0.7 if has_technical else 0.1,
                'formal': 0.6 if has_formal else 0.2,
                'casual': 0.8 if has_casual else 0.1
            }

            # Normalize
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}

            # Create mixture with computed weights
            mixture = (
                weights['technical'] * technical +
                weights['formal'] * formal +
                weights['casual'] * casual
            )

            return mixture.predict(context, top_k)

    smart = SmartMixer()

    # Test with different contexts
    contexts = [
        ["The", "algorithm", "processes"],      # Technical
        ["Hey", "that's", "really", "cool"],     # Casual
        ["Therefore", "we", "must", "consider"], # Formal
    ]

    for ctx in contexts:
        print(f"Context: {' '.join(ctx)}")
        result = smart.predict(ctx, 3)
        print(f"  Adapted prediction: {list(result.keys())[0]}")
    print()


# ============================================================================
# Example 4: Mathematical Laws in Action
# ============================================================================

def example_mathematical_properties():
    """
    Demonstrate that our algebra follows mathematical laws.
    """
    print("Example 4: Mathematical Laws")
    print("-" * 50)

    # Create models
    a = AlgebraicModelWrapper(MockLLM("a"))
    b = AlgebraicModelWrapper(MockLLM("b"))
    c = AlgebraicModelWrapper(MockLLM("c"))

    context = ["test"]

    # Associativity of addition
    left_assoc = (a + b) + c
    right_assoc = a + (b + c)
    print("Associativity: (a + b) + c = a + (b + c)")
    print(f"  Left: {list(left_assoc.predict(context, 2).keys())[:2]}")
    print(f"  Right: {list(right_assoc.predict(context, 2).keys())[:2]}")

    # Distributivity
    distributed = 0.5 * (a + b)
    expanded = (0.5 * a) + (0.5 * b)
    print("\nDistributivity: α*(a + b) = α*a + α*b")
    print(f"  Distributed: {list(distributed.predict(context, 2).keys())[:2]}")
    print(f"  Expanded: {list(expanded.predict(context, 2).keys())[:2]}")

    # De Morgan's Laws (for union/intersection)
    # Not exactly equivalent due to normalization, but demonstrates the concept
    union = a | b
    intersection = a & b
    print("\nSet Operations:")
    print(f"  a | b (union/max): {list(union.predict(context, 2).keys())[:2]}")
    print(f"  a & b (intersection/min): {list(intersection.predict(context, 2).keys())[:2]}")

    # Identity elements
    identity = a + (0 * b)  # Adding zero-weighted model
    print("\nIdentity: a + 0*b = a")
    print(f"  Original a: {list(a.predict(context, 2).keys())[:2]}")
    print(f"  a + 0*b: {list(identity.predict(context, 2).keys())[:2]}")
    print()


# ============================================================================
# Example 5: Building Complex Models with the Builder Pattern
# ============================================================================

def example_builder_pattern():
    """
    Use the fluent builder pattern for complex compositions.
    """
    print("Example 5: Fluent Builder Pattern")
    print("-" * 50)

    corpus = "Machine learning is a subset of artificial intelligence. " * 5
    sa = SuffixArray(corpus)

    # Create base components
    llm = AlgebraicModelWrapper(MockLLM())
    ngram = AlgebraicModelWrapper(LightweightNGramModel())
    suffix = SuffixArrayModel(sa)

    # Build a sophisticated model step by step
    model = (ModelBuilder()
            .with_base(llm)                        # Start with LLM
            .multiply(0.7)                         # Weight it 70%
            .add(ngram, 0.2)                       # Add 20% n-gram
            .add(suffix, 0.1)                      # Add 10% suffix
            .transform(MaxKWordsTransform(10))     # Limit context
            .temperature(0.9)                      # Slight sharpening
            .filter(lambda t: len(t) > 2)          # Remove short tokens
            .threshold(0.001)                      # Remove low probability
            .top_k(50)                             # Keep top 50
            .build())

    print("Built model with:")
    print("  - 70% LLM base")
    print("  - 20% n-gram grounding")
    print("  - 10% suffix array matching")
    print("  - Context limited to 10 words")
    print("  - Temperature 0.9 (slight sharpening)")
    print("  - Filtered short tokens")
    print("  - Threshold 0.001 probability")
    print("  - Top-50 selection")

    context = ["Machine", "learning", "is"]
    result = model.predict(context, 5)
    print(f"\nPrediction: {list(result.keys())[:3]}")
    print()


# ============================================================================
# Example 6: Practical Grounding Equation
# ============================================================================

def example_grounding_equation():
    """
    Implement the paper's core grounding equation elegantly.
    """
    print("Example 6: Elegant Grounding Equation")
    print("-" * 50)

    # The equation from the paper:
    # P(x_t | context) = α_LLM * P_LLM + α_ngram * P_ngram + α_suffix * P_suffix

    # Initialize components
    corpus = """
    Neural networks learn patterns from data.
    Deep learning uses neural networks.
    Machine learning includes deep learning.
    """
    sa = SuffixArray(corpus)

    llm = AlgebraicModelWrapper(MockLLM())
    ngram = AlgebraicModelWrapper(LightweightNGramModel(n=3))
    suffix = SuffixArrayModel(sa)

    # Express the equation directly in code
    # This reads like the mathematical equation!
    grounded = 0.95 * llm + 0.03 * ngram + 0.02 * suffix

    print("Equation: P = 0.95*LLM + 0.03*NGram + 0.02*Suffix")
    print("In code:  grounded = 0.95 * llm + 0.03 * ngram + 0.02 * suffix")
    print()

    # Add sophistication with transformations
    sophisticated = (
        0.95 * llm +
        0.03 * (ngram << LongestSuffixTransform(sa)) +
        0.02 * (suffix << MaxKWordsTransform(5))
    ) ** 0.9  # Apply temperature

    print("With transformations:")
    print("sophisticated = (")
    print("    0.95 * llm +")
    print("    0.03 * (ngram << LongestSuffixTransform) +")
    print("    0.02 * (suffix << MaxKWordsTransform(5))")
    print(") ** 0.9  # temperature")

    context = ["Neural", "networks"]
    result = sophisticated.predict(context, 5)
    print(f"\nResult: {list(result.keys())[:3]}")
    print()


# ============================================================================
# Example 7: Ensemble Methods
# ============================================================================

def example_ensemble_methods():
    """
    Show different ensemble combination strategies.
    """
    print("Example 7: Ensemble Strategies")
    print("-" * 50)

    # Create diverse models
    models = [
        AlgebraicModelWrapper(MockLLM(f"model_{i}"))
        for i in range(3)
    ]

    context = ["The", "answer", "is"]

    # Different ensemble strategies using operators
    print("Ensemble strategies using operators:")

    # Average (mixture)
    from functools import reduce
    average = reduce(lambda a, b: a + b, models) / len(models)
    print(f"Average: reduce(add, models) / len(models)")

    # Weighted average
    weighted = 0.5 * models[0] + 0.3 * models[1] + 0.2 * models[2]
    print(f"Weighted: 0.5*m0 + 0.3*m1 + 0.2*m2")

    # Voting (union of all)
    voting = models[0] | models[1] | models[2]
    print(f"Voting: m0 | m1 | m2 (takes max)")

    # Conservative (intersection)
    conservative = models[0] & models[1] & models[2]
    print(f"Conservative: m0 & m1 & m2 (takes min)")

    # Product (geometric mean via composition)
    product = models[0].compose(models[1], models[2])
    print(f"Product: m0.compose(m1, m2)")

    print("\nResults preview:")
    for name, model in [("Average", average), ("Voting", voting)]:
        result = model.predict(context, 3)
        print(f"  {name}: {list(result.keys())[0] if result else 'None'}")
    print()


# ============================================================================
# Example 8: Recency and Caching
# ============================================================================

def example_recency_caching():
    """
    Demonstrate temporal modeling with recency and caching.
    """
    print("Example 8: Temporal Modeling")
    print("-" * 50)

    base = AlgebraicModelWrapper(MockLLM())

    # Add recency bias - recent context matters more
    recency_model = RecencyBiasedModel(base, decay_rate=0.9)

    # Add caching - remember recent predictions
    cached_model = CacheModel(base, cache_size=100, cache_weight=0.1)

    # Combine both
    temporal = CacheModel(
        RecencyBiasedModel(base, decay_rate=0.95),
        cache_size=50,
        cache_weight=0.05
    )

    print("Temporal modeling:")
    print("  RecencyBiasedModel: weights recent context more")
    print("  CacheModel: remembers and reuses recent predictions")
    print("  Combined: both recency and memory")

    # Simulate conversation
    contexts = [
        ["Tell", "me", "about"],
        ["machine", "learning", "and"],
        ["how", "it", "relates", "to"],
    ]

    print("\nSimulating conversation flow:")
    for ctx in contexts:
        result = temporal.predict(ctx, 3)
        print(f"  Context: {' '.join(ctx)}")
        print(f"  Next: {list(result.keys())[0] if result else 'None'}")
    print()


# ============================================================================
# Example 9: Custom Operators
# ============================================================================

def example_custom_operators():
    """
    Create custom operators for specialized behavior.
    """
    print("Example 9: Custom Operators")
    print("-" * 50)

    class ContrastiveModel(AlgebraicModel):
        """Model that emphasizes differences between two models."""

        def __init__(self, positive: AlgebraicModel, negative: AlgebraicModel,
                     contrast_weight: float = 2.0):
            self.positive = positive
            self.negative = negative
            self.contrast_weight = contrast_weight

        def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
            pos_preds = self.positive.predict(context, top_k * 2)
            neg_preds = self.negative.predict(context, top_k * 2)

            # Emphasize tokens that positive predicts but negative doesn't
            contrastive = {}
            for token in pos_preds:
                pos_score = pos_preds[token]
                neg_score = neg_preds.get(token, 0.01)  # Small epsilon

                # Contrastive score: high when positive >> negative
                contrastive[token] = pos_score * (pos_score / neg_score) ** self.contrast_weight

            # Normalize
            total = sum(contrastive.values())
            if total > 0:
                contrastive = {k: v/total for k, v in contrastive.items()}

            sorted_preds = sorted(contrastive.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_preds[:top_k])

    # Create models with different biases
    creative = AlgebraicModelWrapper(MockLLM("creative"))
    boring = AlgebraicModelWrapper(MockLLM("boring"))

    # Emphasize creative differences
    contrastive = ContrastiveModel(creative, boring, contrast_weight=1.5)

    print("ContrastiveModel(creative, boring):")
    print("  Emphasizes what creative predicts but boring doesn't")
    print("  Score = P_creative * (P_creative / P_boring)^weight")

    context = ["Generate", "something"]
    result = contrastive.predict(context, 5)
    print(f"\nContrastive result: {list(result.keys())[:3]}")
    print()


# ============================================================================
# Example 10: One-Liner Compositions
# ============================================================================

def example_one_liners():
    """
    Show the power of the API through elegant one-liners.
    """
    print("Example 10: Powerful One-Liners")
    print("-" * 50)

    # Setup
    corpus = "The quick brown fox jumps over the lazy dog."
    sa = SuffixArray(corpus)
    llm = AlgebraicModelWrapper(MockLLM())
    ngram = AlgebraicModelWrapper(LightweightNGramModel())
    suffix = SuffixArrayModel(sa)

    print("Elegant one-liner compositions:\n")

    # Grounded model with temperature
    grounded = (0.9 * llm + 0.1 * ngram) ** 0.8
    print("(0.9 * llm + 0.1 * ngram) ** 0.8")
    print("  → Grounded model with temperature\n")

    # Filtered and normalized
    filtered = ((llm | ngram) - 0.1).filter(lambda t: t.isalpha())
    print("((llm | ngram) - 0.1).filter(lambda t: t.isalpha())")
    print("  → Union with bias and alphabetic filter\n")

    # Context-aware with fallback
    smart = (suffix << LongestSuffixTransform(sa)) | (0.5 * llm)
    print("(suffix << LongestSuffixTransform(sa)) | (0.5 * llm)")
    print("  → Suffix matching with LLM fallback\n")

    # Sharpened ensemble
    from functools import reduce
    ensemble = (reduce(lambda a, b: a + b, [llm, ngram, suffix]) / 3) ** 0.7
    print("(llm + ngram + suffix) / 3 ** 0.7")
    print("  → Average ensemble with sharpening\n")

    # Adaptive threshold
    adaptive = llm.threshold(0.01).top_k(20) + 0.1 * ngram
    print("llm.threshold(0.01).top_k(20) + 0.1 * ngram")
    print("  → Filtered LLM with n-gram boost\n")

    print()


# ============================================================================
# Main Demo
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("ALGEBRAIC LANGUAGE MODEL FRAMEWORK")
    print("Elegant Composition Through Mathematical Operators")
    print("=" * 70)
    print()

    examples = [
        example_basic_algebra,
        example_context_pipeline,
        example_adaptive_weighting,
        example_mathematical_properties,
        example_builder_pattern,
        example_grounding_equation,
        example_ensemble_methods,
        example_recency_caching,
        example_custom_operators,
        example_one_liners,
    ]

    for example in examples:
        example()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("The algebraic framework provides:")
    print("  • Mathematical operators (+, *, |, &, **, <<, >>)")
    print("  • Context transformations (suffix, recency, focus)")
    print("  • Composable building blocks")
    print("  • Elegant one-liner compositions")
    print("  • Production-ready patterns")
    print()
    print("Key insight: Language models are algebraic objects that can")
    print("be composed using mathematical operations, creating powerful")
    print("and interpretable systems from simple components.")
    print()
    print("✓ All examples completed successfully!")


if __name__ == "__main__":
    main()