#!/usr/bin/env python3
"""
Integration of Algebraic Framework with Suffix Arrays and N-gram Models

This module demonstrates practical usage of the algebraic API with real
suffix array implementations and n-gram models for sophisticated
language model composition.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from collections import defaultdict

# Import our modules using langcalc package
from langcalc.algebra import (
    AlgebraicModel, AlgebraicModelWrapper, ContextTransform,
    LongestSuffixTransform, MaxKWordsTransform, RecencyWeightTransform,
    ModelBuilder, MixtureModel, create_ensemble, create_grounded_model
)
from langcalc.data.suffix_array import SuffixArray
from langcalc.grounding import (
    LightweightNGramModel, WikipediaNGram, NewsNGram,
    UserContextNGram, MockLLM, SimpleCorpusIndex
)

# Note: NGramSuffixModel may need to be implemented or imported differently


# ============================================================================
# Advanced Suffix-Based Models
# ============================================================================

class SuffixArrayModel(AlgebraicModel):
    """
    Algebraic model backed by suffix arrays for efficient pattern matching.
    """

    def __init__(self, suffix_array: SuffixArray, n: int = 3):
        self.suffix_array = suffix_array
        self.n = n
        self.separator = " "

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        """Predict using suffix array pattern matching."""
        # Find all matches for the context
        context_str = self.separator.join(context[-(self.n-1):])
        positions = self.suffix_array.find_pattern(context_str)

        if not positions:
            # No matches, return uniform distribution
            return {'<unk>': 1.0}

        # Count what follows
        next_tokens = defaultdict(int)
        text = self.suffix_array.text

        for pos in positions:
            next_pos = pos + len(context_str)
            if next_pos < len(text) - 1:
                # Skip separator
                if text[next_pos] == self.separator:
                    next_pos += 1

                # Find next token
                end_pos = text.find(self.separator, next_pos)
                if end_pos == -1:
                    end_pos = len(text)

                next_token = text[next_pos:end_pos]
                if next_token and next_token != self.separator:
                    next_tokens[next_token] += 1

        # Convert to probabilities
        total = sum(next_tokens.values())
        probs = {token: count/total for token, count in next_tokens.items()}

        sorted_preds = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


class AdaptiveSuffixModel(AlgebraicModel):
    """
    Suffix model that adapts its matching length based on data availability.
    """

    def __init__(self, suffix_array: SuffixArray, min_n: int = 2, max_n: int = 10):
        self.suffix_array = suffix_array
        self.min_n = min_n
        self.max_n = max_n

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        """
        Try different context lengths, starting from longest.
        Combine predictions with exponentially decreasing weights.
        """
        predictions = defaultdict(float)
        weight_sum = 0

        for n in range(min(self.max_n, len(context)), self.min_n - 1, -1):
            sub_context = context[-n:]
            pattern = " ".join(sub_context)

            positions = self.suffix_array.find_pattern(pattern)
            if positions:
                # Weight decreases exponentially with shorter contexts
                weight = 2.0 ** (n - self.min_n)
                weight_sum += weight

                # Get predictions for this context length
                model = SuffixArrayModel(self.suffix_array, n + 1)
                sub_preds = model.predict(context, top_k * 2)

                for token, prob in sub_preds.items():
                    predictions[token] += weight * prob

        if weight_sum > 0:
            # Normalize
            predictions = {k: v/weight_sum for k, v in predictions.items()}

        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


# ============================================================================
# Context-Aware Transformations
# ============================================================================

class DynamicWindowTransform(ContextTransform):
    """
    Dynamic window that adjusts size based on context properties.
    """

    def __init__(self, size_fn: Callable[[List[str]], int]):
        self.size_fn = size_fn

    def transform(self, context: List[str]) -> List[str]:
        window_size = self.size_fn(context)
        return context[-window_size:] if len(context) > window_size else context


class FocusTransform(ContextTransform):
    """
    Focus on specific types of words (e.g., content words, rare words).
    """

    def __init__(self, focus_type: str = 'content'):
        self.focus_type = focus_type
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in',
                          'on', 'at', 'to', 'for', 'of', 'with', 'by'}

    def transform(self, context: List[str]) -> List[str]:
        if self.focus_type == 'content':
            # Keep only content words
            return [w for w in context if w.lower() not in self.stop_words]
        elif self.focus_type == 'rare':
            # Would use frequency analysis in practice
            # For demo, keep words longer than 5 characters
            return [w for w in context if len(w) > 5]
        elif self.focus_type == 'recent':
            # Keep recent words with exponential decay
            if len(context) <= 3:
                return context
            # Keep last 3 words plus some older ones with decreasing probability
            result = context[-3:]
            for i in range(len(context) - 4, -1, -1):
                if np.random.random() < 0.5 ** (len(context) - i - 3):
                    result.insert(0, context[i])
            return result
        else:
            return context


class PatternMatchTransform(ContextTransform):
    """
    Transform context to match known patterns in the corpus.
    """

    def __init__(self, suffix_array: SuffixArray):
        self.suffix_array = suffix_array

    def transform(self, context: List[str]) -> List[str]:
        # Try to find the longest suffix that has many continuations
        best_suffix = context
        max_continuations = 0

        for start in range(len(context)):
            suffix = context[start:]
            pattern = " ".join(suffix)

            positions = self.suffix_array.find_pattern(pattern)
            if len(positions) > max_continuations:
                max_continuations = len(positions)
                best_suffix = suffix

        return best_suffix


# ============================================================================
# Specialized Algebraic Models
# ============================================================================

class RecencyBiasedModel(AlgebraicModel):
    """
    Model that gives more weight to recent context.
    """

    def __init__(self, base_model: AlgebraicModel, decay_rate: float = 0.9):
        self.base_model = base_model
        self.decay_rate = decay_rate

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        # Create multiple predictions with different context lengths
        predictions = defaultdict(float)
        weight_sum = 0

        for i in range(1, min(len(context) + 1, 10)):
            sub_context = context[-i:]
            weight = self.decay_rate ** (i - 1)
            weight_sum += weight

            sub_preds = self.base_model.predict(sub_context, top_k * 2)
            for token, prob in sub_preds.items():
                predictions[token] += weight * prob

        # Normalize
        if weight_sum > 0:
            predictions = {k: v/weight_sum for k, v in predictions.items()}

        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


class CacheModel(AlgebraicModel):
    """
    Model that caches recent predictions and blends them with base model.
    """

    def __init__(self, base_model: AlgebraicModel, cache_size: int = 100,
                 cache_weight: float = 0.1):
        self.base_model = base_model
        self.cache_size = cache_size
        self.cache_weight = cache_weight
        self.cache = []

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        # Get base predictions
        base_preds = self.base_model.predict(context, top_k * 2)

        # Compute cache predictions
        cache_preds = defaultdict(float)
        if self.cache:
            for token in self.cache:
                cache_preds[token] += 1.0
            total = sum(cache_preds.values())
            cache_preds = {k: v/total for k, v in cache_preds.items()}

        # Blend predictions
        combined = defaultdict(float)
        for token, prob in base_preds.items():
            combined[token] = (1 - self.cache_weight) * prob

        for token, prob in cache_preds.items():
            combined[token] += self.cache_weight * prob

        # Update cache with top prediction
        if combined:
            top_token = max(combined.items(), key=lambda x: x[1])[0]
            self.cache.append(top_token)
            if len(self.cache) > self.cache_size:
                self.cache.pop(0)

        sorted_preds = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


# ============================================================================
# Complex Composition Examples
# ============================================================================

def create_hierarchical_model(corpus_text: str) -> AlgebraicModel:
    """
    Create a hierarchical model with multiple levels of grounding.

    Structure:
        - LLM as base (70%)
        - Wikipedia grounding (15%)
        - Suffix array exact matches (10%)
        - Adaptive n-grams (5%)
    """
    # Build suffix array from corpus
    sa = SuffixArray(corpus_text)

    # Create component models
    llm = AlgebraicModelWrapper(MockLLM("base_llm"))
    wiki = AlgebraicModelWrapper(WikipediaNGram())
    suffix_model = SuffixArrayModel(sa, n=3)
    adaptive_model = AdaptiveSuffixModel(sa, min_n=2, max_n=5)

    # Apply transformations
    wiki_transformed = wiki << LongestSuffixTransform(sa, max_length=7)
    suffix_focused = suffix_model << FocusTransform('content')

    # Build hierarchical model
    return (
        0.70 * llm +
        0.15 * wiki_transformed +
        0.10 * suffix_focused +
        0.05 * adaptive_model
    )


def create_context_sensitive_model(models: Dict[str, AlgebraicModel]) -> AlgebraicModel:
    """
    Create a model that switches behavior based on context type.

    Context types:
        - Technical: Use technical model
        - Conversational: Use chat model
        - Factual: Use Wikipedia model
        - Default: Use balanced mixture
    """
    def detect_context_type(context: List[str]) -> str:
        text = " ".join(context).lower()

        technical_words = {'algorithm', 'function', 'code', 'programming',
                          'software', 'neural', 'network', 'data'}
        conversational_words = {'hello', 'hi', 'thanks', 'please', 'sure',
                               'okay', 'yeah', 'hmm'}
        factual_triggers = {'who', 'what', 'when', 'where', 'which', 'fact'}

        if any(word in text for word in technical_words):
            return 'technical'
        elif any(word in text for word in conversational_words):
            return 'conversational'
        elif any(word in text for word in factual_triggers):
            return 'factual'
        else:
            return 'default'

    class ContextSensitiveModel(AlgebraicModel):
        def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
            context_type = detect_context_type(context)

            if context_type == 'technical' and 'technical' in models:
                return models['technical'].predict(context, top_k)
            elif context_type == 'conversational' and 'conversational' in models:
                return models['conversational'].predict(context, top_k)
            elif context_type == 'factual' and 'factual' in models:
                return models['factual'].predict(context, top_k)
            else:
                # Default: balanced mixture
                mixture = create_ensemble(list(models.values()), method='average')
                return mixture.predict(context, top_k)

    return ContextSensitiveModel()


def create_multi_scale_model(corpus_text: str) -> AlgebraicModel:
    """
    Create a model that operates at multiple scales.

    Scales:
        - Character level (for rare words)
        - Word level (standard)
        - Phrase level (for idioms)
        - Sentence level (for coherence)
    """
    sa = SuffixArray(corpus_text)

    # Create models at different scales
    char_model = SuffixArrayModel(sa, n=10)  # Character n-grams
    word_model = SuffixArrayModel(sa, n=3)   # Word trigrams
    phrase_model = SuffixArrayModel(sa, n=5) # Phrase level
    sentence_model = AdaptiveSuffixModel(sa, min_n=5, max_n=15)

    # Apply appropriate transforms
    char_transformed = char_model << MaxKWordsTransform(2)
    word_transformed = word_model << MaxKWordsTransform(5)
    phrase_transformed = phrase_model << PatternMatchTransform(sa)
    sentence_transformed = sentence_model << RecencyWeightTransform(0.9)

    # Combine with scale-appropriate weights
    return (
        0.05 * char_transformed +    # Small weight for character level
        0.50 * word_transformed +     # Main weight for word level
        0.30 * phrase_transformed +   # Significant weight for phrases
        0.15 * sentence_transformed   # Some weight for sentence coherence
    )


# ============================================================================
# Advanced Operators in Practice
# ============================================================================

def create_attention_based_model(models: List[AlgebraicModel],
                                corpus_text: str) -> AlgebraicModel:
    """
    Create a model using attention mechanism to weight components.
    """
    sa = SuffixArray(corpus_text)

    class AttentionModel(AlgebraicModel):
        def __init__(self, models: List[AlgebraicModel]):
            self.models = models

        def compute_attention_weights(self, context: List[str]) -> List[float]:
            """Compute attention weights based on context relevance."""
            # Simple heuristic: weight based on how many matches each model would find
            weights = []

            for model in self.models:
                # Estimate model relevance by checking prediction entropy
                preds = model.predict(context[:3], top_k=10)  # Use short context
                if preds:
                    # Lower entropy = more confident = higher weight
                    entropy = -sum(p * np.log(p + 1e-10) for p in preds.values())
                    weight = np.exp(-entropy)  # Convert entropy to weight
                else:
                    weight = 0.1

                weights.append(weight)

            # Normalize weights
            total = sum(weights)
            if total > 0:
                weights = [w/total for w in weights]
            else:
                weights = [1.0/len(self.models)] * len(self.models)

            return weights

        def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
            weights = self.compute_attention_weights(context)

            combined = defaultdict(float)
            for model, weight in zip(self.models, weights):
                preds = model.predict(context, top_k * 2)
                for token, prob in preds.items():
                    combined[token] += weight * prob

            sorted_preds = sorted(combined.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_preds[:top_k])

    return AttentionModel(models)


# ============================================================================
# Complete System Example
# ============================================================================

def build_production_system(corpus_text: str,
                           wikipedia_text: Optional[str] = None,
                           news_text: Optional[str] = None) -> AlgebraicModel:
    """
    Build a complete production-ready system with all components.

    This demonstrates how to combine everything into a sophisticated
    language model with multiple grounding sources and transformations.
    """
    # Build suffix array
    sa = SuffixArray(corpus_text)

    # Create base models
    llm = AlgebraicModelWrapper(MockLLM("production_llm"))

    # Create specialized n-gram models
    wiki_model = AlgebraicModelWrapper(WikipediaNGram())
    if wikipedia_text:
        wiki_model.wrapped_model.train_on_text(wikipedia_text)

    news_model = AlgebraicModelWrapper(NewsNGram())
    if news_text:
        news_model.wrapped_model.train_on_text(news_text)

    # Create suffix-based models
    exact_match = SuffixArrayModel(sa, n=3)
    adaptive = AdaptiveSuffixModel(sa, min_n=2, max_n=8)
    multi_scale = create_multi_scale_model(corpus_text)

    # Create composed models with transformations
    wiki_grounded = (
        wiki_model
        << LongestSuffixTransform(sa, max_length=10)
        << FocusTransform('content')
    )

    news_recent = (
        news_model
        << MaxKWordsTransform(5)
        << RecencyWeightTransform(0.95)
    )

    exact_filtered = (
        exact_match
        << PatternMatchTransform(sa)
    ).filter(lambda token: len(token) > 2)  # Filter short tokens

    # Build the complete system using the builder pattern
    system = (ModelBuilder()
              # Base LLM with majority weight
              .with_base(llm * 0.70)

              # Add grounding sources
              .add(wiki_grounded, 0.10)
              .add(news_recent, 0.05)
              .add(exact_filtered, 0.08)
              .add(adaptive, 0.05)
              .add(multi_scale, 0.02)

              # Apply global transformations
              .temperature(0.85)  # Slight sharpening
              .threshold(0.001)    # Remove very low probability tokens
              .top_k(100)         # Keep reasonable number of candidates

              .build())

    # Wrap in caching layer for efficiency
    cached_system = CacheModel(system, cache_size=50, cache_weight=0.05)

    # Add recency bias
    final_system = RecencyBiasedModel(cached_system, decay_rate=0.95)

    return final_system


# ============================================================================
# Demonstration
# ============================================================================

def demonstrate_integration():
    """Demonstrate the integrated algebraic system."""

    print("=" * 70)
    print("ALGEBRAIC FRAMEWORK INTEGRATION DEMONSTRATION")
    print("=" * 70)
    print()

    # Sample corpus
    corpus = """
    The quick brown fox jumps over the lazy dog.
    The dog barked at the mailman yesterday.
    Machine learning algorithms process data efficiently.
    Neural networks are inspired by biological systems.
    The algorithm learns patterns from training data.
    Data science combines statistics and programming.
    The fox ran quickly through the forest.
    """

    # Build suffix array
    sa = SuffixArray(corpus)

    print("1. Basic Suffix Array Model")
    print("-" * 40)

    suffix_model = SuffixArrayModel(sa, n=3)
    context = ["The", "quick", "brown"]

    preds = suffix_model.predict(context, 5)
    print(f"Context: {' '.join(context)}")
    print(f"Predictions: {preds}")

    print("\n2. Adaptive Suffix Model")
    print("-" * 40)

    adaptive_model = AdaptiveSuffixModel(sa, min_n=2, max_n=5)
    preds = adaptive_model.predict(context, 5)
    print(f"Adaptive predictions: {preds}")

    print("\n3. Algebraic Compositions")
    print("-" * 40)

    # Create base models
    llm = AlgebraicModelWrapper(MockLLM())
    ngram = AlgebraicModelWrapper(LightweightNGramModel(n=3))
    ngram.wrapped_model.train_on_text(corpus)

    # Compose models algebraically
    composed = (
        0.7 * llm +
        0.2 * (suffix_model << MaxKWordsTransform(3)) +
        0.1 * (adaptive_model << LongestSuffixTransform(sa, max_length=5))
    )

    preds = composed.predict(context, 5)
    print(f"Composed model predictions: {preds}")

    print("\n4. Advanced Transformations")
    print("-" * 40)

    # Apply multiple transformations
    transformed_model = (
        suffix_model
        << FocusTransform('content')
        << PatternMatchTransform(sa)
    )

    # Test with different context
    context2 = ["Machine", "learning", "algorithms", "process"]
    preds = transformed_model.predict(context2, 5)
    print(f"Context: {' '.join(context2)}")
    print(f"Transformed predictions: {preds}")

    print("\n5. Context-Sensitive Behavior")
    print("-" * 40)

    # Create specialized models
    models = {
        'technical': suffix_model << FocusTransform('content'),
        'conversational': llm,
        'factual': adaptive_model,
    }

    context_sensitive = create_context_sensitive_model(models)

    # Test with different context types
    tech_context = ["Neural", "networks", "are"]
    conv_context = ["Hello", "how", "are"]
    fact_context = ["What", "is", "the"]

    print(f"Technical context: {context_sensitive.predict(tech_context, 3)}")
    print(f"Conversational context: {context_sensitive.predict(conv_context, 3)}")
    print(f"Factual context: {context_sensitive.predict(fact_context, 3)}")

    print("\n6. Hierarchical Model")
    print("-" * 40)

    hierarchical = create_hierarchical_model(corpus)
    preds = hierarchical.predict(["The", "algorithm"], 5)
    print(f"Hierarchical model predictions: {preds}")

    print("\n7. Mathematical Properties")
    print("-" * 40)

    # Demonstrate distributivity: a*(b+c) = a*b + a*c
    a = suffix_model
    b = llm
    c = adaptive_model

    left = 0.5 * (b + c)
    right = (0.5 * b) + (0.5 * c)

    print("Distributivity test: 0.5*(b+c) vs 0.5*b + 0.5*c")
    preds_left = left.predict(["The"], 3)
    preds_right = right.predict(["The"], 3)
    print(f"Left side: {list(preds_left.keys())[:3]}")
    print(f"Right side: {list(preds_right.keys())[:3]}")

    print("\n8. Production System")
    print("-" * 40)

    production = build_production_system(corpus)
    context3 = ["Machine", "learning"]

    preds = production.predict(context3, 5)
    print(f"Production system predictions: {preds}")

    print("\n✓ Integration demonstration complete!")
    print("\nThe algebraic framework seamlessly integrates with suffix arrays")
    print("and n-gram models to create sophisticated language model compositions.")


if __name__ == "__main__":
    demonstrate_integration()