#!/usr/bin/env python3
"""
Comprehensive Algebraic Framework for Language Model Composition

This module provides a rich algebraic API for composing language models
with sophisticated operators, context transformations, and functional
composition following mathematical principles and Unix philosophy.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from collections import defaultdict, deque
from dataclasses import dataclass
from abc import ABC, abstractmethod
import functools
import operator
from enum import Enum


# ============================================================================
# Core Algebraic Types
# ============================================================================

class ContextTransform(ABC):
    """Base class for context transformations."""

    @abstractmethod
    def transform(self, context: List[str]) -> List[str]:
        """Transform the context before model evaluation."""
        pass

    def __or__(self, other: 'ContextTransform') -> 'ComposedTransform':
        """Compose transforms using pipe operator: t1 | t2"""
        return ComposedTransform([self, other])

    def __and__(self, other: 'ContextTransform') -> 'ParallelTransform':
        """Apply transforms in parallel: t1 & t2"""
        return ParallelTransform([self, other])


class ModelOperator(ABC):
    """Base class for model operators."""

    @abstractmethod
    def apply(self, models: List['AlgebraicModel'],
              context: List[str]) -> Dict[str, float]:
        """Apply operator to models and return predictions."""
        pass


class AlgebraicModel(ABC):
    """Enhanced base model with rich algebraic operations."""

    @abstractmethod
    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        """Core prediction method."""
        pass

    # Basic arithmetic operations
    def __add__(self, other: Union['AlgebraicModel', float]) -> 'AlgebraicModel':
        """Addition: model1 + model2 or model + scalar"""
        if isinstance(other, (int, float)):
            return ScalarBiasModel(self, other)
        return SumModel([self, other])

    def __mul__(self, scalar: float) -> 'AlgebraicModel':
        """Scalar multiplication: alpha * model"""
        return ScaledModel(self, scalar)

    def __rmul__(self, scalar: float) -> 'AlgebraicModel':
        """Right multiplication: scalar * model"""
        return self.__mul__(scalar)

    def __sub__(self, other: 'AlgebraicModel') -> 'AlgebraicModel':
        """Subtraction: model1 - model2"""
        return SumModel([self, ScaledModel(other, -1.0)])

    def __truediv__(self, other: Union['AlgebraicModel', float]) -> 'AlgebraicModel':
        """Division for normalization: model / normalizer"""
        if isinstance(other, (int, float)):
            return ScaledModel(self, 1.0 / other)
        return NormalizedModel(self, other)

    # Set operations
    def __or__(self, other: 'AlgebraicModel') -> 'AlgebraicModel':
        """Union/Max: model1 | model2 (takes max probability)"""
        return MaxModel([self, other])

    def __and__(self, other: 'AlgebraicModel') -> 'AlgebraicModel':
        """Intersection/Min: model1 & model2 (takes min probability)"""
        return MinModel([self, other])

    def __xor__(self, other: 'AlgebraicModel') -> 'AlgebraicModel':
        """Symmetric difference: model1 ^ model2"""
        return SymmetricDifferenceModel(self, other)

    # Function application
    def __lshift__(self, transform: ContextTransform) -> 'AlgebraicModel':
        """Apply transform: model << transform"""
        return TransformedModel(self, transform)

    def __rshift__(self, func: Callable) -> 'AlgebraicModel':
        """Apply function to predictions: model >> func"""
        return FunctionModel(self, func)

    # Power operations
    def __pow__(self, exponent: float) -> 'AlgebraicModel':
        """Temperature/sharpening: model ** temperature"""
        return TemperatureModel(self, exponent)

    # Composition
    def compose(self, *others: 'AlgebraicModel') -> 'AlgebraicModel':
        """Compose multiple models."""
        return ComposedModel([self] + list(others))

    def mix(self, other: 'AlgebraicModel', alpha: float = 0.5) -> 'AlgebraicModel':
        """Linear mixture with explicit weight."""
        return MixtureModel([self, other], [alpha, 1.0 - alpha])

    def filter(self, predicate: Callable[[str], bool]) -> 'AlgebraicModel':
        """Filter predictions by predicate."""
        return FilteredModel(self, predicate)

    def top_k(self, k: int) -> 'AlgebraicModel':
        """Keep only top k predictions."""
        return TopKModel(self, k)

    def threshold(self, min_prob: float) -> 'AlgebraicModel':
        """Keep only predictions above threshold."""
        return ThresholdModel(self, min_prob)


# ============================================================================
# Context Transformations
# ============================================================================

class LongestSuffixTransform(ContextTransform):
    """Find and use the longest matching suffix from a corpus."""

    def __init__(self, suffix_array, max_length: Optional[int] = None):
        self.suffix_array = suffix_array
        self.max_length = max_length

    def transform(self, context: List[str]) -> List[str]:
        """Return the longest suffix that exists in the corpus."""
        text = " ".join(context)

        # Try suffixes from longest to shortest
        max_len = self.max_length or len(context)
        for length in range(min(max_len, len(context)), 0, -1):
            suffix = context[-length:]
            suffix_text = " ".join(suffix)

            if self.suffix_array.find_pattern(suffix_text):
                return suffix

        return context[-1:]  # At minimum, return last word


class MaxKWordsTransform(ContextTransform):
    """Limit context to last k words."""

    def __init__(self, k: int):
        self.k = k

    def transform(self, context: List[str]) -> List[str]:
        return context[-self.k:] if len(context) > self.k else context


class RecencyWeightTransform(ContextTransform):
    """Apply exponential decay to older context."""

    def __init__(self, decay_rate: float = 0.9):
        self.decay_rate = decay_rate

    def transform(self, context: List[str]) -> List[str]:
        # This returns the context but the model using it should apply weights
        # We'll attach metadata for the model to use
        weighted_context = []
        for i, word in enumerate(context):
            weight = self.decay_rate ** (len(context) - i - 1)
            # Encode weight in a special format the model can parse
            weighted_context.append(f"{word}@{weight:.3f}")
        return weighted_context


class SlidingWindowTransform(ContextTransform):
    """Use a sliding window over the context."""

    def __init__(self, window_size: int, stride: int = 1):
        self.window_size = window_size
        self.stride = stride
        self.position = 0

    def transform(self, context: List[str]) -> List[str]:
        start = self.position
        end = min(start + self.window_size, len(context))

        # Update position for next call
        self.position = (self.position + self.stride) % max(1, len(context) - self.window_size + 1)

        return context[start:end]


class SemanticFilterTransform(ContextTransform):
    """Filter context to semantically relevant words."""

    def __init__(self, relevance_fn: Callable[[str], float], threshold: float = 0.5):
        self.relevance_fn = relevance_fn
        self.threshold = threshold

    def transform(self, context: List[str]) -> List[str]:
        return [word for word in context
                if self.relevance_fn(word) >= self.threshold]


class ComposedTransform(ContextTransform):
    """Compose multiple transforms sequentially."""

    def __init__(self, transforms: List[ContextTransform]):
        self.transforms = transforms

    def transform(self, context: List[str]) -> List[str]:
        result = context
        for t in self.transforms:
            result = t.transform(result)
        return result


class ParallelTransform(ContextTransform):
    """Apply transforms in parallel and combine results."""

    def __init__(self, transforms: List[ContextTransform],
                 combiner: Callable = None):
        self.transforms = transforms
        self.combiner = combiner or self._default_combiner

    def _default_combiner(self, results: List[List[str]]) -> List[str]:
        # Default: concatenate unique words preserving order
        seen = set()
        combined = []
        for result in results:
            for word in result:
                if word not in seen:
                    seen.add(word)
                    combined.append(word)
        return combined

    def transform(self, context: List[str]) -> List[str]:
        results = [t.transform(context) for t in self.transforms]
        return self.combiner(results)


# ============================================================================
# Model Implementations
# ============================================================================

class SumModel(AlgebraicModel):
    """Sum of multiple models."""

    def __init__(self, models: List[AlgebraicModel]):
        self.models = models

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        combined = defaultdict(float)
        for model in self.models:
            predictions = model.predict(context, top_k * 2)  # Get more to ensure coverage
            for token, prob in predictions.items():
                combined[token] += prob

        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {k: v/total for k, v in combined.items()}

        # Return top k
        sorted_preds = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


class ScaledModel(AlgebraicModel):
    """Scaled model: alpha * model"""

    def __init__(self, model: AlgebraicModel, scalar: float):
        self.model = model
        self.scalar = scalar

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        predictions = self.model.predict(context, top_k)
        return {token: prob * self.scalar for token, prob in predictions.items()}


class MaxModel(AlgebraicModel):
    """Maximum of models (union operation)."""

    def __init__(self, models: List[AlgebraicModel]):
        self.models = models

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        combined = defaultdict(float)
        for model in self.models:
            predictions = model.predict(context, top_k * 2)
            for token, prob in predictions.items():
                combined[token] = max(combined[token], prob)

        sorted_preds = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


class MinModel(AlgebraicModel):
    """Minimum of models (intersection operation)."""

    def __init__(self, models: List[AlgebraicModel]):
        self.models = models

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        # Get predictions from all models
        all_predictions = [model.predict(context, top_k * 2) for model in self.models]

        # Find common tokens
        common_tokens = set(all_predictions[0].keys())
        for preds in all_predictions[1:]:
            common_tokens &= set(preds.keys())

        # Take minimum probability for common tokens
        combined = {}
        for token in common_tokens:
            combined[token] = min(preds.get(token, 0) for preds in all_predictions)

        sorted_preds = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


class TransformedModel(AlgebraicModel):
    """Model with context transformation."""

    def __init__(self, model: AlgebraicModel, transform: ContextTransform):
        self.model = model
        self.transform = transform

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        transformed_context = self.transform.transform(context)
        return self.model.predict(transformed_context, top_k)


class TemperatureModel(AlgebraicModel):
    """Apply temperature/sharpening to predictions."""

    def __init__(self, model: AlgebraicModel, temperature: float):
        self.model = model
        self.temperature = temperature

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        predictions = self.model.predict(context, top_k * 2)

        if self.temperature == 0:
            # Argmax
            max_token = max(predictions.items(), key=lambda x: x[1])[0]
            return {max_token: 1.0}

        # Apply temperature
        adjusted = {}
        for token, prob in predictions.items():
            if prob > 0:
                adjusted[token] = prob ** (1.0 / self.temperature)

        # Renormalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v/total for k, v in adjusted.items()}

        sorted_preds = sorted(adjusted.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


class MixtureModel(AlgebraicModel):
    """Weighted mixture of models."""

    def __init__(self, models: List[AlgebraicModel], weights: List[float]):
        self.models = models
        # Normalize weights
        total = sum(weights)
        self.weights = [w/total for w in weights]

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        combined = defaultdict(float)

        for model, weight in zip(self.models, self.weights):
            predictions = model.predict(context, top_k * 2)
            for token, prob in predictions.items():
                combined[token] += weight * prob

        sorted_preds = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


class FilteredModel(AlgebraicModel):
    """Filter predictions by predicate."""

    def __init__(self, model: AlgebraicModel, predicate: Callable[[str], bool]):
        self.model = model
        self.predicate = predicate

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        # Get more predictions to account for filtering
        predictions = self.model.predict(context, top_k * 3)

        # Filter
        filtered = {token: prob for token, prob in predictions.items()
                   if self.predicate(token)}

        # Renormalize
        total = sum(filtered.values())
        if total > 0:
            filtered = {k: v/total for k, v in filtered.items()}

        sorted_preds = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


class TopKModel(AlgebraicModel):
    """Keep only top k predictions."""

    def __init__(self, model: AlgebraicModel, k: int):
        self.model = model
        self.k = k

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        predictions = self.model.predict(context, self.k)
        return predictions


class ThresholdModel(AlgebraicModel):
    """Keep only predictions above threshold."""

    def __init__(self, model: AlgebraicModel, min_prob: float):
        self.model = model
        self.min_prob = min_prob

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        predictions = self.model.predict(context, top_k * 2)

        # Filter by threshold
        filtered = {token: prob for token, prob in predictions.items()
                   if prob >= self.min_prob}

        # Renormalize
        total = sum(filtered.values())
        if total > 0:
            filtered = {k: v/total for k, v in filtered.items()}

        sorted_preds = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


class SymmetricDifferenceModel(AlgebraicModel):
    """Symmetric difference: tokens in one model but not both."""

    def __init__(self, model1: AlgebraicModel, model2: AlgebraicModel):
        self.model1 = model1
        self.model2 = model2

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        preds1 = self.model1.predict(context, top_k * 2)
        preds2 = self.model2.predict(context, top_k * 2)

        tokens1 = set(preds1.keys())
        tokens2 = set(preds2.keys())

        # Symmetric difference
        sym_diff = tokens1 ^ tokens2

        combined = {}
        for token in sym_diff:
            if token in preds1:
                combined[token] = preds1[token]
            else:
                combined[token] = preds2[token]

        # Renormalize
        total = sum(combined.values())
        if total > 0:
            combined = {k: v/total for k, v in combined.items()}

        sorted_preds = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


class NormalizedModel(AlgebraicModel):
    """Normalize one model by another."""

    def __init__(self, numerator: AlgebraicModel, denominator: AlgebraicModel):
        self.numerator = numerator
        self.denominator = denominator

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        num_preds = self.numerator.predict(context, top_k * 2)
        den_preds = self.denominator.predict(context, top_k * 2)

        normalized = {}
        for token in num_preds:
            if token in den_preds and den_preds[token] > 0:
                normalized[token] = num_preds[token] / den_preds[token]

        # Renormalize to probabilities
        total = sum(normalized.values())
        if total > 0:
            normalized = {k: v/total for k, v in normalized.items()}

        sorted_preds = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


class ScalarBiasModel(AlgebraicModel):
    """Add scalar bias to all predictions."""

    def __init__(self, model: AlgebraicModel, bias: float):
        self.model = model
        self.bias = bias

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        predictions = self.model.predict(context, top_k)

        # Add bias
        biased = {token: max(0, prob + self.bias)
                 for token, prob in predictions.items()}

        # Renormalize
        total = sum(biased.values())
        if total > 0:
            biased = {k: v/total for k, v in biased.items()}

        return biased


class FunctionModel(AlgebraicModel):
    """Apply arbitrary function to predictions."""

    def __init__(self, model: AlgebraicModel, func: Callable[[Dict[str, float]], Dict[str, float]]):
        self.model = model
        self.func = func

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        predictions = self.model.predict(context, top_k * 2)
        transformed = self.func(predictions)

        # Ensure valid probabilities
        total = sum(transformed.values())
        if total > 0:
            transformed = {k: v/total for k, v in transformed.items()}

        sorted_preds = sorted(transformed.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


class ComposedModel(AlgebraicModel):
    """Sequential composition of models."""

    def __init__(self, models: List[AlgebraicModel]):
        self.models = models

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        # Each model refines the predictions of the previous
        predictions = self.models[0].predict(context, top_k * 2)

        for model in self.models[1:]:
            # Use predictions as weights for next model
            next_preds = model.predict(context, top_k * 2)

            # Combine by multiplication (AND-like operation)
            combined = {}
            for token in set(predictions.keys()) & set(next_preds.keys()):
                combined[token] = predictions[token] * next_preds[token]

            # Renormalize
            total = sum(combined.values())
            if total > 0:
                predictions = {k: v/total for k, v in combined.items()}
            else:
                predictions = next_preds

        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


# ============================================================================
# Advanced Operators
# ============================================================================

class AttentionOperator(ModelOperator):
    """Attention-based combination of models."""

    def __init__(self, key_fn: Callable[[List[str]], np.ndarray]):
        self.key_fn = key_fn

    def apply(self, models: List[AlgebraicModel],
              context: List[str]) -> Dict[str, float]:
        # Compute attention weights
        key = self.key_fn(context)
        weights = self._compute_attention(models, context, key)

        # Weighted combination
        combined = defaultdict(float)
        for model, weight in zip(models, weights):
            predictions = model.predict(context, 100)
            for token, prob in predictions.items():
                combined[token] += weight * prob

        return dict(combined)

    def _compute_attention(self, models: List[AlgebraicModel],
                          context: List[str], key: np.ndarray) -> List[float]:
        # Simplified attention computation
        num_models = len(models)
        scores = np.random.randn(num_models)  # Would use proper scoring
        weights = np.exp(scores) / np.sum(np.exp(scores))  # Softmax
        return weights.tolist()


class GatingOperator(ModelOperator):
    """Gated combination using a gating function."""

    def __init__(self, gate_fn: Callable[[List[str]], List[float]]):
        self.gate_fn = gate_fn

    def apply(self, models: List[AlgebraicModel],
              context: List[str]) -> Dict[str, float]:
        gates = self.gate_fn(context)

        combined = defaultdict(float)
        for model, gate in zip(models, gates):
            if gate > 0:
                predictions = model.predict(context, 100)
                for token, prob in predictions.items():
                    combined[token] += gate * prob

        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {k: v/total for k, v in combined.items()}

        return dict(combined)


# ============================================================================
# Builder Pattern for Complex Compositions
# ============================================================================

class ModelBuilder:
    """Fluent builder for complex model compositions."""

    def __init__(self, base: Optional[AlgebraicModel] = None):
        self.model = base
        self.transforms = []
        self.operations = []

    def with_base(self, model: AlgebraicModel) -> 'ModelBuilder':
        """Set base model."""
        self.model = model
        return self

    def add(self, model: AlgebraicModel, weight: float = 1.0) -> 'ModelBuilder':
        """Add another model."""
        if self.model is None:
            self.model = model * weight
        else:
            self.model = self.model + (model * weight)
        return self

    def multiply(self, scalar: float) -> 'ModelBuilder':
        """Multiply by scalar."""
        self.model = self.model * scalar
        return self

    def transform(self, transform: ContextTransform) -> 'ModelBuilder':
        """Apply context transformation."""
        self.model = self.model << transform
        return self

    def filter(self, predicate: Callable) -> 'ModelBuilder':
        """Filter predictions."""
        self.model = self.model.filter(predicate)
        return self

    def temperature(self, temp: float) -> 'ModelBuilder':
        """Apply temperature."""
        self.model = self.model ** temp
        return self

    def top_k(self, k: int) -> 'ModelBuilder':
        """Keep top k."""
        self.model = self.model.top_k(k)
        return self

    def threshold(self, min_prob: float) -> 'ModelBuilder':
        """Apply probability threshold."""
        self.model = self.model.threshold(min_prob)
        return self

    def max_with(self, other: AlgebraicModel) -> 'ModelBuilder':
        """Take maximum with another model."""
        self.model = self.model | other
        return self

    def min_with(self, other: AlgebraicModel) -> 'ModelBuilder':
        """Take minimum with another model."""
        self.model = self.model & other
        return self

    def build(self) -> AlgebraicModel:
        """Build the final model."""
        return self.model


# ============================================================================
# Practical Wrapper for Integration
# ============================================================================

class AlgebraicModelWrapper(AlgebraicModel):
    """Wrapper to convert existing models to algebraic models."""

    def __init__(self, model):
        self.wrapped_model = model

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        # Call the wrapped model's predict method
        if hasattr(self.wrapped_model, 'predict'):
            return self.wrapped_model.predict(context, top_k)
        else:
            raise NotImplementedError(f"Wrapped model {type(self.wrapped_model)} doesn't have predict method")


# ============================================================================
# Example Compositions
# ============================================================================

def create_grounded_model(llm, ngram, suffix_array,
                          llm_weight: float = 0.95,
                          ngram_weight: float = 0.03,
                          suffix_weight: float = 0.02) -> AlgebraicModel:
    """
    Create a sophisticated grounded model with multiple components.

    Example:
        grounded = llm_weight * llm +
                  ngram_weight * (ngram << LongestSuffixTransform(sa)) +
                  suffix_weight * (suffix_model << MaxKWordsTransform(5))
    """
    # Wrap models if needed
    if not isinstance(llm, AlgebraicModel):
        llm = AlgebraicModelWrapper(llm)
    if not isinstance(ngram, AlgebraicModel):
        ngram = AlgebraicModelWrapper(ngram)

    # Build sophisticated model
    return (ModelBuilder()
            .with_base(llm * llm_weight)
            .add(ngram << LongestSuffixTransform(suffix_array), ngram_weight)
            .add(ngram << MaxKWordsTransform(3), suffix_weight)
            .temperature(0.8)
            .build())


def create_adaptive_model(models: List[AlgebraicModel],
                          context_analyzer: Callable) -> AlgebraicModel:
    """
    Create an adaptive model that changes behavior based on context.

    The context_analyzer returns weights for each model based on context.
    """
    class AdaptiveModel(AlgebraicModel):
        def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
            weights = context_analyzer(context)
            mixture = MixtureModel(models, weights)
            return mixture.predict(context, top_k)

    return AdaptiveModel()


def create_ensemble(models: List[AlgebraicModel],
                   method: str = 'average') -> AlgebraicModel:
    """
    Create an ensemble with various combination methods.

    Methods:
        - 'average': Simple average
        - 'max': Take maximum probability
        - 'min': Take minimum (conservative)
        - 'product': Geometric mean
    """
    if method == 'average':
        weights = [1.0 / len(models)] * len(models)
        return MixtureModel(models, weights)
    elif method == 'max':
        return MaxModel(models)
    elif method == 'min':
        return MinModel(models)
    elif method == 'product':
        # Geometric mean via log-space averaging
        class ProductModel(AlgebraicModel):
            def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
                all_preds = [m.predict(context, top_k * 2) for m in models]

                # Compute geometric mean
                combined = defaultdict(lambda: 1.0)
                for preds in all_preds:
                    for token, prob in preds.items():
                        if prob > 0:
                            combined[token] *= prob ** (1.0 / len(models))

                sorted_preds = sorted(combined.items(), key=lambda x: x[1], reverse=True)
                return dict(sorted_preds[:top_k])

        return ProductModel()
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


# ============================================================================
# Demonstration
# ============================================================================

def demonstrate_algebra():
    """Demonstrate the algebraic API."""

    print("=" * 70)
    print("ALGEBRAIC MODEL COMPOSITION DEMONSTRATION")
    print("=" * 70)
    print()

    # Create mock models for demonstration
    class MockModel(AlgebraicModel):
        def __init__(self, name: str, bias: Dict[str, float] = None):
            self.name = name
            self.bias = bias or {}

        def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
            base = {'the': 0.2, 'a': 0.15, 'and': 0.1, 'of': 0.1,
                   'to': 0.08, 'in': 0.07, 'that': 0.05}
            base.update(self.bias)

            # Normalize
            total = sum(base.values())
            base = {k: v/total for k, v in base.items()}

            sorted_preds = sorted(base.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_preds[:top_k])

    # Create models
    llm = MockModel("LLM", {'neural': 0.3, 'network': 0.2})
    ngram = MockModel("NGram", {'statistical': 0.3, 'probability': 0.2})
    wiki = MockModel("Wikipedia", {'encyclopedia': 0.4, 'knowledge': 0.3})

    context = ["The", "artificial", "intelligence"]

    print("1. Basic Algebraic Operations")
    print("-" * 40)

    # Addition
    combined = llm + ngram
    print(f"LLM + NGram: {dict(list(combined.predict(context, 5).items())[:3])}")

    # Scalar multiplication
    weighted = 0.7 * llm + 0.3 * ngram
    print(f"0.7*LLM + 0.3*NGram: {dict(list(weighted.predict(context, 5).items())[:3])}")

    # Union (max)
    union = llm | ngram
    print(f"LLM | NGram (max): {dict(list(union.predict(context, 5).items())[:3])}")

    # Intersection (min)
    intersection = llm & ngram
    print(f"LLM & NGram (min): {dict(list(intersection.predict(context, 5).items())[:3])}")

    print("\n2. Context Transformations")
    print("-" * 40)

    # Apply transformations
    transform1 = MaxKWordsTransform(2)
    transform2 = MaxKWordsTransform(3)

    model_t1 = llm << transform1
    model_t2 = llm << transform2

    print(f"Original context: {context}")
    print(f"After MaxK(2): {transform1.transform(context)}")
    print(f"After MaxK(3): {transform2.transform(context)}")

    # Composed transforms
    composed_transform = transform1 | transform2  # Sequential composition

    print("\n3. Complex Compositions")
    print("-" * 40)

    # Build complex model using builder
    complex_model = (ModelBuilder()
                     .with_base(llm)
                     .multiply(0.7)
                     .add(ngram, 0.2)
                     .add(wiki, 0.1)
                     .temperature(0.8)
                     .top_k(10)
                     .build())

    preds = complex_model.predict(context, 5)
    print(f"Complex composition: {dict(list(preds.items())[:3])}")

    # Function application
    def boost_technical(preds: Dict[str, float]) -> Dict[str, float]:
        """Boost technical terms."""
        technical = ['neural', 'network', 'algorithm', 'statistical']
        boosted = preds.copy()
        for term in technical:
            if term in boosted:
                boosted[term] *= 2.0
        return boosted

    boosted_model = llm >> boost_technical
    boosted_preds = boosted_model.predict(context, 5)
    print(f"After boosting technical: {dict(list(boosted_preds.items())[:3])}")

    print("\n4. Mathematical Properties")
    print("-" * 40)

    # Demonstrate associativity: (a + b) + c = a + (b + c)
    model_abc1 = (llm + ngram) + wiki
    model_abc2 = llm + (ngram + wiki)

    preds1 = model_abc1.predict(context, 3)
    preds2 = model_abc2.predict(context, 3)

    print(f"(LLM + NGram) + Wiki: {list(preds1.keys())[:3]}")
    print(f"LLM + (NGram + Wiki): {list(preds2.keys())[:3]}")

    # Demonstrate commutativity: a + b = b + a
    model_ab = llm + ngram
    model_ba = ngram + llm

    print(f"LLM + NGram: {list(model_ab.predict(context, 3).keys())[:3]}")
    print(f"NGram + LLM: {list(model_ba.predict(context, 3).keys())[:3]}")

    print("\n5. Practical Examples")
    print("-" * 40)

    # Create ensemble
    ensemble_avg = create_ensemble([llm, ngram, wiki], method='average')
    ensemble_max = create_ensemble([llm, ngram, wiki], method='max')

    print(f"Ensemble (average): {dict(list(ensemble_avg.predict(context, 3).items())[:3])}")
    print(f"Ensemble (max): {dict(list(ensemble_max.predict(context, 3).items())[:3])}")

    # Temperature effects
    sharp = llm ** 0.5   # Sharpen distribution
    smooth = llm ** 2.0  # Smooth distribution

    print(f"\nOriginal: {dict(list(llm.predict(context, 3).items())[:3])}")
    print(f"Sharpened (T=0.5): {dict(list(sharp.predict(context, 3).items())[:3])}")
    print(f"Smoothed (T=2.0): {dict(list(smooth.predict(context, 3).items())[:3])}")

    print("\n✓ Demonstration complete!")
    print("\nThe algebraic API enables elegant composition of models")
    print("using mathematical operators and functional transformations.")


if __name__ == "__main__":
    demonstrate_algebra()