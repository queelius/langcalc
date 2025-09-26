"""
Functional combinators for elegant model composition.

This module provides higher-order functions for composing
language models in sophisticated ways, inspired by functional
programming and category theory.
"""

from typing import List, Callable, Optional, Tuple, Any
import numpy as np
from functools import reduce

from ngram_projections.models.base import LanguageModel
from ngram_projections.projections.base import Projection


def compose(*models: LanguageModel) -> LanguageModel:
    """
    Compose multiple models sequentially.

    compose(f, g, h) creates h ∘ g ∘ f
    (reads right-to-left like mathematical composition)

    Args:
        *models: Models to compose

    Returns:
        Composed model
    """
    if not models:
        raise ValueError("At least one model required")

    if len(models) == 1:
        return models[0]

    return reduce(lambda m1, m2: m1 >> m2, models)


def parallel(*models: LanguageModel) -> List[LanguageModel]:
    """
    Run models in parallel (conceptually).

    Returns results from all models for further processing.

    Args:
        *models: Models to run in parallel

    Returns:
        List of models (for use with other combinators)
    """
    return list(models)


def choose(condition: Callable[[Optional[List[int]]], bool],
          if_true: LanguageModel,
          if_false: LanguageModel) -> LanguageModel:
    """
    Conditional model selection based on context.

    Args:
        condition: Function that takes context and returns bool
        if_true: Model to use when condition is true
        if_false: Model to use when condition is false

    Returns:
        Conditional model
    """
    class ConditionalModel(LanguageModel):
        def logprobs(self, tokens: List[int], context: Optional[List[int]] = None) -> np.ndarray:
            model = if_true if condition(context) else if_false
            return model.logprobs(tokens, context)

        def sample(self, context: Optional[List[int]] = None,
                  temperature: float = 1.0,
                  max_tokens: int = 100) -> List[int]:
            model = if_true if condition(context) else if_false
            return model.sample(context, temperature, max_tokens)

        def score(self, sequence: List[int]) -> float:
            # Use if_true for scoring by default
            return if_true.score(sequence)

        def __repr__(self) -> str:
            return f"choose(·, {if_true}, {if_false})"

    return ConditionalModel()


def adapt(model: LanguageModel,
         adapter: Callable[[List[int]], List[int]]) -> LanguageModel:
    """
    Adapt a model's input/output using a transformation function.

    Args:
        model: Base model
        adapter: Function to transform token sequences

    Returns:
        Adapted model
    """
    class AdaptedModel(LanguageModel):
        def logprobs(self, tokens: List[int], context: Optional[List[int]] = None) -> np.ndarray:
            adapted_context = adapter(context) if context else None
            return model.logprobs(tokens, adapted_context)

        def sample(self, context: Optional[List[int]] = None,
                  temperature: float = 1.0,
                  max_tokens: int = 100) -> List[int]:
            adapted_context = adapter(context) if context else None
            return model.sample(adapted_context, temperature, max_tokens)

        def score(self, sequence: List[int]) -> float:
            adapted_sequence = adapter(sequence)
            return model.score(adapted_sequence)

        def __repr__(self) -> str:
            return f"adapt({model}, ·)"

    return AdaptedModel()


def ensemble(models: List[LanguageModel],
            aggregator: Callable[[List[np.ndarray]], np.ndarray] = None) -> LanguageModel:
    """
    Create an ensemble with custom aggregation.

    Args:
        models: List of models
        aggregator: Function to aggregate predictions (default: mean)

    Returns:
        Ensemble model
    """
    if aggregator is None:
        aggregator = lambda probs: np.mean(probs, axis=0)

    class EnsembleModel(LanguageModel):
        def logprobs(self, tokens: List[int], context: Optional[List[int]] = None) -> np.ndarray:
            all_logprobs = []
            for model in models:
                logprobs = model.logprobs(tokens, context)
                all_logprobs.append(np.exp(logprobs))  # Convert to probs

            aggregated = aggregator(all_logprobs)
            return np.log(np.maximum(aggregated, 1e-10))

        def sample(self, context: Optional[List[int]] = None,
                  temperature: float = 1.0,
                  max_tokens: int = 100) -> List[int]:
            # Sample from random model
            idx = np.random.randint(len(models))
            return models[idx].sample(context, temperature, max_tokens)

        def score(self, sequence: List[int]) -> float:
            scores = [model.score(sequence) for model in models]
            # Average in probability space
            probs = np.exp(scores)
            return np.log(np.mean(probs))

        def __repr__(self) -> str:
            return f"ensemble([{', '.join(str(m) for m in models)}])"

    return EnsembleModel()


def cascade(models: List[LanguageModel],
           confidence_thresholds: Optional[List[float]] = None) -> LanguageModel:
    """
    Create a cascade of models with confidence-based fallback.

    Each model is tried in order until one exceeds its confidence threshold.

    Args:
        models: List of models in order of preference
        confidence_thresholds: Confidence thresholds for each model

    Returns:
        Cascaded model
    """
    if confidence_thresholds is None:
        confidence_thresholds = [-10.0] * (len(models) - 1) + [-np.inf]

    class CascadeModel(LanguageModel):
        def logprobs(self, tokens: List[int], context: Optional[List[int]] = None) -> np.ndarray:
            for model, threshold in zip(models, confidence_thresholds):
                logprobs = model.logprobs(tokens, context)
                if np.max(logprobs) > threshold:
                    return logprobs
            return models[-1].logprobs(tokens, context)

        def sample(self, context: Optional[List[int]] = None,
                  temperature: float = 1.0,
                  max_tokens: int = 100) -> List[int]:
            for model, threshold in zip(models, confidence_thresholds):
                # Try sampling and check quality
                result = model.sample(context, temperature, max_tokens)
                if result:  # Simple check - could be more sophisticated
                    return result
            return models[-1].sample(context, temperature, max_tokens)

        def score(self, sequence: List[int]) -> float:
            for model, threshold in zip(models, confidence_thresholds):
                score = model.score(sequence)
                if score / len(sequence) > threshold:
                    return score
            return models[-1].score(sequence)

        def __repr__(self) -> str:
            return f"cascade([{', '.join(str(m) for m in models)}])"

    return CascadeModel()


def pipe(data_source: Callable[[], List[int]],
        *transformations: Callable[[List[int]], List[int]]) -> Callable[[], List[int]]:
    """
    Create a pipeline of transformations.

    Useful for preprocessing/postprocessing around models.

    Args:
        data_source: Function that produces initial tokens
        *transformations: Sequence of transformation functions

    Returns:
        Function that applies the full pipeline
    """
    def pipeline():
        data = data_source()
        for transform in transformations:
            data = transform(data)
        return data

    return pipeline


def memoize(model: LanguageModel, cache_size: int = 1000) -> LanguageModel:
    """
    Add memoization to a model for efficiency.

    Args:
        model: Model to memoize
        cache_size: Maximum cache size

    Returns:
        Memoized model
    """
    from functools import lru_cache

    class MemoizedModel(LanguageModel):
        def __init__(self):
            # Memoize the expensive operations
            self._cached_logprobs = lru_cache(maxsize=cache_size)(self._compute_logprobs)
            self._cached_score = lru_cache(maxsize=cache_size)(self._compute_score)

        def _compute_logprobs(self, tokens_tuple: tuple, context_tuple: tuple) -> np.ndarray:
            tokens = list(tokens_tuple)
            context = list(context_tuple) if context_tuple else None
            return model.logprobs(tokens, context)

        def _compute_score(self, sequence_tuple: tuple) -> float:
            return model.score(list(sequence_tuple))

        def logprobs(self, tokens: List[int], context: Optional[List[int]] = None) -> np.ndarray:
            tokens_tuple = tuple(tokens)
            context_tuple = tuple(context) if context else ()
            return self._cached_logprobs(tokens_tuple, context_tuple)

        def sample(self, context: Optional[List[int]] = None,
                  temperature: float = 1.0,
                  max_tokens: int = 100) -> List[int]:
            # Sampling is not memoized as it's stochastic
            return model.sample(context, temperature, max_tokens)

        def score(self, sequence: List[int]) -> float:
            return self._cached_score(tuple(sequence))

        def __repr__(self) -> str:
            return f"memoize({model})"

    return MemoizedModel()


def lift(function: Callable[[Any], Any]) -> Callable[[LanguageModel], LanguageModel]:
    """
    Lift a regular function to operate on models.

    Args:
        function: Function to lift

    Returns:
        Function that operates on models
    """
    def lifted(model: LanguageModel) -> LanguageModel:
        return adapt(model, function)

    return lifted