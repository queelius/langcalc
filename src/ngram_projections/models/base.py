"""
Base language model abstraction with algebraic operations.

This module provides the fundamental abstraction for all language models,
supporting elegant composition through operator overloading.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Any
import numpy as np


class LanguageModel(ABC):
    """
    Abstract base class for all language models.

    Supports algebraic operations for model composition:
    - Addition (+): Mixture models
    - Multiplication (*): Weighted models
    - Right shift (>>): Sequential composition
    - Or (|): Fallback/ensemble
    - Matmul (@): Projection application
    """

    @abstractmethod
    def logprobs(self, tokens: List[int], context: Optional[List[int]] = None) -> np.ndarray:
        """
        Compute log probabilities for tokens given context.

        Args:
            tokens: List of token ids to score
            context: Optional context tokens

        Returns:
            Array of log probabilities
        """
        pass

    @abstractmethod
    def sample(self, context: Optional[List[int]] = None,
               temperature: float = 1.0,
               max_tokens: int = 100) -> List[int]:
        """
        Sample tokens from the model.

        Args:
            context: Optional context tokens
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate

        Returns:
            List of generated token ids
        """
        pass

    @abstractmethod
    def score(self, sequence: List[int]) -> float:
        """
        Score a complete sequence.

        Args:
            sequence: List of token ids

        Returns:
            Log probability of the sequence
        """
        pass

    def __add__(self, other: 'LanguageModel') -> 'LanguageModel':
        """Create an equal-weight mixture model."""
        from ngram_projections.models.mixture import MixtureModel
        return MixtureModel([self, other], weights=[0.5, 0.5])

    def __mul__(self, weight: float) -> 'WeightedModel':
        """Create a weighted model for use in mixtures."""
        return WeightedModel(self, weight)

    def __rmul__(self, weight: float) -> 'WeightedModel':
        """Support weight * model syntax."""
        return self.__mul__(weight)

    def __rshift__(self, other: 'LanguageModel') -> 'SequentialModel':
        """
        Chain models sequentially (composition).
        model1 >> model2 means: use model1's output as input to model2
        """
        return SequentialModel(self, other)

    def __or__(self, other: 'LanguageModel') -> 'FallbackModel':
        """
        Create a fallback model.
        model1 | model2 means: use model2 if model1 fails or has low confidence
        """
        return FallbackModel(self, other)

    def __matmul__(self, projection: 'Projection') -> 'ProjectedModel':
        """
        Apply a projection to the model.
        model @ projection creates a model with projected context
        """
        from ngram_projections.projections.base import Projection
        if not isinstance(projection, Projection):
            raise TypeError(f"Can only apply Projection objects, got {type(projection)}")
        return ProjectedModel(self, projection)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class WeightedModel(LanguageModel):
    """A model with an associated weight for mixture composition."""

    def __init__(self, model: LanguageModel, weight: float):
        self.model = model
        self.weight = weight

    def logprobs(self, tokens: List[int], context: Optional[List[int]] = None) -> np.ndarray:
        return self.model.logprobs(tokens, context)

    def sample(self, context: Optional[List[int]] = None,
               temperature: float = 1.0,
               max_tokens: int = 100) -> List[int]:
        return self.model.sample(context, temperature, max_tokens)

    def score(self, sequence: List[int]) -> float:
        return self.model.score(sequence)

    def __add__(self, other: Union[LanguageModel, 'WeightedModel']) -> LanguageModel:
        """Combine weighted models into a mixture."""
        from ngram_projections.models.mixture import MixtureModel

        if isinstance(other, WeightedModel):
            # Both are weighted
            total_weight = self.weight + other.weight
            return MixtureModel(
                [self.model, other.model],
                [self.weight / total_weight, other.weight / total_weight]
            )
        else:
            # Other is unweighted (implicitly weight 1)
            total_weight = self.weight + 1.0
            return MixtureModel(
                [self.model, other],
                [self.weight / total_weight, 1.0 / total_weight]
            )

    def __repr__(self) -> str:
        return f"{self.weight} * {self.model}"


class SequentialModel(LanguageModel):
    """Sequential composition of two models."""

    def __init__(self, first: LanguageModel, second: LanguageModel):
        self.first = first
        self.second = second

    def logprobs(self, tokens: List[int], context: Optional[List[int]] = None) -> np.ndarray:
        # Generate intermediate context from first model
        if context is not None:
            intermediate = self.first.sample(context, max_tokens=10)
            extended_context = context + intermediate
        else:
            extended_context = self.first.sample(max_tokens=10)

        # Use as context for second model
        return self.second.logprobs(tokens, extended_context)

    def sample(self, context: Optional[List[int]] = None,
               temperature: float = 1.0,
               max_tokens: int = 100) -> List[int]:
        # Generate from first model
        first_output = self.first.sample(context, temperature, max_tokens // 2)

        # Use as context for second model
        extended_context = (context or []) + first_output
        second_output = self.second.sample(extended_context, temperature, max_tokens - len(first_output))

        return first_output + second_output

    def score(self, sequence: List[int]) -> float:
        # Score as a combined sequence
        # This is a simplified implementation
        return self.first.score(sequence[:len(sequence)//2]) + \
               self.second.score(sequence[len(sequence)//2:])

    def __repr__(self) -> str:
        return f"({self.first} >> {self.second})"


class FallbackModel(LanguageModel):
    """Fallback/ensemble model that uses secondary model when primary has low confidence."""

    def __init__(self, primary: LanguageModel, fallback: LanguageModel,
                 confidence_threshold: float = -10.0):
        self.primary = primary
        self.fallback = fallback
        self.confidence_threshold = confidence_threshold

    def logprobs(self, tokens: List[int], context: Optional[List[int]] = None) -> np.ndarray:
        primary_logprobs = self.primary.logprobs(tokens, context)

        # Use fallback for low-confidence predictions
        mask = primary_logprobs < self.confidence_threshold
        if np.any(mask):
            fallback_logprobs = self.fallback.logprobs(tokens, context)
            primary_logprobs[mask] = fallback_logprobs[mask]

        return primary_logprobs

    def sample(self, context: Optional[List[int]] = None,
               temperature: float = 1.0,
               max_tokens: int = 100) -> List[int]:
        # Try primary first
        result = self.primary.sample(context, temperature, max_tokens)

        # Check confidence (simplified: check if result is empty or too short)
        if len(result) < max_tokens // 2:
            # Fallback to secondary
            result = self.fallback.sample(context, temperature, max_tokens)

        return result

    def score(self, sequence: List[int]) -> float:
        primary_score = self.primary.score(sequence)
        if primary_score < self.confidence_threshold * len(sequence):
            return self.fallback.score(sequence)
        return primary_score

    def __repr__(self) -> str:
        return f"({self.primary} | {self.fallback})"


class ProjectedModel(LanguageModel):
    """A model with context projection applied."""

    def __init__(self, model: LanguageModel, projection: 'Projection'):
        self.model = model
        self.projection = projection

    def logprobs(self, tokens: List[int], context: Optional[List[int]] = None) -> np.ndarray:
        # Apply projection to context
        if context is not None:
            projected_context = self.projection.project(context)
        else:
            projected_context = None

        return self.model.logprobs(tokens, projected_context)

    def sample(self, context: Optional[List[int]] = None,
               temperature: float = 1.0,
               max_tokens: int = 100) -> List[int]:
        # Apply projection to context
        if context is not None:
            projected_context = self.projection.project(context)
        else:
            projected_context = None

        return self.model.sample(projected_context, temperature, max_tokens)

    def score(self, sequence: List[int]) -> float:
        # Score with projection applied to prefixes
        total_score = 0.0
        for i in range(1, len(sequence)):
            context = sequence[:i]
            projected_context = self.projection.project(context)
            logprobs = self.model.logprobs([sequence[i]], projected_context)
            total_score += logprobs[0]
        return total_score

    def __repr__(self) -> str:
        return f"({self.model} @ {self.projection})"