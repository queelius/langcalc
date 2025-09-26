"""
Mixture models for combining multiple language models.

This module provides implementations for weighted mixtures
of language models with elegant composition semantics.
"""

from typing import List, Optional
import numpy as np

from ngram_projections.models.base import LanguageModel


class MixtureModel(LanguageModel):
    """
    Weighted mixture of multiple language models.

    Combines models using weighted averaging of probabilities
    in probability space (not log space).
    """

    def __init__(self, models: List[LanguageModel],
                 weights: Optional[List[float]] = None):
        """
        Initialize mixture model.

        Args:
            models: List of component models
            weights: Optional weights (normalized automatically)
        """
        self.models = models
        self.n_models = len(models)

        if weights is None:
            # Equal weights by default
            self.weights = np.ones(self.n_models) / self.n_models
        else:
            # Normalize weights
            weights = np.array(weights)
            self.weights = weights / weights.sum()

    def logprobs(self, tokens: List[int], context: Optional[List[int]] = None) -> np.ndarray:
        """
        Compute log probabilities as weighted mixture.

        Args:
            tokens: List of token ids to score
            context: Optional context tokens

        Returns:
            Array of log probabilities
        """
        # Get probabilities from each model
        all_probs = []
        for model in self.models:
            logprobs = model.logprobs(tokens, context)
            probs = np.exp(logprobs)
            all_probs.append(probs)

        # Weighted average in probability space
        all_probs = np.array(all_probs)
        mixed_probs = np.sum(all_probs * self.weights[:, np.newaxis], axis=0)

        # Convert back to log space
        return np.log(np.maximum(mixed_probs, 1e-10))

    def sample(self, context: Optional[List[int]] = None,
               temperature: float = 1.0,
               max_tokens: int = 100) -> List[int]:
        """
        Sample from the mixture model.

        Args:
            context: Optional context tokens
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate

        Returns:
            List of generated token ids
        """
        # First, select which model to sample from
        model_idx = np.random.choice(self.n_models, p=self.weights)

        # Sample from selected model
        return self.models[model_idx].sample(context, temperature, max_tokens)

    def score(self, sequence: List[int]) -> float:
        """
        Score a sequence using the mixture.

        Args:
            sequence: List of token ids

        Returns:
            Log probability of the sequence
        """
        # Get scores from each model
        scores = []
        for model in self.models:
            scores.append(model.score(sequence))

        # Mixture in probability space
        log_scores = np.array(scores)
        max_score = np.max(log_scores)

        # Numerical stability: subtract max before exp
        probs = np.exp(log_scores - max_score)
        mixed_prob = np.sum(probs * self.weights)

        return max_score + np.log(mixed_prob)

    def update_weights(self, weights: List[float]):
        """
        Update mixture weights.

        Args:
            weights: New weights (will be normalized)
        """
        weights = np.array(weights)
        self.weights = weights / weights.sum()

    def __repr__(self) -> str:
        components = []
        for weight, model in zip(self.weights, self.models):
            components.append(f"{weight:.3f}*{model}")
        return f"MixtureModel([{', '.join(components)}])"


class InterpolatedModel(MixtureModel):
    """
    Linear interpolation between models.

    Special case of mixture with two models and complementary weights.
    """

    def __init__(self, model1: LanguageModel, model2: LanguageModel,
                 lambda_weight: float = 0.5):
        """
        Initialize interpolated model.

        Args:
            model1: First model
            model2: Second model
            lambda_weight: Weight for first model (second gets 1 - lambda_weight)
        """
        super().__init__([model1, model2], [lambda_weight, 1 - lambda_weight])
        self.lambda_weight = lambda_weight

    def set_lambda(self, lambda_weight: float):
        """Update interpolation weight."""
        self.lambda_weight = lambda_weight
        self.update_weights([lambda_weight, 1 - lambda_weight])

    def __repr__(self) -> str:
        return f"InterpolatedModel(λ={self.lambda_weight:.3f})"


class HierarchicalMixture(LanguageModel):
    """
    Hierarchical mixture with dynamic weighting.

    Weights can depend on the context, allowing for more
    sophisticated model combination strategies.
    """

    def __init__(self, models: List[LanguageModel],
                 weight_fn=None):
        """
        Initialize hierarchical mixture.

        Args:
            models: List of component models
            weight_fn: Function that takes context and returns weights
        """
        self.models = models
        self.n_models = len(models)
        self.weight_fn = weight_fn or self._default_weight_fn

    def _default_weight_fn(self, context: Optional[List[int]]) -> np.ndarray:
        """Default: equal weights regardless of context."""
        return np.ones(self.n_models) / self.n_models

    def logprobs(self, tokens: List[int], context: Optional[List[int]] = None) -> np.ndarray:
        """
        Compute log probabilities with context-dependent weights.

        Args:
            tokens: List of token ids to score
            context: Optional context tokens

        Returns:
            Array of log probabilities
        """
        # Get context-dependent weights
        weights = self.weight_fn(context)

        # Get probabilities from each model
        all_probs = []
        for model in self.models:
            logprobs = model.logprobs(tokens, context)
            probs = np.exp(logprobs)
            all_probs.append(probs)

        # Weighted average
        all_probs = np.array(all_probs)
        mixed_probs = np.sum(all_probs * weights[:, np.newaxis], axis=0)

        return np.log(np.maximum(mixed_probs, 1e-10))

    def sample(self, context: Optional[List[int]] = None,
               temperature: float = 1.0,
               max_tokens: int = 100) -> List[int]:
        """Sample with context-dependent model selection."""
        weights = self.weight_fn(context)
        model_idx = np.random.choice(self.n_models, p=weights)
        return self.models[model_idx].sample(context, temperature, max_tokens)

    def score(self, sequence: List[int]) -> float:
        """Score with dynamic weights."""
        total_logprob = 0.0

        for i in range(1, len(sequence)):
            context = sequence[:i]
            token = sequence[i]

            logprobs = self.logprobs([token], context)
            total_logprob += logprobs[0]

        return total_logprob

    def __repr__(self) -> str:
        return f"HierarchicalMixture(n_models={self.n_models})"