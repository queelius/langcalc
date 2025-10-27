"""
N-gram language model with projection support.

This module provides an efficient n-gram model implementation
using suffix arrays for fast retrieval and supporting various
projection strategies.
"""

from typing import List, Optional, Dict, Tuple
import numpy as np
from collections import defaultdict, Counter

from langcalc.models.base import LanguageModel
from langcalc.data.suffix_array import SuffixArray
from langcalc.projections.base import Projection, IdentityProjection


class NGramModel(LanguageModel):
    """
    N-gram language model with efficient suffix array backing.

    Supports various projection strategies for context retrieval
    and can be composed with other models using algebraic operations.
    """

    def __init__(self, corpus: List[int],
                 n: int = 3,
                 projection: Optional[Projection] = None,
                 smoothing: str = 'laplace',
                 alpha: float = 1.0):
        """
        Initialize n-gram model from corpus.

        Args:
            corpus: Training corpus as token sequence
            n: Order of the n-gram model
            projection: Projection function for context transformation
            smoothing: Smoothing method ('laplace', 'kneser_ney', 'none')
            alpha: Smoothing parameter
        """
        self.corpus = corpus
        self.n = n
        self.projection = projection or IdentityProjection()
        self.smoothing = smoothing
        self.alpha = alpha

        # Build suffix array for efficient retrieval
        self.suffix_array = SuffixArray(corpus)

        # Build n-gram counts
        self._build_ngram_counts()

    def _build_ngram_counts(self):
        """Build n-gram frequency tables."""
        self.ngram_counts = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)

        # Count all n-grams
        for i in range(len(self.corpus) - self.n + 1):
            context = tuple(self.corpus[i:i + self.n - 1])
            token = self.corpus[i + self.n - 1]

            self.ngram_counts[context][token] += 1
            self.context_counts[context] += 1

        # Vocabulary for smoothing
        self.vocab = set(self.corpus)
        self.vocab_size = len(self.vocab)

    def logprobs(self, tokens: List[int], context: Optional[List[int]] = None) -> np.ndarray:
        """
        Compute log probabilities for tokens given context.

        Args:
            tokens: List of token ids to score
            context: Optional context tokens

        Returns:
            Array of log probabilities
        """
        if context is None:
            context = []

        # Apply projection to context
        projected = self.projection.project(context)

        # Use only the last n-1 tokens as context
        ngram_context = tuple(projected[-(self.n - 1):]) if len(projected) >= self.n - 1 else tuple(projected)

        logprobs = []
        for token in tokens:
            prob = self._get_probability(ngram_context, token)
            logprobs.append(np.log(prob) if prob > 0 else -np.inf)

        return np.array(logprobs)

    def _get_probability(self, context: Tuple[int, ...], token: int) -> float:
        """
        Get smoothed probability of token given context.

        Args:
            context: Context tuple
            token: Token to score

        Returns:
            Probability value
        """
        if self.smoothing == 'none':
            if context in self.ngram_counts:
                count = self.ngram_counts[context].get(token, 0)
                total = self.context_counts[context]
                return count / total if total > 0 else 0.0
            return 0.0

        elif self.smoothing == 'laplace':
            count = self.ngram_counts[context].get(token, 0) + self.alpha
            total = self.context_counts[context] + self.alpha * self.vocab_size
            return count / total

        elif self.smoothing == 'kneser_ney':
            # Simplified Kneser-Ney smoothing
            return self._kneser_ney_probability(context, token)

        else:
            raise ValueError(f"Unknown smoothing method: {self.smoothing}")

    def _kneser_ney_probability(self, context: Tuple[int, ...], token: int) -> float:
        """Simplified Kneser-Ney smoothing."""
        discount = 0.75  # Fixed discount

        count = self.ngram_counts[context].get(token, 0)
        context_count = self.context_counts[context]

        if context_count == 0:
            # Back off to uniform distribution
            return 1.0 / self.vocab_size

        # Calculate probability with discount
        prob = max(count - discount, 0) / context_count

        # Add continuation probability
        num_continuations = len(self.ngram_counts[context])
        lambda_weight = discount * num_continuations / context_count

        # Simplified continuation probability
        continuation_prob = 1.0 / self.vocab_size

        return prob + lambda_weight * continuation_prob

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
        if context is None:
            context = []
        else:
            context = context.copy()

        generated = []

        for _ in range(max_tokens):
            # Apply projection
            projected = self.projection.project(context)
            ngram_context = tuple(projected[-(self.n - 1):]) if len(projected) >= self.n - 1 else tuple(projected)

            # Get probability distribution
            if ngram_context in self.ngram_counts:
                token_counts = self.ngram_counts[ngram_context]
                tokens = list(token_counts.keys())

                if tokens:  # Check if we have any tokens
                    counts = np.array([token_counts[t] for t in tokens], dtype=np.float32)

                    # Apply temperature
                    if temperature != 1.0:
                        counts = np.power(counts, 1.0 / temperature)

                    # Normalize to probabilities
                    probs = counts / counts.sum()

                    # Sample token
                    token = np.random.choice(tokens, p=probs)
                else:
                    # Fall back to random sampling if no tokens found
                    if self.vocab:
                        token = np.random.choice(list(self.vocab))
                    else:
                        # Emergency fallback: return a random token id
                        token = np.random.randint(0, 100)
            else:
                # Random sampling from vocabulary
                if self.vocab:
                    token = np.random.choice(list(self.vocab))
                else:
                    # Emergency fallback: return a random token id
                    token = np.random.randint(0, 100)

            generated.append(token)
            context.append(token)

        return generated

    def score(self, sequence: List[int]) -> float:
        """
        Score a complete sequence.

        Args:
            sequence: List of token ids

        Returns:
            Log probability of the sequence
        """
        total_logprob = 0.0

        for i in range(1, len(sequence)):
            context = sequence[max(0, i - self.n + 1):i]
            token = sequence[i]

            logprobs = self.logprobs([token], context)
            total_logprob += logprobs[0]

        return total_logprob

    def find_similar_contexts(self, context: List[int], k: int = 5) -> List[Tuple[List[int], float]]:
        """
        Find similar contexts in the corpus.

        Args:
            context: Query context
            k: Number of similar contexts to return

        Returns:
            List of (context, similarity_score) tuples
        """
        # Apply projection
        projected = self.projection.project(context)

        # Find matches using suffix array
        matches = self.suffix_array.search(projected[-self.n + 1:])

        similar_contexts = []
        for match_pos in matches[:k]:
            # Extract context around match
            start = max(0, match_pos - self.n + 1)
            end = min(len(self.corpus), match_pos + self.n)
            matched_context = self.corpus[start:end]

            # Compute similarity
            similarity = self.projection.similarity(projected, matched_context)
            similar_contexts.append((matched_context, similarity))

        # Sort by similarity
        similar_contexts.sort(key=lambda x: -x[1])

        return similar_contexts[:k]

    def perplexity(self, test_corpus: List[int]) -> float:
        """
        Compute perplexity on test corpus.

        Args:
            test_corpus: Test token sequence

        Returns:
            Perplexity value
        """
        log_prob = self.score(test_corpus)
        return np.exp(-log_prob / len(test_corpus))

    def __repr__(self) -> str:
        return f"NGramModel(n={self.n}, smoothing={self.smoothing}, projection={self.projection})"