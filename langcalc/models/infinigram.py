"""
Infinigram language model adapter for LangCalc.

This module provides a LangCalc-compatible wrapper for the external
infinigram package, enabling variable-length n-gram models with suffix
array backing to be composed algebraically with other language models.
"""

from typing import List, Optional
import numpy as np

from langcalc.models.base import LanguageModel

try:
    from infinigram import Infinigram as _Infinigram
except ImportError:
    raise ImportError(
        "The infinigram package is required to use InfinigramModel. "
        "Install it with: pip install infinigram>=0.2.0"
    )


class InfinigramModel(LanguageModel):
    """
    LangCalc adapter for the Infinigram variable-length n-gram model.

    Infinigram uses suffix arrays to efficiently find the longest matching
    context in a corpus and predict the next token based on observed
    continuations. Unlike traditional fixed-order n-grams, Infinigram
    automatically adapts to use as much context as available.

    This adapter wraps the external infinigram package to work within
    LangCalc's algebraic framework, supporting composition with other
    language models.

    Example:
        >>> from langcalc.models import InfinigramModel, MockLLM
        >>>
        >>> # Create from byte-level corpus
        >>> text = "the cat sat on the mat"
        >>> corpus = list(text.encode('utf-8'))
        >>> infinigram = InfinigramModel(corpus, max_length=20)
        >>>
        >>> # Compose with LLM
        >>> llm = MockLLM(vocab_size=256)
        >>> model = 0.95 * llm + 0.05 * infinigram
        >>>
        >>> # Predict next byte
        >>> context = list("the cat".encode('utf-8'))
        >>> logprobs = model.logprobs(list(range(256)), context)
    """

    def __init__(self,
                 corpus: List[int],
                 max_length: Optional[int] = None,
                 min_count: int = 1,
                 smoothing: float = 0.01):
        """
        Initialize Infinigram model from byte-level corpus.

        Args:
            corpus: Token sequence (should be bytes 0-255 for v0.2.0+)
            max_length: Maximum suffix length to consider (None = unlimited)
            min_count: Minimum occurrences for a pattern to be included
            smoothing: Laplace smoothing factor for unseen tokens (default: 0.01)

        Raises:
            ValueError: If corpus is empty or smoothing is negative

        Note:
            Infinigram v0.2.0+ expects byte-level input (values 0-255).
            For text, encode as UTF-8 first:
                corpus = list("hello world".encode('utf-8'))
        """
        # Input validation
        if not corpus:
            raise ValueError("Corpus cannot be empty")

        if smoothing < 0:
            raise ValueError(f"Smoothing must be non-negative, got {smoothing}")

        self.infinigram = _Infinigram(
            corpus=corpus,
            max_length=max_length,
            min_count=min_count
        )

        # Store parameters for representation
        self.max_length = max_length
        self.min_count = min_count
        self.smoothing = smoothing  # Store for use in predict()

        # Get vocabulary size (256 for byte-level models in v0.2.0+)
        self.vocab_size = 256  # Fixed vocabulary for byte-level

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

        # Get probability distribution from infinigram with smoothing
        probs_dict = self.infinigram.predict(
            context,
            top_k=self.vocab_size,
            smoothing=self.smoothing
        )

        # Convert to log probabilities for requested tokens
        logprobs = []
        for token in tokens:
            prob = probs_dict.get(token, self.smoothing / self.vocab_size)
            logprobs.append(np.log(prob) if prob > 0 else -np.inf)

        return np.array(logprobs)

    def sample(self,
               context: Optional[List[int]] = None,
               temperature: float = 1.0,
               max_tokens: int = 100) -> List[int]:
        """
        Sample tokens from the model.

        Args:
            context: Optional context tokens
            temperature: Sampling temperature (higher = more random)
            max_tokens: Maximum number of tokens to generate

        Returns:
            List of generated token ids
        """
        if context is None:
            context = []

        generated = []
        current_context = list(context)

        for _ in range(max_tokens):
            # Get probability distribution with smoothing
            probs_dict = self.infinigram.predict(
                current_context,
                top_k=self.vocab_size,
                smoothing=self.smoothing
            )

            # Convert to arrays for sampling
            tokens = list(probs_dict.keys())
            probs = np.array(list(probs_dict.values()))

            # Apply temperature
            if temperature != 1.0:
                logits = np.log(probs + 1e-10)
                logits = logits / temperature
                probs = np.exp(logits - np.max(logits))
                probs = probs / np.sum(probs)

            # Sample next token
            next_token = np.random.choice(tokens, p=probs)
            generated.append(next_token)

            # Update context for next iteration
            current_context.append(next_token)

        return generated

    def score(self, sequence: List[int]) -> float:
        """
        Score a complete sequence.

        Computes the log probability of the sequence by summing
        log probabilities of each token given its prefix.

        Args:
            sequence: List of token ids

        Returns:
            Log probability of the sequence
        """
        if len(sequence) == 0:
            return 0.0

        total_logprob = 0.0

        # Score each token given its prefix
        for i in range(len(sequence)):
            context = sequence[:i]  # Prefix before current token
            token = sequence[i]

            # Get log probability of this token
            logprob = self.logprobs([token], context)[0]
            total_logprob += logprob

        return total_logprob

    def confidence(self, context: List[int]) -> float:
        """
        Get confidence score for predictions given context.

        This is a pass-through to the underlying Infinigram's
        confidence scoring, which is based on match length and
        frequency.

        Args:
            context: Token sequence

        Returns:
            Confidence score in [0, 1]
        """
        return self.infinigram.confidence(context)

    def longest_suffix(self, context: List[int]) -> tuple:
        """
        Find longest matching suffix in corpus.

        This is a pass-through to the underlying Infinigram's
        suffix matching capability, useful for debugging or
        understanding model behavior.

        Args:
            context: Token sequence

        Returns:
            (position, length) of longest match
        """
        return self.infinigram.longest_suffix(context)

    def update(self, new_tokens: List[int]):
        """
        Dynamically add new tokens to corpus.

        Note: This rebuilds the suffix array, which can be slow
        for large corpora. For incremental updates, consider
        batching additions.

        Args:
            new_tokens: Token sequence to add
        """
        self.infinigram.update(new_tokens)

    def __repr__(self) -> str:
        return (f"InfinigramModel(corpus_size={len(self.infinigram.corpus)}, "
                f"max_length={self.max_length})")
