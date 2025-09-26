"""
Wrappers for external language models (HuggingFace, OpenAI, etc.).

This module provides a uniform interface to various LLM providers,
allowing them to be composed with n-gram models using our algebraic API.
"""

from typing import List, Optional, Any
import numpy as np

from ngram_projections.models.base import LanguageModel


class HuggingFaceModel(LanguageModel):
    """
    Wrapper for HuggingFace transformers models.

    Provides a uniform interface compatible with our algebra.
    """

    def __init__(self, model_name: str = "gpt2",
                 device: str = "cpu",
                 cache_dir: Optional[str] = None):
        """
        Initialize HuggingFace model.

        Args:
            model_name: Name or path of the model
            device: Device to run on ('cpu' or 'cuda')
            cache_dir: Optional cache directory
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers torch")

        self.model_name = model_name
        self.device = device

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            add_bos_token=False,
            add_eos_token=False
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir
        ).to(device)

        self.model.eval()

        # Handle tokenizer without pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def logprobs(self, tokens: List[int], context: Optional[List[int]] = None) -> np.ndarray:
        """
        Compute log probabilities using the transformer model.

        Args:
            tokens: List of token ids to score
            context: Optional context tokens

        Returns:
            Array of log probabilities
        """
        import torch

        if context is None:
            context = []

        # Prepare input
        input_ids = context.copy()

        with torch.no_grad():
            # Get model outputs
            inputs = torch.tensor([input_ids]).to(self.device)
            outputs = self.model(inputs)
            logits = outputs.logits[0, -1, :]  # Last position

            # Convert to log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Extract probabilities for requested tokens
            token_logprobs = []
            for token in tokens:
                if token < len(log_probs):
                    token_logprobs.append(log_probs[token].item())
                else:
                    token_logprobs.append(-np.inf)

        return np.array(token_logprobs)

    def sample(self, context: Optional[List[int]] = None,
               temperature: float = 1.0,
               max_tokens: int = 100) -> List[int]:
        """
        Sample tokens from the transformer model.

        Args:
            context: Optional context tokens
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate

        Returns:
            List of generated token ids
        """
        import torch

        if context is None:
            context = []

        input_ids = torch.tensor([context]).to(self.device)
        generated = []

        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :] / temperature

                # Sample from distribution
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                generated.append(next_token)

                # Update input
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[next_token]]).to(self.device)
                ], dim=1)

                # Stop at EOS token
                if next_token == self.tokenizer.eos_token_id:
                    break

        return generated

    def score(self, sequence: List[int]) -> float:
        """
        Score a complete sequence.

        Args:
            sequence: List of token ids

        Returns:
            Log probability of the sequence
        """
        import torch

        total_logprob = 0.0

        with torch.no_grad():
            for i in range(1, len(sequence)):
                context = sequence[:i]
                token = sequence[i]

                inputs = torch.tensor([context]).to(self.device)
                outputs = self.model(inputs)
                logits = outputs.logits[0, -1, :]

                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                total_logprob += log_probs[token].item()

        return total_logprob

    def __repr__(self) -> str:
        return f"HuggingFaceModel('{self.model_name}')"


class MockLLM(LanguageModel):
    """
    Mock LLM for testing and demonstration.

    Generates random but consistent outputs for testing
    the algebraic composition API.
    """

    def __init__(self, vocab_size: int = 1000,
                 seed: Optional[int] = None,
                 name: str = "MockLLM"):
        """
        Initialize mock LLM.

        Args:
            vocab_size: Size of vocabulary
            seed: Random seed for reproducibility
            name: Name for this mock model
        """
        self.vocab_size = vocab_size
        self.name = name

        if seed is not None:
            np.random.seed(seed)

        # Generate some "learned" patterns
        self.patterns = self._generate_patterns()

    def _generate_patterns(self) -> dict:
        """Generate some fake patterns for consistency."""
        patterns = {}

        # Create some common bigrams
        for i in range(100):
            context = tuple(np.random.randint(0, self.vocab_size, size=2))
            probs = np.random.dirichlet(np.ones(10))
            tokens = np.random.choice(self.vocab_size, size=10, replace=False)
            patterns[context] = (tokens, probs)

        return patterns

    def logprobs(self, tokens: List[int], context: Optional[List[int]] = None) -> np.ndarray:
        """Generate mock log probabilities."""
        if context and len(context) >= 2:
            key = tuple(context[-2:])
            if key in self.patterns:
                pattern_tokens, pattern_probs = self.patterns[key]

                logprobs = []
                for token in tokens:
                    if token in pattern_tokens:
                        idx = np.where(pattern_tokens == token)[0][0]
                        logprobs.append(np.log(pattern_probs[idx]))
                    else:
                        # Random low probability
                        logprobs.append(np.log(0.001 / self.vocab_size))
                return np.array(logprobs)

        # Default: uniform random
        return np.log(np.ones(len(tokens)) / self.vocab_size)

    def sample(self, context: Optional[List[int]] = None,
               temperature: float = 1.0,
               max_tokens: int = 100) -> List[int]:
        """Generate mock tokens."""
        generated = []

        for _ in range(max_tokens):
            if context and len(context) >= 2:
                key = tuple(context[-2:])
                if key in self.patterns:
                    pattern_tokens, pattern_probs = self.patterns[key]

                    # Adjust probabilities with temperature
                    adjusted_probs = np.power(pattern_probs, 1.0 / temperature)
                    adjusted_probs /= adjusted_probs.sum()

                    idx = np.random.choice(len(pattern_tokens), p=adjusted_probs)
                    token = pattern_tokens[idx]
                else:
                    token = np.random.randint(0, self.vocab_size)
            else:
                token = np.random.randint(0, self.vocab_size)

            generated.append(token)

            if context is not None:
                context = context + [token]
            else:
                context = [token]

        return generated

    def score(self, sequence: List[int]) -> float:
        """Score a sequence."""
        total = 0.0
        for i in range(1, len(sequence)):
            context = sequence[max(0, i-10):i]
            logprobs = self.logprobs([sequence[i]], context)
            total += logprobs[0]
        return total

    def __repr__(self) -> str:
        return f"{self.name}(vocab={self.vocab_size})"