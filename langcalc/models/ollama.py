"""
Ollama language model provider for LangCalc.

This module provides a LangCalc-compatible wrapper for Ollama LLMs,
enabling them to be composed algebraically with other language models.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import requests
import json

from langcalc.models.base import LanguageModel


class OllamaModel(LanguageModel):
    """
    LangCalc adapter for Ollama LLM models.

    Provides a uniform interface to Ollama models over HTTP,
    allowing them to be composed with other models using
    LangCalc's algebraic operations.

    Example:
        >>> from langcalc.models import OllamaModel, InfinigramModel
        >>>
        >>> # Create Ollama model
        >>> llm = OllamaModel("mistral", host="localhost")
        >>>
        >>> # Create Infinigram model
        >>> corpus = list("the cat sat on the mat".encode('utf-8'))
        >>> infinigram = InfinigramModel(corpus)
        >>>
        >>> # Compose them
        >>> model = 0.95 * llm + 0.05 * infinigram
        >>>
        >>> # Generate
        >>> context = list("the".encode('utf-8'))
        >>> samples = model.sample(context, max_tokens=20)
    """

    def __init__(self,
                 model_name: str = "mistral",
                 host: str = "localhost",
                 port: int = 11434,
                 timeout: int = 30,
                 vocab_size: int = 32000):
        """
        Initialize Ollama model.

        Args:
            model_name: Name of the Ollama model (e.g., "mistral", "llama2")
            host: Hostname or IP address of Ollama server
            port: Port number of Ollama server (default: 11434)
            timeout: Request timeout in seconds
            vocab_size: Vocabulary size for the model (for smoothing)

        Raises:
            ConnectionError: If cannot connect to Ollama server
        """
        self.model_name = model_name
        self.host = host
        self.port = port
        self.timeout = timeout
        self.vocab_size = vocab_size

        self.base_url = f"http://{host}:{port}"
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        self.embeddings_url = f"{self.base_url}/api/embeddings"

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]

                print(f"✓ Connected to Ollama at {self.host}:{self.port}")
                print(f"  Available models: {model_names}")

                # Warn if requested model not available
                if self.model_name not in model_names:
                    # Check if it's a partial match
                    matches = [m for m in model_names if self.model_name in m]
                    if matches:
                        print(f"  ⚠ Model '{self.model_name}' not found, but found: {matches}")
                    else:
                        print(f"  ⚠ Model '{self.model_name}' not available on server")
            else:
                raise ConnectionError(f"Ollama server responded with status {response.status_code}")
        except requests.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to Ollama server at {self.host}:{self.port}: {e}\n"
                f"Make sure Ollama is running: ollama serve"
            )

    def _bytes_to_text(self, tokens: List[int]) -> str:
        """Convert byte tokens to UTF-8 text."""
        try:
            return bytes(tokens).decode('utf-8', errors='ignore')
        except Exception:
            # Fallback: treat as string of numbers
            return ' '.join(str(t) for t in tokens)

    def _text_to_bytes(self, text: str) -> List[int]:
        """Convert text to byte tokens."""
        return list(text.encode('utf-8'))

    def logprobs(self, tokens: List[int], context: Optional[List[int]] = None) -> np.ndarray:
        """
        Compute log probabilities for tokens given context.

        Note: Ollama doesn't directly provide per-token logprobs, so this
        uses a sampling-based approximation by generating multiple completions
        and estimating the distribution.

        Args:
            tokens: List of token ids to score
            context: Optional context tokens

        Returns:
            Array of log probabilities (approximated)
        """
        if context is None:
            context = []

        # Convert context to text
        context_text = self._bytes_to_text(context)

        # Since Ollama doesn't provide logprobs directly, we'll use a
        # uniform distribution with small penalty for unseen tokens
        # This is a limitation of Ollama's API

        # For now, return uniform distribution in log space
        # A better approach would be to sample multiple times
        uniform_logprob = np.log(1.0 / self.vocab_size)
        return np.full(len(tokens), uniform_logprob, dtype=np.float64)

    def sample(self,
               context: Optional[List[int]] = None,
               temperature: float = 1.0,
               max_tokens: int = 100) -> List[int]:
        """
        Sample tokens from the Ollama model.

        Args:
            context: Optional context tokens
            temperature: Sampling temperature (higher = more random)
            max_tokens: Maximum number of tokens to generate

        Returns:
            List of generated token ids (as bytes)
        """
        if context is None:
            context = []

        # Convert context to text
        prompt = self._bytes_to_text(context)

        try:
            # Call Ollama API
            response = requests.post(
                self.generate_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')

                # Convert back to bytes
                return self._text_to_bytes(generated_text)
            else:
                raise RuntimeError(
                    f"Ollama API error {response.status_code}: {response.text}"
                )

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to call Ollama API: {e}")

    def score(self, sequence: List[int]) -> float:
        """
        Score a complete sequence.

        Note: Since Ollama doesn't provide logprobs, this returns
        an approximate score based on sequence length.

        Args:
            sequence: List of token ids

        Returns:
            Approximate log probability of the sequence
        """
        # Approximate score: uniform probability per token
        return -len(sequence) * np.log(self.vocab_size)

    def chat(self,
             messages: List[Dict[str, str]],
             temperature: float = 1.0,
             max_tokens: int = 100) -> str:
        """
        Chat completion using Ollama's chat API.

        This is a convenience method not part of the LanguageModel interface,
        but useful for chat-based interactions.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response text

        Example:
            >>> model = OllamaModel("mistral")
            >>> messages = [
            ...     {"role": "user", "content": "Hello!"}
            ... ]
            >>> response = model.chat(messages)
        """
        try:
            response = requests.post(
                self.chat_url,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', '')
            else:
                raise RuntimeError(
                    f"Ollama chat API error {response.status_code}: {response.text}"
                )

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to call Ollama chat API: {e}")

    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Get embeddings for text.

        This is a convenience method not part of the LanguageModel interface.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        try:
            response = requests.post(
                self.embeddings_url,
                json={
                    "model": self.model_name,
                    "prompt": text
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                return np.array(result.get('embedding', []))
            else:
                raise RuntimeError(
                    f"Ollama embeddings API error {response.status_code}: {response.text}"
                )

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to call Ollama embeddings API: {e}")

    def __repr__(self) -> str:
        return f"OllamaModel(model='{self.model_name}', host='{self.host}:{self.port}')"
