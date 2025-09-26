"""
Semantic projection using embeddings.

This projection finds semantically similar contexts using
embedding-based similarity measures.
"""

from typing import List, Optional, Dict, Callable, Tuple
import numpy as np
from ngram_projections.projections.base import Projection


class SemanticProjection(Projection):
    """
    Projection based on semantic similarity using embeddings.

    Maps token sequences to embedding space and finds
    similar contexts based on cosine similarity.
    """

    def __init__(self, embedding_fn: Optional[Callable] = None,
                 embedding_dim: int = 128,
                 similarity_threshold: float = 0.7):
        """
        Initialize semantic projection.

        Args:
            embedding_fn: Function to compute embeddings for token sequences
            embedding_dim: Dimension of embeddings
            similarity_threshold: Minimum similarity for retrieval
        """
        self.embedding_fn = embedding_fn or self._default_embedding
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold

        # Cache for embeddings
        self.embedding_cache: Dict[tuple, np.ndarray] = {}

    def _default_embedding(self, tokens: List[int]) -> np.ndarray:
        """
        Default embedding function using random projections.

        In practice, this would use a real embedding model.
        """
        # Simple hash-based embedding for demonstration
        np.random.seed(hash(tuple(tokens)) % (2**32))
        return np.random.randn(self.embedding_dim)

    def get_embedding(self, tokens: List[int]) -> np.ndarray:
        """
        Get embedding for a token sequence with caching.

        Args:
            tokens: Token sequence

        Returns:
            Embedding vector
        """
        key = tuple(tokens)
        if key not in self.embedding_cache:
            self.embedding_cache[key] = self.embedding_fn(tokens)
        return self.embedding_cache[key]

    def project(self, context: List[int]) -> List[int]:
        """
        Project context using semantic similarity.

        For efficiency, this returns a truncated version
        that preserves semantic information.

        Args:
            context: Input context tokens

        Returns:
            Projected context
        """
        if not context:
            return []

        # For projection, we select the most semantically relevant portion
        # In practice, this would use attention or other relevance measures

        # Simple strategy: keep last portion with decay
        max_len = min(len(context), 30)
        return context[-max_len:]

    def similarity(self, seq1: List[int], seq2: List[int]) -> float:
        """
        Compute semantic similarity between sequences.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Cosine similarity in [0, 1]
        """
        if not seq1 or not seq2:
            return 0.0

        # Get embeddings
        emb1 = self.get_embedding(seq1)
        emb2 = self.get_embedding(seq2)

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)

        if norm_product == 0:
            return 0.0

        similarity = dot_product / norm_product

        # Map from [-1, 1] to [0, 1]
        return (similarity + 1) / 2

    def find_similar_contexts(self, query: List[int],
                            candidates: List[List[int]],
                            k: int = 5) -> List[Tuple[List[int], float]]:
        """
        Find semantically similar contexts from candidates.

        Args:
            query: Query context
            candidates: List of candidate contexts
            k: Number of results to return

        Returns:
            List of (context, similarity) tuples
        """
        similarities = []

        for candidate in candidates:
            sim = self.similarity(query, candidate)
            if sim >= self.similarity_threshold:
                similarities.append((candidate, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: -x[1])

        return similarities[:k]

    def __repr__(self) -> str:
        return f"SemanticProjection(dim={self.embedding_dim}, threshold={self.similarity_threshold})"


class AttentionProjection(Projection):
    """
    Projection based on attention weights.

    Selects the most relevant parts of the context
    using learned or heuristic attention patterns.
    """

    def __init__(self, attention_fn: Optional[Callable] = None,
                 window_size: int = 10,
                 top_k: int = 5):
        """
        Initialize attention projection.

        Args:
            attention_fn: Function to compute attention weights
            window_size: Size of attention window
            top_k: Number of top positions to keep
        """
        self.attention_fn = attention_fn or self._default_attention
        self.window_size = window_size
        self.top_k = top_k

    def _default_attention(self, tokens: List[int]) -> np.ndarray:
        """
        Default attention using recency bias.

        More recent tokens get higher attention weights.
        """
        n = len(tokens)
        # Exponential decay from end
        weights = np.exp(np.linspace(-2, 0, n))
        return weights / weights.sum()

    def project(self, context: List[int]) -> List[int]:
        """
        Project context using attention-based selection.

        Args:
            context: Input context tokens

        Returns:
            Projected context with most attended tokens
        """
        if not context:
            return []

        if len(context) <= self.top_k:
            return context

        # Compute attention weights
        weights = self.attention_fn(context)

        # Select top-k positions
        top_indices = np.argsort(weights)[-self.top_k:]
        top_indices = np.sort(top_indices)  # Preserve order

        return [context[i] for i in top_indices]

    def similarity(self, seq1: List[int], seq2: List[int]) -> float:
        """
        Compute similarity based on attention patterns.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Similarity score
        """
        if not seq1 or not seq2:
            return 0.0

        # Project both sequences
        proj1 = self.project(seq1)
        proj2 = self.project(seq2)

        # Compare projected sequences
        min_len = min(len(proj1), len(proj2))
        if min_len == 0:
            return 0.0

        matches = sum(a == b for a, b in zip(proj1, proj2))
        return matches / min_len

    def __repr__(self) -> str:
        return f"AttentionProjection(window={self.window_size}, top_k={self.top_k})"