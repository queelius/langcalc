"""
Recency projection using longest suffix matching.

This projection emphasizes recent context by finding the longest
matching suffix in the training data.
"""

from typing import List, Optional
from ngram_projections.projections.base import Projection
from ngram_projections.data.suffix_array import SuffixArray


class RecencyProjection(Projection):
    """
    Projection that retrieves context based on longest suffix match.

    This implements the recency heuristic from n-gram models,
    finding the longest matching suffix in the corpus.
    """

    def __init__(self, corpus: Optional[List[int]] = None,
                 suffix_array: Optional[SuffixArray] = None,
                 max_context: int = 100):
        """
        Initialize with a corpus or pre-built suffix array.

        Args:
            corpus: Token sequence to build suffix array from
            suffix_array: Pre-built suffix array
            max_context: Maximum context length to consider
        """
        if suffix_array is not None:
            self.suffix_array = suffix_array
        elif corpus is not None:
            self.suffix_array = SuffixArray(corpus)
        else:
            raise ValueError("Must provide either corpus or suffix_array")

        self.max_context = max_context

    def project(self, context: List[int]) -> List[int]:
        """
        Project context by finding longest matching suffix.

        Args:
            context: Input context tokens

        Returns:
            Projected context (the matching suffix)
        """
        if not context:
            return []

        # Limit context to max_context
        context = context[-self.max_context:]

        # Find longest suffix match
        position, length = self.suffix_array.find_longest_suffix(context)

        if length == 0:
            # No match found, return truncated context
            return context[-min(10, len(context)):]

        # Return the matching suffix
        return context[-length:]

    def similarity(self, seq1: List[int], seq2: List[int]) -> float:
        """
        Compute similarity based on longest common suffix.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Similarity score based on suffix overlap
        """
        # Find longest common suffix
        common_length = 0
        min_len = min(len(seq1), len(seq2))

        for i in range(1, min_len + 1):
            if seq1[-i] == seq2[-i]:
                common_length = i
            else:
                break

        if min_len == 0:
            return 0.0

        return common_length / min_len

    def find_continuations(self, context: List[int], k: int = 5) -> List[List[int]]:
        """
        Find the k most likely continuations based on suffix matches.

        Args:
            context: Context to continue from
            k: Number of continuations to return

        Returns:
            List of continuation sequences
        """
        position, length = self.suffix_array.find_longest_suffix(context)

        if position == -1:
            return []

        continuations = []
        # Get continuations after the matching position
        for offset in range(k):
            cont_pos = position + length + offset
            if cont_pos < len(self.suffix_array.tokens):
                # Get next few tokens as continuation
                end_pos = min(cont_pos + 5, len(self.suffix_array.tokens))
                continuation = self.suffix_array.tokens[cont_pos:end_pos]
                if continuation:
                    continuations.append(continuation)

        return continuations

    def __repr__(self) -> str:
        return f"RecencyProjection(max_context={self.max_context})"