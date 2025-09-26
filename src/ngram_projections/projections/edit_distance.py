"""
Edit distance projection for approximate matching.

This projection finds contexts that are similar within a certain
edit distance threshold.
"""

from typing import List, Optional, Tuple, Set
from collections import deque
import numpy as np
from ngram_projections.projections.base import Projection


class EditDistanceProjection(Projection):
    """
    Projection using edit distance for approximate matching.

    Finds contexts that can be reached through a small number
    of insertions, deletions, and substitutions.
    """

    def __init__(self, max_distance: int = 2,
                 vocab_size: Optional[int] = None,
                 operations: Set[str] = {'insert', 'delete', 'substitute'}):
        """
        Initialize edit distance projection.

        Args:
            max_distance: Maximum edit distance to consider
            vocab_size: Size of vocabulary for substitutions
            operations: Allowed edit operations
        """
        self.max_distance = max_distance
        self.vocab_size = vocab_size
        self.operations = operations

    def project(self, context: List[int]) -> List[int]:
        """
        Project context by finding nearby sequences.

        For efficiency, this returns a trimmed version of the context
        that focuses on the most important parts for matching.

        Args:
            context: Input context tokens

        Returns:
            Projected context
        """
        if not context:
            return []

        # For projection, we keep the last portion of context
        # This could be extended to find the "core" of the sequence
        max_len = min(len(context), 20)
        return context[-max_len:]

    def similarity(self, seq1: List[int], seq2: List[int]) -> float:
        """
        Compute similarity based on edit distance.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Similarity score (1 - normalized_edit_distance)
        """
        distance = self.edit_distance(seq1, seq2)
        max_len = max(len(seq1), len(seq2))

        if max_len == 0:
            return 1.0

        normalized_distance = min(distance / max_len, 1.0)
        return 1.0 - normalized_distance

    def edit_distance(self, seq1: List[int], seq2: List[int]) -> int:
        """
        Compute edit distance between two sequences.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Minimum edit distance
        """
        m, n = len(seq1), len(seq2)

        # Dynamic programming table
        dp = np.zeros((m + 1, n + 1), dtype=np.int32)

        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill the table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    candidates = []

                    if 'substitute' in self.operations:
                        candidates.append(dp[i - 1][j - 1] + 1)
                    if 'delete' in self.operations:
                        candidates.append(dp[i - 1][j] + 1)
                    if 'insert' in self.operations:
                        candidates.append(dp[i][j - 1] + 1)

                    dp[i][j] = min(candidates) if candidates else float('inf')

        return dp[m][n]

    def find_neighbors(self, sequence: List[int],
                      max_distance: Optional[int] = None) -> List[List[int]]:
        """
        Find all sequences within edit distance of the input.

        Args:
            sequence: Input sequence
            max_distance: Maximum edit distance (uses self.max_distance if None)

        Returns:
            List of neighbor sequences
        """
        if max_distance is None:
            max_distance = self.max_distance

        # Use BFS to explore edit distance neighborhood
        visited = set()
        queue = deque([(tuple(sequence), 0)])
        neighbors = []

        while queue:
            current, distance = queue.popleft()

            if current in visited:
                continue
            visited.add(current)

            if distance > 0:  # Don't include the original sequence
                neighbors.append(list(current))

            if distance >= max_distance:
                continue

            # Generate all possible edits
            current_list = list(current)

            # Deletions
            if 'delete' in self.operations:
                for i in range(len(current_list)):
                    neighbor = current_list[:i] + current_list[i + 1:]
                    queue.append((tuple(neighbor), distance + 1))

            # Insertions
            if 'insert' in self.operations and self.vocab_size:
                for i in range(len(current_list) + 1):
                    # Sample a few tokens to insert (for efficiency)
                    for token in range(min(5, self.vocab_size)):
                        neighbor = current_list[:i] + [token] + current_list[i:]
                        queue.append((tuple(neighbor), distance + 1))

            # Substitutions
            if 'substitute' in self.operations and self.vocab_size:
                for i in range(len(current_list)):
                    # Sample a few tokens to substitute (for efficiency)
                    for token in range(min(5, self.vocab_size)):
                        if token != current_list[i]:
                            neighbor = current_list[:i] + [token] + current_list[i + 1:]
                            queue.append((tuple(neighbor), distance + 1))

        return neighbors[:100]  # Limit output for efficiency

    def __repr__(self) -> str:
        return f"EditDistanceProjection(max_distance={self.max_distance})"