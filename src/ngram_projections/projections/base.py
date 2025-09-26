"""
Base projection abstraction.

Projections transform context sequences into projected forms for retrieval.
They are first-class citizens that can be composed and applied to models.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any
import numpy as np


class Projection(ABC):
    """
    Abstract base class for all projection functions.

    Projections transform input sequences to facilitate retrieval
    from n-gram databases or other context stores.
    """

    @abstractmethod
    def project(self, context: List[int]) -> List[int]:
        """
        Project a context sequence.

        Args:
            context: Input token sequence

        Returns:
            Projected token sequence
        """
        pass

    @abstractmethod
    def similarity(self, seq1: List[int], seq2: List[int]) -> float:
        """
        Compute similarity between two sequences under this projection.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Similarity score in [0, 1]
        """
        pass

    def __rshift__(self, other: 'Projection') -> 'ComposedProjection':
        """
        Compose projections: proj1 >> proj2 applies proj1 then proj2.
        """
        return ComposedProjection(self, other)

    def __or__(self, other: 'Projection') -> 'UnionProjection':
        """
        Union of projections: proj1 | proj2 returns results from both.
        """
        return UnionProjection(self, other)

    def __and__(self, other: 'Projection') -> 'IntersectionProjection':
        """
        Intersection of projections: proj1 & proj2 applies both and finds overlap.
        """
        return IntersectionProjection(self, other)

    def __mul__(self, weight: float) -> 'WeightedProjection':
        """
        Weight a projection for ensemble methods.
        """
        return WeightedProjection(self, weight)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ComposedProjection(Projection):
    """Sequential composition of two projections."""

    def __init__(self, first: Projection, second: Projection):
        self.first = first
        self.second = second

    def project(self, context: List[int]) -> List[int]:
        intermediate = self.first.project(context)
        return self.second.project(intermediate)

    def similarity(self, seq1: List[int], seq2: List[int]) -> float:
        # Composite similarity through both projections
        proj1_1 = self.first.project(seq1)
        proj1_2 = self.first.project(seq2)
        intermediate_sim = self.first.similarity(seq1, seq2)

        proj2_1 = self.second.project(proj1_1)
        proj2_2 = self.second.project(proj1_2)
        final_sim = self.second.similarity(proj2_1, proj2_2)

        # Combine similarities (geometric mean)
        return np.sqrt(intermediate_sim * final_sim)

    def __repr__(self) -> str:
        return f"({self.first} >> {self.second})"


class UnionProjection(Projection):
    """Union of multiple projections."""

    def __init__(self, *projections: Projection):
        self.projections = list(projections)

    def project(self, context: List[int]) -> List[int]:
        # For union, we concatenate results (with deduplication)
        seen = set()
        result = []
        for proj in self.projections:
            projected = proj.project(context)
            for token in projected:
                if token not in seen:
                    seen.add(token)
                    result.append(token)
        return result

    def similarity(self, seq1: List[int], seq2: List[int]) -> float:
        # Maximum similarity across all projections
        similarities = [proj.similarity(seq1, seq2) for proj in self.projections]
        return max(similarities)

    def __repr__(self) -> str:
        return f"({' | '.join(str(p) for p in self.projections)})"


class IntersectionProjection(Projection):
    """Intersection of multiple projections."""

    def __init__(self, *projections: Projection):
        self.projections = list(projections)

    def project(self, context: List[int]) -> List[int]:
        # For intersection, keep only common tokens
        if not self.projections:
            return context

        results = [set(proj.project(context)) for proj in self.projections]
        common = results[0]
        for result_set in results[1:]:
            common &= result_set

        # Preserve order from first projection
        first_result = self.projections[0].project(context)
        return [token for token in first_result if token in common]

    def similarity(self, seq1: List[int], seq2: List[int]) -> float:
        # Minimum similarity across all projections (conservative)
        similarities = [proj.similarity(seq1, seq2) for proj in self.projections]
        return min(similarities)

    def __repr__(self) -> str:
        return f"({' & '.join(str(p) for p in self.projections)})"


class WeightedProjection(Projection):
    """A projection with an associated weight."""

    def __init__(self, projection: Projection, weight: float):
        self.projection = projection
        self.weight = weight

    def project(self, context: List[int]) -> List[int]:
        return self.projection.project(context)

    def similarity(self, seq1: List[int], seq2: List[int]) -> float:
        return self.weight * self.projection.similarity(seq1, seq2)

    def __repr__(self) -> str:
        return f"{self.weight} * {self.projection}"


class IdentityProjection(Projection):
    """Identity projection that returns input unchanged."""

    def project(self, context: List[int]) -> List[int]:
        return context

    def similarity(self, seq1: List[int], seq2: List[int]) -> float:
        if len(seq1) != len(seq2):
            return 0.0
        return sum(a == b for a, b in zip(seq1, seq2)) / len(seq1)

    def __repr__(self) -> str:
        return "IdentityProjection()"