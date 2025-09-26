"""
ngram_projections: An elegant algebraic API for composing language models.

This package provides a composable language model algebra where different LM providers
can be combined using mathematical operators, following the Unix philosophy of doing
one thing well and composing simple pieces into powerful abstractions.
"""

from ngram_projections.models.base import LanguageModel
from ngram_projections.models.ngram import NGramModel
from ngram_projections.models.mixture import MixtureModel
from ngram_projections.projections.base import Projection
from ngram_projections.projections.recency import RecencyProjection
from ngram_projections.projections.edit_distance import EditDistanceProjection

__version__ = "0.1.0"

__all__ = [
    "LanguageModel",
    "NGramModel",
    "MixtureModel",
    "Projection",
    "RecencyProjection",
    "EditDistanceProjection",
]