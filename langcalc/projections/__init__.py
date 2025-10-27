"""
Context projection functions for language models.

Projections transform contexts before model evaluation, enabling
sophisticated retrieval and grounding strategies.
"""

from langcalc.projections.base import Projection, IdentityProjection
from langcalc.projections.recency import RecencyProjection
from langcalc.projections.semantic import SemanticProjection
from langcalc.projections.edit_distance import EditDistanceProjection

__all__ = [
    "Projection",
    "IdentityProjection",
    "RecencyProjection",
    "SemanticProjection",
    "EditDistanceProjection",
]
