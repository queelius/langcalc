"""
LangCalc: A Calculus for Language Models

An algebraic framework for composing language models through mathematical operations,
featuring variable-length n-grams (Infinigrams), efficient suffix arrays, and
lightweight grounding for factual accuracy.

Example:
    >>> from langcalc import Infinigram, Ollama
    >>> wiki = Infinigram(wikipedia_corpus)
    >>> llm = Ollama("llama2")
    >>> grounded = 0.95 * llm + 0.05 * wiki
    >>> text = grounded.sample(context, max_tokens=50)
"""

__version__ = "0.4.0"
__author__ = "LangCalc Contributors"

# Core models
from langcalc.models.base import LanguageModel

# Algebraic operations (import key classes from algebra module)
try:
    from langcalc.algebra import (
        AlgebraicModel,
        SumModel,
        ScaledModel,
        MaxModel,
        MinModel,
        SymmetricDifferenceModel,
        TransformedModel,
        NormalizedModel,
        PowerModel,
        ComposedModel,
        InvertedModel,
    )
except ImportError:
    # Algebra module not yet migrated
    AlgebraicModel = None

# Model implementations
try:
    from langcalc.models.ngram import NGramModel
    from langcalc.models.llm import HuggingFaceModel, MockLLM
    from langcalc.models.mixture import MixtureModel, InterpolatedModel
    from langcalc.models.base import WeightedModel
except ImportError:
    # Models not yet migrated
    NGramModel = None
    HuggingFaceModel = None
    MockLLM = None
    MixtureModel = None

# Projections
try:
    from langcalc.projections.recency import RecencyProjection
    from langcalc.projections.semantic import SemanticProjection
    from langcalc.projections.edit_distance import EditDistanceProjection
    from langcalc.projections.base import Projection
except ImportError:
    # Projections not yet migrated
    RecencyProjection = None
    SemanticProjection = None
    EditDistanceProjection = None
    Projection = None

# Data structures
try:
    from langcalc.data.suffix_array import SuffixArray
except ImportError:
    # Data structures not yet migrated
    SuffixArray = None

# Public API
__all__ = [
    # Version
    "__version__",

    # Core
    "LanguageModel",

    # Algebra
    "AlgebraicModel",
    "SumModel",
    "ScaledModel",
    "MaxModel",
    "MinModel",
    "TransformedModel",

    # Models
    "NGramModel",
    "HuggingFaceModel",
    "MockLLM",
    "MixtureModel",
    "InterpolatedModel",
    "WeightedModel",

    # Projections
    "RecencyProjection",
    "SemanticProjection",
    "EditDistanceProjection",
    "Projection",

    # Data
    "SuffixArray",
]


def get_version():
    """Return the version string."""
    return __version__


def list_models():
    """List all available model classes."""
    models = []
    if NGramModel:
        models.append("NGramModel")
    if HuggingFaceModel:
        models.append("HuggingFaceModel")
    if MockLLM:
        models.append("MockLLM")
    if MixtureModel:
        models.append("MixtureModel")
    return models


# Convenience imports for common use cases
try:
    from langcalc.grounding import (
        MixtureModel as GroundingMixture,
        WeightedModel as GroundingWeighted,
    )
    __all__.extend(["GroundingMixture", "GroundingWeighted"])
except ImportError:
    pass
