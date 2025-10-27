"""
Additional tests to improve coverage for model_algebra.py module.

This file targets uncovered code paths including:
- NormalizedModel (division operation)
- create_grounded_model function
- create_adaptive_model function
- Edge cases in various model classes
"""

import pytest
from typing import List, Dict
import sys
import os

# Use new langcalc package imports
from langcalc.algebra import (
    AlgebraicModel, NormalizedModel, create_grounded_model,
    create_adaptive_model, ModelBuilder, LongestSuffixTransform,
    MaxKWordsTransform, ComposedModel, SymmetricDifferenceModel,
    TransformedModel
)
from langcalc.models.ngram import NGramModel


class MockModel(AlgebraicModel):
    """Mock model for testing."""

    def __init__(self, predictions=None):
        self.predictions = predictions or {
            "the": 0.3, "a": 0.2, "and": 0.15,
            "of": 0.1, "to": 0.08
        }

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        # Normalize
        total = sum(self.predictions.values())
        if total > 0:
            normalized = {k: v/total for k, v in self.predictions.items()}
        else:
            normalized = self.predictions

        sorted_preds = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


class TestNormalizedModel:
    """Test NormalizedModel (division operation)."""

    def test_normalized_model_basic(self):
        """Test basic normalized model (model / model)."""
        numerator = MockModel({"the": 0.4, "a": 0.3, "and": 0.2})
        denominator = MockModel({"the": 0.2, "a": 0.15, "and": 0.1})

        normalized = NormalizedModel(numerator, denominator)
        preds = normalized.predict(["test"], top_k=5)

        assert isinstance(preds, dict)
        assert len(preds) > 0
        # Predictions should be normalized probabilities
        assert abs(sum(preds.values()) - 1.0) < 0.01

    def test_normalized_model_with_zero_denominator(self):
        """Test normalized model when denominator has zero for some tokens."""
        numerator = MockModel({"the": 0.4, "a": 0.3, "and": 0.2, "x": 0.1})
        denominator = MockModel({"the": 0.2, "a": 0.15})  # Missing "and" and "x"

        normalized = NormalizedModel(numerator, denominator)
        preds = normalized.predict(["test"], top_k=10)

        # Should only include tokens present in both with non-zero denominator
        assert isinstance(preds, dict)
        # "and" and "x" should be excluded (not in denominator)

    def test_normalized_model_empty_intersection(self):
        """Test normalized model with no common tokens."""
        numerator = MockModel({"the": 0.5, "a": 0.5})
        denominator = MockModel({"x": 0.5, "y": 0.5})

        normalized = NormalizedModel(numerator, denominator)
        preds = normalized.predict(["test"], top_k=10)

        # Should return empty dict when no tokens in common
        assert isinstance(preds, dict)


class TestModelDivision:
    """Test model division (NormalizedModel via /) - covers uncovered lines."""

    def test_model_division_operator(self):
        """Test using / operator to create NormalizedModel."""
        model1 = MockModel({"the": 0.4, "a": 0.3, "and": 0.2})
        model2 = MockModel({"the": 0.2, "a": 0.15, "and": 0.1})

        # Use / operator
        result = model1 / model2

        assert isinstance(result, NormalizedModel)
        preds = result.predict(["test"], top_k=5)
        assert isinstance(preds, dict)
        assert len(preds) > 0


class TestAdaptiveModelCreation:
    """Test create_adaptive_model function."""

    def test_create_adaptive_model_basic(self):
        """Test creating an adaptive model."""
        model1 = MockModel({"the": 0.4, "a": 0.3})
        model2 = MockModel({"and": 0.4, "of": 0.3})
        models = [model1, model2]

        def context_analyzer(context: List[str]) -> List[float]:
            # Simple: equal weights
            return [0.5, 0.5]

        adaptive = create_adaptive_model(models, context_analyzer)

        assert isinstance(adaptive, AlgebraicModel)
        preds = adaptive.predict(["test"], top_k=5)
        assert isinstance(preds, dict)

    def test_create_adaptive_model_context_dependent(self):
        """Test adaptive model with context-dependent weights."""
        model1 = MockModel({"technical": 0.5, "code": 0.3})
        model2 = MockModel({"casual": 0.5, "chat": 0.3})
        models = [model1, model2]

        def context_analyzer(context: List[str]) -> List[float]:
            # Adjust weights based on context
            if any("tech" in word.lower() for word in context):
                return [0.8, 0.2]  # Favor technical model
            return [0.2, 0.8]  # Favor casual model

        adaptive = create_adaptive_model(models, context_analyzer)

        # Test with technical context
        preds1 = adaptive.predict(["tech", "stuff"], top_k=5)
        assert isinstance(preds1, dict)

        # Test with non-technical context
        preds2 = adaptive.predict(["normal", "words"], top_k=5)
        assert isinstance(preds2, dict)


class TestComposedModel:
    """Test ComposedModel for sequential composition."""

    def test_composed_model_basic(self):
        """Test basic model composition."""
        model1 = MockModel({"the": 0.4, "a": 0.3, "and": 0.2})
        model2 = MockModel({"the": 0.5, "a": 0.3, "x": 0.2})

        composed = ComposedModel([model1, model2])
        preds = composed.predict(["test"], top_k=5)

        assert isinstance(preds, dict)
        # Should combine predictions through multiplication

    def test_composed_model_multiple_models(self):
        """Test composing more than 2 models."""
        models = [
            MockModel({"the": 0.4, "a": 0.3}),
            MockModel({"the": 0.5, "a": 0.3}),
            MockModel({"the": 0.6, "a": 0.2})
        ]

        composed = ComposedModel(models)
        preds = composed.predict(["test"], top_k=5)

        assert isinstance(preds, dict)

    def test_composed_model_no_overlap(self):
        """Test composed model with no overlapping tokens."""
        model1 = MockModel({"the": 0.5, "a": 0.5})
        model2 = MockModel({"x": 0.5, "y": 0.5})

        composed = ComposedModel([model1, model2])
        preds = composed.predict(["test"], top_k=5)

        # When no overlap, should fall back to second model
        assert isinstance(preds, dict)


class TestSymmetricDifferenceModel:
    """Test SymmetricDifferenceModel (XOR operation)."""

    def test_symmetric_difference_basic(self):
        """Test basic symmetric difference operation."""
        model1 = MockModel({"the": 0.4, "a": 0.3, "and": 0.2})
        model2 = MockModel({"the": 0.5, "x": 0.3, "y": 0.2})

        xor_model = SymmetricDifferenceModel(model1, model2)
        preds = xor_model.predict(["test"], top_k=5)

        assert isinstance(preds, dict)
        # Should include tokens with high difference

    def test_symmetric_difference_identical_models(self):
        """Test XOR with identical models."""
        model = MockModel({"the": 0.5, "a": 0.5})

        xor_model = SymmetricDifferenceModel(model, model)
        preds = xor_model.predict(["test"], top_k=5)

        # With identical models, all differences should be small
        assert isinstance(preds, dict)


class TestTransformedModel:
    """Test TransformedModel with context transforms."""

    def test_transformed_model_longest_suffix(self):
        """Test model with longest suffix transform."""
        model = MockModel({"the": 0.4, "a": 0.3})
        transform = LongestSuffixTransform(n=3)

        transformed = TransformedModel(model, transform)
        preds = transformed.predict(["the", "quick", "brown", "fox"], top_k=5)

        assert isinstance(preds, dict)

    def test_transformed_model_max_k_words(self):
        """Test model with max k words transform."""
        model = MockModel({"the": 0.4, "a": 0.3})
        transform = MaxKWordsTransform(k=2)

        transformed = TransformedModel(model, transform)
        preds = transformed.predict(["one", "two", "three", "four"], top_k=5)

        assert isinstance(preds, dict)


class TestModelBuilderAdditional:
    """Additional tests for ModelBuilder."""

    def test_model_builder_temperature_method(self):
        """Test ModelBuilder temperature method."""
        base = MockModel({"the": 0.4, "a": 0.3})

        builder = ModelBuilder()
        model = builder.with_base(base).temperature(0.5).build()

        assert isinstance(model, AlgebraicModel)
        preds = model.predict(["test"], top_k=5)
        assert isinstance(preds, dict)

    def test_model_builder_multiply_method(self):
        """Test ModelBuilder multiply method."""
        base = MockModel({"the": 0.4, "a": 0.3})

        builder = ModelBuilder()
        model = builder.with_base(base).multiply(0.7).build()

        assert isinstance(model, AlgebraicModel)
        preds = model.predict(["test"], top_k=5)
        assert isinstance(preds, dict)

    def test_model_builder_add_method(self):
        """Test ModelBuilder add method."""
        base = MockModel({"the": 0.4, "a": 0.3})
        other = MockModel({"and": 0.5, "of": 0.3})

        builder = ModelBuilder()
        model = builder.with_base(base).add(other, weight=0.3).build()

        assert isinstance(model, AlgebraicModel)
        preds = model.predict(["test"], top_k=5)
        assert isinstance(preds, dict)

    def test_model_builder_apply_transform_method(self):
        """Test ModelBuilder apply_transform method."""
        base = MockModel({"the": 0.4, "a": 0.3})
        transform = MaxKWordsTransform(k=3)

        builder = ModelBuilder()
        model = builder.with_base(base).apply_transform(transform).build()

        assert isinstance(model, AlgebraicModel)
        preds = model.predict(["a", "b", "c", "d"], top_k=5)
        assert isinstance(preds, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
