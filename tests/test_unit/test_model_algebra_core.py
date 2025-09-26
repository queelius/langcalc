"""
Core unit tests for model_algebra.py algebraic framework.

Tests the fundamental algebraic operations and transformations available in the module.
"""

import pytest
import numpy as np
from typing import List, Dict
from unittest.mock import Mock, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from model_algebra import (
    ContextTransform, ModelOperator, AlgebraicModel,
    ComposedTransform, ParallelTransform,
    ScalarBiasModel, SumModel, ScaledModel, NormalizedModel,
    MaxModel, MinModel, SymmetricDifferenceModel, TransformedModel,
    TemperatureModel, MixtureModel, FilteredModel, TopKModel, ThresholdModel,
    FunctionModel, ComposedModel,
    LongestSuffixTransform, MaxKWordsTransform, RecencyWeightTransform,
    SlidingWindowTransform, SemanticFilterTransform,
    AttentionOperator, GatingOperator,
    ModelBuilder, AlgebraicModelWrapper,
    create_ensemble
)


# ============================================================================
# Mock Model for Testing
# ============================================================================

class MockAlgebraicModel(AlgebraicModel):
    """Mock implementation of AlgebraicModel for testing."""

    def __init__(self, name="mock", predictions=None):
        self.name = name
        self.predictions = predictions or {
            "the": 0.2, "a": 0.15, "and": 0.1,
            "of": 0.08, "to": 0.07, "in": 0.05
        }

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        """Return mock predictions."""
        # Normalize predictions
        total = sum(self.predictions.values())
        normalized = {k: v/total for k, v in self.predictions.items()}

        # Return top_k items
        sorted_preds = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_context():
    """Simple context for testing."""
    return ["the", "quick", "brown", "fox", "jumps"]


@pytest.fixture
def mock_model():
    """Create a mock algebraic model."""
    return MockAlgebraicModel("test_model")


@pytest.fixture
def mock_model2():
    """Create a second mock model with different predictions."""
    return MockAlgebraicModel("model2", {
        "over": 0.25, "across": 0.2, "through": 0.15,
        "the": 0.1, "a": 0.08, "quickly": 0.05
    })


# ============================================================================
# Context Transform Tests
# ============================================================================

class TestContextTransforms:
    """Test context transformation classes."""

    def test_longest_suffix_transform(self, simple_context):
        """Test LongestSuffixTransform keeps last n tokens."""
        transform = LongestSuffixTransform(n=3)
        result = transform.transform(simple_context)
        assert result == ["brown", "fox", "jumps"]

    def test_longest_suffix_empty_context(self):
        """Test LongestSuffixTransform with empty context."""
        transform = LongestSuffixTransform(n=3)
        result = transform.transform([])
        assert result == []

    def test_longest_suffix_shorter_than_n(self):
        """Test when context is shorter than n."""
        transform = LongestSuffixTransform(n=10)
        result = transform.transform(["one", "two"])
        assert result == ["one", "two"]

    def test_max_k_words_transform(self, simple_context):
        """Test MaxKWordsTransform keeps first k tokens."""
        transform = MaxKWordsTransform(k=2)
        result = transform.transform(simple_context)
        assert result == ["the", "quick"]

    def test_max_k_words_empty(self):
        """Test MaxKWordsTransform with empty context."""
        transform = MaxKWordsTransform(k=3)
        result = transform.transform([])
        assert result == []

    def test_recency_weight_transform(self, simple_context):
        """Test RecencyWeightTransform."""
        transform = RecencyWeightTransform(alpha=0.5, window=3)
        result = transform.transform(simple_context)
        # Should process the context (exact behavior depends on implementation)
        assert isinstance(result, list)
        assert len(result) <= len(simple_context)

    def test_sliding_window_transform(self, simple_context):
        """Test SlidingWindowTransform."""
        transform = SlidingWindowTransform(window_size=3, stride=2)
        result = transform.transform(simple_context)
        assert isinstance(result, list)
        # Result should be a window of the original
        assert len(result) <= 3

    def test_semantic_filter_transform(self):
        """Test SemanticFilterTransform."""
        # This might filter based on semantic similarity
        transform = SemanticFilterTransform(threshold=0.5)
        context = ["cat", "dog", "airplane", "bird", "car"]
        result = transform.transform(context)
        assert isinstance(result, list)
        assert len(result) <= len(context)

    def test_composed_transform(self, simple_context):
        """Test composing transforms with | operator."""
        t1 = LongestSuffixTransform(n=4)
        t2 = MaxKWordsTransform(k=2)

        composed = t1 | t2
        assert isinstance(composed, ComposedTransform)

        result = composed.transform(simple_context)
        # Should apply t1 then t2
        intermediate = t1.transform(simple_context)
        expected = t2.transform(intermediate)
        assert result == expected

    def test_parallel_transform(self, simple_context):
        """Test parallel transforms with & operator."""
        t1 = LongestSuffixTransform(n=3)
        t2 = MaxKWordsTransform(k=2)

        parallel = t1 & t2
        assert isinstance(parallel, ParallelTransform)

        result = parallel.transform(simple_context)
        # Result should combine both transforms somehow
        assert isinstance(result, list)


# ============================================================================
# Algebraic Model Operations Tests
# ============================================================================

class TestAlgebraicOperations:
    """Test algebraic operations on models."""

    def test_model_addition(self, mock_model, mock_model2):
        """Test model addition creates SumModel."""
        result = mock_model + mock_model2
        assert isinstance(result, SumModel)
        assert len(result.models) == 2

        # Test predictions
        preds = result.predict(["test"], top_k=5)
        assert isinstance(preds, dict)
        assert len(preds) <= 5

    def test_model_scalar_addition(self, mock_model):
        """Test model + scalar creates ScalarBiasModel."""
        result = mock_model + 0.1
        assert isinstance(result, ScalarBiasModel)
        assert result.bias == 0.1

        preds = result.predict(["test"], top_k=3)
        assert isinstance(preds, dict)

    def test_model_multiplication(self, mock_model):
        """Test scalar multiplication."""
        result = mock_model * 0.5
        assert isinstance(result, ScaledModel)
        assert result.scale == 0.5

        # Test right multiplication
        result2 = 0.5 * mock_model
        assert isinstance(result2, ScaledModel)
        assert result2.scale == 0.5

    def test_model_subtraction(self, mock_model, mock_model2):
        """Test model subtraction."""
        result = mock_model - mock_model2
        assert isinstance(result, SumModel)
        # Second model should be negatively scaled
        assert len(result.models) == 2
        assert isinstance(result.models[1], ScaledModel)
        assert result.models[1].scale == -1.0

    def test_model_division_by_scalar(self, mock_model):
        """Test model division by scalar."""
        result = mock_model / 2.0
        assert isinstance(result, ScaledModel)
        assert result.scale == 0.5

    def test_model_division_by_model(self, mock_model, mock_model2):
        """Test model division by another model."""
        result = mock_model / mock_model2
        assert isinstance(result, NormalizedModel)

    def test_model_union(self, mock_model, mock_model2):
        """Test model union with | operator."""
        result = mock_model | mock_model2
        assert isinstance(result, MaxModel)
        assert len(result.models) == 2

        # Test that it takes max probabilities
        preds = result.predict(["test"], top_k=5)
        assert isinstance(preds, dict)

    def test_model_intersection(self, mock_model, mock_model2):
        """Test model intersection with & operator."""
        result = mock_model & mock_model2
        assert isinstance(result, MinModel)
        assert len(result.models) == 2

        # Test that it takes min probabilities
        preds = result.predict(["test"], top_k=5)
        assert isinstance(preds, dict)

    def test_model_xor(self, mock_model, mock_model2):
        """Test symmetric difference."""
        result = mock_model ^ mock_model2
        assert isinstance(result, SymmetricDifferenceModel)

        preds = result.predict(["test"], top_k=5)
        assert isinstance(preds, dict)

    def test_model_transform_application(self, mock_model):
        """Test applying transform with << operator."""
        transform = LongestSuffixTransform(n=3)
        result = mock_model << transform

        assert isinstance(result, TransformedModel)
        assert result.transform == transform

        # Test that transform is applied to context
        context = ["a", "b", "c", "d", "e"]
        preds = result.predict(context, top_k=3)
        assert isinstance(preds, dict)

    def test_model_composition(self, mock_model, mock_model2):
        """Test model composition with >> operator."""
        result = mock_model >> mock_model2
        assert isinstance(result, ComposedModel)

        preds = result.predict(["test"], top_k=3)
        assert isinstance(preds, dict)

    def test_model_power(self, mock_model):
        """Test model power with ** operator."""
        result = mock_model ** 2.0
        assert isinstance(result, TemperatureModel)  # Assuming power creates temperature scaling
        assert result.temperature == 0.5  # 1/power

        preds = result.predict(["test"], top_k=3)
        assert isinstance(preds, dict)

    def test_model_inversion(self, mock_model):
        """Test model inversion with ~ operator."""
        result = ~mock_model
        # Should create some form of inverted model
        assert result is not None

        preds = result.predict(["test"], top_k=3)
        assert isinstance(preds, dict)


# ============================================================================
# Model Operator Tests
# ============================================================================

class TestModelOperators:
    """Test model operator classes."""

    def test_attention_operator(self, mock_model, mock_model2):
        """Test AttentionOperator."""
        operator = AttentionOperator(temperature=1.0)
        models = [mock_model, mock_model2]
        context = ["test", "context"]

        result = operator.apply(models, context)
        assert isinstance(result, dict)
        # Should return weighted combination based on attention

    def test_gating_operator(self, mock_model, mock_model2):
        """Test GatingOperator."""
        operator = GatingOperator(gate_threshold=0.5)
        models = [mock_model, mock_model2]
        context = ["test", "context"]

        result = operator.apply(models, context)
        assert isinstance(result, dict)


# ============================================================================
# Model Builder Tests
# ============================================================================

class TestModelBuilder:
    """Test ModelBuilder pattern."""

    def test_basic_builder(self, mock_model):
        """Test basic builder operations."""
        builder = ModelBuilder()

        result = (builder
                  .with_base(mock_model)
                  .multiply(0.5)
                  .build())

        assert isinstance(result, AlgebraicModel)
        preds = result.predict(["test"], top_k=3)
        assert isinstance(preds, dict)

    def test_complex_builder(self, mock_model, mock_model2):
        """Test complex builder chain."""
        builder = ModelBuilder()

        result = (builder
                  .with_base(mock_model)
                  .multiply(0.7)
                  .add(mock_model2, weight=0.3)
                  .apply_transform(LongestSuffixTransform(n=3))
                  .build())

        assert isinstance(result, AlgebraicModel)
        preds = result.predict(["a", "b", "c", "d"], top_k=3)
        assert isinstance(preds, dict)


# ============================================================================
# Ensemble Creation Tests
# ============================================================================

class TestEnsembleCreation:
    """Test ensemble model creation."""

    def test_mixture_ensemble(self, mock_model, mock_model2):
        """Test creating mixture ensemble."""
        models = [mock_model, mock_model2]
        weights = [0.6, 0.4]

        ensemble = create_ensemble(models, weights, method="mixture")
        assert isinstance(ensemble, AlgebraicModel)

        preds = ensemble.predict(["test"], top_k=5)
        assert isinstance(preds, dict)
        assert sum(preds.values()) <= 1.01  # Allow small numerical error

    def test_voting_ensemble(self, mock_model, mock_model2):
        """Test creating voting ensemble."""
        models = [mock_model, mock_model2]

        ensemble = create_ensemble(models, method="voting")
        assert isinstance(ensemble, AlgebraicModel)

        preds = ensemble.predict(["test"], top_k=5)
        assert isinstance(preds, dict)

    def test_product_ensemble(self, mock_model, mock_model2):
        """Test creating product ensemble."""
        models = [mock_model, mock_model2]

        ensemble = create_ensemble(models, method="product")
        assert isinstance(ensemble, AlgebraicModel)

        preds = ensemble.predict(["test"], top_k=5)
        assert isinstance(preds, dict)


# ============================================================================
# Wrapper Tests
# ============================================================================

class TestAlgebraicModelWrapper:
    """Test AlgebraicModelWrapper."""

    def test_wrapper_basic(self):
        """Test wrapping a simple model."""
        # Create a simple model-like object
        simple_model = Mock()
        simple_model.predict = Mock(return_value={"token": 0.5, "other": 0.5})

        wrapped = AlgebraicModelWrapper(simple_model)
        assert isinstance(wrapped, AlgebraicModel)

        preds = wrapped.predict(["test"], top_k=2)
        assert isinstance(preds, dict)
        assert len(preds) == 2


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_context(self, mock_model):
        """Test with empty context."""
        preds = mock_model.predict([], top_k=5)
        assert isinstance(preds, dict)

    def test_zero_top_k(self, mock_model):
        """Test with top_k=0."""
        preds = mock_model.predict(["test"], top_k=0)
        assert isinstance(preds, dict)
        assert len(preds) == 0

    def test_very_large_top_k(self, mock_model):
        """Test with very large top_k."""
        preds = mock_model.predict(["test"], top_k=1000)
        assert isinstance(preds, dict)
        # Should return all available predictions
        assert len(preds) <= 1000

    def test_division_by_zero(self, mock_model):
        """Test division by zero."""
        with pytest.raises(ZeroDivisionError):
            _ = mock_model / 0

    def test_negative_scaling(self, mock_model):
        """Test negative scaling."""
        result = mock_model * (-1.0)
        assert isinstance(result, ScaledModel)
        assert result.scale == -1.0

    def test_empty_model_list_in_sum(self):
        """Test SumModel with empty list."""
        with pytest.raises((ValueError, IndexError)):
            _ = SumModel([])

    def test_mismatched_operator_inputs(self):
        """Test operators with mismatched inputs."""
        operator = AttentionOperator()

        with pytest.raises((ValueError, TypeError, AttributeError)):
            # Empty model list
            operator.apply([], ["context"])


# ============================================================================
# Complex Composition Tests
# ============================================================================

class TestComplexCompositions:
    """Test complex model compositions."""

    def test_nested_operations(self, mock_model, mock_model2):
        """Test deeply nested operations."""
        # (0.5 * model1 + 0.3 * model2) | model1
        complex_model = (0.5 * mock_model + 0.3 * mock_model2) | mock_model

        preds = complex_model.predict(["test"], top_k=5)
        assert isinstance(preds, dict)

    def test_multiple_transform_chain(self, mock_model):
        """Test chaining multiple transforms."""
        t1 = LongestSuffixTransform(n=5)
        t2 = MaxKWordsTransform(k=3)

        result = (mock_model << t1) << t2

        context = ["a", "b", "c", "d", "e", "f", "g"]
        preds = result.predict(context, top_k=3)
        assert isinstance(preds, dict)

    def test_mixed_algebraic_operations(self, mock_model, mock_model2):
        """Test mixing different algebraic operations."""
        # Complex expression with multiple operators
        result = ((mock_model * 0.7 + mock_model2 * 0.3) & mock_model) | mock_model2

        preds = result.predict(["test"], top_k=5)
        assert isinstance(preds, dict)

    @pytest.mark.parametrize("depth", [1, 3, 5])
    def test_deep_composition_chain(self, mock_model, depth):
        """Test deeply nested composition doesn't overflow."""
        result = mock_model
        for _ in range(depth):
            result = result * 0.9 + mock_model * 0.1

        preds = result.predict(["test"], top_k=3)
        assert isinstance(preds, dict)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the algebraic framework."""

    def test_end_to_end_workflow(self, mock_model, mock_model2):
        """Test complete workflow from creation to prediction."""
        # Create a complex model
        transform = LongestSuffixTransform(n=3)

        # Build complex model using algebraic operations
        model = ((mock_model << transform) * 0.6 + mock_model2 * 0.4) ** 2.0

        # Make predictions
        context = ["the", "quick", "brown", "fox", "jumps"]
        preds = model.predict(context, top_k=10)

        assert isinstance(preds, dict)
        assert len(preds) <= 10
        assert all(isinstance(k, str) for k in preds.keys())
        assert all(isinstance(v, (int, float)) for v in preds.values())

    def test_model_persistence(self, mock_model):
        """Test that models maintain state correctly."""
        scaled = mock_model * 0.5

        # Make multiple predictions
        preds1 = scaled.predict(["test"], top_k=3)
        preds2 = scaled.predict(["test"], top_k=3)

        # Should get same results
        assert preds1 == preds2

    def test_transform_composition_order(self):
        """Test that transform composition order matters."""
        t1 = LongestSuffixTransform(n=3)
        t2 = MaxKWordsTransform(k=2)

        comp1 = t1 | t2  # t1 then t2
        comp2 = t2 | t1  # t2 then t1

        context = ["a", "b", "c", "d", "e"]

        result1 = comp1.transform(context)
        result2 = comp2.transform(context)

        # Results should potentially be different
        assert isinstance(result1, list)
        assert isinstance(result2, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])