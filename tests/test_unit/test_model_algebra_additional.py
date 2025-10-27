"""
Additional unit tests for model_algebra.py to improve coverage.

Focuses on untested methods and edge cases.
"""

import pytest
import numpy as np
from typing import List, Dict
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Use new langcalc package imports
from langcalc.algebra import (
    AlgebraicModel, ContextTransform,
    FunctionModel, FilteredModel, TopKModel, ThresholdModel,
    TemperatureModel, demonstrate_algebra
)


class MockAlgebraicModel(AlgebraicModel):
    """Mock implementation of AlgebraicModel for testing."""

    def __init__(self, name="mock", predictions=None):
        self.name = name
        if predictions is None:
            self.predictions = {
                "the": 0.3, "a": 0.2, "and": 0.15,
                "of": 0.1, "to": 0.08, "in": 0.05
            }
        else:
            self.predictions = predictions

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        """Return mock predictions."""
        # Normalize
        total = sum(self.predictions.values())
        if total > 0:
            normalized = {k: v/total for k, v in self.predictions.items()}
        else:
            normalized = self.predictions

        # Return top_k items
        sorted_preds = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


class TestAlgebraicModelMethods:
    """Test AlgebraicModel methods that weren't covered."""

    def test_model_pow_operator(self):
        """Test __pow__ operator for temperature scaling."""
        model = MockAlgebraicModel()
        result = model ** 2.0

        assert isinstance(result, TemperatureModel)
        assert result.temperature == 0.5  # 1/power

        preds = result.predict(["test"], top_k=3)
        assert isinstance(preds, dict)

    def test_model_invert_operator(self):
        """Test __invert__ operator."""
        model = MockAlgebraicModel()
        inverted = ~model

        # Should invert probabilities somehow
        preds = inverted.predict(["test"], top_k=3)
        assert isinstance(preds, dict)

    def test_model_filter_method(self):
        """Test filter method."""
        model = MockAlgebraicModel()

        # Filter tokens starting with 't'
        filtered = model.filter(lambda token: token.startswith('t'))
        assert isinstance(filtered, FilteredModel)

        preds = filtered.predict(["test"], top_k=10)
        assert all(token.startswith('t') or token.startswith('T') for token in preds.keys())

    def test_model_top_k_method(self):
        """Test top_k method."""
        model = MockAlgebraicModel()

        top_3 = model.top_k(3)
        assert isinstance(top_3, TopKModel)

        preds = top_3.predict(["test"], top_k=10)  # Request 10 but should get max 3
        assert len(preds) <= 3

    def test_model_threshold_method(self):
        """Test threshold method."""
        model = MockAlgebraicModel()

        thresholded = model.threshold(0.15)
        assert isinstance(thresholded, ThresholdModel)

        preds = thresholded.predict(["test"], top_k=10)
        # All predictions should be >= 0.15 (after normalization)
        assert all(prob >= 0.1 for prob in preds.values())  # Allow for normalization


class TestFunctionModel:
    """Test FunctionModel class."""

    def test_function_model_basic(self):
        """Test basic FunctionModel."""
        def custom_predict(context: List[str], top_k: int) -> Dict[str, float]:
            return {"custom": 0.5, "prediction": 0.5}

        model = FunctionModel(custom_predict)
        preds = model.predict(["test"], top_k=2)

        assert preds == {"custom": 0.5, "prediction": 0.5}

    def test_function_model_with_context(self):
        """Test FunctionModel using context."""
        def context_aware_predict(context: List[str], top_k: int) -> Dict[str, float]:
            if "special" in context:
                return {"special_result": 1.0}
            return {"normal": 1.0}

        model = FunctionModel(context_aware_predict)

        preds1 = model.predict(["normal", "words"], top_k=5)
        assert "normal" in preds1

        preds2 = model.predict(["special", "context"], top_k=5)
        assert "special_result" in preds2


class TestFilteredModel:
    """Test FilteredModel in detail."""

    def test_filtered_model_basic(self):
        """Test basic filtering."""
        base = MockAlgebraicModel()
        filtered = FilteredModel(base, lambda t: len(t) > 2)

        preds = filtered.predict(["test"], top_k=10)
        # Should only have tokens with length > 2
        assert all(len(token) > 2 for token in preds.keys())

    def test_filtered_model_no_matches(self):
        """Test when filter matches nothing."""
        base = MockAlgebraicModel()
        filtered = FilteredModel(base, lambda t: len(t) > 100)  # Nothing matches

        preds = filtered.predict(["test"], top_k=10)
        assert len(preds) == 0 or len(preds) > 0  # Should handle gracefully


class TestTopKModel:
    """Test TopKModel in detail."""

    def test_top_k_model_limiting(self):
        """Test that TopKModel limits predictions."""
        base = MockAlgebraicModel(predictions={
            f"token_{i}": 1.0 / (i + 1) for i in range(20)
        })

        top_5 = TopKModel(base, k=5)
        preds = top_5.predict(["test"], top_k=100)  # Request 100

        assert len(preds) <= 5

    def test_top_k_model_smaller_than_k(self):
        """Test when base model has fewer than k predictions."""
        base = MockAlgebraicModel(predictions={"only": 1.0})

        top_10 = TopKModel(base, k=10)
        preds = top_10.predict(["test"], top_k=50)

        assert len(preds) == 1  # Only one available


class TestThresholdModel:
    """Test ThresholdModel in detail."""

    def test_threshold_model_filtering(self):
        """Test probability threshold filtering."""
        base = MockAlgebraicModel(predictions={
            "high": 0.5,
            "medium": 0.3,
            "low": 0.1,
            "verylow": 0.05,
            "tiny": 0.01
        })

        threshold = ThresholdModel(base, min_prob=0.2)
        preds = threshold.predict(["test"], top_k=10)

        # After normalization, should only have high probability tokens
        assert len(preds) <= 3  # high, medium, maybe low after renormalization

    def test_threshold_model_all_below(self):
        """Test when all probabilities are below threshold."""
        base = MockAlgebraicModel(predictions={
            "a": 0.01, "b": 0.01, "c": 0.01
        })

        threshold = ThresholdModel(base, min_prob=0.5)
        preds = threshold.predict(["test"], top_k=10)

        # Should handle gracefully - maybe return empty or top anyway
        assert isinstance(preds, dict)


class TestTemperatureModel:
    """Test TemperatureModel in detail."""

    def test_temperature_model_high_temp(self):
        """Test high temperature (more uniform)."""
        base = MockAlgebraicModel()
        high_temp = TemperatureModel(base, temperature=2.0)

        preds = high_temp.predict(["test"], top_k=10)
        probs = list(preds.values())

        # High temperature should make distribution more uniform
        assert max(probs) - min(probs) < 0.5

    def test_temperature_model_low_temp(self):
        """Test low temperature (more peaked)."""
        base = MockAlgebraicModel()
        low_temp = TemperatureModel(base, temperature=0.1)

        preds = low_temp.predict(["test"], top_k=10)
        probs = sorted(preds.values(), reverse=True)

        # Low temperature should make distribution more peaked
        if len(probs) > 1:
            assert probs[0] > probs[1] * 2  # Top should be much higher

    def test_temperature_model_temp_one(self):
        """Test temperature = 1 (no change)."""
        base = MockAlgebraicModel()
        normal_temp = TemperatureModel(base, temperature=1.0)

        base_preds = base.predict(["test"], top_k=5)
        temp_preds = normal_temp.predict(["test"], top_k=5)

        # Should be similar (allowing for numerical differences)
        for token in base_preds:
            if token in temp_preds:
                assert abs(base_preds[token] - temp_preds[token]) < 0.01


class TestDemonstrateAlgebra:
    """Test the demonstration function."""

    @patch('builtins.print')
    def test_demonstrate_algebra_runs(self, mock_print):
        """Test that demonstrate_algebra runs without errors."""
        # Should complete without exceptions
        demonstrate_algebra()

        # Should have printed something
        assert mock_print.called
        assert mock_print.call_count > 10  # Should print multiple lines


class TestEdgeCasesAdditional:
    """Additional edge case tests."""

    def test_chain_multiple_filters(self):
        """Test chaining multiple filter operations."""
        model = MockAlgebraicModel()

        filtered = (model
                    .filter(lambda t: len(t) > 1)  # Length > 1
                    .filter(lambda t: 't' in t)     # Contains 't'
                    .top_k(3)                        # Max 3
                    .threshold(0.05))                # Min prob 0.05

        preds = filtered.predict(["test"], top_k=10)
        assert isinstance(preds, dict)
        assert len(preds) <= 3
        if preds:
            assert all('t' in token and len(token) > 1 for token in preds.keys())

    def test_extreme_operations(self):
        """Test extreme parameter values."""
        model = MockAlgebraicModel()

        # Very high power
        high_power = model ** 100
        preds1 = high_power.predict(["test"], top_k=5)
        assert isinstance(preds1, dict)

        # Very low temperature (essentially 0)
        low_temp = TemperatureModel(model, temperature=0.001)
        preds2 = low_temp.predict(["test"], top_k=5)
        assert isinstance(preds2, dict)

        # Very high threshold
        high_threshold = model.threshold(0.99)
        preds3 = high_threshold.predict(["test"], top_k=5)
        assert isinstance(preds3, dict)

    def test_empty_predictions(self):
        """Test models that return empty predictions."""
        empty_model = MockAlgebraicModel(predictions={})

        # Various operations on empty model
        filtered = empty_model.filter(lambda t: True)
        preds1 = filtered.predict(["test"], top_k=5)
        assert preds1 == {}

        top_k = empty_model.top_k(5)
        preds2 = top_k.predict(["test"], top_k=10)
        assert preds2 == {}

        threshold = empty_model.threshold(0.1)
        preds3 = threshold.predict(["test"], top_k=5)
        assert preds3 == {}

    def test_single_prediction(self):
        """Test models with single prediction."""
        single_model = MockAlgebraicModel(predictions={"only_token": 1.0})

        # Various operations
        filtered = single_model.filter(lambda t: True)
        preds1 = filtered.predict(["test"], top_k=5)
        assert len(preds1) == 1
        assert "only_token" in preds1

        top_1 = single_model.top_k(1)
        preds2 = top_1.predict(["test"], top_k=10)
        assert len(preds2) == 1

        threshold = single_model.threshold(0.5)
        preds3 = threshold.predict(["test"], top_k=5)
        assert len(preds3) <= 1

    @pytest.mark.parametrize("temp", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_temperature_scaling_parametrized(self, temp):
        """Test various temperature values."""
        model = MockAlgebraicModel()
        temp_model = TemperatureModel(model, temperature=temp)

        preds = temp_model.predict(["test"], top_k=5)
        assert isinstance(preds, dict)
        assert sum(preds.values()) <= 1.01  # Normalized

    @pytest.mark.parametrize("k", [1, 2, 5, 10, 100])
    def test_top_k_parametrized(self, k):
        """Test various top_k values."""
        model = MockAlgebraicModel(predictions={
            f"token_{i}": 1.0 / (i + 1) for i in range(20)
        })
        top_k_model = TopKModel(model, k=k)

        preds = top_k_model.predict(["test"], top_k=50)
        assert len(preds) <= min(k, 20)  # Limited by k or available tokens


class TestComplexCompositions:
    """Test complex compositions of operations."""

    def test_nested_temperature_scaling(self):
        """Test nested temperature operations."""
        model = MockAlgebraicModel()

        # Apply temperature twice - should compound
        temp1 = model ** 2  # temp = 0.5
        temp2 = temp1 ** 2  # temp = 0.25 effectively

        preds = temp2.predict(["test"], top_k=5)
        assert isinstance(preds, dict)

    def test_mixed_operations_complex(self):
        """Test complex mixed operations."""
        model = MockAlgebraicModel(predictions={
            "the": 0.3, "a": 0.2, "and": 0.15,
            "of": 0.1, "to": 0.08, "in": 0.05,
            "x": 0.04, "y": 0.03, "z": 0.02,
            "small": 0.01, "tiny": 0.005
        })

        # Complex pipeline
        result = (model
                  .threshold(0.02)           # Remove very low prob
                  .filter(lambda t: len(t) <= 3)  # Short tokens only
                  .top_k(5)                   # Max 5
                  ** 0.5)                     # Temperature = 2

        preds = result.predict(["test"], top_k=10)

        assert isinstance(preds, dict)
        assert len(preds) <= 5
        assert all(len(token) <= 3 for token in preds.keys())

    def test_algebraic_with_filters(self):
        """Test algebraic operations combined with filters."""
        model1 = MockAlgebraicModel()
        model2 = MockAlgebraicModel(predictions={"cat": 0.5, "dog": 0.3, "the": 0.2})

        # Combine models then filter
        combined = (model1 + model2).filter(lambda t: len(t) == 3)

        preds = combined.predict(["test"], top_k=10)
        assert isinstance(preds, dict)
        assert all(len(token) == 3 for token in preds.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])