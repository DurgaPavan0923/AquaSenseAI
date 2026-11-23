"""Unit tests for Inductive Conformal Prediction (ICP) uncertainty bounds."""

import numpy as np
import pytest

from src.ml.conformal import ConformalPredictor


@pytest.fixture
def mock_conformal_predictor():
    """Initializes conformal predictor with 100 synthetic calibration non-conformity scores."""
    # Calibration scores distributed between 0.10 and 0.70
    scores = np.linspace(0.10, 0.70, 100)
    return ConformalPredictor(calibration_scores=scores, classes=["Good", "Moderate", "Polluted"])


def test_conformal_quantile_calculation(mock_conformal_predictor):
    """Verifies that conformal quantile threshold strictly bounds confidence level."""
    # At alpha = 0.10 (90% confidence), threshold should be around the 90th percentile
    q_90 = mock_conformal_predictor.get_quantile(alpha=0.10)
    assert 0.60 <= q_90 <= 0.70

    # At alpha = 0.05 (95% confidence), threshold must be greater than or equal to 90%
    q_95 = mock_conformal_predictor.get_quantile(alpha=0.05)
    assert q_95 >= q_90


def test_conformal_prediction_set_generation(mock_conformal_predictor):
    """Verifies single sample prediction set construction under varying confidence."""
    # Clear high-confidence prediction
    probs_clear = np.array([0.92, 0.06, 0.02])
    _, set_clear = mock_conformal_predictor.predict_set(probs_clear, alpha=0.10)
    assert set_clear == ["Good"]

    # Borderline ambiguous sample
    probs_ambiguous = np.array([0.05, 0.48, 0.47])
    _, set_ambiguous = mock_conformal_predictor.predict_set(probs_ambiguous, alpha=0.10)
    # Both Moderate and Polluted should be included
    assert "Moderate" in set_ambiguous
    assert "Polluted" in set_ambiguous


def test_conformal_guarantees_non_empty_set(mock_conformal_predictor):
    """Verifies that an extreme uniform outlier still yields a non-empty set (argmax fallback)."""
    probs_flat = np.array([0.34, 0.33, 0.33])
    # Forcing a very high threshold
    _, pred_set = mock_conformal_predictor.predict_set(probs_flat, alpha=0.50)
    assert len(pred_set) >= 1
    assert "Good" in pred_set  # Argmax index 0


def test_coverage_evaluation(mock_conformal_predictor):
    """Verifies batch coverage calculation on test probabilities."""
    test_probs = np.array([
        [0.90, 0.08, 0.02],
        [0.10, 0.85, 0.05],
        [0.02, 0.10, 0.88],
        [0.05, 0.50, 0.45]
    ])
    y_true = np.array([0, 1, 2, 2])

    res = mock_conformal_predictor.evaluate_coverage(test_probs, y_true, alpha=0.10)
    assert res["empirical_coverage"] >= 0.75
    assert res["mean_prediction_set_size"] >= 1.0
