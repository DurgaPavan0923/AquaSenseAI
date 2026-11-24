"""Unit tests for Inductive Conformal Prediction (LAC) uncertainty bounds."""

import numpy as np

from src.aquasense.uncertainty import ConformalCalibrator, assess_prediction_confidence


def test_conformal_calibration_and_coverage():
    # 50 synthetic calibration probabilities
    np.random.seed(42)
    y_calib = np.random.choice([0, 1, 2], size=100)
    probs_calib = np.zeros((100, 3))
    for i in range(100):
        c = y_calib[i]
        probs_calib[i, c] = 0.85
        remaining = 0.15 / 2
        for other in range(3):
            if other != c:
                probs_calib[i, other] = remaining

    calibrator = ConformalCalibrator(alpha=0.10)
    calibrator.calibrate(probs_calib, y_calib)

    assert calibrator.calibrated_ is True
    assert 0.0 < calibrator.quantile_ <= 1.0

    eval_res = calibrator.evaluate_coverage(probs_calib, y_calib)
    assert eval_res["empirical_coverage"] >= 0.88
    assert eval_res["average_set_size"] >= 1.0


def test_conformal_prediction_set_generation():
    calibrator = ConformalCalibrator(alpha=0.10)
    calibrator.quantile_ = 0.15
    calibrator.calibrated_ = True

    # High confidence: P(0) = 0.95 >= 1 - 0.15 (0.85)
    c_set = calibrator.predict_set(np.array([0.95, 0.03, 0.02]))
    assert c_set == [0]

    # Ambiguous: P(1)=0.50, P(2)=0.48. Both >= 1 - 0.60
    calibrator.quantile_ = 0.60
    amb_set = calibrator.predict_set(np.array([0.02, 0.50, 0.48]))
    assert 1 in amb_set and 2 in amb_set


def test_assess_prediction_confidence():
    # Low confidence / ambiguous
    needs_lab, msg = assess_prediction_confidence(np.array([0.02, 0.50, 0.48]), [1, 2])
    assert needs_lab is True
    assert "Needs laboratory test" in msg

    # High confidence
    needs_lab_clear, msg_clear = assess_prediction_confidence(np.array([0.95, 0.03, 0.02]), [0])
    assert needs_lab_clear is False
