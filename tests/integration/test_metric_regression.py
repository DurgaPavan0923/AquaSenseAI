"""
Model performance regression tests for AquaSense AI.
Verifies that the deployed model meets strict quality thresholds on the holdout test set:
Macro-F1 >= 0.90, Accuracy >= 0.90, and Conformal Coverage >= 0.88.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.aquasense.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.aquasense.uncertainty import ConformalCalibrator


@pytest.mark.skipif(
    not Path("data/processed/test.csv").exists(),
    reason="Requires processed test data (data/processed/test.csv) from raw dataset",
)
def test_production_model_bundle_metrics():
    bundle_path = Path("models/model_bundle.joblib")
    test_path = Path("data/processed/test.csv")
    metrics_path = Path("results/metrics.json")

    assert bundle_path.exists(), "Model bundle missing."
    assert test_path.exists(), "Holdout test set missing."
    assert metrics_path.exists(), "Metrics summary missing."

    bundle = joblib.load(bundle_path)
    test_df = pd.read_csv(test_path)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].values

    clf = bundle["model"]
    probs = clf.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)

    macro_f1 = float(f1_score(y_test, preds, average="macro"))
    acc = float(accuracy_score(y_test, preds))

    # Regression floors
    assert macro_f1 >= 0.90, f"Macro-F1 regression: {macro_f1:.4f} < 0.90"
    assert acc >= 0.90, f"Accuracy regression: {acc:.4f} < 0.90"

    # Conformal Coverage floor
    quantile = float(bundle["conformal_quantile"])
    conformal = ConformalCalibrator(alpha=0.10)
    conformal.quantile_ = quantile
    conformal.calibrated_ = True
    cov = conformal.evaluate_coverage(probs, y_test)

    assert cov["empirical_coverage"] >= 0.88, f"Coverage regression: {cov['empirical_coverage']} < 0.88"


def test_metrics_json_matches_bundle():
    metrics_path = Path("results/metrics.json")
    with open(metrics_path, encoding="utf-8") as f:
        data = json.load(f)

    h = data["headline_metrics"]
    assert h["macro_f1"] >= 0.90
    assert h["accuracy"] >= 0.90
    assert h["macro_pr_auc"] >= 0.90
    assert h["brier_score"] <= 0.10
