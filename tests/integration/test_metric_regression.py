"""Model regression testing enforcing statistical and predictive performance floors."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.common.config import yaml_config
from src.ml.conformal import ConformalPredictor


def test_active_model_performance_floors():
    """Asserts that production model candidate exceeds required statistical performance thresholds."""
    artifacts_dir = Path(yaml_config["models"]["artifacts_dir"]) / "xgboost_active"
    model_path = artifacts_dir / "model.joblib"
    calib_path = artifacts_dir / "calibration_scores.npy"
    test_path = Path(yaml_config["dataset"]["processed_test_path"])

    assert model_path.exists(), "Active model artifact missing."
    assert calib_path.exists(), "Calibration scores artifact missing."
    assert test_path.exists(), "Holdout test dataset missing."

    model = joblib.load(model_path)
    calib_scores = np.load(calib_path)
    test_df = pd.read_csv(test_path)

    numerical_cols = [f["name"] for f in yaml_config["features"]["numerical"]]
    target_col = yaml_config["target"]["name"]

    X_test = test_df[numerical_cols]
    y_test = test_df[target_col].to_numpy()

    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)

    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    acc = accuracy_score(y_test, preds)

    # 1. Performance Floor Assertions
    assert macro_f1 >= 0.90, f"Model Macro-F1 regression: {macro_f1:.4f} < threshold 0.90"
    assert acc >= 0.90, f"Model accuracy regression: {acc:.4f} < threshold 0.90"

    # 2. Conformal Coverage Floor
    conformal = ConformalPredictor(calib_scores)
    cov_res = conformal.evaluate_coverage(probs, y_test, alpha=0.10)
    assert cov_res["empirical_coverage"] >= 0.85, (
        f"Conformal coverage regression: {cov_res['empirical_coverage']:.4f} < 0.85 nominal target"
    )
