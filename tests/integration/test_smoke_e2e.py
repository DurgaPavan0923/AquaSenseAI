"""End-to-End smoke test verifying complete pipeline integration."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import generate_benchmark_dataset
from src.ml.anomaly import AnomalyEngine
from src.ml.conformal import ConformalPredictor
from src.ml.explainability import WaterQualityExplainer
from src.ml.train import train_and_calibrate_active_model


def test_full_pipeline_smoke_e2e(tmp_path):
    """Executes full operational pipeline: synthetic generation -> train -> calibration -> conformal -> SHAP -> anomaly."""
    # 1. Dataset Generation
    df = generate_benchmark_dataset(n_samples=120, seed=42)
    train_df = df.iloc[:90].copy()
    test_df = df.iloc[90:].copy()

    # 2. Train and Calibrate Model
    custom_cfg = {
        "project": {"random_seed": 42},
        "target": {"name": "Pollution Level"},
        "features": {
            "numerical": [
                {"name": "pH"}, {"name": "Turbidity"}, {"name": "Temperature"},
                {"name": "Dissolved Oxygen"}, {"name": "Biochemical Oxygen Demand"},
                {"name": "Lead"}, {"name": "Mercury"}, {"name": "Arsenic"}
            ]
        },
        "split": {"strategy": "stratified_kfold", "n_splits": 3, "random_state": 42, "shuffle": True},
        "models": {
            "active_model_type": "xgboost",
            "registry_dir": str(tmp_path / "registry"),
            "artifacts_dir": str(tmp_path / "artifacts")
        }
    }

    calibrated_clf, metrics, calib_scores = train_and_calibrate_active_model(train_df, test_df, config=custom_cfg)
    assert metrics["test_macro_f1"] > 0.80

    # 3. Conformal Prediction Set
    conformal = ConformalPredictor(calib_scores)
    sample_feat = test_df.iloc[[0]][[f["name"] for f in custom_cfg["features"]["numerical"]]]
    probs = calibrated_clf.predict_proba(sample_feat)[0]
    _, conf_set = conformal.predict_set(probs, alpha=0.10)
    assert len(conf_set) >= 1

    # 4. Local SHAP Attribution
    explainer = WaterQualityExplainer(calibrated_clf)
    expl = explainer.explain_local_sample(sample_feat)
    assert "dominant_pollutant_driver" in expl

    # 5. Anomaly Detection
    anom_engine = AnomalyEngine()
    anom_engine.fit(train_df)
    anom_res = anom_engine.score_sample(sample_feat.iloc[0].to_dict())
    assert "is_anomaly" in anom_res
