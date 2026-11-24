"""
End-to-end smoke test verifying complete pipeline integration in AquaSense AI.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.aquasense.anomaly import AnomalyDetector
from src.aquasense.data import FEATURE_COLUMNS, TARGET_COLUMN, load_raw_dataset, split_dataset
from src.aquasense.explain import explain_sample_occlusion
from src.aquasense.features import DomainFeatureEngineer
from src.aquasense.uncertainty import ConformalCalibrator


def test_full_pipeline_smoke_e2e():
    """Tests loading, feature transformation, training, calibration, conformal sets, occlusion, and anomalies."""
    df = load_raw_dataset()
    train_df, test_df = split_dataset(df, test_size=0.20, random_state=42)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].values
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].values

    # 1. Pipeline execution
    pipe = Pipeline(
        [
            ("features", DomainFeatureEngineer()),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=20, random_state=42)),
        ]
    )
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)
    assert len(preds) == len(y_test)

    # 2. Conformal calibration
    conformal = ConformalCalibrator(alpha=0.10)
    conformal.calibrate(probs, y_test)
    assert conformal.calibrated_ is True
    assert 0.0 <= conformal.quantile_ <= 1.0

    cov_eval = conformal.evaluate_coverage(probs, y_test)
    assert cov_eval["empirical_coverage"] >= 0.85

    # 3. Local occlusion explanation
    medians = X_train.median().values
    sample = X_test.iloc[0].values
    exp = explain_sample_occlusion(pipe, sample, medians)
    assert exp["method"] == "occlusion-based, approximate"
    assert len(exp["attributions"]) == 8

    # 4. Anomaly detection
    detector = AnomalyDetector()
    detector.fit(X_train.values)
    audit = detector.audit_sample(X_test.iloc[0].to_dict())
    assert "severity" in audit
    assert audit["severity"] in ["NORMAL", "WARNING", "CRITICAL"]
