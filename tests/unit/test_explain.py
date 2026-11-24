"""Unit tests for occlusion-based feature attribution and rank stability."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.aquasense.explain import evaluate_rank_stability, explain_sample_occlusion


def test_explain_sample_occlusion():
    # Fit simple mock classifier on 8 features
    X = np.random.uniform(1.0, 10.0, (50, 8))
    y = np.random.choice([0, 1, 2], size=50)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    medians = np.median(X, axis=0)
    sample = X[0]

    res = explain_sample_occlusion(clf, sample, medians)
    assert res["method"] == "occlusion-based, approximate"
    assert "feature_attributions" in res or "attributions" in res
    attrs = res.get("feature_attributions", res.get("attributions"))
    assert len(attrs) == 8


def test_evaluate_rank_stability():
    f1 = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    f2 = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    rep = evaluate_rank_stability([f1, f2])
    assert rep["mean_spearman_rank_correlation"] == 1.0
    assert rep["stability_level"] == "High"
