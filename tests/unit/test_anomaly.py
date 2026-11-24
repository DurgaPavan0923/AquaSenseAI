"""Unit tests for sensor fault detection and multivariate anomaly scoring in AquaSense AI."""

import pytest

from src.aquasense.anomaly import AnomalyDetector


@pytest.fixture
def detector():
    return AnomalyDetector(contamination=0.03, random_state=42)


def test_physical_boundary_violations(detector):
    """Verifies that impossible physical values trigger CRITICAL severity."""
    sample_bad_ph = {"pH": 16.5, "Turbidity": 2.0, "Temperature": 22.0, "Dissolved Oxygen": 7.0}
    audit = detector.audit_sample(sample_bad_ph)
    assert audit["is_anomaly"] is True
    assert audit["severity"] == "CRITICAL"
    assert any("pH" in v for v in audit["physical_violations"])


def test_negative_physical_values(detector):
    """Verifies that negative turbidity triggers physical violation."""
    sample_neg = {"pH": 7.0, "Turbidity": -5.0, "Temperature": 22.0, "Dissolved Oxygen": 7.0}
    audit = detector.audit_sample(sample_neg)
    assert audit["is_anomaly"] is True
    assert audit["severity"] == "CRITICAL"
    assert any("Turbidity" in v for v in audit["physical_violations"])


def test_clean_sample_normal_severity(detector, sample_feature_dict):
    """Verifies clean nominal sample results in NORMAL severity."""
    audit = detector.audit_sample(sample_feature_dict)
    assert audit["severity"] == "NORMAL"
    assert audit["is_anomaly"] is False
    assert len(audit["physical_violations"]) == 0
    assert len(audit["statutory_violations"]) == 0


def test_statutory_violation_triggers_warning(detector, contaminated_feature_dict):
    """Verifies contaminated sample triggers WARNING or CRITICAL."""
    audit = detector.audit_sample(contaminated_feature_dict)
    assert audit["is_anomaly"] is True
    assert audit["severity"] in ["WARNING", "CRITICAL"]
    assert len(audit["statutory_violations"]) > 0
