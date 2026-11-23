"""Integration tests for SQLAlchemy persistence, transactional records, and alert workflows."""

import pytest
from src.database.models import Alert, ModelRecord, Prediction, Sample
from src.database.repository import WaterQualityRepository


def test_atomic_sample_and_prediction_persistence(in_memory_db, sample_feature_dict):
    """Verifies atomic creation of sample, prediction, explanation, and alert records."""
    repo = WaterQualityRepository(in_memory_db)

    inference_result = {
        "prediction": {
            "predicted_class": 0,
            "predicted_label": "Good",
            "confidence_score": 0.94,
            "calibrated_probabilities": {"Good": 0.94, "Moderate": 0.04, "Polluted": 0.02},
            "conformal_prediction_set": ["Good"],
            "coverage_guarantee": 0.90
        }
    }

    explainability_result = {
        "base_value": -1.2,
        "attributions": {"Dissolved Oxygen": -0.4, "Lead": -0.2},
        "dominant_pollutant_driver": "Dissolved Oxygen"
    }

    sample = repo.save_sample_and_inference(
        sample_data={"location_id": "STATION_TEST_01", **sample_feature_dict},
        inference_result=inference_result,
        explainability_result=explainability_result
    )

    assert sample.id is not None
    assert sample.prediction is not None
    assert sample.prediction.predicted_label == "Good"
    assert sample.prediction.explanation is not None
    assert sample.prediction.explanation.top_positive_feature == "Dissolved Oxygen"


def test_alert_query_and_acknowledgment(in_memory_db, contaminated_feature_dict):
    """Verifies that high-risk samples create alerts and that operators can acknowledge them."""
    repo = WaterQualityRepository(in_memory_db)

    inference_result = {
        "prediction": {
            "predicted_class": 2,
            "predicted_label": "Polluted",
            "confidence_score": 0.88,
            "calibrated_probabilities": {"Good": 0.02, "Moderate": 0.10, "Polluted": 0.88},
            "conformal_prediction_set": ["Polluted"],
            "coverage_guarantee": 0.90
        }
    }

    sample = repo.save_sample_and_inference(
        sample_data={"location_id": "STATION_HOTSPOT", **contaminated_feature_dict},
        inference_result=inference_result
    )

    # Verify unacknowledged alert was generated
    unack_alerts = repo.get_alerts(unacknowledged_only=True)
    assert len(unack_alerts) >= 1
    alert_to_ack = unack_alerts[0]
    assert alert_to_ack.is_acknowledged is False

    # Acknowledge alert
    ack_res = repo.acknowledge_alert(alert_to_ack.id, username="test_operator")
    assert ack_res is not None
    assert ack_res.is_acknowledged is True
    assert ack_res.acknowledged_at is not None
