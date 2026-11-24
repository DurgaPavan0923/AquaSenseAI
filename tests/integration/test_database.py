"""
Integration tests for AquaSense AI SQLAlchemy database records and transaction workflows.
"""

from src.aquasense.db import AlertRecord, PredictionRecord, SampleRecord


def test_atomic_sample_and_prediction_persistence(in_memory_db, sample_feature_dict):
    """Verifies creation and retrieval of sample and prediction records."""
    sample = SampleRecord(
        sample_code="WQS_TEST_01",
        ph=sample_feature_dict["pH"],
        turbidity=sample_feature_dict["Turbidity"],
        temperature=sample_feature_dict["Temperature"],
        dissolved_oxygen=sample_feature_dict["Dissolved Oxygen"],
        biochemical_oxygen_demand=sample_feature_dict["Biochemical Oxygen Demand"],
        lead=sample_feature_dict["Lead"],
        mercury=sample_feature_dict["Mercury"],
        arsenic=sample_feature_dict["Arsenic"],
    )
    in_memory_db.add(sample)
    in_memory_db.commit()
    in_memory_db.refresh(sample)

    assert sample.id is not None
    assert sample.ph == 7.3

    prediction = PredictionRecord(
        sample_id=sample.id,
        predicted_class=0,
        class_label="Good",
        prob_good=0.98,
        prob_moderate=0.01,
        prob_polluted=0.01,
        confidence=0.98,
        conformal_set="['Good']",
        needs_lab_test=False,
    )
    in_memory_db.add(prediction)
    in_memory_db.commit()
    in_memory_db.refresh(prediction)

    assert prediction.id is not None
    assert prediction.predicted_class == 0
    assert prediction.class_label == "Good"


def test_alert_persistence(in_memory_db):
    """Verifies that sensor audits and statutory alerts persist properly."""
    alert = AlertRecord(
        sample_id=1,
        severity="WARNING",
        status_message="Lead concentration exceeds WHO limit",
        is_anomaly=True,
        violations_json='["Lead > 0.01 mg/L"]',
    )
    in_memory_db.add(alert)
    in_memory_db.commit()
    in_memory_db.refresh(alert)

    assert alert.id is not None
    assert alert.severity == "WARNING"
    assert alert.is_anomaly is True
