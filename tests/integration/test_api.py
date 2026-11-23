"""Integration tests for FastAPI REST microservice contracts and schema validations."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app as fastapi_app


@pytest.fixture
def client():
    """Initializes FastAPI test client."""
    with TestClient(fastapi_app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Verifies that the root discovery endpoint returns service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "/docs" in data["documentation"]
    assert "disclaimer" in data


def test_health_alias_endpoint(client):
    """Verifies convenience alias /health matches /api/v1/health."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database" in data



def test_health_endpoint(client):
    """Verifies that system health probe returns 200 and valid schema."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "model_loaded" in data


def test_predict_single_sample_valid(client, sample_feature_dict):
    """Verifies end-to-end inference request generates calibrated probabilities and conformal set."""
    payload = {
        "location_id": "STATION_INTEGRATION_TEST",
        "sensor_id": "PROBE_PYTEST",
        "features": sample_feature_dict,
        "confidence_level": 0.90
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "sample_id" in data
    assert data["prediction"]["predicted_label"] in ["Good", "Moderate", "Polluted"]
    assert "conformal_prediction_set" in data["prediction"]
    assert len(data["prediction"]["conformal_prediction_set"]) >= 1
    assert "disclaimer" in data


def test_predict_single_sample_invalid_range_raises_422(client, sample_feature_dict):
    """Verifies that out-of-range physical inputs fail Pydantic validation with 422."""
    bad_features = sample_feature_dict.copy()
    bad_features["pH"] = 18.5  # Exceeds maximum 14.0

    payload = {
        "location_id": "STATION_FAIL",
        "features": bad_features
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_standalone_anomaly_endpoint(client, sample_feature_dict):
    """Verifies standalone anomaly detection endpoint."""
    response = client.post("/api/v1/anomaly", json=sample_feature_dict)
    assert response.status_code == 200
    data = response.json()
    assert "is_anomaly" in data
    assert "severity" in data
