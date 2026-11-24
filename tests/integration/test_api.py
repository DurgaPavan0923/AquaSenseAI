"""
Integration tests for AquaSense AI serverless API endpoints.
Tests /health, /predict, /explain, /anomaly, /models, and /predict/batch.
"""

import io

from fastapi.testclient import TestClient

from api.index import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "notice" in data


def test_models_endpoint():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "features" in data
    assert "classes" in data
    assert "conformal_quantile_90" in data


def test_predict_endpoint_valid(sample_feature_dict):
    response = client.post("/predict", json=sample_feature_dict)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_tier"] in ["Good", "Moderate", "Polluted"]
    assert "probabilities" in data
    assert "conformal_prediction_set" in data
    assert len(data["conformal_prediction_set"]) >= 1
    assert "needs_laboratory_test" in data
    assert "disclaimer" in data


def test_predict_endpoint_out_of_bounds(sample_feature_dict):
    invalid = sample_feature_dict.copy()
    invalid["pH"] = 25.0  # Physical max 14.0
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_explain_endpoint(sample_feature_dict):
    response = client.post("/explain", json=sample_feature_dict)
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "occlusion-based, approximate"
    assert "feature_attributions" in data
    assert len(data["feature_attributions"]) == 8


def test_anomaly_endpoint(sample_feature_dict, contaminated_feature_dict):
    res_normal = client.post("/anomaly", json=sample_feature_dict)
    assert res_normal.status_code == 200
    assert res_normal.json()["severity"] == "NORMAL"

    res_polluted = client.post("/anomaly", json=contaminated_feature_dict)
    assert res_polluted.status_code == 200
    assert res_polluted.json()["severity"] in ["WARNING", "CRITICAL"]


def test_batch_predict_csv():
    csv_content = (
        "pH,Turbidity,Temperature,Dissolved Oxygen,Biochemical Oxygen Demand,Lead,Mercury,Arsenic\n"
        "7.2,2.5,21.0,7.0,1.5,0.003,0.0002,0.003\n"
        "4.8,35.0,28.0,2.0,15.0,0.04,0.004,0.03\n"
    )
    files = {"file": ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["results"]) == 2
