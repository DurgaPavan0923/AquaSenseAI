# AquaSense AI - REST API Specification

The AquaSense AI inference microservice is built with FastAPI and runs on port `8000`.

Base URL: `http://localhost:8000/api/v1`

---

## 1. Endpoints

### A. Health & Readiness Probe
`GET /health`
* **Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "model_loaded": true,
  "conformal_calibrated": true,
  "version": "1.0.0"
}
```

---

### B. Single Sample Screening
`POST /predict`
* **Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: aquasense-dev-secret-key-2026" \
  -d '{
    "location_id": "STATION_RIVER_01",
    "sensor_id": "PROBE_01",
    "features": {
      "pH": 6.8,
      "Turbidity": 4.5,
      "Temperature": 22.0,
      "Dissolved Oxygen": 5.8,
      "Biochemical Oxygen Demand": 3.2,
      "Lead": 0.012,
      "Mercury": 0.0008,
      "Arsenic": 0.009
    },
    "confidence_level": 0.90
  }'
```

* **Response:**
```json
{
  "sample_id": "8d6a4bb0-0199-4a92-b05c-df03b4eb89f5",
  "location_id": "STATION_RIVER_01",
  "recorded_at": "2026-11-23T15:00:00Z",
  "prediction": {
    "predicted_class": 1,
    "predicted_label": "Moderate",
    "confidence_score": 0.7842,
    "calibrated_probabilities": {
      "Good": 0.0812,
      "Moderate": 0.7842,
      "Polluted": 0.1346
    },
    "conformal_prediction_set": ["Moderate", "Polluted"],
    "coverage_guarantee": 0.90
  },
  "anomaly": {
    "is_anomaly": false,
    "anomaly_score": 0.124,
    "has_sensor_fault": false,
    "severity": "LOW",
    "violations": []
  },
  "explainability": {
    "base_value": -0.842,
    "shap_attributions": {
      "Lead": 0.542,
      "Biochemical Oxygen Demand": 0.412,
      "Dissolved Oxygen": 0.321
    },
    "dominant_pollutant_driver": "Lead"
  },
  "regulatory_audit": {
    "who_bis_violations": [
      "Dissolved Oxygen below recommended limit (5.8 < 6.5 mg/L)",
      "Lead concentration exceeds maximum limit (0.012 > 0.01 mg/L)"
    ]
  },
  "disclaimer": "Screening support only; not a substitute for certified laboratory analysis."
}
```

---

### C. Batch CSV Screening
`POST /predict/batch`
* Ingests multipart CSV file (`max 10MB`).
* Returns summary counts, preview rows with conformal sets, and flagged anomalies.

---

### D. Active Alerts & Acknowledgment
* `GET /alerts?unacknowledged_only=true`
* `POST /alerts/{alert_id}/acknowledge`
