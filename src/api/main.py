"""AquaSense AI Core FastAPI Application and Endpoint Routers."""

import io
import os
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path for serverless runtimes (Vercel / AWS Lambda)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from src.api.dependencies import ModelServicesContainer, verify_api_key
from src.api.schemas import (
    AlertAcknowledgeRequest,
    BatchInferenceResponse,
    HealthResponse,
    SingleInferenceRequest,
    SingleInferenceResponse,
    WaterFeaturesPayload,
)
from src.common.config import yaml_config
from src.common.constants import STATUTORY_DISCLAIMER, TARGET_CLASS_MAPPING, WHO_BIS_STANDARDS
from src.common.logger import logger
from src.database.connection import get_db, init_db
from src.database.models import ModelRecord, Sample
from src.database.repository import WaterQualityRepository

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context: initializes database tables and pre-loads model services."""
    logger.info("Initializing AquaSense AI microservice components...")
    init_db()
    container = ModelServicesContainer.get_instance()
    container.load_services()
    yield
    logger.info("Shutting down AquaSense AI microservice...")


app = FastAPI(
    title="AquaSense AI Microservice",
    description="Explainable Water-Quality Monitoring & Pollution-Risk Prediction REST API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def sanitize_csv_cell(value: Any) -> Any:
    """Neutralizes CSV formula injection (CWE-1236) by quoting starting operators."""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


# ---------------------------------------------------------
# 0. Root & Discovery Probes (for Vercel & Browser Navigation)
# ---------------------------------------------------------
@app.get("/", tags=["System"])
def root():
    """Root health and service discovery endpoint for Vercel deployment."""
    return {
        "service": "AquaSense AI Microservice",
        "status": "online",
        "documentation": "/docs",
        "openapi_schema": "/openapi.json",
        "health_check": "/api/v1/health",
        "version": "1.0.0",
        "disclaimer": STATUTORY_DISCLAIMER
    }


@app.get("/health", response_model=HealthResponse, tags=["System"], include_in_schema=False)
def health_alias(db: Session = Depends(get_db)):
    """Convenience alias for /api/v1/health."""
    return health_check(db)


# ---------------------------------------------------------
# 1. Health & Readiness Probe
# ---------------------------------------------------------
@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """Verifies system health, database accessibility, and model loading state."""
    db_ok = True
    try:
        db.execute(Sample.__table__.select().limit(1))
    except Exception:
        db_ok = False

    container = ModelServicesContainer.get_instance()
    return HealthResponse(
        status="healthy" if (db_ok and container.is_ready) else "degraded",
        database="connected" if db_ok else "disconnected",
        model_loaded=container.model is not None,
        conformal_calibrated=container.conformal_predictor is not None,
        version="1.0.0"
    )


# ---------------------------------------------------------
# 2. Single Sample Risk Prediction
# ---------------------------------------------------------
@app.post("/api/v1/predict", response_model=SingleInferenceResponse, tags=["Inference"])
@limiter.limit("60/minute")
def predict_single_sample(
    request: Request,
    payload: SingleInferenceRequest,
    db: Session = Depends(get_db),
    api_key: str | None = Depends(verify_api_key)
):
    """Screens single water sample: predicts risk tier, calibrated probabilities, conformal set, and SHAP drivers."""
    container = ModelServicesContainer.get_instance()
    if not container.is_ready or container.model is None or container.conformal_predictor is None:
        raise HTTPException(status_code=503, detail="Model service not ready or uninitialized.")

    # Convert payload features to dict
    features_dict = payload.features.model_dump(by_alias=True)
    df_sample = pd.DataFrame([features_dict])

    # 1. Anomaly & Sensor Fault Screening
    anomaly_result = container.anomaly_engine.score_sample(features_dict) if container.anomaly_engine else {
        "is_anomaly": False, "anomaly_score": 0.0, "has_sensor_fault": False, "severity": "LOW", "violations": []
    }

    # 2. Model Prediction & Probabilities
    probs = container.model.predict_proba(df_sample)[0]
    pred_class = int(np.argmax(probs))
    pred_label = TARGET_CLASS_MAPPING[pred_class]
    calibrated_probs = {TARGET_CLASS_MAPPING[i]: float(round(probs[i], 4)) for i in range(len(probs))}

    # 3. Conformal Prediction Set
    alpha = 1.0 - payload.confidence_level
    _, conformal_set = container.conformal_predictor.predict_set(probs, alpha=alpha)

    # 4. Local SHAP Attribution
    explanation = container.explainer.explain_local_sample(df_sample) if container.explainer else {
        "base_value": 0.0, "attributions": {}, "dominant_pollutant_driver": "N/A"
    }

    # 5. Regulatory WHO/BIS Exceedance Audit
    violations = []
    if payload.features.ph < WHO_BIS_STANDARDS["pH"]["min"] or payload.features.ph > WHO_BIS_STANDARDS["pH"]["max"]:
        violations.append(f"pH level {payload.features.ph} outside desirable limit [6.5, 8.5].")
    if payload.features.turbidity > WHO_BIS_STANDARDS["Turbidity"]["max"]:
        violations.append(f"Turbidity {payload.features.turbidity} NTU exceeds maximum limit of 5.0 NTU.")
    if payload.features.dissolved_oxygen < WHO_BIS_STANDARDS["Dissolved Oxygen"]["min"]:
        violations.append(f"Dissolved oxygen {payload.features.dissolved_oxygen} mg/L is below minimum potable threshold (6.5 mg/L).")
    if payload.features.bod > WHO_BIS_STANDARDS["Biochemical Oxygen Demand"]["clean_max"]:
        violations.append(f"BOD {payload.features.bod} mg/L indicates organic pollution (>2.0 mg/L).")
    if payload.features.lead > WHO_BIS_STANDARDS["Lead"]["max"]:
        violations.append(f"Lead {payload.features.lead} mg/L exceeds statutory toxicity limit (0.01 mg/L).")
    if payload.features.mercury > WHO_BIS_STANDARDS["Mercury"]["max"]:
        violations.append(f"Mercury {payload.features.mercury} mg/L exceeds statutory toxicity limit (0.001 mg/L).")
    if payload.features.arsenic > WHO_BIS_STANDARDS["Arsenic"]["max"]:
        violations.append(f"Arsenic {payload.features.arsenic} mg/L exceeds statutory toxicity limit (0.01 mg/L).")

    inference_dict = {
        "prediction": {
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "confidence_score": float(round(max(probs), 4)),
            "calibrated_probabilities": calibrated_probs,
            "conformal_prediction_set": conformal_set,
            "coverage_guarantee": payload.confidence_level
        }
    }

    # 6. Database Persistence
    repo = WaterQualityRepository(db)
    sample_record = repo.save_sample_and_inference(
        sample_data={
            "location_id": payload.location_id,
            "sensor_id": payload.sensor_id,
            "source_type": payload.source_type,
            **features_dict
        },
        inference_result=inference_dict,
        explainability_result=explanation,
        anomaly_result=anomaly_result
    )

    # 6. Conformal Abstention & Action Protocol
    is_ambiguous = len(conformal_set) > 1 or anomaly_result.get("is_anomaly", False)
    if is_ambiguous:
        triage_rec = (
            "AMBIGUOUS / HIGH UNCERTAINTY: Prediction set contains multiple classes or sensor anomaly flagged. "
            "Automated classification abstained; mandatory ISO/IEC 17025 certified laboratory assay required."
        )
    else:
        triage_rec = f"SCREENING PASS: Clear risk classification as '{pred_label}' at {payload.confidence_level*100:.0f}% confidence."

    return SingleInferenceResponse(
        sample_id=sample_record.id,
        location_id=payload.location_id,
        recorded_at=datetime.now(UTC).isoformat(),
        prediction=inference_dict["prediction"],
        anomaly=anomaly_result,
        explainability={
            "base_value": float(round(explanation["base_value"], 4)),
            "shap_attributions": {k: float(round(v, 4)) for k, v in explanation["attributions"].items()},
            "dominant_pollutant_driver": explanation["dominant_pollutant_driver"]
        },
        regulatory_audit={"who_bis_violations": violations},
        abstain_recommended=is_ambiguous,
        triage_recommendation=triage_rec,
        disclaimer=STATUTORY_DISCLAIMER
    )


# ---------------------------------------------------------
# 3. Batch CSV Inference Endpoint
# ---------------------------------------------------------
@app.post("/api/v1/predict/batch", response_model=BatchInferenceResponse, tags=["Inference"])
@limiter.limit("10/minute")
async def predict_batch_csv(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    api_key: str | None = Depends(verify_api_key)
):
    """Processes uploaded CSV: validates columns, runs batch inferences, conformal sets, and returns preview."""
    # Enforce 10MB limit
    MAX_SIZE = 10 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="File size exceeds maximum allowable limit of 10MB.")

    # Check MIME / extension
    if not (file.filename.endswith(".csv") or file.content_type in ["text/csv", "application/vnd.ms-excel"]):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are accepted.")

    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        # Sanitize against CSV formula injection (CWE-1236)
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].apply(sanitize_csv_cell)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Malformed CSV content: {str(e)}")

    container = ModelServicesContainer.get_instance()
    numerical_cols = [f["name"] for f in yaml_config["features"]["numerical"]]
    missing = [c for c in numerical_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"CSV missing mandatory feature columns: {missing}")

    # Vectorized Predictions
    X = df[numerical_cols]
    probs = container.model.predict_proba(X)
    preds = np.argmax(probs, axis=1)
    conformal_sets = container.conformal_predictor.predict_sets_batch(probs, alpha=0.10)

    # Compile Preview Rows
    preview = []
    risk_counts = {"Good": 0, "Moderate": 0, "Polluted": 0}
    anomalies_flagged = 0

    repo = WaterQualityRepository(db)
    for i in range(len(df)):
        c_label = TARGET_CLASS_MAPPING[preds[i]]
        risk_counts[c_label] += 1

        sample_dict = df.iloc[i].to_dict()
        anom_res = container.anomaly_engine.score_sample(sample_dict) if container.anomaly_engine else {"is_anomaly": False}
        if anom_res.get("is_anomaly"):
            anomalies_flagged += 1

        row_meta = {
            "index": i + 1,
            "predicted_risk": c_label,
            "confidence": float(round(max(probs[i]), 3)),
            "conformal_set": conformal_sets[i],
            "is_anomaly": anom_res.get("is_anomaly", False)
        }
        if i < 20:  # Preview top 20
            preview.append(row_meta)

    return BatchInferenceResponse(
        total_processed=len(df),
        anomalies_flagged=anomalies_flagged,
        pollution_risk_counts=risk_counts,
        preview=preview,
        disclaimer=STATUTORY_DISCLAIMER
    )


# ---------------------------------------------------------
# 4. Standalone Anomaly & Range Check Endpoint
# ---------------------------------------------------------
@app.post("/api/v1/anomaly", tags=["Diagnostics"])
def check_anomaly(payload: WaterFeaturesPayload):
    """Evaluates sample for sensor faults and multivariate statistical anomalies without full ML pipeline."""
    container = ModelServicesContainer.get_instance()
    features = payload.model_dump(by_alias=True)
    if container.anomaly_engine:
        res = container.anomaly_engine.score_sample(features)
    else:
        res = {"is_anomaly": False, "anomaly_score": 0.0, "has_sensor_fault": False, "severity": "LOW", "violations": []}
    return res


# ---------------------------------------------------------
# 5. Global Feature Importance & Explainability Data
# ---------------------------------------------------------
@app.get("/api/v1/explain/global", tags=["Explainability"])
def get_global_explanations():
    """Returns precomputed global feature importances and metadata."""
    feature_names = [f["name"] for f in yaml_config["features"]["numerical"]] + [
        "OSDR", "HMHI", "pH_Neutrality_Deviation", "Turbidity_DO_Ratio", "Toxic_Breach_Count"
    ]
    container = ModelServicesContainer.get_instance()
    if container.model and hasattr(container.model, "calibrated_classifiers_"):
        base_clf = container.model.calibrated_classifiers_[0].estimator.named_steps["classifier"]
        importances = getattr(base_clf, "feature_importances_", None)
        if importances is not None:
            sorted_feats = sorted(
                [{"feature": feature_names[i], "importance": float(round(importances[i], 4))}
                 for i in range(min(len(feature_names), len(importances)))],
                key=lambda x: x["importance"],
                reverse=True
            )
            return {"features": sorted_feats}
    return {"features": [{"feature": f, "importance": 0.1} for f in feature_names]}


# ---------------------------------------------------------
# 6. Historical Data & Alert Endpoints
# ---------------------------------------------------------
@app.get("/api/v1/history", tags=["Persistence"])
def get_historical_telemetry(limit: int = 50, db: Session = Depends(get_db)):
    """Fetches recent water samples with linked predictions."""
    repo = WaterQualityRepository(db)
    samples = repo.get_recent_samples(limit=limit)
    out = []
    for s in samples:
        pred_label = s.prediction.predicted_label if s.prediction else "Unknown"
        conf = s.prediction.confidence_score if s.prediction else 0.0
        out.append({
            "id": s.id,
            "recorded_at": s.recorded_at.isoformat() if s.recorded_at else None,
            "location_id": s.location_id,
            "pH": s.ph,
            "Turbidity": s.turbidity,
            "Temperature": s.temperature,
            "Dissolved Oxygen": s.dissolved_oxygen,
            "BOD": s.bod,
            "Lead": s.lead,
            "Mercury": s.mercury,
            "Arsenic": s.arsenic,
            "predicted_label": pred_label,
            "confidence": conf
        })
    return out


@app.get("/api/v1/alerts", tags=["Alerts"])
def get_active_alerts(unacknowledged_only: bool = False, db: Session = Depends(get_db)):
    """Fetches system alerts with severity and acknowledgment status."""
    repo = WaterQualityRepository(db)
    alerts = repo.get_alerts(unacknowledged_only=unacknowledged_only)
    return [
        {
            "id": a.id,
            "sample_id": a.sample_id,
            "severity": a.severity,
            "alert_type": a.alert_type,
            "message": a.message,
            "is_acknowledged": a.is_acknowledged,
            "created_at": a.created_at.isoformat() if a.created_at else None
        }
        for a in alerts
    ]


@app.post("/api/v1/alerts/{alert_id}/acknowledge", tags=["Alerts"])
def acknowledge_alert(alert_id: str, payload: AlertAcknowledgeRequest, db: Session = Depends(get_db)):
    """Marks an alert as acknowledged by an operator."""
    repo = WaterQualityRepository(db)
    alert = repo.acknowledge_alert(alert_id=alert_id, username=payload.acknowledged_by)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert ID not found.")
    return {"status": "success", "alert_id": alert.id, "acknowledged_at": alert.acknowledged_at.isoformat()}


@app.get("/api/v1/models", tags=["System"])
def get_registered_models(db: Session = Depends(get_db)):
    """Lists registered model records and active production model."""
    models = db.query(ModelRecord).all()
    return [
        {
            "id": m.id,
            "name": m.model_name,
            "version": m.version,
            "algorithm": m.algorithm,
            "is_active": m.is_active,
            "created_at": m.created_at.isoformat() if m.created_at else None
        }
        for m in models
    ]
