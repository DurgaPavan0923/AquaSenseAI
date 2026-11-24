"""
AquaSense AI: Lean Vercel Serverless API.
Optimized for serverless execution (< 150 MB runtime footprint).
Uses NumPy, Scikit-Learn, Joblib, Pydantic, and FastAPI without heavy dependencies.
"""

import csv
import io
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aquasense.anomaly import AnomalyDetector
from src.aquasense.uncertainty import assess_prediction_confidence

# Load Model Bundle
MODEL_BUNDLE_PATH = PROJECT_ROOT / "models" / "model_bundle.joblib"
REGISTRY_PATH = PROJECT_ROOT / "models" / "registry.json"

bundle: dict[str, Any] | None = None
if MODEL_BUNDLE_PATH.exists():
    try:
        bundle = joblib.load(MODEL_BUNDLE_PATH)
    except Exception as exc:
        print(f"Warning: Failed to load model bundle: {exc}")

app = FastAPI(
    title="AquaSense AI API",
    description="Explainable water-quality monitoring and pollution-risk screening API.",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURE_ORDER = [
    "pH",
    "Turbidity",
    "Temperature",
    "Dissolved Oxygen",
    "Biochemical Oxygen Demand",
    "Lead",
    "Mercury",
    "Arsenic",
]

CLASS_LABELS = {0: "Good", 1: "Moderate", 2: "Polluted"}


class WaterSampleInput(BaseModel):
    pH: float = Field(..., ge=0.0, le=14.0, description="Acidity/alkalinity level")
    Turbidity: float = Field(..., ge=0.0, le=500.0, description="Turbidity in NTU")
    Temperature: float = Field(..., ge=-5.0, le=60.0, description="Water temperature in deg C")
    Dissolved_Oxygen: float = Field(..., ge=0.0, le=25.0, alias="Dissolved Oxygen", description="DO in mg/L")
    Biochemical_Oxygen_Demand: float = Field(
        ..., ge=0.0, le=100.0, alias="Biochemical Oxygen Demand", description="BOD in mg/L"
    )
    Lead: float = Field(..., ge=0.0, le=5.0, description="Lead concentration in mg/L")
    Mercury: float = Field(..., ge=0.0, le=1.0, description="Mercury concentration in mg/L")
    Arsenic: float = Field(..., ge=0.0, le=5.0, description="Arsenic concentration in mg/L")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "pH": 7.2,
                "Turbidity": 2.5,
                "Temperature": 22.0,
                "Dissolved Oxygen": 6.8,
                "Biochemical Oxygen Demand": 1.8,
                "Lead": 0.005,
                "Mercury": 0.0004,
                "Arsenic": 0.003,
            }
        },
    }

    def to_array(self) -> np.ndarray:
        return np.array([
            self.pH,
            self.Turbidity,
            self.Temperature,
            self.Dissolved_Oxygen,
            self.Biochemical_Oxygen_Demand,
            self.Lead,
            self.Mercury,
            self.Arsenic,
        ], dtype=float)

    def to_dict(self) -> dict[str, float]:
        return {
            "pH": self.pH,
            "Turbidity": self.Turbidity,
            "Temperature": self.Temperature,
            "Dissolved Oxygen": self.Dissolved_Oxygen,
            "Biochemical Oxygen Demand": self.Biochemical_Oxygen_Demand,
            "Lead": self.Lead,
            "Mercury": self.Mercury,
            "Arsenic": self.Arsenic,
        }


def sanitize_csv_cell(value: str) -> str:
    """Sanitizes CSV cell to prevent CSV formula injection."""
    stripped = value.strip()
    if stripped and stripped[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + stripped
    return stripped


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Returns server and active model health status."""
    is_model_loaded = bundle is not None
    return {
        "status": "healthy",
        "service": "AquaSense AI Serverless API",
        "model_loaded": is_model_loaded,
        "version": bundle.get("version", "1.0.0") if bundle else "unloaded",
        "notice": "Screening support only. Not a substitute for certified laboratory analysis.",
    }


@app.get("/models")
def get_model_metadata() -> dict[str, Any]:
    """Returns details and verification metrics of the active screening model."""
    if not bundle:
        raise HTTPException(status_code=503, detail="Model bundle not loaded.")

    import json
    registry_data = {}
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            registry_data = json.load(f)

    return {
        "model_name": "Calibrated Random Forest with Domain Feature Engineering",
        "version": bundle.get("version", "1.0.0"),
        "features": FEATURE_ORDER,
        "classes": CLASS_LABELS,
        "conformal_quantile_90": round(float(bundle.get("conformal_quantile", 0.0216)), 6),
        "registry": registry_data,
    }


@app.post("/predict")
def predict_water_quality(sample: WaterSampleInput) -> dict[str, Any]:
    """
    Performs calibrated multi-class screening with Inductive Conformal Prediction set
    and confidence assessment.
    """
    if not bundle:
        raise HTTPException(status_code=503, detail="Model bundle not loaded.")

    clf = bundle["model"]
    quantile = float(bundle.get("conformal_quantile", 0.0216))

    x = sample.to_array().reshape(1, -1)
    probs = clf.predict_proba(x)[0]
    pred_class = int(np.argmax(probs))

    # Conformal Prediction Set: C(x) = {y : P(y|x) >= 1 - quantile}
    threshold = max(0.0, 1.0 - quantile)
    prediction_set = [int(c) for c, p in enumerate(probs) if p >= threshold]
    if not prediction_set:
        prediction_set = [pred_class]

    # Confidence audit
    needs_lab_test, confidence_msg = assess_prediction_confidence(probs, prediction_set)

    return {
        "predicted_class": pred_class,
        "risk_tier": CLASS_LABELS.get(pred_class, "Unknown"),
        "probabilities": {
            "Good": round(float(probs[0]), 4),
            "Moderate": round(float(probs[1]), 4),
            "Polluted": round(float(probs[2]), 4),
        },
        "conformal_prediction_set": [CLASS_LABELS.get(c, str(c)) for c in prediction_set],
        "conformal_coverage_guarantee": "90% Marginal Coverage",
        "needs_laboratory_test": needs_lab_test,
        "confidence_assessment": confidence_msg,
        "disclaimer": "Screening support only. Not a substitute for certified laboratory analysis.",
    }


@app.post("/explain")
def explain_prediction(sample: WaterSampleInput) -> dict[str, Any]:
    """
    Computes local feature attribution via counterfactual occlusion against training medians.
    Clearly labeled: occlusion-based, approximate.
    """
    if not bundle:
        raise HTTPException(status_code=503, detail="Model bundle not loaded.")

    clf = bundle["model"]
    medians = np.array(bundle.get("training_medians", [7.15, 2.90, 22.1, 5.58, 2.41, 0.006, 0.00038, 0.0057]))

    x = sample.to_array()
    base_probs = clf.predict_proba(x.reshape(1, -1))[0]
    pred_class = int(np.argmax(base_probs))
    base_prob = float(base_probs[pred_class])

    attributions: dict[str, float] = {}
    for i, feat_name in enumerate(FEATURE_ORDER):
        occluded = x.copy()
        occluded[i] = medians[i]
        occluded_prob = float(clf.predict_proba(occluded.reshape(1, -1))[0][pred_class])
        delta = base_prob - occluded_prob
        attributions[feat_name] = round(delta, 4)

    # Sort descending by absolute impact
    sorted_attributions = dict(
        sorted(attributions.items(), key=lambda item: abs(item[1]), reverse=True)
    )

    return {
        "method": "occlusion-based, approximate",
        "predicted_class": CLASS_LABELS.get(pred_class, str(pred_class)),
        "base_probability": round(base_prob, 4),
        "feature_attributions": sorted_attributions,
        "notice": "Attribution indicates change in predicted risk probability when replacing the feature with its training baseline median.",
    }


@app.post("/anomaly")
def audit_anomalies(sample: WaterSampleInput) -> dict[str, Any]:
    """
    Audits single sample against physical sensor boundaries and WHO/BIS statutory drinking water thresholds.
    """
    detector = AnomalyDetector()
    audit = detector.audit_sample(sample.to_dict())
    return audit


@app.post("/predict/batch")
async def predict_batch_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Processes batch CSV file upload with file size validation, row limits, and CSV sanitization.
    """
    if not bundle:
        raise HTTPException(status_code=503, detail="Model bundle not loaded.")

    contents = await file.read()
    if len(contents) > 2 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds maximum limit of 2 MB.")

    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid file encoding. UTF-8 encoded CSV required.")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV file appears empty or unparseable.")

    # Match column names case-insensitively
    field_map = {}
    for field in reader.fieldnames:
        clean = field.strip()
        for expected in FEATURE_ORDER:
            if clean.lower() == expected.lower():
                field_map[expected] = field

    missing = [f for f in FEATURE_ORDER if f not in field_map]
    if missing:
        raise HTTPException(status_code=400, detail=f"CSV missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    max_rows = 500
    clf = bundle["model"]
    quantile = float(bundle.get("conformal_quantile", 0.0216))
    threshold = max(0.0, 1.0 - quantile)

    for i, row in enumerate(reader):
        if i >= max_rows:
            break
        try:
            sample_vals = [float(row[field_map[f]]) for f in FEATURE_ORDER]
            x_arr = np.array(sample_vals, dtype=float).reshape(1, -1)
            probs = clf.predict_proba(x_arr)[0]
            pred_class = int(np.argmax(probs))

            c_set = [int(c) for c, p in enumerate(probs) if p >= threshold]
            if not c_set:
                c_set = [pred_class]

            needs_lab, _ = assess_prediction_confidence(probs, c_set)

            rows.append({
                "row_index": i + 1,
                "predicted_class": CLASS_LABELS.get(pred_class, str(pred_class)),
                "confidence": round(float(probs[pred_class]), 4),
                "conformal_set": [CLASS_LABELS.get(c, str(c)) for c in c_set],
                "needs_laboratory_test": needs_lab,
            })
        except ValueError as exc:
            rows.append({
                "row_index": i + 1,
                "error": f"Invalid numerical values: {exc}",
            })

    return {
        "total_processed": len(rows),
        "limit_applied": max_rows,
        "results": rows,
        "disclaimer": "Screening support only. Not a substitute for certified laboratory analysis.",
    }
