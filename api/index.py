"""
AquaSense AI: Lean Vercel Serverless API.
Optimized for serverless execution (< 150 MB runtime footprint).
Uses NumPy, Scikit-Learn, Joblib, Pydantic, and FastAPI without heavy dependencies.
"""

import csv
import datetime
import io
import json
import sys
import urllib.parse
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
from src.aquasense.features import DomainFeatureEngineer  # noqa: F401
from src.aquasense.uncertainty import assess_prediction_confidence

# Load Model Bundle
MODEL_BUNDLE_PATH = PROJECT_ROOT / "models" / "model_bundle.joblib"
REGISTRY_PATH = PROJECT_ROOT / "models" / "registry.json"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.json"
WEB_DIR = PROJECT_ROOT / "web"

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


class VercelRouteMiddleware:
    """Raw ASGI middleware to normalize request paths rewritten by Vercel serverless."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            qs = scope.get("query_string", b"").decode("latin-1")
            route_applied = False
            if qs:
                params = urllib.parse.parse_qs(qs)
                route = params.get("_route", [None])[0]
                if route:
                    clean_path = "/" + route.lstrip("/")
                    scope["path"] = clean_path
                    scope["raw_path"] = clean_path.encode("latin-1")
                    route_applied = True
            if not route_applied:
                path = scope.get("path", "")
                if path.startswith("/api/index.py"):
                    clean_path = "/" + path[len("/api/index.py") :].lstrip("/")
                    scope["path"] = clean_path
                    scope["raw_path"] = clean_path.encode("latin-1")
        await self.app(scope, receive, send)


app.add_middleware(VercelRouteMiddleware)

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
    sample_id: str | None = Field(default=None, description="Sample identifier")
    location: str | None = Field(default=None, description="Sampling station or GPS coordinates")
    collection_time: str | None = Field(default=None, description="Sample collection timestamp")
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
                "sample_id": "STN-DEL-001",
                "location": "Upper Yamuna Basin, Station 3",
                "collection_time": "2026-11-24T09:00:00Z",
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
        return np.array(
            [
                self.pH,
                self.Turbidity,
                self.Temperature,
                self.Dissolved_Oxygen,
                self.Biochemical_Oxygen_Demand,
                self.Lead,
                self.Mercury,
                self.Arsenic,
            ],
            dtype=float,
        )

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
    stripped = str(value).strip()
    if stripped and stripped[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + stripped
    return stripped


def serve_html_page(filename: str):
    """Serves static HTML pages with utf-8 encoding."""
    target = WEB_DIR / filename
    if target.exists():
        from starlette.responses import HTMLResponse

        return HTMLResponse(content=target.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail=f"Page {filename} not found.")


@app.get("/")
@app.get("/index.html")
def get_home_page():
    """Serves AquaSense AI Web Homepage."""
    return serve_html_page("index.html")


@app.get("/analyze")
@app.get("/analyze.html")
def get_analyze_page():
    """Serves Water Analysis Workspace Page."""
    return serve_html_page("analyze.html")


@app.get("/batch")
@app.get("/batch.html")
def get_batch_page():
    """Serves Batch CSV Screening Page."""
    return serve_html_page("batch.html")


@app.get("/results")
@app.get("/results.html")
def get_results_page():
    """Serves Model Research & Metrics Page."""
    return serve_html_page("results.html")


@app.get("/monitoring")
@app.get("/monitoring.html")
def get_monitoring_page():
    """Serves Geographic Monitoring Map Page."""
    return serve_html_page("monitoring.html")


@app.get("/about")
@app.get("/about.html")
def get_about_page():
    """Serves Methodology & Standards Page."""
    return serve_html_page("about.html")


@app.get("/privacy")
@app.get("/privacy.html")
def get_privacy_page():
    """Serves Privacy Policy Page."""
    return serve_html_page("privacy.html")


@app.get("/terms")
@app.get("/terms.html")
def get_terms_page():
    """Serves Terms & Conditions Page."""
    return serve_html_page("terms.html")


@app.get("/health")
@app.get("/health/")
@app.get("/api/health")
@app.get("/api/health/")
@app.get("/api")
@app.get("/api/")
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
@app.get("/models/")
@app.get("/api/models")
@app.get("/api/models/")
def get_model_metadata() -> dict[str, Any]:
    """Returns details and verification metrics of the active screening model."""
    if not bundle:
        raise HTTPException(status_code=503, detail="Model bundle not loaded.")

    registry_data = {}
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            registry_data = json.load(f)

    metrics_data = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            metrics_data = json.load(f)

    return {
        "model_name": "Calibrated Random Forest with Domain Feature Engineering (4-Way Disjoint Partition)",
        "version": bundle.get("version", "1.0.0"),
        "features": FEATURE_ORDER,
        "classes": CLASS_LABELS,
        "conformal_quantile_90": round(float(bundle.get("conformal_quantile", 0.0230)), 6),
        "registry": registry_data,
        "verified_metrics": metrics_data.get("headline_metrics", {}),
        "partition_breakdown": metrics_data.get("partition_breakdown", {}),
        "dataset_leakage_audit": metrics_data.get("dataset_leakage_audit", {}),
        "conformal_uncertainty": metrics_data.get("conformal_uncertainty", {}),
    }


@app.post("/predict")
@app.post("/predict/")
@app.post("/api/predict")
@app.post("/api/predict/")
def predict_water_quality(sample: WaterSampleInput) -> dict[str, Any]:
    """
    Performs calibrated multi-class screening with Inductive Conformal Prediction set,
    granulated parameter observations, and statutory laboratory recommendations.
    """
    if not bundle:
        raise HTTPException(status_code=503, detail="Model bundle not loaded.")

    clf = bundle["model"]
    quantile = float(bundle.get("conformal_quantile", 0.0230))

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

    # Sensor & statutory bounds audit
    detector = AnomalyDetector()
    anomaly_audit = detector.audit_sample(sample.to_dict())

    # Formulate laboratory recommendation
    if needs_lab_test or anomaly_audit["severity"] == "CRITICAL":
        lab_recommendation = (
            "Immediate certified environmental laboratory testing required (ICP-MS heavy metal panel, 5-day standard "
            "BOD assay, and membrane filtration microbial screening). Prediction confidence is ambiguous or sensor bounds breached."
        )
    elif pred_class == 2:
        lab_recommendation = (
            "High pollution risk detected. Immediate contamination tracing and accredited laboratory confirmation "
            "recommended prior to any municipal, agricultural, or potable consumption."
        )
    elif pred_class == 1:
        lab_recommendation = "Moderate risk profile. Secondary verification advised prior to sensitive use."
    else:
        lab_recommendation = (
            "Nominal low-risk screening profile. Routine scheduled laboratory verification recommended."
        )

    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    sample_id = sample.sample_id or f"AQ-{int(datetime.datetime.now(datetime.UTC).timestamp())}"

    return {
        "sample_metadata": {
            "sample_id": sample_id,
            "location": sample.location or "Monitored Station",
            "collection_time": sample.collection_time or now_iso,
            "analysis_timestamp": now_iso,
        },
        "predicted_class": pred_class,
        "risk_tier": CLASS_LABELS.get(pred_class, "Unknown"),
        "probabilities": {
            "Good": round(float(probs[0]), 4),
            "Moderate": round(float(probs[1]), 4),
            "Polluted": round(float(probs[2]), 4),
        },
        "conformal_prediction_set": [CLASS_LABELS.get(c, str(c)) for c in prediction_set],
        "conformal_coverage_guarantee": "90% Marginal Coverage (Finite-Sample LAC)",
        "conformal_inclusion_threshold": round(threshold, 4),
        "needs_laboratory_test": needs_lab_test,
        "confidence_assessment": confidence_msg,
        "laboratory_recommendation": lab_recommendation,
        "anomaly_audit": anomaly_audit,
        "parameter_observations": anomaly_audit.get("parameter_observations", []),
        "model_version": bundle.get("version", "1.0.0"),
        "disclaimer": "Screening support only. Predictions do not replace certified laboratory environmental analysis.",
    }


@app.post("/explain")
@app.post("/explain/")
@app.post("/api/explain")
@app.post("/api/explain/")
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
    sorted_attributions = dict(sorted(attributions.items(), key=lambda item: abs(item[1]), reverse=True))

    return {
        "method": "occlusion-based, approximate",
        "predicted_class": CLASS_LABELS.get(pred_class, str(pred_class)),
        "base_probability": round(base_prob, 4),
        "feature_attributions": sorted_attributions,
        "notice": "Attribution indicates change in predicted risk probability when replacing the feature with its training baseline median.",
    }


@app.post("/anomaly")
@app.post("/anomaly/")
@app.post("/api/anomaly")
@app.post("/api/anomaly/")
def audit_anomalies(sample: WaterSampleInput) -> dict[str, Any]:
    """
    Audits single sample against physical sensor boundaries and WHO/BIS statutory drinking water thresholds.
    """
    detector = AnomalyDetector()
    audit = detector.audit_sample(sample.to_dict())
    return audit


@app.post("/predict/batch")
@app.post("/predict/batch/")
@app.post("/api/predict/batch")
@app.post("/api/predict/batch/")
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
    quantile = float(bundle.get("conformal_quantile", 0.0230))
    threshold = max(0.0, 1.0 - quantile)

    risk_counts = {"Good": 0, "Moderate": 0, "Polluted": 0}
    needs_lab_count = 0
    anomalous_count = 0

    detector = AnomalyDetector()

    # Look for optional sample identifier column
    sample_id_col = None
    for field in reader.fieldnames:
        if field.strip().lower() in ("sample_id", "sampleid", "id", "sample"):
            sample_id_col = field
            break

    for i, row in enumerate(reader):
        if i >= max_rows:
            break
        try:
            sample_vals = [float(row[field_map[f]]) for f in FEATURE_ORDER]
            sample_dict = {f: float(row[field_map[f]]) for f in FEATURE_ORDER}
            x_arr = np.array(sample_vals, dtype=float).reshape(1, -1)
            probs = clf.predict_proba(x_arr)[0]
            pred_class = int(np.argmax(probs))
            tier_name = CLASS_LABELS.get(pred_class, "Unknown")
            risk_counts[tier_name] = risk_counts.get(tier_name, 0) + 1

            c_set = [int(c) for c, p in enumerate(probs) if p >= threshold]
            if not c_set:
                c_set = [pred_class]

            needs_lab, conf_msg = assess_prediction_confidence(probs, c_set)
            if needs_lab:
                needs_lab_count += 1

            # Quick statutory limit check
            audit = detector.audit_sample(sample_dict)
            if audit["is_anomaly"]:
                anomalous_count += 1

            sid = sanitize_csv_cell(row[sample_id_col]) if sample_id_col and row.get(sample_id_col) else f"ROW-{i + 1}"

            rows.append(
                {
                    "row_index": i + 1,
                    "sample_id": sid,
                    "predicted_class": tier_name,
                    "confidence": round(float(probs[pred_class]), 4),
                    "probabilities": {
                        "Good": round(float(probs[0]), 4),
                        "Moderate": round(float(probs[1]), 4),
                        "Polluted": round(float(probs[2]), 4),
                    },
                    "conformal_set": [CLASS_LABELS.get(c, str(c)) for c in c_set],
                    "needs_laboratory_test": needs_lab,
                    "confidence_message": conf_msg,
                    "anomaly_severity": audit["severity"],
                    "is_anomaly": audit["is_anomaly"],
                }
            )
        except (ValueError, KeyError) as exc:
            rows.append(
                {
                    "row_index": i + 1,
                    "sample_id": f"ROW-{i + 1}",
                    "error": f"Invalid numerical values: {exc}",
                }
            )

    return {
        "batch_id": f"BATCH-{int(datetime.datetime.now(datetime.UTC).timestamp())}",
        "total_processed": len(rows),
        "limit_applied": max_rows,
        "batch_summary": {
            "risk_distribution": risk_counts,
            "polluted_samples_count": risk_counts.get("Polluted", 0),
            "moderate_samples_count": risk_counts.get("Moderate", 0),
            "good_samples_count": risk_counts.get("Good", 0),
            "needs_laboratory_test_count": needs_lab_count,
            "anomalous_samples_count": anomalous_count,
        },
        "results": rows,
        "disclaimer": "Screening support only. Predictions do not replace certified laboratory environmental analysis.",
    }


# Mount static web directory for local execution and frontend serving
if WEB_DIR.exists():
    from starlette.staticfiles import StaticFiles

    figures_dir = WEB_DIR / "figures"
    if figures_dir.exists():
        app.mount("/figures", StaticFiles(directory=str(figures_dir)), name="figures")

    css_dir = WEB_DIR / "css"
    if css_dir.exists():
        app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")

    js_dir = WEB_DIR / "js"
    if js_dir.exists():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")

    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web_root")

