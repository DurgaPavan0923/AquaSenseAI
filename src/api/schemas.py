"""Pydantic v2 REST API Request and Response Data Transfer Schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.common.constants import STATUTORY_DISCLAIMER


class WaterFeaturesPayload(BaseModel):
    """Raw physical and chemical water sample metrics."""
    ph: float = Field(..., alias="pH", ge=0.0, le=14.0, description="pH level (0.0 to 14.0)")
    turbidity: float = Field(..., alias="Turbidity", ge=0.0, le=500.0, description="Turbidity in NTU")
    temperature: float = Field(..., alias="Temperature", ge=-5.0, le=60.0, description="Temperature in °C")
    dissolved_oxygen: float = Field(..., alias="Dissolved Oxygen", ge=0.0, le=25.0, description="Dissolved Oxygen in mg/L")
    bod: float = Field(..., alias="Biochemical Oxygen Demand", ge=0.0, le=100.0, description="BOD in mg/L")
    lead: float = Field(..., alias="Lead", ge=0.0, le=5.0, description="Lead in mg/L")
    mercury: float = Field(..., alias="Mercury", ge=0.0, le=1.0, description="Mercury in mg/L")
    arsenic: float = Field(..., alias="Arsenic", ge=0.0, le=5.0, description="Arsenic in mg/L")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "pH": 7.3,
                "Turbidity": 2.5,
                "Temperature": 21.0,
                "Dissolved Oxygen": 7.5,
                "Biochemical Oxygen Demand": 1.8,
                "Lead": 0.003,
                "Mercury": 0.0002,
                "Arsenic": 0.004
            }
        }
    }


class SingleInferenceRequest(BaseModel):
    """Single water sample inference request."""
    location_id: str = Field(default="STATION_RIVER_01", description="Monitoring station identifier")
    sensor_id: str = Field(default="PROBE_ALPHA", description="Sensor probe hardware ID")
    source_type: str = Field(default="iot_stream", description="Source: iot_stream, manual_lab, or batch_upload")
    features: WaterFeaturesPayload
    confidence_level: float = Field(default=0.90, ge=0.50, le=0.99, description="Conformal confidence (1 - alpha)")


class PredictionDetail(BaseModel):
    predicted_class: int
    predicted_label: str
    confidence_score: float
    calibrated_probabilities: Dict[str, float]
    conformal_prediction_set: List[str]
    coverage_guarantee: float


class AnomalyDetail(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    has_sensor_fault: bool
    severity: str
    violations: List[str]


class ExplainabilityDetail(BaseModel):
    base_value: float
    shap_attributions: Dict[str, float]
    dominant_pollutant_driver: str


class SingleInferenceResponse(BaseModel):
    sample_id: str
    location_id: str
    recorded_at: str
    prediction: PredictionDetail
    anomaly: AnomalyDetail
    explainability: ExplainabilityDetail
    regulatory_audit: Dict[str, List[str]]
    abstain_recommended: bool = Field(default=False, description="True if conformal prediction set is ambiguous (>1 class) or an anomaly was flagged.")
    triage_recommendation: str = Field(default="", description="Operational action protocol (Screening clear vs Mandatory lab assay required).")
    disclaimer: str = Field(default=STATUTORY_DISCLAIMER)


class BatchInferenceResponse(BaseModel):
    total_processed: int
    anomalies_flagged: int
    pollution_risk_counts: Dict[str, int]
    preview: List[Dict[str, Any]]
    disclaimer: str = Field(default=STATUTORY_DISCLAIMER)


class HealthResponse(BaseModel):
    status: str
    database: str
    model_loaded: bool
    conformal_calibrated: bool
    version: str


class AlertAcknowledgeRequest(BaseModel):
    acknowledged_by: Optional[str] = "operator"
