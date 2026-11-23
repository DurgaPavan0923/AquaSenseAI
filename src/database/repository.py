"""Data Access Object (DAO) repository implementing transactional persistence."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from src.common.logger import logger
from src.database.models import Alert, Explanation, ModelRecord, Prediction, Sample


class WaterQualityRepository:
    """Repository handling atomic transactions for sensor samples, ML inferences, and alerts."""

    def __init__(self, db: Session):
        self.db = db

    def get_or_create_active_model(self, model_name: str, version: str = "1.0.0", algorithm: str = "XGBoost") -> ModelRecord:
        """Retrieves or creates active model registry record."""
        active_model = self.db.query(ModelRecord).filter(ModelRecord.model_name == model_name, ModelRecord.is_active == True).first()
        if not active_model:
            active_model = ModelRecord(
                model_name=model_name,
                version=version,
                algorithm=algorithm,
                artifact_path=f"artifacts/models/{model_name}/model.joblib",
                metrics_json=json.dumps({"status": "active"}),
                is_active=True
            )
            self.db.add(active_model)
            self.db.commit()
            self.db.refresh(active_model)
        return active_model

    def save_sample_and_inference(
        self,
        sample_data: Dict[str, Any],
        inference_result: Dict[str, Any],
        explainability_result: Optional[Dict[str, Any]] = None,
        anomaly_result: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None
    ) -> Sample:
        """Atomically saves sample, prediction record, SHAP explanations, and generated alerts."""
        if not model_id:
            active_model = self.get_or_create_active_model(model_name="xgboost_active")
            model_id = active_model.id

        # 1. Create Sample
        sample = Sample(
            location_id=sample_data.get("location_id", "STATION_01"),
            sensor_id=sample_data.get("sensor_id", "PROBE_01"),
            source_type=sample_data.get("source_type", "iot_stream"),
            ph=float(sample_data["pH"]),
            turbidity=float(sample_data["Turbidity"]),
            temperature=float(sample_data["Temperature"]),
            dissolved_oxygen=float(sample_data["Dissolved Oxygen"]),
            bod=float(sample_data["Biochemical Oxygen Demand"]),
            lead=float(sample_data["Lead"]),
            mercury=float(sample_data["Mercury"]),
            arsenic=float(sample_data["Arsenic"]),
            is_synthetic=sample_data.get("is_synthetic", False)
        )
        self.db.add(sample)
        self.db.flush()

        # 2. Create Prediction
        pred_meta = inference_result.get("prediction", {})
        probabilities = pred_meta.get("calibrated_probabilities", {})
        conformal_set = pred_meta.get("conformal_prediction_set", [])
        is_anom = anomaly_result.get("is_anomaly", False) if anomaly_result else False
        anom_score = anomaly_result.get("anomaly_score", 0.0) if anomaly_result else 0.0

        prediction = Prediction(
            sample_id=sample.id,
            model_id=model_id,
            predicted_class=int(pred_meta.get("predicted_class", 0)),
            predicted_label=str(pred_meta.get("predicted_label", "Good")),
            confidence_score=float(max(probabilities.values())) if probabilities else 1.0,
            probabilities_json=json.dumps(probabilities),
            conformal_set_json=json.dumps(conformal_set),
            is_anomaly=is_anom,
            anomaly_score=float(anom_score)
        )
        self.db.add(prediction)
        self.db.flush()

        # 3. Create Explanation
        if explainability_result:
            explanation = Explanation(
                prediction_id=prediction.id,
                base_value=float(explainability_result.get("base_value", 0.0)),
                shap_values_json=json.dumps(explainability_result.get("attributions", {})),
                top_positive_feature=explainability_result.get("dominant_pollutant_driver"),
                top_negative_feature=None
            )
            self.db.add(explanation)

        # 4. Create Alerts if triggered
        if anomaly_result and anomaly_result.get("is_anomaly"):
            for msg in anomaly_result.get("violations", []):
                alert = Alert(
                    sample_id=sample.id,
                    prediction_id=prediction.id,
                    severity=anomaly_result.get("severity", "MEDIUM"),
                    alert_type="SENSOR_ANOMALY" if anomaly_result.get("has_sensor_fault") else "POLLUTION_OUTLIER",
                    message=msg
                )
                self.db.add(alert)
        elif prediction.predicted_label == "Polluted":
            alert = Alert(
                sample_id=sample.id,
                prediction_id=prediction.id,
                severity="HIGH",
                alert_type="HIGH_POLLUTION_RISK",
                message=f"High pollution risk predicted at {sample.location_id}."
            )
            self.db.add(alert)

        self.db.commit()
        self.db.refresh(sample)
        return sample

    def get_recent_samples(self, limit: int = 50) -> List[Sample]:
        """Fetches recent samples ordered by timestamp descending."""
        return self.db.query(Sample).order_by(Sample.recorded_at.desc()).limit(limit).all()

    def get_alerts(self, unacknowledged_only: bool = False, limit: int = 100) -> List[Alert]:
        """Fetches system alerts with optional acknowledgment filter."""
        query = self.db.query(Alert)
        if unacknowledged_only:
            query = query.filter(Alert.is_acknowledged == False)
        return query.order_by(Alert.created_at.desc()).limit(limit).all()

    def acknowledge_alert(self, alert_id: str, username: Optional[str] = None) -> Optional[Alert]:
        """Acknowledges an existing alert."""
        alert = self.db.query(Alert).filter(Alert.id == alert_id).first()
        if alert:
            alert.is_acknowledged = True
            alert.acknowledged_at = datetime.now(timezone.utc)
            self.db.commit()
            self.db.refresh(alert)
        return alert
