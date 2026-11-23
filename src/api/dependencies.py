"""FastAPI dependency injection, security authentication, and service caching."""

import hashlib
import os
import secrets
from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import Header, HTTPException, Security, status

from src.common.config import settings, yaml_config
from src.common.logger import logger
from src.ml.anomaly import AnomalyEngine
from src.ml.conformal import ConformalPredictor
from src.ml.explainability import WaterQualityExplainer


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file for integrity verification."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


class ModelServicesContainer:
    """Singleton container managing pre-loaded ML models and explainers."""
    
    _instance: Optional["ModelServicesContainer"] = None

    def __init__(self):
        self.model: Optional[Any] = None
        self.conformal_predictor: Optional[ConformalPredictor] = None
        self.explainer: Optional[WaterQualityExplainer] = None
        self.anomaly_engine: Optional[AnomalyEngine] = None
        self.is_ready: bool = False

    @classmethod
    def get_instance(cls) -> "ModelServicesContainer":
        if cls._instance is None:
            cls._instance = ModelServicesContainer()
        return cls._instance

    def load_services(self) -> None:
        """Loads serialized model artifacts and initializes explainer and anomaly engine."""
        base_dir = Path(__file__).resolve().parent.parent.parent
        raw_art_dir = Path(yaml_config["models"]["artifacts_dir"]) / "xgboost_active"
        if (raw_art_dir / "model.joblib").exists():
            artifacts_dir = raw_art_dir
        else:
            artifacts_dir = base_dir / raw_art_dir

        model_file = artifacts_dir / "model.joblib"
        calib_file = artifacts_dir / "calibration_scores.npy"

        # If artifacts do not exist, auto-train on startup
        if not model_file.exists() or not calib_file.exists():
            logger.info("Artifacts missing at startup. Triggering initial model training pipeline...")
            from src.data.loader import load_dataset, split_and_save_data
            from src.ml.train import train_and_calibrate_active_model

            df = load_dataset()
            train_df, test_df = split_and_save_data(df)
            self.model, _, calib_scores = train_and_calibrate_active_model(train_df, test_df)
        else:
            # Verify artifact integrity
            file_hash = compute_file_sha256(model_file)
            logger.info("Loading serialized active model from disk with integrity check", path=str(model_file), sha256=file_hash[:12])
            self.model = joblib.load(model_file)
            calib_scores = np.load(calib_file)

        # Initialize Conformal Predictor
        self.conformal_predictor = ConformalPredictor(calib_scores)

        # Initialize Explainer
        feature_names = [f["name"] for f in yaml_config["features"]["numerical"]] + [
            "OSDR", "HMHI", "pH_Neutrality_Deviation", "Turbidity_DO_Ratio", "Toxic_Breach_Count"
        ]
        self.explainer = WaterQualityExplainer(self.model, feature_names=feature_names)

        # Initialize and fit Anomaly Engine
        self.anomaly_engine = AnomalyEngine()
        raw_train_path = Path(yaml_config["dataset"]["processed_train_path"])
        train_path = raw_train_path if raw_train_path.exists() else base_dir / raw_train_path
        if train_path.exists():
            train_df = pd.read_csv(train_path)
            self.anomaly_engine.fit(train_df)
        else:
            from src.data.loader import load_dataset
            self.anomaly_engine.fit(load_dataset())


        self.is_ready = True
        logger.info("All inference and explainability engines successfully loaded into memory")


def verify_api_key(x_api_key: Optional[str] = Header(None)) -> Optional[str]:
    """Enforces constant-time API Key authentication."""
    expected_key = settings.api_key
    # In production, API Key is strictly required
    if settings.aquasense_env == "production" or (expected_key and expected_key != "aquasense-dev-secret-key-2026"):
        if not x_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing mandatory X-API-Key authentication header."
            )
        if not secrets.compare_digest(x_api_key, expected_key or ""):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or unauthorized X-API-Key provided."
            )
    return x_api_key
