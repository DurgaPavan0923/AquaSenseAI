"""Leakage-safe preprocessing pipelines, imputation, scaling, and diagnostic audit reporting."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.common.config import yaml_config
from src.common.logger import logger
from src.ml.features import DomainFeatureEngineer


class PreprocessingAuditLogger(BaseEstimator, TransformerMixin):
    """Transformer that inspects missing values and extreme outlier counts prior to transformations."""

    def __init__(self, report_path: str = "reports/preprocessing_report.json"):
        self.report_path = report_path
        self.audit_summary_: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame | np.ndarray, y: Optional[Any] = None) -> "PreprocessingAuditLogger":
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X)

        missing_per_col = df.isnull().sum().to_dict()
        
        # Outlier counts via IQR: Q1 - 1.5*IQR or Q3 + 1.5*IQR
        outlier_counts = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_counts[str(col)] = int(outliers)

        self.audit_summary_ = {
            "imputation_policy": "median",
            "scaling_policy": "RobustScaler (IQR-based to preserve heavy metal spikes)",
            "total_rows_inspected": len(df),
            "missing_values_detected": {str(k): int(v) for k, v in missing_per_col.items()},
            "iqr_outliers_flagged": outlier_counts
        }

        # Persist audit summary report
        try:
            out_file = Path(self.report_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(self.audit_summary_, f, indent=2)
            logger.info("Preprocessing audit report logged", path=self.report_path)
        except Exception as e:
            logger.warning("Could not persist preprocessing report", error=str(e))

        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        return X


def build_preprocessor(numerical_features: List[str]) -> Pipeline:
    """Builds a leakage-safe scikit-learn Pipeline with feature engineering, imputation, and scaling."""
    pipeline = Pipeline([
        ("audit", PreprocessingAuditLogger()),
        ("imputer", SimpleImputer(strategy="median")),
        ("feature_engineer", DomainFeatureEngineer()),
        ("scaler", RobustScaler(with_centering=True, with_scaling=True))
    ])
    return pipeline


def get_cv_splitter(config: Optional[dict] = None) -> Any:
    """Instantiates the leakage-safe cross-validation splitter defined in configuration."""
    cfg = config or yaml_config
    split_cfg = cfg["split"]
    strategy = split_cfg.get("strategy", "stratified_kfold")
    n_splits = split_cfg.get("n_splits", 5)
    random_state = split_cfg.get("random_state", 42)
    shuffle = split_cfg.get("shuffle", True)

    if strategy == "stratified_kfold":
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    elif strategy == "group_kfold":
        return GroupKFold(n_splits=n_splits)
    elif strategy == "time_series_split":
        return TimeSeriesSplit(n_splits=n_splits)
    else:
        logger.warning(f"Unrecognized split strategy '{strategy}'. Defaulting to StratifiedKFold.")
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
