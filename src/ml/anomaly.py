"""Dual-tier anomaly detection: physical sensor-fault heuristics plus unsupervised Isolation Forest."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.common.config import yaml_config
from src.common.logger import logger


class AnomalyEngine:
    """Combines deterministic sensor failure rules with an unsupervised Isolation Forest."""

    def __init__(self, contamination: float = 0.03, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            n_jobs=-1
        )
        self.is_fitted = False
        self.numerical_cols = [f["name"] for f in yaml_config["features"]["numerical"]]

    def fit(self, X: pd.DataFrame | np.ndarray) -> "AnomalyEngine":
        """Fits Isolation Forest on historical baseline feature distributions."""
        if isinstance(X, pd.DataFrame):
            df_feats = X[self.numerical_cols] if all(c in X.columns for c in self.numerical_cols) else X
            data = df_feats.to_numpy()
        else:
            data = X

        self.isolation_forest.fit(data)
        self.is_fitted = True
        logger.info("Isolation Forest fitted successfully", n_samples=len(data))
        return self

    def check_deterministic_rules(
        self,
        sample: dict[str, float],
        previous_sample: dict[str, float] | None = None
    ) -> tuple[bool, list[str], str]:
        """Checks for hardware sensor faults, impossible physics, and immediate toxic spikes.
        
        Returns:
            (has_fault, list_of_violations, severity_tier)
        """
        violations = []
        severity = "LOW"

        # 1. Physics boundaries (impossible measurements)
        ph = sample.get("pH", 7.0)
        turb = sample.get("Turbidity", 0.0)
        temp = sample.get("Temperature", 20.0)
        do = sample.get("Dissolved Oxygen", 7.0)
        bod = sample.get("Biochemical Oxygen Demand", 1.0)
        lead = sample.get("Lead", 0.0)
        mercury = sample.get("Mercury", 0.0)
        arsenic = sample.get("Arsenic", 0.0)

        if ph < 0.0 or ph > 14.0:
            violations.append(f"Physical failure: pH {ph} outside [0, 14] boundary.")
            severity = "CRITICAL"
        if turb < 0.0:
            violations.append(f"Physical failure: Negative turbidity ({turb} NTU).")
            severity = "HIGH"
        if do < 0.0:
            violations.append(f"Physical failure: Negative dissolved oxygen ({do} mg/L).")
            severity = "HIGH"
        if temp < -5.0 or temp > 60.0:
            violations.append(f"Environmental extreme: Temperature ({temp} °C) outside physical envelope.")
            severity = "MEDIUM"

        # 2. Rate-of-change / instantaneous sensor jumps (indicating air-locks or cable disconnects)
        if previous_sample:
            delta_limits = {
                "pH": 2.5,
                "Turbidity": 50.0,
                "Temperature": 10.0,
                "Dissolved Oxygen": 6.0
            }
            for param, max_delta in delta_limits.items():
                if param in sample and param in previous_sample:
                    delta = abs(sample[param] - previous_sample[param])
                    if delta > max_delta:
                        violations.append(
                            f"Sensor jump failure: {param} delta of {delta:.2f} exceeds dynamic threshold {max_delta}."
                        )
                        severity = "CRITICAL"

        # 3. Severe statutory toxic spikes
        if lead > 0.05:  # 5x statutory limit
            violations.append(f"EXTREME TOXICITY SPIKE: Lead concentration ({lead:.4f} mg/L) > 5x limit.")
            severity = "CRITICAL"
        if mercury > 0.005:  # 5x statutory limit
            violations.append(f"EXTREME TOXICITY SPIKE: Mercury concentration ({mercury:.5f} mg/L) > 5x limit.")
            severity = "CRITICAL"

        has_fault = len(violations) > 0
        return has_fault, violations, severity

    def score_sample(
        self,
        sample: dict[str, float],
        previous_sample: dict[str, float] | None = None
    ) -> dict[str, Any]:
        """Runs complete dual-tier anomaly audit on a single sample."""
        # Tier 1: Rules
        has_fault, violations, rule_severity = self.check_deterministic_rules(sample, previous_sample)

        # Tier 2: Isolation Forest
        if not self.is_fitted:
            # Fallback heuristic if not fitted yet
            is_if_outlier = False
            raw_score = 0.0
        else:
            feature_vector = np.array([[sample.get(col, 0.0) for col in self.numerical_cols]])
            raw_score = float(self.isolation_forest.decision_function(feature_vector)[0])
            pred = self.isolation_forest.predict(feature_vector)[0]
            is_if_outlier = bool(pred == -1)

        # Compound Decision
        is_anomaly = has_fault or is_if_outlier

        # Determine aggregate severity
        if rule_severity == "CRITICAL" or (is_if_outlier and raw_score < -0.20):
            aggregate_severity = "CRITICAL"
        elif rule_severity == "HIGH" or is_if_outlier:
            aggregate_severity = "HIGH"
        elif has_fault:
            aggregate_severity = "MEDIUM"
        else:
            aggregate_severity = "LOW"

        if is_if_outlier and not has_fault:
            violations.append("Multivariate statistical anomaly detected via Isolation Forest.")

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": raw_score,
            "has_sensor_fault": has_fault,
            "severity": aggregate_severity,
            "violations": violations
        }
