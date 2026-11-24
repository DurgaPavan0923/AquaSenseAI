"""
Anomaly detection and sensor health auditing for AquaSense AI.
Combines Isolation Forest statistical density estimation with physical and regulatory threshold rules.
"""

from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

PHYSICAL_BOUNDS: dict[str, dict[str, float]] = {
    "pH": {"min": 0.0, "max": 14.0},
    "Turbidity": {"min": 0.0, "max": 500.0},
    "Temperature": {"min": -5.0, "max": 60.0},
    "Dissolved Oxygen": {"min": 0.0, "max": 25.0},
    "Biochemical Oxygen Demand": {"min": 0.0, "max": 100.0},
    "Lead": {"min": 0.0, "max": 5.0},
    "Mercury": {"min": 0.0, "max": 1.0},
    "Arsenic": {"min": 0.0, "max": 5.0},
}

STATUTORY_LIMITS: dict[str, dict[str, float]] = {
    "pH": {"min": 6.5, "max": 8.5},
    "Turbidity": {"max": 5.0},
    "Dissolved Oxygen": {"min": 6.5},
    "Biochemical Oxygen Demand": {"max": 2.0},
    "Lead": {"max": 0.01},
    "Mercury": {"max": 0.001},
    "Arsenic": {"max": 0.01},
}

FEATURE_ORDER: list[str] = [
    "pH",
    "Turbidity",
    "Temperature",
    "Dissolved Oxygen",
    "Biochemical Oxygen Demand",
    "Lead",
    "Mercury",
    "Arsenic",
]


class AnomalyDetector:
    """
    Two-stage anomaly detector:
    Stage 1: Physical domain and statutory limit validation.
    Stage 2: Unsupervised statistical isolation forest density estimation.
    """

    def __init__(self, contamination: float = 0.03, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        """Fits the Isolation Forest on clean training samples."""
        if hasattr(X, "values"):
            X = X.values
        self.model.fit(X)
        self.is_fitted = True
        return self

    def audit_sample(self, sample_dict: dict[str, float]) -> dict[str, Any]:
        """
        Audits a single sample across physical limits, statutory limits, and statistical anomaly score.
        """
        physical_violations: list[str] = []
        statutory_violations: list[str] = []

        # Check physical sensor bounds
        for feat, bounds in PHYSICAL_BOUNDS.items():
            val = float(sample_dict.get(feat, 0.0))
            if "min" in bounds and val < bounds["min"]:
                physical_violations.append(f"{feat} below physical minimum {bounds['min']} (got {val:.4f})")
            if "max" in bounds and val > bounds["max"]:
                physical_violations.append(f"{feat} above physical maximum {bounds['max']} (got {val:.4f})")

        # Check statutory limits (WHO / BIS)
        for feat, limits in STATUTORY_LIMITS.items():
            val = float(sample_dict.get(feat, 0.0))
            if "min" in limits and val < limits["min"]:
                statutory_violations.append(f"{feat} below potable threshold {limits['min']} (got {val:.4f})")
            if "max" in limits and val > limits["max"]:
                statutory_violations.append(f"{feat} exceeds maximum statutory limit {limits['max']} (got {val:.4f})")

        # Check statistical isolation score if fitted
        is_statistical_anomaly = False
        anomaly_score = 0.0
        if self.is_fitted:
            row_vals = np.array([[float(sample_dict.get(f, 0.0)) for f in FEATURE_ORDER]])
            pred = self.model.predict(row_vals)[0]
            anomaly_score = float(self.model.score_samples(row_vals)[0])
            is_statistical_anomaly = bool(pred == -1)

        # Determine overall severity
        if physical_violations:
            severity = "CRITICAL"
            status_message = "Physical sensor fault or uncalibrated reading detected."
        elif statutory_violations or is_statistical_anomaly:
            severity = "WARNING"
            status_message = "Potential water hazard or unusual sample profile detected."
        else:
            severity = "NORMAL"
            status_message = "All parameters within standard expected ranges."

        return {
            "severity": severity,
            "status_message": status_message,
            "is_anomaly": bool(severity != "NORMAL"),
            "is_statistical_anomaly": is_statistical_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "physical_violations": physical_violations,
            "statutory_violations": statutory_violations,
        }
