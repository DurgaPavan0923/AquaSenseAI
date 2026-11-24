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

STATUTORY_LIMITS: dict[str, dict[str, Any]] = {
    "pH": {
        "min": 6.5,
        "max": 8.5,
        "unit": "pH",
        "category": "Drinking & Surface Water",
        "authority": "WHO 4th Ed. / BIS IS 10500:2012 / US EPA Secondary Standard",
        "rationale": "pH < 6.5 accelerates pipe corrosion and heavy metal leaching; pH > 8.5 causes scaling, bitter taste, and reduced chlorination efficacy.",
    },
    "Turbidity": {
        "max": 5.0,
        "unit": "NTU",
        "category": "Drinking Water",
        "authority": "WHO Guidelines / BIS IS 10500:2012 Permissible Limit",
        "rationale": "High turbidity shields biological pathogens from disinfection and indicates particulate or colloidal suspension.",
    },
    "Dissolved Oxygen": {
        "min": 6.5,
        "unit": "mg/L",
        "category": "Surface Water & Aquatic Health",
        "authority": "CPCB Class A Surface Water / US EPA Freshwater Aquatic Life Criteria",
        "rationale": "DO below 6.5 mg/L indicates organic decomposition, hypoxia, and respiratory stress for aquatic organisms.",
    },
    "Biochemical Oxygen Demand": {
        "max": 2.0,
        "unit": "mg/L",
        "category": "Surface Water & Organic Load",
        "authority": "CPCB Class A Raw Water Source / European Environment Agency Benchmark",
        "rationale": "5-day BOD > 2.0 mg/L signifies untreated organic pollution, sewage discharge, or agricultural runoff.",
    },
    "Lead": {
        "max": 0.01,
        "unit": "mg/L",
        "category": "Drinking Water Toxicant",
        "authority": "WHO Guideline Value (0.01 mg/L) / BIS IS 10500:2012 / US EPA Action Level (0.015 mg/L)",
        "rationale": "Cumulative neurotoxicant causing cognitive impairment in children, hypertension, and nephrotoxicity.",
    },
    "Mercury": {
        "max": 0.001,
        "unit": "mg/L",
        "category": "Drinking Water Toxicant",
        "authority": "BIS IS 10500:2012 (0.001 mg/L) / WHO Guideline (0.006 mg/L inorganic) / US EPA MCL (0.002 mg/L)",
        "rationale": "Potent neurotoxin and renal toxicant with bioaccumulative methylmercury pathways in aquatic food chains.",
    },
    "Arsenic": {
        "max": 0.01,
        "unit": "mg/L",
        "category": "Drinking Water Toxicant",
        "authority": "WHO Provisional Guideline (0.01 mg/L) / BIS IS 10500:2012 / US EPA MCL (0.010 mg/L)",
        "rationale": "Group 1 human carcinogen causing vascular disease, skin lesions, and internal cancers upon chronic ingestion.",
    },
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


def audit_parameter_observations(sample_dict: dict[str, float]) -> list[dict[str, Any]]:
    """
    Generates granular scientific observations for each parameter with verified statutory citations.
    """
    observations: list[dict[str, Any]] = []
    for feat in FEATURE_ORDER:
        val = float(sample_dict.get(feat, 0.0))
        spec = STATUTORY_LIMITS.get(feat)
        if not spec:
            # Temperature
            observations.append(
                {
                    "parameter": feat,
                    "value": round(val, 2),
                    "unit": "deg C",
                    "status": "Ambient",
                    "authority": "Operational / Physical Characteristic",
                    "observation": f"Measured temperature of {val:.1f} deg C within normal ambient hydrological range.",
                }
            )
            continue

        unit = spec["unit"]
        status = "Compliant"
        note = f"Within standard benchmark criteria ({spec['authority']})."

        if "min" in spec and val < spec["min"]:
            status = "Deficient" if "Oxygen" in feat else "Below Permissible Limit"
            note = f"Measured {val:.4f} {unit} is below statutory minimum {spec['min']} {unit}. {spec['rationale']}"
        elif "max" in spec and val > spec["max"]:
            status = "Exceeded"
            note = f"Measured {val:.4f} {unit} exceeds statutory limit of {spec['max']} {unit}. {spec['rationale']}"

        observations.append(
            {
                "parameter": feat,
                "value": round(val, 5),
                "unit": unit,
                "status": status,
                "category": spec["category"],
                "authority": spec["authority"],
                "observation": note,
            }
        )
    return observations


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
            "parameter_observations": audit_parameter_observations(sample_dict),
        }
