"""Statutory drinking-water quality thresholds based on WHO and BIS IS 10500:2012 standards."""

# WHO Guidelines for Drinking-water Quality (4th ed.) and BIS IS 10500:2012 parameters
WHO_BIS_STANDARDS = {
    "pH": {"min": 6.5, "max": 8.5, "unit": "dimensionless"},
    "Turbidity": {"min": 0.0, "max": 5.0, "ideal_max": 1.0, "unit": "NTU"},
    "Temperature": {"min": 10.0, "max": 30.0, "thermal_risk": 32.0, "unit": "°C"},
    "Dissolved Oxygen": {"min": 6.5, "hypoxic_limit": 4.0, "unit": "mg/L"},
    "Biochemical Oxygen Demand": {"clean_max": 2.0, "polluted_min": 5.0, "unit": "mg/L"},
    "Lead": {"max": 0.01, "unit": "mg/L"},
    "Mercury": {"max": 0.001, "unit": "mg/L"},
    "Arsenic": {"max": 0.01, "unit": "mg/L"},
}

TARGET_CLASS_MAPPING = {
    0: "Good",
    1: "Moderate",
    2: "Polluted"
}

CLASS_NAME_TO_INT = {v: k for k, v in TARGET_CLASS_MAPPING.items()}

SEVERITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

STATUTORY_DISCLAIMER = (
    "Screening support only; not a substitute for certified laboratory analysis. "
    "AquaSense AI outputs are probabilistic screening indicators designed to triage "
    "environmental risk and cannot replace analytical assays certified under ISO/IEC 17025."
)
