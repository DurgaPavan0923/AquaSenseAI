"""
Domain-guided feature engineering for AquaSense AI based on WHO and BIS IS 10500:2012 guidelines.
Compatible with both NumPy arrays (serverless runtime) and pandas DataFrames (offline training).
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

WHO_BIS_STANDARDS = {
    "pH": {"min": 6.5, "max": 8.5},
    "Turbidity": {"max": 5.0},
    "Dissolved Oxygen": {"min": 6.5},
    "Biochemical Oxygen Demand": {"clean_max": 2.0},
    "Lead": {"max": 0.01},
    "Mercury": {"max": 0.001},
    "Arsenic": {"max": 0.01},
}

BASE_FEATURE_NAMES: list[str] = [
    "pH",
    "Turbidity",
    "Temperature",
    "Dissolved Oxygen",
    "Biochemical Oxygen Demand",
    "Lead",
    "Mercury",
    "Arsenic",
]

ENGINEERED_FEATURE_NAMES: list[str] = [
    "OSDR",
    "HMHI",
    "pH_Neutrality_Deviation",
    "Turbidity_DO_Ratio",
    "Toxic_Breach_Count",
]


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms raw physicochemical water parameters into domain ratio and toxicological features.
    Accepts both pd.DataFrame and np.ndarray inputs.
    """

    def __init__(self, epsilon: float = 1e-5):
        self.epsilon = epsilon
        self.feature_names_in_: list[str] | None = None

    def fit(self, X: Any, y: Any = None) -> "DomainFeatureEngineer":
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = BASE_FEATURE_NAMES
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Transforms input into base features plus 5 domain-engineered features as a 2D numpy array."""
        is_df = hasattr(X, "columns")

        if is_df:
            ph = np.asarray(X["pH"], dtype=float)
            turb = np.asarray(X["Turbidity"], dtype=float)
            temp = np.asarray(X["Temperature"], dtype=float)
            do = np.asarray(X["Dissolved Oxygen"], dtype=float)
            bod = np.asarray(X["Biochemical Oxygen Demand"], dtype=float)
            pb = np.asarray(X["Lead"], dtype=float)
            hg = np.asarray(X["Mercury"], dtype=float)
            as_metal = np.asarray(X["Arsenic"], dtype=float)
        else:
            arr = np.asarray(X, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            ph = arr[:, 0]
            turb = arr[:, 1]
            temp = arr[:, 2]
            do = arr[:, 3]
            bod = arr[:, 4]
            pb = arr[:, 5]
            hg = arr[:, 6]
            as_metal = arr[:, 7]

        # 1. Oxygen Saturation Deficit Ratio (OSDR)
        osdr = bod / (do + self.epsilon)

        # 2. Heavy Metal Hazard Index (HMHI)
        hmhi = (
            (pb / WHO_BIS_STANDARDS["Lead"]["max"])
            + (hg / WHO_BIS_STANDARDS["Mercury"]["max"])
            + (as_metal / WHO_BIS_STANDARDS["Arsenic"]["max"])
        )

        # 3. pH Neutrality Deviation
        ph_dev = np.abs(ph - 7.0)

        # 4. Turbidity-DO Ratio
        turb_clipped = np.clip(turb, 0.0, None)
        turb_do_ratio = np.log10(turb_clipped + 1.0) / (do + self.epsilon)

        # 5. Statutory Breach Count
        breaches = (
            (ph < WHO_BIS_STANDARDS["pH"]["min"]).astype(float)
            + (ph > WHO_BIS_STANDARDS["pH"]["max"]).astype(float)
            + (turb > WHO_BIS_STANDARDS["Turbidity"]["max"]).astype(float)
            + (do < WHO_BIS_STANDARDS["Dissolved Oxygen"]["min"]).astype(float)
            + (bod > WHO_BIS_STANDARDS["Biochemical Oxygen Demand"]["clean_max"]).astype(float)
            + (pb > WHO_BIS_STANDARDS["Lead"]["max"]).astype(float)
            + (hg > WHO_BIS_STANDARDS["Mercury"]["max"]).astype(float)
            + (as_metal > WHO_BIS_STANDARDS["Arsenic"]["max"]).astype(float)
        )

        # Stack into 2D array: base features (8) + engineered (5) = 13 features
        output = np.column_stack([
            ph, turb, temp, do, bod, pb, hg, as_metal,
            osdr, hmhi, ph_dev, turb_do_ratio, breaches
        ])
        return output

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        base = list(input_features) if input_features is not None else BASE_FEATURE_NAMES
        return np.array(base + ENGINEERED_FEATURE_NAMES)
