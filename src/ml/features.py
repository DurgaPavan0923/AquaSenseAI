"""Domain-guided feature engineering based on WHO and BIS IS 10500:2012 guidelines."""


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.common.constants import WHO_BIS_STANDARDS


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom Scikit-Learn Transformer for environmental & toxicological ratio features."""

    def __init__(self, epsilon: float = 1e-5):
        self.epsilon = epsilon
        self.feature_names_in_: list[str] | None = None
        self.engineered_feature_names_: list[str] = [
            "OSDR",                     # Oxygen Saturation Deficit Ratio: BOD / (DO + eps)
            "HMHI",                     # Heavy Metal Hazard Index: normalized sum of Pb, Hg, As
            "pH_Neutrality_Deviation",   # |pH - 7.0|
            "Turbidity_DO_Ratio",       # log10(Turbidity + 1) / (DO + eps)
            "Toxic_Breach_Count"        # Count of statutory limit exceedances
        ]

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None) -> "DomainFeatureEngineer":
        """Fits transformer and registers feature names."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Computes domain ratio and exceedance features."""
        if not isinstance(X, pd.DataFrame):
            # Convert numpy array to DataFrame using recorded or default names
            cols = self.feature_names_in_ or [
                "pH", "Turbidity", "Temperature", "Dissolved Oxygen",
                "Biochemical Oxygen Demand", "Lead", "Mercury", "Arsenic"
            ]
            df = pd.DataFrame(X, columns=cols[:X.shape[1]])
        else:
            df = X.copy()

        # 1. Oxygen Saturation Deficit Ratio (OSDR)
        # High BOD coupled with depleted DO indicates severe organic wastewater decomposition
        bod = df["Biochemical Oxygen Demand"] if "Biochemical Oxygen Demand" in df.columns else 0.0
        do = df["Dissolved Oxygen"] if "Dissolved Oxygen" in df.columns else 1.0
        df["OSDR"] = bod / (do + self.epsilon)

        # 2. Heavy Metal Hazard Index (HMHI)
        # Normalizes each heavy metal to its statutory WHO/BIS maximum limit (0.01 for Pb, 0.001 for Hg, 0.01 for As)
        pb = df["Lead"] if "Lead" in df.columns else 0.0
        hg = df["Mercury"] if "Mercury" in df.columns else 0.0
        as_metal = df["Arsenic"] if "Arsenic" in df.columns else 0.0
        df["HMHI"] = (
            (pb / WHO_BIS_STANDARDS["Lead"]["max"]) +
            (hg / WHO_BIS_STANDARDS["Mercury"]["max"]) +
            (as_metal / WHO_BIS_STANDARDS["Arsenic"]["max"])
        )

        # 3. pH Neutrality Deviation
        # Quantifies acid-base buffer collapse
        ph = df["pH"] if "pH" in df.columns else 7.0
        df["pH_Neutrality_Deviation"] = np.abs(ph - 7.0)

        # 4. Turbidity-DO Risk Index
        turb = df["Turbidity"] if "Turbidity" in df.columns else 0.0
        df["Turbidity_DO_Ratio"] = np.log10(np.clip(turb, 0, None) + 1.0) / (do + self.epsilon)

        # 5. Statutory Threshold Exceedance Count
        breaches = (
            (ph < WHO_BIS_STANDARDS["pH"]["min"]).astype(int) +
            (ph > WHO_BIS_STANDARDS["pH"]["max"]).astype(int) +
            (turb > WHO_BIS_STANDARDS["Turbidity"]["max"]).astype(int) +
            (do < WHO_BIS_STANDARDS["Dissolved Oxygen"]["min"]).astype(int) +
            (bod > WHO_BIS_STANDARDS["Biochemical Oxygen Demand"]["clean_max"]).astype(int) +
            (pb > WHO_BIS_STANDARDS["Lead"]["max"]).astype(int) +
            (hg > WHO_BIS_STANDARDS["Mercury"]["max"]).astype(int) +
            (as_metal > WHO_BIS_STANDARDS["Arsenic"]["max"]).astype(int)
        )
        df["Toxic_Breach_Count"] = breaches

        return df

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Returns complete list of feature names including engineered metrics."""
        base_features = list(input_features) if input_features is not None else (self.feature_names_in_ or [])
        return np.array(base_features + self.engineered_feature_names_)
