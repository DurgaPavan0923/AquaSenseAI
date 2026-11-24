"""Unit tests for preprocessing pipeline and domain feature engineering."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.aquasense.features import DomainFeatureEngineer


def test_preprocessing_pipeline_shape():
    """Verifies pipeline expands 8 base features to 13 scaled features."""
    pipe = Pipeline(
        [
            ("features", DomainFeatureEngineer()),
            ("scaler", StandardScaler()),
        ]
    )

    data = pd.DataFrame(
        {
            "pH": [7.0, 7.2, 7.4],
            "Turbidity": [2.0, 3.0, 4.0],
            "Temperature": [20.0, 21.0, 22.0],
            "Dissolved Oxygen": [8.0, 7.5, 7.0],
            "Biochemical Oxygen Demand": [1.5, 1.2, 1.8],
            "Lead": [0.002, 0.003, 0.004],
            "Mercury": [0.0001, 0.0002, 0.0003],
            "Arsenic": [0.002, 0.002, 0.002],
        }
    )

    transformed = pipe.fit_transform(data)
    assert transformed.shape == (3, 13)
    assert not np.isnan(transformed).any()
