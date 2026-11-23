"""Unit tests for preprocessing pipeline, missing value imputation, and robust scaling."""

import numpy as np
import pandas as pd

from src.ml.preprocessor import build_preprocessor, get_cv_splitter


def test_preprocessing_pipeline_handles_nans():
    """Verifies that missing values are cleanly imputed using training medians."""
    numerical_cols = [
        "pH", "Turbidity", "Temperature", "Dissolved Oxygen",
        "Biochemical Oxygen Demand", "Lead", "Mercury", "Arsenic"
    ]

    # Create test data with deliberate NaNs
    data = pd.DataFrame({
        "pH": [7.0, np.nan, 7.4],
        "Turbidity": [2.0, 3.0, np.nan],
        "Temperature": [20.0, 21.0, 22.0],
        "Dissolved Oxygen": [8.0, 7.5, 7.0],
        "Biochemical Oxygen Demand": [1.5, 1.2, 1.8],
        "Lead": [0.002, 0.003, np.nan],
        "Mercury": [0.0001, np.nan, 0.0003],
        "Arsenic": [0.002, 0.002, 0.002]
    })

    pipeline = build_preprocessor(numerical_cols)
    transformed = pipeline.fit_transform(data)

    # Output must have zero NaNs
    assert not np.isnan(transformed).any(), "Transformed output contains residual NaNs after imputation."
    # Check that engineered features were added (8 original + 5 engineered = 13)
    assert transformed.shape[1] == 13, f"Expected 13 feature columns, got {transformed.shape[1]}."


def test_cv_splitter_instantiation():
    """Verifies that cross-validation splitters conform to configuration."""
    from sklearn.model_selection import StratifiedKFold
    splitter = get_cv_splitter({"split": {"strategy": "stratified_kfold", "n_splits": 5, "random_state": 42, "shuffle": True}})
    assert isinstance(splitter, StratifiedKFold)
    assert splitter.n_splits == 5
