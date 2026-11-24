"""Unit tests for domain-specific WHO/BIS feature engineering formulations."""

import numpy as np
import pandas as pd

from src.aquasense.features import (
    DomainFeatureEngineer,
)


def test_domain_feature_engineering_math(sample_dataframe):
    """Verifies precision of mathematical formulations for engineered ratios."""
    transformer = DomainFeatureEngineer(epsilon=1e-5)
    transformed_arr = transformer.fit_transform(sample_dataframe)

    # Output shape should be (N, 13)
    assert transformed_arr.shape == (len(sample_dataframe), 13)

    # Mathematical checks on first row (nominal)
    row0 = sample_dataframe.iloc[0]
    expected_osdr = row0["Biochemical Oxygen Demand"] / (row0["Dissolved Oxygen"] + 1e-5)
    # OSDR is at index 8
    assert np.isclose(transformed_arr[0, 8], expected_osdr, atol=1e-4)

    expected_hmhi = (
        (row0["Lead"] / 0.01)
        + (row0["Mercury"] / 0.001)
        + (row0["Arsenic"] / 0.01)
    )
    # HMHI is at index 9
    assert np.isclose(transformed_arr[0, 9], expected_hmhi, atol=1e-4)

    # pH neutrality deviation at index 10
    expected_ph_dev = abs(row0["pH"] - 7.0)
    assert np.isclose(transformed_arr[0, 10], expected_ph_dev, atol=1e-4)

    # Clean sample toxic breach count at index 12 should be 0
    assert transformed_arr[0, 12] == 0.0


def test_contaminated_sample_breach_counter(contaminated_feature_dict):
    """Verifies that contaminated sample registers expected statutory threshold violations."""
    df = pd.DataFrame([contaminated_feature_dict])
    transformer = DomainFeatureEngineer()
    out = transformer.fit_transform(df)

    # Toxic breach count at index 12
    assert out[0, 12] >= 5.0
    # HMHI at index 9
    assert out[0, 9] > 5.0


def test_numpy_array_input_support():
    """Verifies transformer handles raw 2D numpy arrays seamlessly (serverless runtime compatibility)."""
    raw_array = np.array([
        [7.2, 3.5, 22.0, 7.5, 1.8, 0.002, 0.0001, 0.002],
        [6.0, 15.0, 26.0, 4.0, 6.0, 0.020, 0.0020, 0.020],
    ])
    transformer = DomainFeatureEngineer()
    out = transformer.fit_transform(raw_array)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 13)
