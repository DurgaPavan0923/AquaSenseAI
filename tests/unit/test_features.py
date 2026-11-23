"""Unit tests for domain-specific WHO/BIS feature engineering formulations."""

import numpy as np
import pandas as pd
import pytest

from src.common.constants import WHO_BIS_STANDARDS
from src.ml.features import DomainFeatureEngineer


def test_domain_feature_engineering_math(sample_dataframe):
    """Verifies precision of mathematical formulations for engineered ratios."""
    transformer = DomainFeatureEngineer(epsilon=1e-5)
    transformed_df = transformer.fit_transform(sample_dataframe)

    # 1. Verify all new columns exist
    expected_cols = ["OSDR", "HMHI", "pH_Neutrality_Deviation", "Turbidity_DO_Ratio", "Toxic_Breach_Count"]
    for col in expected_cols:
        assert col in transformed_df.columns, f"Engineered column {col} missing from output."

    # 2. Mathematical checks on first row (nominal)
    row0 = sample_dataframe.iloc[0]
    expected_osdr = row0["Biochemical Oxygen Demand"] / (row0["Dissolved Oxygen"] + 1e-5)
    assert np.isclose(transformed_df.loc[0, "OSDR"], expected_osdr, atol=1e-4)

    expected_hmhi = (
        (row0["Lead"] / 0.01) +
        (row0["Mercury"] / 0.001) +
        (row0["Arsenic"] / 0.01)
    )
    assert np.isclose(transformed_df.loc[0, "HMHI"], expected_hmhi, atol=1e-4)

    expected_ph_dev = abs(row0["pH"] - 7.0)
    assert np.isclose(transformed_df.loc[0, "pH_Neutrality_Deviation"], expected_ph_dev, atol=1e-4)

    # Clean sample should have 0 toxic breaches
    assert transformed_df.loc[0, "Toxic_Breach_Count"] == 0


def test_contaminated_sample_breach_counter(contaminated_feature_dict):
    """Verifies that contaminated sample registers expected statutory threshold violations."""
    df = pd.DataFrame([contaminated_feature_dict])
    transformer = DomainFeatureEngineer()
    out = transformer.fit_transform(df)

    # Contaminated sample violates:
    # pH (5.2 < 6.5), Turbidity (28.5 > 5.0), DO (2.8 < 6.5), BOD (12.0 > 2.0),
    # Lead (0.035 > 0.01), Mercury (0.003 > 0.001), Arsenic (0.025 > 0.01) -> 7 breaches
    assert out.loc[0, "Toxic_Breach_Count"] >= 5
    assert out.loc[0, "HMHI"] > 5.0  # Normalized sum exceeds threshold substantially


def test_numpy_array_input_support():
    """Verifies transformer handles raw 2D numpy arrays without crashing."""
    raw_array = np.array([
        [7.2, 3.5, 22.0, 7.5, 1.8, 0.002, 0.0001, 0.002],
        [6.0, 15.0, 26.0, 4.0, 6.0, 0.020, 0.0020, 0.020]
    ])
    transformer = DomainFeatureEngineer()
    out = transformer.fit_transform(raw_array)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == 2
    assert "OSDR" in out.columns
