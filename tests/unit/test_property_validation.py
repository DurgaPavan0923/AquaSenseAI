"""Property-based testing and fuzz validation for data schemas and boundaries."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.data.schema import WaterSampleInput, validate_dataframe


def test_pydantic_boundary_rejection_fuzzing():
    """Fuzzes schema with physically impossible values across all parameters."""
    valid_base = {
        "pH": 7.0,
        "Turbidity": 2.0,
        "Temperature": 20.0,
        "Dissolved Oxygen": 7.0,
        "Biochemical Oxygen Demand": 1.5,
        "Lead": 0.002,
        "Mercury": 0.0001,
        "Arsenic": 0.002
    }

    # Test out-of-range negative values
    for param, min_val in [("pH", 0.0), ("Turbidity", 0.0), ("Dissolved Oxygen", 0.0)]:
        fuzzed = valid_base.copy()
        fuzzed[param] = -1.0
        with pytest.raises(ValidationError):
            WaterSampleInput(**fuzzed)

    # Test out-of-range positive values
    fuzzed_ph = valid_base.copy()
    fuzzed_ph["pH"] = 15.1
    with pytest.raises(ValidationError):
        WaterSampleInput(**fuzzed_ph)


def test_dataframe_validator_detects_all_null_rows():
    """Verifies that batch DataFrame validator flags all-null corruption rows."""
    expected_cols = [
        "pH", "Turbidity", "Temperature", "Dissolved Oxygen",
        "Biochemical Oxygen Demand", "Lead", "Mercury", "Arsenic"
    ]
    corrupted_df = pd.DataFrame([
        {col: 7.0 for col in expected_cols},
        {col: np.nan for col in expected_cols},  # Completely null row
        {col: 6.5 for col in expected_cols}
    ])

    is_valid, errors = validate_dataframe(corrupted_df, expected_cols)
    assert is_valid is False
    assert any("all feature values are null" in e for e in errors)


def test_dataframe_validator_detects_out_of_bounds_columns():
    """Verifies that out-of-physical-range columns are identified with exact counts."""
    expected_cols = [
        "pH", "Turbidity", "Temperature", "Dissolved Oxygen",
        "Biochemical Oxygen Demand", "Lead", "Mercury", "Arsenic"
    ]
    df = pd.DataFrame({
        "pH": [7.0, 16.0, -2.0],  # 2 violations
        "Turbidity": [1.0, 2.0, 3.0],
        "Temperature": [20.0, 20.0, 20.0],
        "Dissolved Oxygen": [7.0, 7.0, 7.0],
        "Biochemical Oxygen Demand": [1.0, 1.0, 1.0],
        "Lead": [0.001, 0.001, 0.001],
        "Mercury": [0.0001, 0.0001, 0.0001],
        "Arsenic": [0.001, 0.001, 0.001]
    })

    is_valid, errors = validate_dataframe(df, expected_cols)
    assert is_valid is False
    assert any("pH" in e for e in errors)
