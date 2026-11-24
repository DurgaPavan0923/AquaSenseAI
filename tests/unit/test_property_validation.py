"""Property-based testing and boundary fuzz validation for AquaSense AI."""

import pytest
from pydantic import ValidationError

from api.index import WaterSampleInput


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
        "Arsenic": 0.002,
    }

    # Test out-of-range negative values
    for param in ["pH", "Turbidity", "Dissolved Oxygen", "Biochemical Oxygen Demand"]:
        fuzzed = valid_base.copy()
        fuzzed[param] = -1.0
        with pytest.raises(ValidationError):
            WaterSampleInput(**fuzzed)

    # Test out-of-range positive pH (> 14.0)
    fuzzed_ph = valid_base.copy()
    fuzzed_ph["pH"] = 15.1
    with pytest.raises(ValidationError):
        WaterSampleInput(**fuzzed_ph)


def test_valid_sample_conversion():
    """Tests to_array and to_dict helpers on valid inputs."""
    sample = WaterSampleInput(
        pH=7.4,
        Turbidity=3.0,
        Temperature=21.0,
        **{"Dissolved Oxygen": 6.8, "Biochemical Oxygen Demand": 1.9},
        Lead=0.004,
        Mercury=0.0002,
        Arsenic=0.003,
    )
    arr = sample.to_array()
    assert len(arr) == 8
    assert arr[0] == 7.4
    d = sample.to_dict()
    assert d["pH"] == 7.4
    assert d["Dissolved Oxygen"] == 6.8
