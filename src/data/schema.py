"""Data validation schemas and range assertions using Pydantic v2."""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from pydantic import BaseModel, Field, field_validator


class WaterSampleInput(BaseModel):
    """Pydantic model validating physical ranges and boundaries for a single water sample."""
    
    ph: float = Field(..., alias="pH", ge=0.0, le=14.0, description="pH level (0.0 to 14.0)")
    turbidity: float = Field(..., alias="Turbidity", ge=0.0, le=500.0, description="Turbidity in NTU")
    temperature: float = Field(..., alias="Temperature", ge=-5.0, le=60.0, description="Water temperature in °C")
    dissolved_oxygen: float = Field(..., alias="Dissolved Oxygen", ge=0.0, le=25.0, description="DO in mg/L")
    bod: float = Field(..., alias="Biochemical Oxygen Demand", ge=0.0, le=100.0, description="BOD in mg/L")
    lead: float = Field(..., alias="Lead", ge=0.0, le=5.0, description="Lead in mg/L")
    mercury: float = Field(..., alias="Mercury", ge=0.0, le=1.0, description="Mercury in mg/L")
    arsenic: float = Field(..., alias="Arsenic", ge=0.0, le=5.0, description="Arsenic in mg/L")
    
    # Optional metadata
    sample_id: Optional[str] = Field(default=None, alias="Sample_ID")
    location_id: Optional[str] = Field(default="STATION_MAIN", alias="Location_ID")
    timestamp: Optional[str] = Field(default=None, alias="Timestamp")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "pH": 7.2,
                "Turbidity": 2.1,
                "Temperature": 22.5,
                "Dissolved Oxygen": 7.8,
                "Biochemical Oxygen Demand": 1.5,
                "Lead": 0.005,
                "Mercury": 0.0002,
                "Arsenic": 0.003
            }
        }
    }

    @field_validator("ph")
    @classmethod
    def validate_ph(cls, v: float) -> float:
        if v < 0.0 or v > 14.0:
            raise ValueError(f"pH value {v} is outside possible physical boundaries [0, 14].")
        return round(v, 4)


def validate_dataframe(df: pd.DataFrame, expected_features: List[str]) -> Tuple[bool, List[str]]:
    """Validates an entire pandas DataFrame against required columns, nulls, and boundary conditions.
    
    Returns:
        (is_valid, list_of_error_messages)
    """
    errors: List[str] = []
    
    # 1. Check for empty dataframe
    if df.empty:
        return False, ["DataFrame is completely empty."]
    
    # 2. Check for missing columns
    missing_cols = [col for col in expected_features if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required feature columns: {missing_cols}")
        return False, errors
    
    # 3. Check for severe all-null rows
    null_rows = df[expected_features].isnull().all(axis=1)
    if null_rows.any():
        errors.append(f"Found {null_rows.sum()} rows where all feature values are null.")
    
    # 4. Check boundaries per feature
    range_bounds = {
        "pH": (0.0, 14.0),
        "Turbidity": (0.0, 500.0),
        "Temperature": (-5.0, 60.0),
        "Dissolved Oxygen": (0.0, 25.0),
        "Biochemical Oxygen Demand": (0.0, 100.0),
        "Lead": (0.0, 5.0),
        "Mercury": (0.0, 1.0),
        "Arsenic": (0.0, 5.0),
    }
    
    for feat, (lower, upper) in range_bounds.items():
        if feat in df.columns:
            invalid_lower = (df[feat] < lower).sum()
            invalid_upper = (df[feat] > upper).sum()
            if invalid_lower > 0:
                errors.append(f"Column '{feat}' has {invalid_lower} values below physical limit ({lower}).")
            if invalid_upper > 0:
                errors.append(f"Column '{feat}' has {invalid_upper} values above physical limit ({upper}).")
                
    return len(errors) == 0, errors
