"""Pytest fixtures for unit and integration testing."""

import pytest
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base
from src.ml.features import DomainFeatureEngineer
from src.ml.preprocessor import build_preprocessor


@pytest.fixture
def sample_feature_dict():
    """Returns a single clean nominal water sample dictionary."""
    return {
        "pH": 7.3,
        "Turbidity": 2.1,
        "Temperature": 21.5,
        "Dissolved Oxygen": 7.8,
        "Biochemical Oxygen Demand": 1.4,
        "Lead": 0.003,
        "Mercury": 0.0002,
        "Arsenic": 0.004
    }


@pytest.fixture
def contaminated_feature_dict():
    """Returns a highly contaminated water sample exceeding multiple statutory limits."""
    return {
        "pH": 5.2,
        "Turbidity": 28.5,
        "Temperature": 27.0,
        "Dissolved Oxygen": 2.8,
        "Biochemical Oxygen Demand": 12.0,
        "Lead": 0.035,   # > 0.01 limit
        "Mercury": 0.003, # > 0.001 limit
        "Arsenic": 0.025  # > 0.01 limit
    }


@pytest.fixture
def sample_dataframe(sample_feature_dict, contaminated_feature_dict):
    """Returns a DataFrame containing nominal, borderline, and contaminated samples."""
    borderline = {
        "pH": 6.4,
        "Turbidity": 6.2,
        "Temperature": 24.0,
        "Dissolved Oxygen": 5.8,
        "Biochemical Oxygen Demand": 3.2,
        "Lead": 0.009,
        "Mercury": 0.0008,
        "Arsenic": 0.008
    }
    return pd.DataFrame([sample_feature_dict, borderline, contaminated_feature_dict])


@pytest.fixture
def in_memory_db():
    """Creates an isolated in-memory SQLite database for test transactions."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
