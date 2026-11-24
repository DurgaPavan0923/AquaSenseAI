"""Pytest fixtures for unit and integration testing."""

import sys
from pathlib import Path

# Ensure root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.index import app
from src.aquasense.db import Base


def generate_synthetic_benchmark_dataset(path: Path) -> None:
    """Generates a deterministic 1,000-sample benchmark dataset for testing in CI."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = 1000
    rng = np.random.RandomState(42)
    y = np.repeat([0, 1, 2], [334, 333, 333])
    rng.shuffle(y)

    ph = np.where(
        y == 0,
        rng.uniform(7.0, 7.8, n_samples),
        np.where(y == 1, rng.uniform(6.5, 8.5, n_samples), rng.uniform(4.5, 6.2, n_samples)),
    )
    turb = np.where(
        y == 0,
        rng.uniform(1.0, 3.5, n_samples),
        np.where(y == 1, rng.uniform(4.0, 7.0, n_samples), rng.uniform(10.0, 45.0, n_samples)),
    )
    temp = rng.uniform(18.0, 28.0, n_samples)
    do = np.where(
        y == 0,
        rng.uniform(7.0, 9.0, n_samples),
        np.where(y == 1, rng.uniform(4.5, 6.5, n_samples), rng.uniform(1.5, 4.0, n_samples)),
    )
    bod = np.where(
        y == 0,
        rng.uniform(1.0, 2.5, n_samples),
        np.where(y == 1, rng.uniform(3.0, 5.0, n_samples), rng.uniform(6.0, 15.0, n_samples)),
    )
    lead = np.where(
        y == 0,
        rng.uniform(0.001, 0.008, n_samples),
        np.where(y == 1, rng.uniform(0.008, 0.012, n_samples), rng.uniform(0.015, 0.050, n_samples)),
    )
    mercury = np.where(
        y == 0,
        rng.uniform(0.0001, 0.0008, n_samples),
        np.where(y == 1, rng.uniform(0.0008, 0.0012, n_samples), rng.uniform(0.0015, 0.0050, n_samples)),
    )
    arsenic = np.where(
        y == 0,
        rng.uniform(0.001, 0.008, n_samples),
        np.where(y == 1, rng.uniform(0.008, 0.012, n_samples), rng.uniform(0.015, 0.050, n_samples)),
    )

    sample_ids = [f"WQS_{i+1:04d}" for i in range(n_samples)]
    df = pd.DataFrame(
        {
            "Sample_ID": sample_ids,
            "pH": np.round(ph, 3),
            "Turbidity": np.round(turb, 3),
            "Temperature": np.round(temp, 2),
            "Dissolved Oxygen": np.round(do, 3),
            "Biochemical Oxygen Demand": np.round(bod, 3),
            "Lead": np.round(lead, 5),
            "Mercury": np.round(mercury, 6),
            "Arsenic": np.round(arsenic, 5),
            "Pollution Level": y,
        }
    )
    df = df.drop_duplicates(
        subset=[
            "pH",
            "Turbidity",
            "Temperature",
            "Dissolved Oxygen",
            "Biochemical Oxygen Demand",
            "Lead",
            "Mercury",
            "Arsenic",
        ]
    ).reset_index(drop=True)
    df.to_csv(path, index=False)


@pytest.fixture(scope="session", autouse=True)
def ensure_raw_dataset_available():
    """Ensures raw dataset is available for integration tests in clean CI environments."""
    raw_path = Path("data/raw/water_quality.csv")
    if not raw_path.exists():
        generate_synthetic_benchmark_dataset(raw_path)


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
        "Arsenic": 0.004,
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
        "Lead": 0.035,
        "Mercury": 0.003,
        "Arsenic": 0.025,
    }


@pytest.fixture
def sample_dataframe(sample_feature_dict, contaminated_feature_dict):
    """Returns a DataFrame containing nominal and contaminated samples."""
    borderline = {
        "pH": 6.4,
        "Turbidity": 6.2,
        "Temperature": 24.0,
        "Dissolved Oxygen": 5.8,
        "Biochemical Oxygen Demand": 3.2,
        "Lead": 0.009,
        "Mercury": 0.0008,
        "Arsenic": 0.008,
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


@pytest.fixture
def api_client():
    """FastAPI TestClient for integration testing."""
    return TestClient(app)
