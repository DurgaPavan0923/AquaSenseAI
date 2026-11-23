"""Data leakage prevention tests across cross-validation splits and preprocessor pipelines."""

import numpy as np
import pandas as pd

from src.data.loader import generate_benchmark_dataset, split_and_save_data
from src.ml.preprocessor import build_preprocessor


def test_zero_leakage_between_train_and_test():
    """Asserts strict disjointness between train and test sample partitions."""
    df = generate_benchmark_dataset(n_samples=200, seed=42)
    train_df, test_df = split_and_save_data(df, config={
        "split": {"test_size": 0.25, "random_state": 42},
        "target": {"name": "Pollution Level"},
        "dataset": {"processed_train_path": "data/processed/train.csv", "processed_test_path": "data/processed/test.csv"}
    })

    # Assert no index overlap
    train_ids = set(train_df["Sample_ID"])
    test_ids = set(test_df["Sample_ID"])
    overlap = train_ids.intersection(test_ids)
    assert len(overlap) == 0, f"Critical Data Leakage: Found {len(overlap)} overlapping IDs between train and test sets."


def test_preprocessor_does_not_leak_test_statistics():
    """Verifies that imputer and scaler statistics are derived solely from training folds."""
    numerical_cols = [
        "pH", "Turbidity", "Temperature", "Dissolved Oxygen",
        "Biochemical Oxygen Demand", "Lead", "Mercury", "Arsenic"
    ]

    # Train set with low values
    train_data = pd.DataFrame({col: [1.0, 2.0, 3.0] for col in numerical_cols})
    # Test set with extreme values
    test_data = pd.DataFrame({col: [100.0, 200.0, 300.0] for col in numerical_cols})

    pipeline = build_preprocessor(numerical_cols)
    pipeline.fit(train_data)

    # Imputer statistics should equal median of train_data (2.0), unaffected by test_data
    imputer_stats = pipeline.named_steps["imputer"].statistics_
    assert np.allclose(imputer_stats, 2.0), f"Data Leakage in Imputer: Expected 2.0, got {imputer_stats}"
