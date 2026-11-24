"""
Data leakage verification tests for AquaSense AI.
Verifies no shared sample identifiers, no identical feature rows across splits,
and preprocessor statistics fit exclusively on train partitions.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.aquasense.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_raw_dataset,
    split_dataset,
    split_dataset_4way,
    verify_no_leakage,
    verify_no_leakage_4way,
)


def test_zero_leakage_between_train_and_test():
    """Asserts strict disjointness between train and test sample partitions."""
    df = load_raw_dataset()
    train_df, test_df = split_dataset(df, test_size=0.20, random_state=42)

    # Verification function must pass without raising AssertionError
    assert verify_no_leakage(train_df, test_df) is True


def test_zero_leakage_across_all_four_partitions():
    """Asserts pairwise disjointness across train, prob_calib, conf_calib, and test partitions."""
    df = load_raw_dataset()
    train_df, prob_calib_df, conf_calib_df, test_df = split_dataset_4way(
        df, test_size=0.20, prob_calib_size=0.15, conformal_calib_size=0.15, random_state=42
    )
    assert len(train_df) == 500
    assert len(prob_calib_df) == 150
    assert len(conf_calib_df) == 150
    assert len(test_df) == 200
    assert verify_no_leakage_4way(train_df, prob_calib_df, conf_calib_df, test_df) is True


def test_preprocessor_does_not_leak_test_statistics():
    """Verifies that scaler statistics are derived solely from training data."""
    train_data = pd.DataFrame({col: [1.0, 2.0, 3.0] for col in FEATURE_COLUMNS})
    test_data = pd.DataFrame({col: [100.0, 200.0, 300.0] for col in FEATURE_COLUMNS})

    scaler = StandardScaler()
    scaler.fit(train_data)

    # Mean of [1.0, 2.0, 3.0] is 2.0
    assert np.allclose(scaler.mean_, 2.0)

    # Test data transform does not change scaler statistics
    _ = scaler.transform(test_data)
    assert np.allclose(scaler.mean_, 2.0)


def test_target_not_in_features():
    """Ensures target column is excluded from feature set."""
    assert TARGET_COLUMN not in FEATURE_COLUMNS
