"""
Data loading, validation, and leakage-safe dataset splitting for AquaSense AI.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_COLUMNS: list[str] = [
    "Sample_ID",
    "pH",
    "Turbidity",
    "Temperature",
    "Dissolved Oxygen",
    "Biochemical Oxygen Demand",
    "Lead",
    "Mercury",
    "Arsenic",
    "Pollution Level",
]

FEATURE_COLUMNS: list[str] = [
    "pH",
    "Turbidity",
    "Temperature",
    "Dissolved Oxygen",
    "Biochemical Oxygen Demand",
    "Lead",
    "Mercury",
    "Arsenic",
]

TARGET_COLUMN: str = "Pollution Level"


def load_raw_dataset(filepath: str | Path = "data/raw/water_quality.csv") -> pd.DataFrame:
    """Loads and validates the raw water quality dataset."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {path.resolve()}")

    df = pd.read_csv(path)

    # Verify column presence
    missing_cols = set(RAW_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Remove any duplicate rows prior to splitting
    initial_len = len(df)
    df = df.drop_duplicates(subset=FEATURE_COLUMNS).reset_index(drop=True)
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"Removed {dropped} duplicate feature rows before splitting.")

    return df


def audit_dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Generates structural summary of the audited dataset."""
    class_counts = df[TARGET_COLUMN].value_counts().to_dict()
    missing_per_col = df.isnull().sum().to_dict()

    return {
        "total_samples": int(len(df)),
        "columns": list(df.columns),
        "target_column": TARGET_COLUMN,
        "class_distribution": class_counts,
        "missing_values_count": int(df.isnull().sum().sum()),
        "missing_per_column": missing_per_col,
        "feature_summary": df[FEATURE_COLUMNS].describe().to_dict(),
    }


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs stratified train-test split on the deduplicated dataset.
    Preserves class balance without data leakage.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET_COLUMN],
        random_state=random_state,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_dataset_4way(
    df: pd.DataFrame,
    test_size: float = 0.20,
    prob_calib_size: float = 0.15,
    conformal_calib_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs statistically rigorous 4-way stratified partitioning:
    1. Independent holdout test partition (default: 20%, 200 samples)
    2. Probability calibration partition (Platt scaling, default: 15%, 150 samples)
    3. Conformal calibration partition (non-conformity quantile, default: 15%, 150 samples)
    4. Base model training partition (default: 50%, 500 samples)
    
    Guarantees that conformal calibration observations are never seen during
    base model training or probability calibrator parameter fitting.
    """
    # 1. Split off independent test partition
    dev_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET_COLUMN],
        random_state=random_state,
        shuffle=True,
    )

    # 2. From dev_df (80%), split off conformal calibration partition
    # 0.15 / 0.80 = 0.1875
    conf_frac = conformal_calib_size / (1.0 - test_size)
    rem_dev_df, conf_calib_df = train_test_split(
        dev_df,
        test_size=conf_frac,
        stratify=dev_df[TARGET_COLUMN],
        random_state=random_state,
        shuffle=True,
    )

    # 3. From remaining dev_df (65%), split off probability calibration partition
    # 0.15 / 0.65 = 0.230769...
    prob_frac = prob_calib_size / (1.0 - test_size - conformal_calib_size)
    train_df, prob_calib_df = train_test_split(
        rem_dev_df,
        test_size=prob_frac,
        stratify=rem_dev_df[TARGET_COLUMN],
        random_state=random_state,
        shuffle=True,
    )

    return (
        train_df.reset_index(drop=True),
        prob_calib_df.reset_index(drop=True),
        conf_calib_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def verify_no_leakage_4way(
    train_df: pd.DataFrame,
    prob_calib_df: pd.DataFrame,
    conf_calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> bool:
    """Verifies that all 4 partitions are strictly pairwise disjoint."""
    splits = {
        "train": train_df,
        "prob_calib": prob_calib_df,
        "conf_calib": conf_calib_df,
        "test": test_df,
    }

    names = list(splits.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            df1, df2 = splits[names[i]], splits[names[j]]
            if "Sample_ID" in df1.columns and "Sample_ID" in df2.columns:
                shared = set(df1["Sample_ID"]).intersection(set(df2["Sample_ID"]))
                if shared:
                    raise AssertionError(f"Leakage between {names[i]} and {names[j]}: {len(shared)} shared Sample_IDs.")
            
            f1 = df1[FEATURE_COLUMNS].round(6)
            f2 = df2[FEATURE_COLUMNS].round(6)
            merged = pd.merge(f1, f2, how="inner")
            if len(merged) > 0:
                raise AssertionError(f"Leakage between {names[i]} and {names[j]}: {len(merged)} identical rows.")

    return True


def verify_no_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """
    Checks that there is zero overlap between train and test splits:
    1. No shared Sample_IDs if present.
    2. No identical feature vectors.
    """
    if "Sample_ID" in train_df.columns and "Sample_ID" in test_df.columns:
        shared_ids = set(train_df["Sample_ID"]).intersection(set(test_df["Sample_ID"]))
        if shared_ids:
            raise AssertionError(f"Data leakage detected: {len(shared_ids)} shared Sample_IDs between splits.")

    train_feats = train_df[FEATURE_COLUMNS].round(6)
    test_feats = test_df[FEATURE_COLUMNS].round(6)

    merged = pd.merge(train_feats, test_feats, how="inner")
    if len(merged) > 0:
        raise AssertionError(f"Data leakage detected: {len(merged)} identical feature rows in train and test splits.")

    return True


def save_processed_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_path: str | Path = "data/processed/train.csv",
    test_path: str | Path = "data/processed/test.csv",
    prob_calib_df: pd.DataFrame | None = None,
    conf_calib_df: pd.DataFrame | None = None,
) -> None:
    """Saves split datasets to data/processed directory."""
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    if prob_calib_df is not None:
        prob_calib_df.to_csv(Path(train_path).parent / "prob_calib.csv", index=False)
    if conf_calib_df is not None:
        conf_calib_df.to_csv(Path(train_path).parent / "conformal_calib.csv", index=False)
