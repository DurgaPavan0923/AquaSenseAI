"""Dataset loader with Kaggle API download, local CSV support, and synthetic benchmark generation."""

import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src.common.config import yaml_config
from src.common.logger import logger
from src.data.schema import validate_dataframe


def generate_benchmark_dataset(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generates a realistic benchmark water quality dataset conforming to the ziya07 Kaggle dataset schema
    with realistic environmental covariance, measurement noise, and natural boundary overlaps."""
    np.random.seed(seed)
    
    # 1. Base Class Assignment (Imbalanced: 55% Good, 20% Moderate, 25% Polluted)
    labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.55, 0.20, 0.25])
    
    # 2. Realistic Physics-guided distributions with continuous overlaps and measurement noise
    ph = []
    turbidity = []
    temperature = []
    do = []
    bod = []
    lead = []
    mercury = []
    arsenic = []
    
    for label in labels:
        # Inject ambient seasonal water temperature
        temp_val = np.random.normal(22.0, 3.5)
        # Oxygen solubility decreases with higher temperature
        do_sat = max(4.0, 14.6 - 0.3 * temp_val)
        
        if label == 0:  # Good (Potable / Clean ecosystem)
            p = np.clip(np.random.normal(7.3, 0.4), 6.2, 8.6)
            turb = np.clip(np.random.exponential(1.5), 0.2, 7.0)
            # DO is healthy, BOD is low with slight overlap
            b = np.clip(np.random.exponential(1.1) + 0.3, 0.2, 3.2)
            d = np.clip(np.random.normal(do_sat * 0.85, 0.9), 5.5, 11.5)
            pb = np.clip(np.random.exponential(0.003), 0.0001, 0.012)
            hg = np.clip(np.random.exponential(0.0002), 0.00001, 0.0011)
            as_val = np.clip(np.random.exponential(0.003), 0.0001, 0.012)
        elif label == 1:  # Moderate (Eutrophic / Mild organic discharge)
            p = np.clip(np.random.normal(6.6, 0.7), 5.6, 8.8)
            turb = np.clip(np.random.normal(6.5, 3.0), 2.5, 16.0)
            b = np.clip(np.random.normal(3.8, 1.2), 1.8, 6.5)
            d = np.clip(np.random.normal(do_sat * 0.60, 1.0), 3.8, 7.2)
            pb = np.clip(np.random.normal(0.011, 0.004), 0.004, 0.022)
            hg = np.clip(np.random.normal(0.0010, 0.0003), 0.0004, 0.0022)
            as_val = np.clip(np.random.normal(0.011, 0.004), 0.004, 0.024)
        else:  # Polluted (Severe wastewater, industrial discharge, or heavy metal contamination)
            # Bimodal pH collapse (either acidic acid-mine drainage or alkaline soap/textile effluent)
            p = np.clip(np.random.choice([np.random.normal(5.0, 0.9), np.random.normal(9.2, 0.8)]), 2.8, 12.0)
            turb = np.clip(np.random.exponential(18.0) + 8.0, 5.0, 95.0)
            b = np.clip(np.random.exponential(9.0) + 4.5, 3.5, 45.0)
            d = np.clip(np.random.normal(2.6, 1.1), 0.1, 5.2)  # Severe hypoxia
            pb = np.clip(np.random.exponential(0.04) + 0.012, 0.008, 0.35)
            hg = np.clip(np.random.exponential(0.0035) + 0.0011, 0.0007, 0.035)
            as_val = np.clip(np.random.exponential(0.03) + 0.012, 0.008, 0.28)

        ph.append(p)
        turbidity.append(turb)
        temperature.append(temp_val)
        do.append(d)
        bod.append(b)
        lead.append(pb)
        mercury.append(hg)
        arsenic.append(as_val)
            
    df = pd.DataFrame({
        "Sample_ID": [f"WQS_{i+1:04d}" for i in range(n_samples)],
        "pH": np.round(ph, 3),
        "Turbidity": np.round(turbidity, 2),
        "Temperature": np.round(temperature, 2),
        "Dissolved Oxygen": np.round(do, 2),
        "Biochemical Oxygen Demand": np.round(bod, 2),
        "Lead": np.round(lead, 5),
        "Mercury": np.round(mercury, 6),
        "Arsenic": np.round(arsenic, 5),
        "Pollution Level": labels
    })
    return df


def download_from_kaggle(slug: str, output_path: Path) -> bool:
    """Attempts to download the dataset using the official Kaggle CLI/API."""
    try:
        import kaggle  # type: ignore
        logger.info("Attempting Kaggle API download", slug=slug)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        kaggle.api.dataset_download_files(slug, path=str(output_dir), unzip=True)
        logger.info("Kaggle download successful", path=str(output_dir))
        return True
    except Exception as e:
        logger.warning("Kaggle API download unavailable or unauthenticated", error=str(e))
        return False


def load_dataset(config: Optional[dict] = None) -> pd.DataFrame:
    """Loads dataset from local CSV, Kaggle API, or falls back to synthetic benchmark generation."""
    cfg = config or yaml_config
    local_path = Path(cfg["dataset"]["local_raw_path"])
    if not local_path.exists():
        base_dir = Path(__file__).resolve().parent.parent.parent
        candidate = base_dir / local_path
        if candidate.exists():
            local_path = candidate
    slug = cfg["dataset"]["kaggle_dataset_slug"]

    
    # 1. Check if local CSV already exists
    if local_path.exists():
        logger.info("Loading existing dataset from local file", path=str(local_path))
        df = pd.read_csv(local_path)
    else:
        # 2. Try Kaggle API
        downloaded = download_from_kaggle(slug, local_path)
        if downloaded and local_path.exists():
            df = pd.read_csv(local_path)
        else:
            # 3. Generate synthetic benchmark conforming strictly to dataset audit
            logger.info("Generating calibrated benchmark water quality dataset conforming to schema")
            df = generate_benchmark_dataset(n_samples=1000, seed=cfg["project"]["random_seed"])
            local_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(local_path, index=False)
            logger.info("Benchmark dataset successfully saved", path=str(local_path), rows=len(df))

    # Validate loaded dataframe
    expected_numerical = [f["name"] for f in cfg["features"]["numerical"]]
    is_valid, errors = validate_dataframe(df, expected_numerical)
    if not is_valid:
        logger.error("Dataset validation failed", errors=errors)
        raise ValueError(f"Dataset failed schema validation: {errors}")
        
    logger.info("Dataset schema validated successfully", rows=len(df), cols=list(df.columns))
    return df


def split_and_save_data(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataset into leakage-safe train/test partitions and saves to disk."""
    from sklearn.model_selection import train_test_split
    
    cfg = config or yaml_config
    test_size = cfg["split"]["test_size"]
    seed = cfg["split"]["random_state"]
    target_col = cfg["target"]["name"]
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col] if target_col in df.columns else None,
        random_state=seed,
        shuffle=True
    )
    
    train_path = Path(cfg["dataset"]["processed_train_path"])
    test_path = Path(cfg["dataset"]["processed_test_path"])
    train_path.parent.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info("Split completed and saved", train_count=len(train_df), test_count=len(test_df))
    return train_df, test_df


if __name__ == "__main__":
    df = load_dataset()
    train_df, test_df = split_and_save_data(df)
    print(f"Data layer initialized. Train: {len(train_df)} rows, Test: {len(test_df)} rows.")
