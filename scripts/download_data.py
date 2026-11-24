#!/usr/bin/env python3
"""
Dataset acquisition and verification script for AquaSense AI.
"""

import os
import sys
from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_FILE = RAW_DATA_DIR / "water_quality.csv"
KAGGLE_DATASET = "ziya07/water-quality-and-pollution-monitoring-dataset"

EXPECTED_COLUMNS = [
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


def verify_dataset(filepath: Path) -> bool:
    if not filepath.exists():
        return False
    try:
        df = pd.read_csv(filepath)
        missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            print(f"Error: Dataset at {filepath} is missing expected columns: {missing_cols}")
            return False
        print(f"Dataset verified successfully: {len(df)} records, {len(df.columns)} columns.")
        return True
    except Exception as exc:
        print(f"Error reading dataset at {filepath}: {exc}")
        return False


def download_from_kaggle() -> bool:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")

    if not username or not key:
        print("Kaggle credentials (KAGGLE_USERNAME / KAGGLE_KEY) not found in environment.")
        return False

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        print(f"Downloading dataset '{KAGGLE_DATASET}' to '{RAW_DATA_DIR}'...")
        api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DATA_DIR), unzip=True)
        return verify_dataset(RAW_DATA_FILE)
    except Exception as exc:
        print(f"Kaggle download failed: {exc}")
        return False


def main() -> int:
    print("Checking dataset availability...")
    if verify_dataset(RAW_DATA_FILE):
        return 0

    print("Raw dataset not found locally. Attempting Kaggle download...")
    if download_from_kaggle():
        return 0

    print("\nDataset missing: Please place 'water_quality.csv' into 'data/raw/' directory.")
    print("Source URL: https://www.kaggle.com/datasets/ziya07/water-quality-and-pollution-monitoring-dataset")
    return 1


if __name__ == "__main__":
    sys.exit(main())
