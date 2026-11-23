"""Automated Data Drift and Population Stability Index (PSI) monitoring job."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.common.config import yaml_config
from src.common.logger import logger


def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
    """Calculates Population Stability Index (PSI) between baseline and production distributions."""
    # Drop NaNs
    b = baseline[~np.isnan(baseline)]
    c = current[~np.isnan(current)]

    if len(b) == 0 or len(c) == 0:
        return 0.0

    # Quantile bin edges based on baseline
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(b, percentiles)
    bins[0] -= 1e-5
    bins[-1] += 1e-5

    # Bucket counts
    b_counts, _ = np.histogram(b, bins=bins)
    c_counts, _ = np.histogram(c, bins=bins)

    # Fractions with smoothing
    b_fractions = np.where(b_counts == 0, 1e-4, b_counts) / len(b)
    c_fractions = np.where(c_counts == 0, 1e-4, c_counts) / len(c)

    # PSI equation: sum((actual - expected) * ln(actual / expected))
    psi_val = np.sum((c_fractions - b_fractions) * np.log(c_fractions / b_fractions))
    return float(psi_val)


def evaluate_dataset_drift(baseline_df: pd.DataFrame, incoming_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Evaluates PSI and KS-test statistics across all continuous parameters."""
    numerical_cols = [f["name"] for f in yaml_config["features"]["numerical"]]
    drift_results = {}

    for col in numerical_cols:
        if col in baseline_df.columns and col in incoming_df.columns:
            b_vals = baseline_df[col].to_numpy()
            c_vals = incoming_df[col].to_numpy()

            psi = calculate_psi(b_vals, c_vals)
            ks_stat, ks_pvalue = ks_2samp(b_vals, c_vals)

            # PSI threshold rules: < 0.1 No drift | 0.1 - 0.2 Moderate shift | > 0.2 Significant drift
            status = "STABLE" if psi < 0.10 else "MODERATE_SHIFT" if psi < 0.20 else "DRIFT_ALERT"

            drift_results[col] = {
                "psi": round(psi, 4),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pvalue), 4),
                "status": status
            }

    return drift_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AquaSense AI Drift Detection Job")
    parser.add_argument("--incoming", default="data/processed/test.csv", help="Path to incoming batch CSV")
    parser.add_argument("--output", default="reports/drift_report.json", help="Path to output report JSON")
    args = parser.parse_args()

    train_path = Path(yaml_config["dataset"]["processed_train_path"])
    inc_path = Path(args.incoming)

    if train_path.exists() and inc_path.exists():
        train_df = pd.read_csv(train_path)
        inc_df = pd.read_csv(inc_path)
        report = evaluate_dataset_drift(train_df, inc_df)
        
        out_file = Path(args.output)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            
        logger.info("Drift evaluation report generated", path=args.output)
        print(json.dumps(report, indent=2))
    else:
        logger.warning("Baseline or incoming dataset not found on disk.")
