#!/usr/bin/env python3
"""
Model bundle export script for AquaSense AI.
Exports the calibrated model pipeline, feature order, conformal quantiles,
and training medians to models/model_bundle.joblib and models/registry.json.
"""

import sys
from pathlib import Path

# Ensure root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aquasense.train import run_training_and_ablations


def main() -> int:
    print("Exporting production model bundle...")
    metrics = run_training_and_ablations()
    bundle_path = Path("models/model_bundle.joblib")
    if bundle_path.exists():
        size_kb = bundle_path.stat().st_size / 1024
        print(f"Verified exported bundle at {bundle_path} ({size_kb:.2f} KB).")
        return 0
    else:
        print("Export failed: models/model_bundle.joblib not found.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
