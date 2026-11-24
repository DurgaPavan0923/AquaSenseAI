"""
Rerun and refresh all AquaSense AI research notebooks in sequence.
Ensures:
- Outputs are cleared first
- Executed from project root in order (01 -> 05)
- All figures saved to outputs/figures/
- All tables and metrics saved to results/
"""

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

NOTEBOOK_SEQUENCE = [
    "01_dataset_audit.ipynb",
    "02_exploratory_data_analysis.ipynb",
    "03_model_training_and_ablations.ipynb",
    "04_explainability_analysis.ipynb",
    "05_uncertainty_quantification.ipynb",
]


def clear_notebook_outputs(nb_path: Path):
    """Clears all cell outputs and execution counts in place."""
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)


def execute_notebook(nb_name: str) -> bool:
    nb_path = NOTEBOOKS_DIR / nb_name
    print(f"\n[{time.strftime('%H:%M:%S')}] Clearing outputs: {nb_name}...")
    clear_notebook_outputs(nb_path)

    print(f"[{time.strftime('%H:%M:%S')}] Executing: {nb_name}...")
    cmd = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=600",
        str(nb_path),
    ]

    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"ERROR executing {nb_name}:\n{res.stderr}\n{res.stdout}")
        return False

    print(f"[{time.strftime('%H:%M:%S')}] SUCCESS: {nb_name} executed completely.")
    return True


def main():
    print("=" * 60)
    print("AquaSense AI Notebook Execution & Results Refresh Pipeline")
    print("=" * 60)

    # Ensure output directories exist
    (PROJECT_ROOT / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)

    failed = []
    for nb in NOTEBOOK_SEQUENCE:
        success = execute_notebook(nb)
        if not success:
            failed.append(nb)
            break

    if failed:
        print(f"\nPipeline FAILED on notebook(s): {failed}")
        return 1

    print("\nAll 5 notebooks executed successfully in sequence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
