#!/usr/bin/env python3
"""
Bundle size verification script for AquaSense AI Vercel serverless deployment.
Ensures that runtime code, models, and runtime dependencies stay strictly below 250 MB (target < 150 MB).
"""

import importlib.util
import sys
from pathlib import Path

MAX_BUNDLE_MB = 250.0
TARGET_BUNDLE_MB = 150.0

RUNTIME_PACKAGES = [
    "fastapi",
    "pydantic",
    "pydantic_core",
    "starlette",
    "numpy",
    "sklearn",
    "scipy",
    "joblib",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_dir_size_mb(path: Path) -> float:
    total_bytes = 0
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    for p in path.rglob("*"):
        if p.is_file() and not any(part.startswith((".", "__pycache__")) for part in p.parts):
            total_bytes += p.stat().st_size
    return total_bytes / (1024 * 1024)


def get_package_size_mb(pkg_name: str) -> float:
    try:
        spec = importlib.util.find_spec(pkg_name)
        if spec and spec.origin:
            origin_path = Path(spec.origin)
            pkg_dir = origin_path.parent if origin_path.name.startswith("__init__") else origin_path
            return get_dir_size_mb(pkg_dir)
    except Exception:
        pass
    return 0.0


def check_bundle_size() -> float:
    print("=" * 60)
    print("AquaSense AI Serverless Bundle Size Audit")
    print("=" * 60)

    # 1. Project runtime directories
    api_size = get_dir_size_mb(PROJECT_ROOT / "api")
    models_size = get_dir_size_mb(PROJECT_ROOT / "models")
    core_pkg_size = get_dir_size_mb(PROJECT_ROOT / "src" / "aquasense")

    app_total = api_size + models_size + core_pkg_size
    print(
        f"App Files:  api/ ({api_size:.2f} MB), models/ ({models_size:.2f} MB), src/aquasense/ ({core_pkg_size:.2f} MB)"
    )
    print(f"Subtotal Application Files: {app_total:.2f} MB")

    # 2. Runtime dependency packages
    dep_total = 0.0
    print("\nRuntime Dependencies (from root requirements.txt):")
    for pkg in RUNTIME_PACKAGES:
        pkg_size = get_package_size_mb(pkg)
        dep_total += pkg_size
        print(f"  - {pkg:<18}: {pkg_size:>6.2f} MB")
    print(f"Subtotal Runtime Dependencies: {dep_total:.2f} MB")

    total_bundle_mb = app_total + dep_total
    print("-" * 60)
    print(f"Total Serverless Bundle Size: {total_bundle_mb:.2f} MB")
    print(f"Maximum Allowed CI Threshold: {MAX_BUNDLE_MB:.1f} MB (Target: < {TARGET_BUNDLE_MB:.1f} MB)")
    print("-" * 60)

    if total_bundle_mb > MAX_BUNDLE_MB:
        print(f"FAILED: Bundle size ({total_bundle_mb:.2f} MB) exceeds limit of {MAX_BUNDLE_MB} MB!")
        return total_bundle_mb

    print(f"PASSED: Bundle size is well within the {MAX_BUNDLE_MB} MB limit.")
    return total_bundle_mb


if __name__ == "__main__":
    size = check_bundle_size()
    if size > MAX_BUNDLE_MB:
        sys.exit(1)
    sys.exit(0)
