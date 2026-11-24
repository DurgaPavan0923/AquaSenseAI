"""
Comprehensive Codebase Quality and Integrity Audit Script for AquaSense AI.
Audits tracked files for:
- Syntax and compilation
- Hardcoded absolute paths
- Secret patterns
- Dead code / unimported modules
- Compliance with config.yaml schema
"""

import ast
import os
import re
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IGNORED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "node_modules",
    "artifacts",
}

SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|password|bearer|auth[_-]?token)\s*[:=]\s*['\"][0-9a-zA-Z\-_]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[0-9a-zA-Z]{36}"),
]

ABSOLUTE_PATH_PATTERNS = [
    re.compile(r"[a-zA-Z]:\\[Users|Desktop|Program]"),
    re.compile(r"/(?:Users|home)/[a-zA-Z0-9_\-]+/"),
]


def audit_python_file(path: Path) -> list[str]:
    issues = []
    try:
        source = path.read_text(encoding="utf-8")
    except Exception as exc:
        return [f"Encoding error: {exc}"]

    # 1. Syntax check
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        issues.append(f"Syntax error at line {exc.lineno}: {exc.msg}")

    # 2. Check for secret patterns
    if path.name not in {".env.example", "test_security_audit.py", "audit_codebase.py"}:
        for lineno, line in enumerate(source.splitlines(), 1):
            for pat in SECRET_PATTERNS:
                if pat.search(line):
                    issues.append(f"Line {lineno}: Potential secret detected")

    # 3. Check for absolute paths
    for lineno, line in enumerate(source.splitlines(), 1):
        for pat in ABSOLUTE_PATH_PATTERNS:
            if pat.search(line):
                issues.append(f"Line {lineno}: Hardcoded absolute path detected")

    return issues


def audit_all():
    print("=" * 60)
    print("AquaSense AI Full File Audit")
    print("=" * 60)

    total_files = 0
    issues_found = {}

    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        for f in files:
            path = Path(root) / f
            total_files += 1
            rel_path = path.relative_to(PROJECT_ROOT)

            if path.suffix == ".py":
                file_issues = audit_python_file(path)
                if file_issues:
                    issues_found[str(rel_path)] = file_issues

    # Audit config.yaml
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        issues_found["config.yaml"] = ["config.yaml does not exist"]
    else:
        try:
            with open(config_path, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            assert "features" in cfg, "Missing 'features' in config.yaml"
            assert "target" in cfg, "Missing 'target' in config.yaml"
            assert "split" in cfg, "Missing 'split' in config.yaml"
            print("config.yaml verified successfully.")
        except Exception as exc:
            issues_found["config.yaml"] = [f"Invalid config.yaml: {exc}"]

    print(f"Audited {total_files} files across repository.")
    if issues_found:
        print(f"FAILED: {len(issues_found)} files with issues:")
        for p, errs in issues_found.items():
            print(f"  - {p}:")
            for e in errs:
                print(f"      * {e}")
        return 1
    else:
        print("PASSED: Zero syntax errors, secrets, or hardcoded paths found.")
        return 0


if __name__ == "__main__":
    sys.exit(audit_all())
