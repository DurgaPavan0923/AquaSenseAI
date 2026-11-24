#!/usr/bin/env python3
"""
Design rules linter for AquaSense AI.
Verifies compliance with Phase 4 strict styling and copy standards across web/, README.md, and docs/.
"""

import re
import sys
from pathlib import Path

TARGET_DIRS = [Path("web"), Path("docs")]
TARGET_FILES = [Path("README.md")]

# Forbidden Patterns
FORBIDDEN_PATTERNS = [
    (r"\u2014", "Em dash (U+2014) is forbidden. Use hyphens, colons, commas, or periods."),
    (r"[\U00010000-\U0010ffff]", "Emoji characters are forbidden."),
    (r"[\u2600-\u27BF]", "Miscellaneous symbol/pictograph emojis are forbidden."),
    (r"(?i)\bgradient\b", "CSS gradients are forbidden. Use solid colours."),
    (r"(?i)\blinear-gradient\b", "Linear gradients are forbidden."),
    (r"(?i)\bradial-gradient\b", "Radial gradients are forbidden."),
    (r"(?i)\b(purple|violet)\b", "Purple and violet color palettes are forbidden."),
    (r"(?i)#(800080|8a2be2|9333ea|a855f7|7c3aed|6d28d9|5b21b6|4c1d95)", "Purple/violet hex codes forbidden."),
    (r"(?i)border-radius\s*:\s*([7-9]|\d{2,})px", "Border-radius must be <= 6px (target <= 4px)."),
    (r"(?i)border-radius\s*:\s*(50%|9999px|pill)", "Pill or circular border-radius is forbidden."),
    (r"(?i)made with\s+(ai|claude|chatgpt|antigravity|vercel)", "Generator badges ('Made with...') are forbidden."),
    (r"(?i)built with\s+(ai|claude|chatgpt|antigravity)", "Generator badges are forbidden."),
    (r"(?i)lorem ipsum", "Placeholder filler text is forbidden."),
]


def lint_file(filepath: Path) -> list[str]:
    violations = []
    try:
        text = filepath.read_text(encoding="utf-8")
    except Exception as exc:
        return [f"{filepath}: Read error: {exc}"]

    lines = text.splitlines()
    for line_num, line in enumerate(lines, start=1):
        for pattern, msg in FORBIDDEN_PATTERNS:
            if re.search(pattern, line):
                # Allow references inside lint_design.py itself or regex definitions
                if filepath.name == "lint_design.py":
                    continue
                violations.append(f"{filepath}:{line_num}: {msg} -> Line preview: {line.strip()[:80]}")

    return violations


def main() -> int:
    print("=" * 60)
    print("AquaSense AI Design & Typography Compliance Audit")
    print("=" * 60)

    files_to_check: list[Path] = []
    for d in TARGET_DIRS:
        if d.exists():
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in (".html", ".css", ".js", ".md", ".txt"):
                    files_to_check.append(p)

    for f in TARGET_FILES:
        if f.exists():
            files_to_check.append(f)

    all_violations: list[str] = []
    for f in files_to_check:
        v = lint_file(f)
        all_violations.extend(v)

    if all_violations:
        print(f"FAILED: Found {len(all_violations)} design/copy rule violations:")
        for v in all_violations:
            print(f"  - {v}")
        return 1

    print(f"PASSED: Audited {len(files_to_check)} files. Zero design rule violations found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
