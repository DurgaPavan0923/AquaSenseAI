"""Unit test verifying design rules compliance across web/, docs/, and README.md."""

from scripts.lint_design import main as run_linter


def test_design_and_typography_lint():
    """Asserts that zero design rule violations exist in web/, docs/, or README.md."""
    status_code = run_linter()
    assert status_code == 0, "Design linter detected violations (see stdout for details)."
