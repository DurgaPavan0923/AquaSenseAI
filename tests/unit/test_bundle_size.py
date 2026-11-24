"""Unit test verifying serverless bundle size constraint (< 250 MB)."""

from scripts.check_bundle_size import MAX_BUNDLE_MB, check_bundle_size


def test_bundle_size_within_limits():
    """Asserts that total serverless bundle size is below 250 MB."""
    size_mb = check_bundle_size()
    assert size_mb <= MAX_BUNDLE_MB, f"Bundle size {size_mb:.2f} MB exceeds {MAX_BUNDLE_MB} MB limit!"
