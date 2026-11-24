"""
Security audit tests for AquaSense AI API.
Tests CSV formula injection sanitization, upload size limits, and input parsing safety.
"""

import io

from fastapi.testclient import TestClient

from api.index import app, sanitize_csv_cell

client = TestClient(app)


def test_csv_formula_injection_sanitization():
    """Verifies that malicious CSV spreadsheet formulas are neutralized."""
    payloads = [
        "=cmd|' /C calc'!A0",
        "+1+1",
        "-5+2",
        "@SUM(1,2)",
        "\t=2+2",
    ]
    for p in payloads:
        sanitized = sanitize_csv_cell(p)
        assert sanitized.startswith("'"), f"Formula injection payload '{p}' not neutralized: got '{sanitized}'"

    # Benign text remains safe
    assert sanitize_csv_cell("WaterSample1") == "WaterSample1"


def test_batch_upload_file_size_limit():
    """Verifies that uploaded CSVs exceeding 2 MB are rejected with HTTP 400."""
    oversized_bytes = b"pH,Turbidity\n" + b"7.0,2.0\n" * 300_000
    assert len(oversized_bytes) > 2 * 1024 * 1024

    files = {"file": ("huge_dataset.csv", io.BytesIO(oversized_bytes), "text/csv")}
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 400
    assert "2 MB" in response.json()["detail"]


def test_batch_upload_invalid_encoding():
    """Verifies non-utf8 files are rejected."""
    bad_bytes = b"\xff\xfe\x00\x00"
    files = {"file": ("corrupt.csv", io.BytesIO(bad_bytes), "text/csv")}
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 400
