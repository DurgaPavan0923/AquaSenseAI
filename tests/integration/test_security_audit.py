"""Security audit tests: CSV formula injection, API Key auth enforcement, and upload limits."""

import io

import pytest
from fastapi.testclient import TestClient

from src.api.main import app as fastapi_app
from src.api.main import sanitize_csv_cell


@pytest.fixture
def client():
    with TestClient(fastapi_app) as test_client:
        yield test_client



def test_csv_formula_injection_sanitization():
    """Verifies that malicious CSV spreadsheet formulas are neutralized with a prepended single-quote."""
    payloads = [
        "=cmd|' /C calc'!A0",
        "+1+1",
        "-5+2",
        "@SUM(1,2)",
        "\t=2+2"
    ]
    for p in payloads:
        sanitized = sanitize_csv_cell(p)
        assert sanitized.startswith("'"), f"Formula injection payload '{p}' was not neutralized: got '{sanitized}'"

    # Benign text should remain unmodified
    assert sanitize_csv_cell("STATION_ALPHA") == "STATION_ALPHA"
    assert sanitize_csv_cell(7.4) == 7.4


def test_batch_upload_file_size_limit(client):
    """Verifies that uploaded CSVs exceeding 10MB are rejected with HTTP 413."""
    # Generate 11MB dummy buffer (exceeding 10MB limit)
    oversized_bytes = b"pH,Turbidity\n" + b"7.0,2.0\n" * 1_500_000
    assert len(oversized_bytes) > 10 * 1024 * 1024

    files = {"file": ("huge_dataset.csv", io.BytesIO(oversized_bytes), "text/csv")}
    response = client.post("/api/v1/predict/batch", files=files)
    assert response.status_code == 413
    assert "10MB" in response.json()["detail"]


def test_batch_upload_non_csv_mime_rejected(client):
    """Verifies that non-CSV executable or binary uploads are rejected with HTTP 400."""
    fake_exe = b"MZ\x90\x00\x03\x00\x00\x00"
    files = {"file": ("malware.exe", io.BytesIO(fake_exe), "application/x-dsexec")}
    response = client.post("/api/v1/predict/batch", files=files)
    assert response.status_code == 400
