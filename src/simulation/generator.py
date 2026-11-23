"""IoT edge telemetry streaming generator with realistic sensor noise, drift, and fault injection."""

import argparse
import time
from datetime import datetime
from typing import Dict, Generator
import numpy as np
import requests

from src.common.logger import logger


def generate_telemetry_stream(
    station_id: str = "STATION_RIVER_01",
    probe_id: str = "PROBE_EDGE_101",
    inject_drift: bool = False,
    inject_fault_every: int = 20
) -> Generator[Dict, None, None]:
    """Generates continuous realistic aquatic telemetry."""
    tick = 0
    drift_ph = 0.0

    while True:
        tick += 1
        
        # Nominal physical parameters
        ph = 7.35 + np.random.normal(0, 0.08) + drift_ph
        turbidity = max(0.2, np.random.exponential(1.2))
        temp = 21.0 + np.random.normal(0, 0.5)
        do = max(1.0, 7.8 + np.random.normal(0, 0.3))
        bod = max(0.2, 1.4 + np.random.normal(0, 0.2))
        lead = max(0.0001, np.random.exponential(0.002))
        mercury = max(0.00001, np.random.exponential(0.0002))
        arsenic = max(0.0001, np.random.exponential(0.002))

        # 1. Slow Sensor Calibration Drift Injection
        if inject_drift:
            drift_ph += 0.02

        # 2. Periodic Fault or Plume Injection
        if inject_fault_every > 0 and (tick % inject_fault_every == 0):
            fault_type = np.random.choice(["sensor_airlock", "chemical_plume"])
            if fault_type == "sensor_airlock":
                logger.warning("Simulating sensor hardware air-lock / jump failure", tick=tick)
                ph = 15.5  # Unphysical value
                turbidity = 250.0
            else:
                logger.warning("Simulating toxic industrial chemical discharge plume", tick=tick)
                bod = 18.5
                do = 2.1  # Severe hypoxia
                lead = 0.045  # 4.5x statutory limit

        payload = {
            "location_id": station_id,
            "sensor_id": probe_id,
            "source_type": "iot_stream",
            "features": {
                "pH": round(float(ph), 3),
                "Turbidity": round(float(turbidity), 2),
                "Temperature": round(float(temp), 2),
                "Dissolved Oxygen": round(float(do), 2),
                "Biochemical Oxygen Demand": round(float(bod), 2),
                "Lead": round(float(lead), 5),
                "Mercury": round(float(mercury), 6),
                "Arsenic": round(float(arsenic), 5)
            },
            "confidence_level": 0.90
        }

        yield payload


def run_streamer(api_url: str = "http://localhost:8000/api/v1/predict", interval: float = 2.0, count: int = 50) -> None:
    """Streams generated packets to the target FastAPI endpoint."""
    logger.info("Starting IoT edge probe simulator", api_url=api_url, interval=interval, count=count)
    stream = generate_telemetry_stream()
    
    sent = 0
    for packet in stream:
        sent += 1
        try:
            res = requests.post(api_url, json=packet, timeout=3)
            if res.status_code == 200:
                body = res.json()
                logger.info(
                    "Telemetry packet dispatched",
                    tick=sent,
                    predicted=body["prediction"]["predicted_label"],
                    conformal_set=body["prediction"]["conformal_prediction_set"],
                    is_anomaly=body["anomaly"]["is_anomaly"]
                )
            else:
                logger.warning("API returned error status", status_code=res.status_code, text=res.text)
        except Exception as e:
            logger.warning("Failed to reach API endpoint", error=str(e))

        if count > 0 and sent >= count:
            break
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AquaSense AI IoT Telemetry Streamer")
    parser.add_argument("--url", default="http://localhost:8000/api/v1/predict", help="FastAPI endpoint URL")
    parser.add_argument("--interval", type=float, default=1.5, help="Seconds between packets")
    parser.add_argument("--count", type=int, default=30, help="Total packets to send (0 = infinite)")
    args = parser.parse_args()

    run_streamer(api_url=args.url, interval=args.interval, count=args.count)
