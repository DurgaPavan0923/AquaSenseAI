#!/usr/bin/env python3
"""
IoT Water Sensor Telemetry Simulator for AquaSense AI.
Simulates real-time sensor streams transmitting physicochemical readings to the AquaSense AI API.
"""

import argparse
import random
import time

import requests

DEFAULT_API_URL = "http://localhost:8000"


def generate_synthetic_reading(scenario: str = "normal") -> dict[str, float]:
    """Generates synthetic water quality reading based on desired scenario."""
    if scenario == "polluted":
        return {
            "pH": round(random.uniform(9.0, 11.5), 2),
            "Turbidity": round(random.uniform(8.0, 45.0), 2),
            "Temperature": round(random.uniform(25.0, 34.0), 1),
            "Dissolved Oxygen": round(random.uniform(1.5, 4.0), 2),
            "Biochemical Oxygen Demand": round(random.uniform(8.0, 25.0), 2),
            "Lead": round(random.uniform(0.02, 0.08), 4),
            "Mercury": round(random.uniform(0.002, 0.006), 5),
            "Arsenic": round(random.uniform(0.02, 0.05), 4),
        }
    elif scenario == "sensor_fault":
        # Physically impossible sensor fault
        return {
            "pH": 15.8,
            "Turbidity": -2.0,
            "Temperature": 95.0,
            "Dissolved Oxygen": 0.0,
            "Biochemical Oxygen Demand": 50.0,
            "Lead": 0.01,
            "Mercury": 0.001,
            "Arsenic": 0.01,
        }
    else:  # Normal potable/good
        return {
            "pH": round(random.uniform(6.8, 7.8), 2),
            "Turbidity": round(random.uniform(0.5, 3.2), 2),
            "Temperature": round(random.uniform(18.0, 24.0), 1),
            "Dissolved Oxygen": round(random.uniform(6.5, 9.0), 2),
            "Biochemical Oxygen Demand": round(random.uniform(0.8, 1.9), 2),
            "Lead": round(random.uniform(0.001, 0.008), 4),
            "Mercury": round(random.uniform(0.0001, 0.0008), 5),
            "Arsenic": round(random.uniform(0.001, 0.007), 4),
        }


def stream_telemetry(api_url: str, count: int, interval: float, scenario: str):
    print(f"Starting IoT telemetry stream to {api_url} ({count} readings, scenario: {scenario})...")
    session = requests.Session()

    for i in range(1, count + 1):
        reading = generate_synthetic_reading(scenario)
        try:
            pred_res = session.post(f"{api_url}/predict", json=reading, timeout=5)
            anom_res = session.post(f"{api_url}/anomaly", json=reading, timeout=5)

            if pred_res.ok and anom_res.ok:
                pred_data = pred_res.json()
                anom_data = anom_res.json()
                print(
                    f"[{i}/{count}] pH={reading['pH']:.2f}, Turb={reading['Turbidity']:.2f} NTU | "
                    f"Tier: {pred_data['risk_tier']:<8} | "
                    f"Audit: {anom_data['severity']:<8} | "
                    f"Conformal Set: {pred_data['conformal_prediction_set']}"
                )
            else:
                print(f"[{i}/{count}] Error: Status code {pred_res.status_code}")
        except Exception as exc:
            print(f"[{i}/{count}] Connection failed: {exc}")

        if i < count:
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="AquaSense AI IoT Telemetry Simulator")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="Base API URL")
    parser.add_argument("--count", type=int, default=10, help="Number of telemetry packets")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval in seconds between packets")
    parser.add_argument("--scenario", choices=["normal", "polluted", "sensor_fault"], default="normal")
    args = parser.parse_args()

    stream_telemetry(args.url, args.count, args.interval, args.scenario)


if __name__ == "__main__":
    main()
