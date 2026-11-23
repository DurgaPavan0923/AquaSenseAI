"""Unit tests for sensor fault detection and multivariate anomaly scoring."""

import pytest
from src.ml.anomaly import AnomalyEngine


@pytest.fixture
def anomaly_engine():
    """Returns an instantiated AnomalyEngine."""
    return AnomalyEngine(contamination=0.05, random_state=42)


def test_physical_boundary_violations(anomaly_engine):
    """Verifies that impossible physical values immediately trigger critical fault flags."""
    # Impossible pH
    sample_bad_ph = {"pH": 16.5, "Turbidity": 2.0, "Dissolved Oxygen": 7.0}
    has_fault, violations, sev = anomaly_engine.check_deterministic_rules(sample_bad_ph)
    assert has_fault is True
    assert sev == "CRITICAL"
    assert any("pH" in v for v in violations)

    # Negative Turbidity
    sample_neg_turb = {"pH": 7.0, "Turbidity": -4.2, "Dissolved Oxygen": 7.0}
    has_fault, violations, sev = anomaly_engine.check_deterministic_rules(sample_neg_turb)
    assert has_fault is True
    assert any("Negative turbidity" in v for v in violations)


def test_sensor_jump_delta(anomaly_engine):
    """Verifies that rapid unphysical step changes between consecutive ticks trigger alerts."""
    prev_tick = {"pH": 7.2, "Turbidity": 3.0, "Temperature": 20.0, "Dissolved Oxygen": 8.0}
    curr_tick = {"pH": 10.5, "Turbidity": 3.2, "Temperature": 20.5, "Dissolved Oxygen": 7.9}  # Delta pH = 3.3 > 2.5 max

    has_fault, violations, sev = anomaly_engine.check_deterministic_rules(curr_tick, previous_sample=prev_tick)
    assert has_fault is True
    assert sev == "CRITICAL"
    assert any("Sensor jump" in v for v in violations)


def test_extreme_toxic_spike_severity(anomaly_engine):
    """Verifies that catastrophic heavy metal concentrations trigger critical alerts."""
    toxic_spike = {
        "pH": 7.0, "Turbidity": 2.0, "Dissolved Oxygen": 7.0,
        "Lead": 0.065,  # > 5x statutory limit of 0.01
        "Mercury": 0.0002
    }
    has_fault, violations, sev = anomaly_engine.check_deterministic_rules(toxic_spike)
    assert has_fault is True
    assert sev == "CRITICAL"
    assert any("EXTREME TOXICITY SPIKE" in v for v in violations)
