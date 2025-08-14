import pytest

from src.ddr5_models import (
    DDR5Configuration,
    DDR5TimingParameters,
    DDR5VoltageParameters,
)
from src.ddr5_simulator import DDR5Simulator


def make_config(
    freq: int = 5600,
    cl: int = 32,
    trcd: int = 32,
    trp: int = 32,
    tras: int = 64,
    trc: int = 96,
    trfc: int = 295,
    vddq: float = 1.1,
    vpp: float = 1.8,
) -> DDR5Configuration:
    return DDR5Configuration(
        frequency=freq,
        timings=DDR5TimingParameters(
            cl=cl, trcd=trcd, trp=trp, tras=tras, trc=trc, trfc=trfc
        ),
        voltages=DDR5VoltageParameters(vddq=vddq, vpp=vpp),
    )


def test_command_overhead_increases_with_frequency():
    sim = DDR5Simulator()
    # Baseline freq 5600
    base_over = sim._calculate_command_overhead()
    # Increase frequency to max JEDEC and compare
    sim.current_config = make_config(freq=8400)
    high_over = sim._calculate_command_overhead()
    assert high_over > base_over
    # Expected numeric check: 2.5 + (8400-5600)*0.001 = 5.3
    assert pytest.approx(5.3, rel=1e-3) == high_over


def test_stability_recommendations_and_violations():
    sim = DDR5Simulator()
    # Very low score triggers all generic recommendations
    rec = sim._get_stability_recommendation(0.55, [
        "timing_violations",
        "voltage_violations",
        "temperature_violations",
        "frequency_violations",
    ])
    # Check presence of generic and specific guidance
    assert "Increase voltage slightly" in rec
    assert "Reduce frequency" in rec
    assert "Consider better cooling" in rec or "cooling" in rec
    assert "Relax timings" in rec or "timings" in rec
    assert "Adjust voltage" in rec or "voltage" in rec
    assert "Lower the memory frequency" in rec or "frequency" in rec


def test_simulate_power_consumption_cached():
    sim = DDR5Simulator()
    sim.load_configuration(make_config(freq=6000))
    first = sim.simulate_power_consumption()
    second = sim.simulate_power_consumption()
    # Cached object should be the same dict instance
    assert first is second
    assert first["total_power_w"] > 0


def test_estimate_power_does_not_mutate_current_config():
    sim = DDR5Simulator()
    original = sim.current_config.model_copy()
    cfg = make_config(freq=7200, vddq=1.15)
    power_w = sim.estimate_power(cfg)
    assert power_w > 0
    # Simulator's current_config should be unchanged
    assert sim.current_config.frequency == original.frequency
    assert sim.current_config.timings.cl == original.timings.cl


def test_bandwidth_cache_key_includes_frequency():
    sim = DDR5Simulator()
    sim.load_configuration(make_config(freq=5600))
    r1 = sim.simulate_bandwidth("sequential", queue_depth=16)
    # Change frequency and load
    sim.load_configuration(make_config(freq=7200))
    r2 = sim.simulate_bandwidth("sequential", queue_depth=16)
    assert r1 != r2
    assert r2["effective_bandwidth_gbps"] > r1["effective_bandwidth_gbps"]
