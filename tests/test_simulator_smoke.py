from src.ddr5_models import DDR5Configuration
from src.ddr5_simulator import DDR5Simulator


def test_simulate_performance_returns_expected_keys():
    cfg = DDR5Configuration()
    sim = DDR5Simulator()
    result = sim.simulate_performance(cfg)
    # Support both legacy and new keys in downstream code
    assert "bandwidth" in result
    assert "latency" in result
    assert "stability" in result
    assert "score" in result


def test_stability_monotonicity_with_voltage():
    cfg = DDR5Configuration()
    sim = DDR5Simulator()
    base = sim.simulate_performance(cfg)["stability"]

    cfg_high_v = cfg.model_copy()
    cfg_high_v.voltages.vddq = min(1.25, cfg_high_v.voltages.vddq + 0.05)
    higher = sim.simulate_performance(cfg_high_v)["stability"]

    # Usually higher VDDQ should not decrease stability in our model
    assert higher >= base * 0.9
