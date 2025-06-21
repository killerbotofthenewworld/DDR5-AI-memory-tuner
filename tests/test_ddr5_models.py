"""
Unit tests for DDR5 models and validation.
"""

import pytest
import sys
from pathlib import Path

# Add src directory to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from ddr5_models import DDR5Configuration, DDR5TimingParameters, DDR5VoltageParameters


class TestDDR5TimingParameters:
    """Test DDR5 timing parameter validation."""
    
    def test_valid_timings(self):
        """Test valid timing relationships."""
        timings = DDR5TimingParameters(
            cl=32,
            trcd=32,
            trp=32,
            tras=64,  # Updated tRAS to satisfy constraints
            trc=96  # Updated tRC to satisfy constraints
        )
        violations = timings.validate_relationships()
        assert len(violations) == 0
    
    def test_invalid_tras(self):
        """Test invalid tRAS relationship."""
        timings = DDR5TimingParameters(
            cl=32, trcd=32, trp=32, tras=50, trc=84  # tRAS < tRCD + CL
        )
        violations = timings.validate_relationships()
        assert len(violations) > 0
        assert "tRAS" in violations[0]
    
    def test_invalid_trc(self):
        """Test invalid tRC relationship."""
        timings = DDR5TimingParameters(
            cl=32, trcd=32, trp=32, tras=52, trc=80  # tRC < tRAS + tRP
        )
        violations = timings.validate_relationships()
        assert len(violations) > 0
        assert "tRC" in violations[0]


class TestDDR5VoltageParameters:
    """Test DDR5 voltage parameter validation."""
    
    def test_valid_voltages(self):
        """Test valid voltage ranges."""
        voltages = DDR5VoltageParameters(vddq=1.1, vpp=1.8)
        violations = voltages.validate_ranges()
        assert len(violations) == 0
    
    def test_invalid_vddq_low(self):
        """Test VDDQ too low."""
        voltages = DDR5VoltageParameters(vddq=0.9, vpp=1.8)
        violations = voltages.validate_ranges()
        assert len(violations) > 0
        assert "VDDQ" in violations[0]
    
    def test_invalid_vddq_high(self):
        """Test VDDQ too high."""
        voltages = DDR5VoltageParameters(vddq=1.3, vpp=1.8)
        violations = voltages.validate_ranges()
        assert len(violations) > 0
        assert "VDDQ" in violations[0]
    
    def test_invalid_vpp(self):
        """Test VPP out of range."""
        voltages = DDR5VoltageParameters(vddq=1.1, vpp=2.0)
        violations = voltages.validate_ranges()
        assert len(violations) > 0
        assert "VPP" in violations[0]


class TestDDR5Configuration:
    """Test complete DDR5 configuration."""
    
    def test_valid_configuration(self):
        """Test a valid DDR5 configuration."""
        config = DDR5Configuration(frequency=5600)
        config.calculate_performance_metrics()
        
        assert config.bandwidth_gbps is not None
        assert config.latency_ns is not None
        assert config.bandwidth_gbps > 0
        assert config.latency_ns > 0
    
    def test_frequency_validation(self):
        """Test frequency validation and rounding."""
        config = DDR5Configuration(frequency=5555)  # Invalid frequency
        # Should be rounded to nearest valid frequency
        assert config.frequency in [5200, 5600]
    
    def test_stability_estimate(self):
        """Test stability estimation."""
        config = DDR5Configuration(frequency=5600)
        stability = config.get_stability_estimate()
        
        assert 0 <= stability <= 100
        assert config.stability_score == stability
    
    def test_performance_calculation(self):
        """Test performance metric calculation."""
        config = DDR5Configuration(frequency=5600)
        config.calculate_performance_metrics()
        
        # Check bandwidth calculation
        expected_bandwidth = (5600 * 8 * 2) / 1000
        assert abs(config.bandwidth_gbps - expected_bandwidth) < 0.1
        
        # Check latency calculation
        clock_period = 2000 / 5600
        expected_latency = config.timings.cl * clock_period
        assert abs(config.latency_ns - expected_latency) < 0.1


if __name__ == "__main__":
    pytest.main([__file__])
