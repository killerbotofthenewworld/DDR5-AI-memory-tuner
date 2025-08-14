"""
Comprehensive test suite for DDR5 AI Sandbox Simulator
Includes unit tests, integration tests, and performance tests.
"""

import pytest
import numpy as np
import pandas as pd  # noqa: F401  # used in potential future extensions
import tempfile  # noqa: F401
import os  # noqa: F401
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock  # noqa: F401

# Add src directory to path for testing (allow E402 for path setup before imports)
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))  # noqa: E402

from ddr5_models import (  # noqa: E402
    DDR5Configuration, DDR5TimingParameters, DDR5VoltageParameters,
    validate_ddr5_configuration
)
from ddr5_simulator import DDR5Simulator  # noqa: E402
from perfect_ai_optimizer import PerfectDDR5Optimizer  # noqa: E402


class TestDDR5Models:
    """Comprehensive tests for DDR5 models and validation."""
    
    def test_timing_parameters_creation(self):
        """Test DDR5 timing parameter creation."""
        timings = DDR5TimingParameters(
            cl=32, trcd=32, trp=32, tras=64, trc=96
        )
        assert timings.cl == 32
        assert timings.trcd == 32
        assert timings.trp == 32
        assert timings.tras == 64
        assert timings.trc == 96
    
    def test_voltage_parameters_creation(self):
        """Test DDR5 voltage parameter creation."""
        voltages = DDR5VoltageParameters(
            vddq=1.1, vpp=1.8, vddq_tx=1.1, vddq_rx=1.1
        )
        assert voltages.vddq == 1.1
        assert voltages.vpp == 1.8
        assert voltages.vddq_tx == 1.1
        assert voltages.vddq_rx == 1.1
    
    def test_ddr5_configuration_creation(self):
        """Test DDR5 configuration creation."""
        timings = DDR5TimingParameters(
            cl=32, trcd=32, trp=32, tras=64, trc=96
        )
        voltages = DDR5VoltageParameters(
            vddq=1.1, vpp=1.8, vddq_tx=1.1, vddq_rx=1.1
        )
        config = DDR5Configuration(
            frequency=5600,
            timings=timings,
            voltages=voltages
        )
        assert config.frequency == 5600
        assert config.timings.cl == 32
        assert config.voltages.vddq == 1.1
    
    def test_timing_relationships_valid(self):
        """Test valid timing relationships."""
        timings = DDR5TimingParameters(
            cl=32, trcd=32, trp=32, tras=64, trc=96
        )
        violations = timings.validate_relationships()
        assert len(violations) == 0
    
    def test_timing_relationships_invalid_tras(self):
        """Test invalid tRAS relationship."""
        timings = DDR5TimingParameters(
            cl=32, trcd=32, trp=32, tras=50, trc=96  # tRAS < tRCD + CL
        )
        violations = timings.validate_relationships()
        assert len(violations) > 0
        assert any("tRAS" in violation for violation in violations)
    
    def test_timing_relationships_invalid_trc(self):
        """Test invalid tRC relationship."""
        timings = DDR5TimingParameters(
            cl=32, trcd=32, trp=32, tras=64, trc=80  # tRC < tRAS + tRP
        )
        violations = timings.validate_relationships()
        assert len(violations) > 0
        assert any("tRC" in violation for violation in violations)
    
    def test_voltage_validation_valid(self):
        """Test valid voltage ranges."""
        voltages = DDR5VoltageParameters(
            vddq=1.1, vpp=1.8, vddq_tx=1.1, vddq_rx=1.1
        )
        violations = voltages.validate_ranges()
        assert len(violations) == 0
    
    def test_voltage_validation_invalid_vddq(self):
        """Test invalid VDDQ voltage."""
        voltages = DDR5VoltageParameters(
            vddq=1.5, vpp=1.8, vddq_tx=1.1, vddq_rx=1.1  # VDDQ too high
        )
        violations = voltages.validate_ranges()
        assert len(violations) > 0
        assert any("VDDQ" in violation for violation in violations)
    
    def test_voltage_validation_invalid_vpp(self):
        """Test invalid VPP voltage."""
        voltages = DDR5VoltageParameters(
            vddq=1.1, vpp=2.5, vddq_tx=1.1, vddq_rx=1.1  # VPP too high
        )
        violations = voltages.validate_ranges()
        assert len(violations) > 0
        assert any("VPP" in violation for violation in violations)
    
    def test_configuration_validation_complete(self):
        """Test complete configuration validation."""
        timings = DDR5TimingParameters(
            cl=32, trcd=32, trp=32, tras=64, trc=96
        )
        voltages = DDR5VoltageParameters(
            vddq=1.1, vpp=1.8, vddq_tx=1.1, vddq_rx=1.1
        )
        config = DDR5Configuration(
            frequency=5600,
            timings=timings,
            voltages=voltages
        )
        
        is_valid, violations = validate_ddr5_configuration(config)
        assert is_valid
        # Check that all violation categories are empty
        total_violations = sum(len(v) for v in violations.values())
        assert total_violations == 0


class TestDDR5Simulator:
    """Comprehensive tests for DDR5 simulator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.simulator = DDR5Simulator()
        self.test_config = DDR5Configuration(
            frequency=5600,
            timings=DDR5TimingParameters(
                cl=32, trcd=32, trp=32, tras=64, trc=96
            ),
            voltages=DDR5VoltageParameters(
                vddq=1.1, vpp=1.8, vddq_tx=1.1, vddq_rx=1.1
            )
        )
    
    def test_simulator_initialization(self):
        """Test simulator initialization."""
        assert self.simulator is not None
        assert hasattr(self.simulator, 'simulate_performance')
    
    def test_performance_simulation(self):
        """Test performance simulation."""
        result = self.simulator.simulate_performance(self.test_config)
        
        assert 'bandwidth' in result
        assert 'latency' in result
        assert 'stability' in result
        assert 'power' in result
        
        # Check reasonable ranges
        assert 0 < result['bandwidth'] < 200000  # MB/s
        assert 0 < result['latency'] < 1000  # ns
        assert 0 < result['stability'] < 100  # percentage
        assert 0 < result['power'] < 10000  # mW
    
    def test_stability_prediction(self):
        """Test stability prediction."""
        stability = self.simulator.predict_stability(self.test_config)
        assert 0 <= stability <= 100
    
    def test_power_estimation(self):
        """Test power estimation."""
        power = self.simulator.estimate_power(self.test_config)
        assert power > 0
    
    def test_temperature_impact(self):
        """Test temperature impact on performance."""
        result_normal = self.simulator.simulate_performance(
            self.test_config, temperature=50
        )
        result_hot = self.simulator.simulate_performance(
            self.test_config, temperature=80
        )
        
        # Performance should degrade at higher temperatures
        assert result_hot['stability'] <= result_normal['stability']
    
    def test_frequency_scaling(self):
        """Test performance scaling with frequency."""
        low_freq_config = self.test_config.model_copy()
        low_freq_config.frequency = 4800
        
        high_freq_config = self.test_config.model_copy()
        high_freq_config.frequency = 6400
        
        low_result = self.simulator.simulate_performance(low_freq_config)
        high_result = self.simulator.simulate_performance(high_freq_config)
        
        # Higher frequency should have higher bandwidth
        assert high_result['bandwidth'] > low_result['bandwidth']


class TestPerfectAIOptimizer:
    """Comprehensive tests for AI optimizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = PerfectDDR5Optimizer()
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer is not None
        assert hasattr(self.optimizer, 'optimize')
        assert hasattr(self.optimizer, 'train_models')
    
    def test_database_initialization(self):
        """Test performance database initialization."""
        assert self.optimizer.performance_database is not None
        assert 'ddr5_5600' in self.optimizer.performance_database
        assert len(self.optimizer.performance_database['ddr5_5600']) > 0
    
    @patch('src.perfect_ai_optimizer.PerfectDDR5Optimizer._generate_training_data')
    def test_model_training(self, mock_training_data):
        """Test AI model training."""
        # Mock training data
        mock_training_data.return_value = (
            np.random.rand(100, 10),  # X
            np.random.rand(100),      # y_performance
            np.random.rand(100)       # y_stability
        )
        
        self.optimizer.train_models()
        assert self.optimizer.is_trained
    
    def test_configuration_generation(self):
        """Test configuration generation."""
        configs = self.optimizer._generate_population(
            target_frequency=5600,
            population_size=10
        )
        
        assert len(configs) == 10
        for config in configs:
            assert isinstance(config, DDR5Configuration)
            assert config.frequency == 5600
    
    def test_fitness_evaluation(self):
        """Test fitness evaluation."""
        config = DDR5Configuration(
            frequency=5600,
            timings=DDR5TimingParameters(
                cl=32, trcd=32, trp=32, tras=64, trc=96
            ),
            voltages=DDR5VoltageParameters(
                vddq=1.1, vpp=1.8, vddq_tx=1.1, vddq_rx=1.1
            )
        )
        
        fitness = self.optimizer._evaluate_fitness(config, 'balanced')
        assert isinstance(fitness, (int, float))
        assert fitness >= 0
    
    def test_mutation_operation(self):
        """Test genetic algorithm mutation."""
        original_config = DDR5Configuration(
            frequency=5600,
            timings=DDR5TimingParameters(
                cl=32, trcd=32, trp=32, tras=64, trc=96
            ),
            voltages=DDR5VoltageParameters(
                vddq=1.1, vpp=1.8, vddq_tx=1.1, vddq_rx=1.1
            )
        )
        
        mutated_config = self.optimizer._mutate_configuration(
            original_config, mutation_rate=0.5
        )
        
        # Should be different from original (with high mutation rate)
        assert mutated_config != original_config
    
    def test_crossover_operation(self):
        """Test genetic algorithm crossover."""
        parent1 = DDR5Configuration(
            frequency=5600,
            timings=DDR5TimingParameters(
                cl=32, trcd=32, trp=32, tras=64, trc=96
            ),
            voltages=DDR5VoltageParameters(
                vddq=1.1, vpp=1.8, vddq_tx=1.1, vddq_rx=1.1
            )
        )
        
        parent2 = DDR5Configuration(
            frequency=5600,
            timings=DDR5TimingParameters(
                cl=36, trcd=36, trp=36, tras=72, trc=108
            ),
            voltages=DDR5VoltageParameters(
                vddq=1.2, vpp=1.9, vddq_tx=1.2, vddq_rx=1.2
            )
        )
        
        child1, child2 = self.optimizer._crossover_configurations(parent1, parent2)
        
        assert isinstance(child1, DDR5Configuration)
        assert isinstance(child2, DDR5Configuration)
        assert child1.frequency == 5600
        assert child2.frequency == 5600


class TestPerformanceTests:
    """Performance and regression tests."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        self.simulator = DDR5Simulator()
        self.optimizer = PerfectDDR5Optimizer()
    
    def test_simulation_performance(self):
        """Test simulation performance requirements."""
        import time
        
        config = DDR5Configuration(
            frequency=5600,
            timings=DDR5TimingParameters(
                cl=32, trcd=32, trp=32, tras=64, trc=96
            ),
            voltages=DDR5VoltageParameters(
                vddq=1.1, vpp=1.8, vddq_tx=1.1, vddq_rx=1.1
            )
        )
        
        start_time = time.time()
        for _ in range(100):
            self.simulator.simulate_performance(config)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.1  # Should be less than 100ms per simulation
    
    def test_memory_usage(self):
        """Test memory usage requirements."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create many configurations
        configs = []
        for _ in range(1000):
            config = DDR5Configuration(
                frequency=5600,
                timings=DDR5TimingParameters(
                    cl=32, trcd=32, trp=32, tras=64, trc=96
                ),
                voltages=DDR5VoltageParameters(
                    vddq=1.1, vpp=1.8, vddq_tx=1.1, vddq_rx=1.1
                )
            )
            configs.append(config)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not use more than 100MB for 1000 configurations
        assert memory_increase < 100
    
    def test_optimization_convergence(self):
        """Test optimization convergence."""
        # This is a simplified test - would need more iterations in real scenario
        with patch.object(self.optimizer, '_generate_training_data') as mock_data:
            mock_data.return_value = (
                np.random.rand(50, 10),
                np.random.rand(50),
                np.random.rand(50)
            )
            
            self.optimizer.train_models()
            
            result = self.optimizer.optimize(
                target_frequency=5600,
                optimization_goal='balanced',
                generations=10,  # Reduced for testing
                population_size=20  # Reduced for testing
            )
            
            assert 'best_config' in result
            assert 'optimization_history' in result
            assert len(result['optimization_history']) > 0


class TestIntegrationTests:
    """Integration tests for complete workflows."""
    
    def test_complete_optimization_workflow(self):
        """Test complete optimization workflow."""
        optimizer = PerfectDDR5Optimizer()
        
        # Mock training data
        with patch.object(optimizer, '_generate_training_data') as mock_data:
            mock_data.return_value = (
                np.random.rand(50, 10),
                np.random.rand(50),
                np.random.rand(50)
            )
            
            # Train models
            optimizer.train_models()
            assert optimizer.is_trained
            
            # Run optimization
            result = optimizer.optimize(
                target_frequency=5600,
                optimization_goal='balanced',
                generations=5,
                population_size=10
            )
            
            # Verify results
            assert 'best_config' in result
            assert 'fitness_score' in result
            assert 'optimization_history' in result
            assert isinstance(result['best_config'], DDR5Configuration)
            assert result['fitness_score'] > 0
    
    def test_configuration_validation_integration(self):
        """Test configuration validation integration."""
        config = DDR5Configuration(
            frequency=5600,
            timings=DDR5TimingParameters(
                cl=32, trcd=32, trp=32, tras=64, trc=96
            ),
            voltages=DDR5VoltageParameters(
                vddq=1.1, vpp=1.8, vddq_tx=1.1, vddq_rx=1.1
            )
        )
        
        # Test validation
        is_valid, violations = validate_ddr5_configuration(config)
        assert is_valid
        
        # Test simulation with valid config
        simulator = DDR5Simulator()
        result = simulator.simulate_performance(config)
        assert all(
            key in result for key in ['bandwidth', 'latency', 'stability', 'power']
        )
    
    def test_error_handling_integration(self):
        """Test error handling across components."""
        # Test invalid configuration
        invalid_config = DDR5Configuration(
            frequency=5600,
            timings=DDR5TimingParameters(
                cl=32, trcd=32, trp=32, tras=30, trc=50  # Invalid relationships
            ),
            voltages=DDR5VoltageParameters(
                vddq=2.0, vpp=3.0, vddq_tx=2.0, vddq_rx=2.0  # Invalid voltages
            )
        )
        
        is_valid, violations = validate_ddr5_configuration(invalid_config)
        assert not is_valid
        assert len(violations) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
