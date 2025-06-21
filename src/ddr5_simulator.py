"""
DDR5 Memory Simulator
Core simulation engine for DDR5 memory behavior and performance.
"""

import numpy as np
from typing import Dict, List, Any
from .ddr5_models import DDR5Configuration, DDR5TimingParameters


class DDR5Simulator:
    """DDR5 memory behavior simulator."""
    
    def __init__(self):
        """Initialize the DDR5 simulator."""
        self.current_config = DDR5Configuration()
        self.simulation_cache = {}
        
        # Performance coefficients (derived from empirical data)
        self.performance_coefficients = {
            'bandwidth_efficiency': 0.85,  # Real vs theoretical bandwidth
            'latency_penalty': 1.15,       # Additional latency factors
            'power_coefficient': 0.012,    # Power per MHz per volt
            'thermal_coefficient': 0.008,  # Temperature rise per watt
        }
    
    def load_configuration(self, config: DDR5Configuration) -> None:
        """Load a DDR5 configuration for simulation."""
        self.current_config = config
        self.current_config.calculate_performance_metrics()
        self.simulation_cache.clear()
    
    def simulate_bandwidth(
        self, 
        access_pattern: str = "sequential",
        queue_depth: int = 32
    ) -> Dict[str, float]:
        """
        Simulate memory bandwidth under different access patterns.
        
        Args:
            access_pattern: "sequential", "random", "mixed"
            queue_depth: Command queue depth
            
        Returns:
            Dictionary with bandwidth metrics
        """
        cache_key = f"bandwidth_{access_pattern}_{queue_depth}"
        if cache_key in self.simulation_cache:
            return self.simulation_cache[cache_key]
        
        base_bandwidth = self.current_config.bandwidth_gbps
        efficiency = self.performance_coefficients['bandwidth_efficiency']
        
        # Access pattern efficiency
        pattern_efficiency = {
            'sequential': 0.95,
            'random': 0.60,
            'mixed': 0.75
        }
        
        # Queue depth impact
        queue_efficiency = min(1.0, 0.6 + (queue_depth / 64) * 0.4)
        
        # Timing impact
        timing_efficiency = self._calculate_timing_efficiency()
        
        effective_bandwidth = (
            base_bandwidth *
            efficiency *
            pattern_efficiency[access_pattern] *
            queue_efficiency *
            timing_efficiency
        )
        
        result = {
            'theoretical_bandwidth_gbps': base_bandwidth,
            'effective_bandwidth_gbps': effective_bandwidth,
            'efficiency_percent': (effective_bandwidth / base_bandwidth) * 100,
            'access_pattern': access_pattern,
            'queue_depth': queue_depth
        }
        
        self.simulation_cache[cache_key] = result
        return result
    
    def simulate_latency(
        self, 
        access_pattern: str = "random",
        bank_conflicts: bool = True
    ) -> Dict[str, float]:
        """
        Simulate memory latency characteristics.
        
        Args:
            access_pattern: "random", "sequential", "worst_case"
            bank_conflicts: Whether to simulate bank conflicts
            
        Returns:
            Dictionary with latency metrics
        """
        cache_key = f"latency_{access_pattern}_{bank_conflicts}"
        if cache_key in self.simulation_cache:
            return self.simulation_cache[cache_key]
        
        base_latency = self.current_config.latency_ns
        
        # Access pattern impact
        pattern_penalty = {
            'sequential': 1.0,
            'random': 1.2,
            'worst_case': 1.5
        }
        
        # Bank conflict penalty
        conflict_penalty = 1.3 if bank_conflicts else 1.0
        
        # Command overhead
        command_overhead = self._calculate_command_overhead()
        
        effective_latency = (
            base_latency * 
            pattern_penalty[access_pattern] * 
            conflict_penalty +
            command_overhead
        )
        
        result = {
            'base_latency_ns': base_latency,
            'effective_latency_ns': effective_latency,
            'overhead_ns': command_overhead,
            'access_pattern': access_pattern,
            'bank_conflicts_enabled': bank_conflicts
        }
        
        self.simulation_cache[cache_key] = result
        return result
    
    def simulate_power_consumption(self) -> Dict[str, float]:
        """
        Simulate power consumption based on configuration.
        
        Returns:
            Dictionary with power metrics
        """
        cache_key = "power_consumption"
        if cache_key in self.simulation_cache:
            return self.simulation_cache[cache_key]
        
        # Base power consumption (mW)
        frequency_mhz = self.current_config.frequency
        vddq = self.current_config.voltages.vddq
        vpp = self.current_config.voltages.vpp
        
        # Dynamic power (frequency and voltage dependent)
        dynamic_power = (
            frequency_mhz * 
            self.performance_coefficients['power_coefficient'] * 
            (vddq ** 2)
        )
        
        # Static power (voltage dependent)
        static_power = 200 + (vddq - 1.1) * 500 + (vpp - 1.8) * 300
        
        # Additional power for higher frequencies
        if frequency_mhz > 5600:
            frequency_penalty = (frequency_mhz - 5600) * 0.1
            dynamic_power += frequency_penalty
        
        total_power = dynamic_power + static_power
        
        result = {
            'dynamic_power_mw': dynamic_power,
            'static_power_mw': static_power,
            'total_power_mw': total_power,
            'power_efficiency_mb_per_mw': 
                (self.current_config.bandwidth_gbps * 1000) / total_power
        }
        
        self.simulation_cache[cache_key] = result
        return result
    
    def run_stability_test(
        self, 
        test_duration_minutes: int = 30,
        stress_level: str = "medium"
    ) -> Dict[str, Any]:
        """
        Simulate memory stability testing.
        
        Args:
            test_duration_minutes: Duration of stability test
            stress_level: "light", "medium", "heavy", "extreme"
            
        Returns:
            Dictionary with stability test results
        """
        config_violations = self.current_config.validate_configuration()
        stability_score = self.current_config.get_stability_estimate()
        
        # Stress level impact
        stress_factors = {
            'light': 0.9,
            'medium': 1.0,
            'heavy': 1.2,
            'extreme': 1.5
        }
        
        stress_factor = stress_factors[stress_level]
        adjusted_stability = stability_score / stress_factor
        
        # Duration impact
        duration_factor = 1.0 - (test_duration_minutes / 1000)  # Slight degradation over time
        final_stability = adjusted_stability * duration_factor
        
        # Determine test result
        if final_stability >= 90:
            test_result = "EXCELLENT"
            error_rate = 0.0
        elif final_stability >= 75:
            test_result = "STABLE"
            error_rate = 0.001
        elif final_stability >= 60:
            test_result = "MARGINAL"
            error_rate = 0.01
        elif final_stability >= 40:
            test_result = "UNSTABLE"
            error_rate = 0.1
        else:
            test_result = "FAILED"
            error_rate = 1.0
        
        return {
            'test_result': test_result,
            'stability_score': final_stability,
            'error_rate': error_rate,
            'violations': config_violations,
            'test_duration_minutes': test_duration_minutes,
            'stress_level': stress_level,
            'recommendation': self._get_stability_recommendation(final_stability, config_violations)
        }
    
    def calculate_stability(self) -> float:
        """
        Calculate the stability score for the current DDR5 configuration.

        Returns:
            Stability score as a float (0.0 to 1.0).
        """
        from .advanced_ai_engine import AdvancedAIEngine

        ai_engine = AdvancedAIEngine()
        return ai_engine.calculate_stability_score(self.current_config)
    
    def _calculate_timing_efficiency(self) -> float:
        """Calculate efficiency based on timing parameters."""
        timings = self.current_config.timings
        
        # Compare against JEDEC baseline
        baseline = DDR5TimingParameters()
        
        # Calculate relative tightness
        cl_ratio = timings.cl / baseline.cl
        trcd_ratio = timings.trcd / baseline.trcd
        trp_ratio = timings.trp / baseline.trp
        
        # Tighter timings = better efficiency but more risk
        avg_ratio = (cl_ratio + trcd_ratio + trp_ratio) / 3
        
        # Efficiency bonus for tighter timings
        if avg_ratio < 0.8:
            return 1.15  # 15% bonus for very tight timings
        elif avg_ratio < 0.9:
            return 1.08  # 8% bonus for tight timings
        elif avg_ratio < 1.0:
            return 1.03  # 3% bonus for slightly tight timings
        else:
            return max(0.95, 1.1 - avg_ratio)  # Penalty for loose timings
    
    def _calculate_command_overhead(self) -> float:
        """
        Calculate command processing overhead.

        Returns:
            Command overhead in nanoseconds.
        """
        # Base command overhead in nanoseconds
        base_overhead = 2.5

        # Additional overhead for high frequencies
        frequency_overhead = max(
            0, (self.current_config.frequency - 5600) * 0.001
        )

        return base_overhead + frequency_overhead
    
    def _get_stability_recommendation(
        self, 
        stability_score: float,
        config_violations: List[str]
    ) -> str:
        """
        Generate stability recommendations based on score and violations.

        Args:
            stability_score: Calculated stability score
            config_violations: List of configuration violations

        Returns:
            Recommendation string
        """
        recommendations = []
        
        if stability_score < 0.6:
            recommendations.append("Increase voltage slightly")
            recommendations.append("Reduce frequency")
        if stability_score < 0.7:
            recommendations.append("Consider better cooling solutions")
        if stability_score < 0.8:
            recommendations.append("Check memory seating and motherboard slots")
        
        # Specific recommendations based on common violations
        for violation in config_violations:
            if violation == "timing_violations":
                recommendations.append("Relax timings in BIOS")
            elif violation == "voltage_violations":
                recommendations.append("Adjust voltage settings")
            elif violation == "temperature_violations":
                recommendations.append("Improve cooling or reduce load")
            elif violation == "frequency_violations":
                recommendations.append("Lower the memory frequency")
        
        if not recommendations:
            return "No action needed. System is stable."
        
        return " | ".join(recommendations)
