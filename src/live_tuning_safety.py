"""
Live Tuning Safety Simulation Module
Comprehensive safety testing and validation for live DDR5 tuning capabilities.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time
import random

from .ddr5_models import DDR5Configuration, DDR5TimingParameters, DDR5VoltageParameters
from hardware_detection import DetectedRAMModule


class SafetyLevel(Enum):
    """Safety levels for live tuning operations."""
    CRITICAL_UNSAFE = "critical_unsafe"
    UNSAFE = "unsafe" 
    CAUTION = "caution"
    SAFE = "safe"
    VERIFIED_SAFE = "verified_safe"


class TestType(Enum):
    """Types of safety tests."""
    VOLTAGE_VALIDATION = "voltage_validation"
    TIMING_VALIDATION = "timing_validation"
    THERMAL_SIMULATION = "thermal_simulation"
    STABILITY_PREDICTION = "stability_prediction"
    ROLLBACK_VERIFICATION = "rollback_verification"
    BIOS_COMPATIBILITY = "bios_compatibility"
    HARDWARE_LIMITS = "hardware_limits"


@dataclass
class SafetyTestResult:
    """Result of a safety test."""
    test_type: TestType
    safety_level: SafetyLevel
    score: float  # 0.0 to 1.0
    warnings: List[str]
    recommendations: List[str]
    details: Dict[str, any]
    execution_time: float


@dataclass
class LiveTuningSafetyReport:
    """Comprehensive safety assessment report."""
    overall_safety: SafetyLevel
    overall_score: float
    test_results: List[SafetyTestResult]
    detected_hardware: List[DetectedRAMModule]
    recommended_config: Optional[DDR5Configuration]
    critical_warnings: List[str]
    safety_recommendations: List[str]
    rollback_plan: Dict[str, str]
    estimated_risk_level: str


class LiveTuningSafetyValidator:
    """
    Advanced safety validation system for live DDR5 tuning.
    Tests multiple safety aspects before allowing real hardware changes.
    """
    
    def __init__(self):
        """Initialize the safety validator."""
        self.test_results: List[SafetyTestResult] = []
        self.hardware_limits = self._load_hardware_limits()
        
    def _load_hardware_limits(self) -> Dict[str, Dict[str, float]]:
        """Load known hardware safety limits for different manufacturers."""
        return {
            "Kingston": {
                "max_voltage_ddr5": 1.35,  # Conservative limit
                "max_frequency": 8400,
                "min_cl": 20,
                "max_cl": 52,
                "thermal_limit": 85.0
            },
            "Generic": {
                "max_voltage_ddr5": 1.25,  # Very conservative
                "max_frequency": 6400,
                "min_cl": 30,
                "max_cl": 50,
                "thermal_limit": 80.0
            }
        }
    
    def run_comprehensive_safety_test(
        self, 
        target_config: DDR5Configuration,
        detected_modules: List[DetectedRAMModule]
    ) -> LiveTuningSafetyReport:
        """
        Run comprehensive safety validation for live tuning.
        
        Args:
            target_config: Proposed DDR5 configuration
            detected_modules: Currently detected hardware
            
        Returns:
            Comprehensive safety report
        """
        print("🔒 Starting Live Tuning Safety Validation...")
        self.test_results = []
        
        # Run all safety tests
        tests = [
            self._test_voltage_safety,
            self._test_timing_safety,
            self._test_thermal_safety,
            self._test_stability_prediction,
            self._test_rollback_capability,
            self._test_bios_compatibility,
            self._test_hardware_limits
        ]
        
        for test in tests:
            try:
                result = test(target_config, detected_modules)
                self.test_results.append(result)
                print(f"✅ {result.test_type.value}: {result.safety_level.value}")
            except Exception as e:
                # Create error result
                error_result = SafetyTestResult(
                    test_type=TestType.VOLTAGE_VALIDATION,  # Default
                    safety_level=SafetyLevel.CRITICAL_UNSAFE,
                    score=0.0,
                    warnings=[f"Test failed: {str(e)}"],
                    recommendations=["Skip live tuning due to test failure"],
                    details={"error": str(e)},
                    execution_time=0.0
                )
                self.test_results.append(error_result)
        
        # Generate comprehensive report
        return self._generate_safety_report(target_config, detected_modules)
    
    def _test_voltage_safety(
        self, 
        config: DDR5Configuration, 
        modules: List[DetectedRAMModule]
    ) -> SafetyTestResult:
        """Test voltage safety parameters."""
        start_time = time.time()
        warnings = []
        recommendations = []
        
        # Get voltage limits for detected hardware
        voltage_limits = self._get_voltage_limits(modules)
        
        vddq = config.voltage_parameters.vddq
        vpp = config.voltage_parameters.vpp
        
        safety_score = 1.0
        
        # Check VDDQ safety
        if vddq > voltage_limits["max_vddq"]:
            safety_score *= 0.2
            warnings.append(f"VDDQ {vddq}V exceeds safe limit {voltage_limits['max_vddq']}V")
        elif vddq > voltage_limits["recommended_vddq"]:
            safety_score *= 0.7
            warnings.append(f"VDDQ {vddq}V above recommended {voltage_limits['recommended_vddq']}V")
        
        # Check VPP safety
        if vpp > voltage_limits["max_vpp"]:
            safety_score *= 0.3
            warnings.append(f"VPP {vpp}V exceeds safe limit {voltage_limits['max_vpp']}V")
        
        # Voltage stability check
        voltage_stability = self._simulate_voltage_stability(vddq, vpp)
        safety_score *= voltage_stability
        
        if voltage_stability < 0.8:
            warnings.append("Voltage stability concerns detected")
            recommendations.append("Consider lower voltages for better stability")
        
        # Determine safety level
        if safety_score >= 0.9:
            safety_level = SafetyLevel.VERIFIED_SAFE
        elif safety_score >= 0.7:
            safety_level = SafetyLevel.SAFE
        elif safety_score >= 0.5:
            safety_level = SafetyLevel.CAUTION
        elif safety_score >= 0.3:
            safety_level = SafetyLevel.UNSAFE
        else:
            safety_level = SafetyLevel.CRITICAL_UNSAFE
        
        if safety_level in [SafetyLevel.SAFE, SafetyLevel.VERIFIED_SAFE]:
            recommendations.append("Voltage parameters are within safe limits")
        
        return SafetyTestResult(
            test_type=TestType.VOLTAGE_VALIDATION,
            safety_level=safety_level,
            score=safety_score,
            warnings=warnings,
            recommendations=recommendations,
            details={
                "vddq": vddq,
                "vpp": vpp,
                "limits": voltage_limits,
                "stability": voltage_stability
            },
            execution_time=time.time() - start_time
        )
    
    def _test_timing_safety(
        self, 
        config: DDR5Configuration, 
        modules: List[DetectedRAMModule]
    ) -> SafetyTestResult:
        """Test timing parameter safety."""
        start_time = time.time()
        warnings = []
        recommendations = []
        
        timings = config.timing_parameters
        frequency = config.frequency
        
        safety_score = 1.0
        
        # Check timing relationships
        timing_violations = self._check_timing_relationships(timings, frequency)
        if timing_violations:
            safety_score *= 0.4
            warnings.extend(timing_violations)
        
        # Check against hardware capabilities
        hardware_compatibility = self._check_hardware_timing_limits(timings, modules)
        safety_score *= hardware_compatibility
        
        # Stability prediction based on timings
        stability_prediction = self._predict_timing_stability(timings, frequency)
        safety_score *= stability_prediction
        
        if stability_prediction < 0.8:
            warnings.append("Aggressive timings may cause instability")
            recommendations.append("Consider more conservative timing values")
        
        # Determine safety level
        if safety_score >= 0.9:
            safety_level = SafetyLevel.VERIFIED_SAFE
        elif safety_score >= 0.7:
            safety_level = SafetyLevel.SAFE
        elif safety_score >= 0.5:
            safety_level = SafetyLevel.CAUTION
        elif safety_score >= 0.3:
            safety_level = SafetyLevel.UNSAFE
        else:
            safety_level = SafetyLevel.CRITICAL_UNSAFE
        
        return SafetyTestResult(
            test_type=TestType.TIMING_VALIDATION,
            safety_level=safety_level,
            score=safety_score,
            warnings=warnings,
            recommendations=recommendations,
            details={
                "timing_violations": timing_violations,
                "hardware_compatibility": hardware_compatibility,
                "stability_prediction": stability_prediction
            },
            execution_time=time.time() - start_time
        )
    
    def _test_thermal_safety(
        self, 
        config: DDR5Configuration, 
        modules: List[DetectedRAMModule]
    ) -> SafetyTestResult:
        """Simulate thermal safety under proposed configuration."""
        start_time = time.time()
        warnings = []
        recommendations = []
        
        # Simulate thermal load
        thermal_load = self._simulate_thermal_load(config, modules)
        
        safety_score = 1.0
        
        # Check thermal limits
        for module in modules:
            manufacturer = module.manufacturer
            limits = self.hardware_limits.get(manufacturer, self.hardware_limits["Generic"])
            
            if thermal_load > limits["thermal_limit"]:
                safety_score *= 0.2
                warnings.append(f"Thermal load {thermal_load:.1f}°C exceeds limit {limits['thermal_limit']}°C")
            elif thermal_load > limits["thermal_limit"] - 10:
                safety_score *= 0.7
                warnings.append(f"Thermal load {thermal_load:.1f}°C approaching limit")
        
        if thermal_load > 70:
            recommendations.append("Consider improved cooling before live tuning")
        
        # Determine safety level
        if safety_score >= 0.9:
            safety_level = SafetyLevel.VERIFIED_SAFE
        elif safety_score >= 0.7:
            safety_level = SafetyLevel.SAFE
        elif safety_score >= 0.5:
            safety_level = SafetyLevel.CAUTION
        else:
            safety_level = SafetyLevel.UNSAFE
        
        return SafetyTestResult(
            test_type=TestType.THERMAL_SIMULATION,
            safety_level=safety_level,
            score=safety_score,
            warnings=warnings,
            recommendations=recommendations,
            details={"thermal_load": thermal_load},
            execution_time=time.time() - start_time
        )
    
    def _test_stability_prediction(
        self, 
        config: DDR5Configuration, 
        modules: List[DetectedRAMModule]
    ) -> SafetyTestResult:
        """Predict system stability with proposed configuration."""
        start_time = time.time()
        warnings = []
        recommendations = []
        
        # AI-based stability prediction
        stability_score = self._predict_system_stability(config, modules)
        
        if stability_score < 0.6:
            warnings.append("Low stability prediction - high risk of crashes")
            recommendations.append("Use more conservative settings")
        elif stability_score < 0.8:
            warnings.append("Moderate stability concerns")
            recommendations.append("Extensive testing recommended")
        
        # Determine safety level based on stability prediction
        if stability_score >= 0.95:
            safety_level = SafetyLevel.VERIFIED_SAFE
        elif stability_score >= 0.85:
            safety_level = SafetyLevel.SAFE
        elif stability_score >= 0.7:
            safety_level = SafetyLevel.CAUTION
        elif stability_score >= 0.5:
            safety_level = SafetyLevel.UNSAFE
        else:
            safety_level = SafetyLevel.CRITICAL_UNSAFE
        
        return SafetyTestResult(
            test_type=TestType.STABILITY_PREDICTION,
            safety_level=safety_level,
            score=stability_score,
            warnings=warnings,
            recommendations=recommendations,
            details={"predicted_stability": stability_score},
            execution_time=time.time() - start_time
        )
    
    def _test_rollback_capability(
        self, 
        config: DDR5Configuration, 
        modules: List[DetectedRAMModule]
    ) -> SafetyTestResult:
        """Test ability to rollback changes safely."""
        start_time = time.time()
        warnings = []
        recommendations = []
        
        # Simulate rollback mechanisms
        rollback_score = 1.0
        
        # Check if BIOS supports safe rollback
        bios_rollback = self._check_bios_rollback_support()
        if not bios_rollback:
            rollback_score *= 0.5
            warnings.append("BIOS rollback capability uncertain")
        
        # Check for backup configuration storage
        backup_capability = self._check_backup_capability()
        rollback_score *= backup_capability
        
        if backup_capability < 0.8:
            warnings.append("Limited backup/restore capability")
            recommendations.append("Manual BIOS reset may be required")
        
        safety_level = SafetyLevel.SAFE if rollback_score >= 0.8 else SafetyLevel.CAUTION
        
        return SafetyTestResult(
            test_type=TestType.ROLLBACK_VERIFICATION,
            safety_level=safety_level,
            score=rollback_score,
            warnings=warnings,
            recommendations=recommendations,
            details={
                "bios_rollback": bios_rollback,
                "backup_capability": backup_capability
            },
            execution_time=time.time() - start_time
        )
    
    def _test_bios_compatibility(
        self, 
        config: DDR5Configuration, 
        modules: List[DetectedRAMModule]
    ) -> SafetyTestResult:
        """Test BIOS/UEFI compatibility with proposed changes."""
        start_time = time.time()
        warnings = []
        recommendations = []
        
        # Simulate BIOS compatibility check
        compatibility_score = self._check_bios_compatibility(config)
        
        if compatibility_score < 0.7:
            warnings.append("BIOS may not support all requested parameters")
            recommendations.append("Verify BIOS version and capabilities")
        
        safety_level = SafetyLevel.SAFE if compatibility_score >= 0.8 else SafetyLevel.CAUTION
        
        return SafetyTestResult(
            test_type=TestType.BIOS_COMPATIBILITY,
            safety_level=safety_level,
            score=compatibility_score,
            warnings=warnings,
            recommendations=recommendations,
            details={"compatibility_score": compatibility_score},
            execution_time=time.time() - start_time
        )
    
    def _test_hardware_limits(
        self, 
        config: DDR5Configuration, 
        modules: List[DetectedRAMModule]
    ) -> SafetyTestResult:
        """Test against absolute hardware limits."""
        start_time = time.time()
        warnings = []
        recommendations = []
        
        safety_score = 1.0
        
        for module in modules:
            manufacturer = module.manufacturer
            limits = self.hardware_limits.get(manufacturer, self.hardware_limits["Generic"])
            
            # Check frequency limits
            if config.frequency > limits["max_frequency"]:
                safety_score *= 0.1
                warnings.append(f"Frequency {config.frequency} exceeds hardware limit {limits['max_frequency']}")
            
            # Check CL limits
            if config.timing_parameters.cl < limits["min_cl"]:
                safety_score *= 0.3
                warnings.append(f"CL {config.timing_parameters.cl} too aggressive")
        
        safety_level = SafetyLevel.SAFE if safety_score >= 0.8 else SafetyLevel.UNSAFE
        
        return SafetyTestResult(
            test_type=TestType.HARDWARE_LIMITS,
            safety_level=safety_level,
            score=safety_score,
            warnings=warnings,
            recommendations=recommendations,
            details={"hardware_limits_check": safety_score},
            execution_time=time.time() - start_time
        )
    
    # Helper methods for simulations
    def _get_voltage_limits(self, modules: List[DetectedRAMModule]) -> Dict[str, float]:
        """Get voltage limits for detected modules."""
        return {
            "max_vddq": 1.35,
            "recommended_vddq": 1.25,
            "max_vpp": 2.0,
            "recommended_vpp": 1.8
        }
    
    def _simulate_voltage_stability(self, vddq: float, vpp: float) -> float:
        """Simulate voltage stability."""
        base_stability = 1.0
        
        # Higher voltages reduce stability
        if vddq > 1.2:
            base_stability *= (1.4 - vddq)
        if vpp > 1.9:
            base_stability *= (2.1 - vpp)
        
        return max(0.0, min(1.0, base_stability + random.uniform(-0.1, 0.05)))
    
    def _check_timing_relationships(self, timings: DDR5TimingParameters, frequency: int) -> List[str]:
        """Check DDR5 timing relationships."""
        violations = []
        
        # Basic DDR5 timing rules
        if timings.tras < (timings.trcd + timings.cl):
            violations.append("tRAS must be >= tRCD + CL")
        
        if timings.trc < (timings.tras + timings.trp):
            violations.append("tRC must be >= tRAS + tRP")
        
        if timings.cl < 20:  # Conservative DDR5 minimum
            violations.append("CL too aggressive for DDR5")
        
        return violations
    
    def _check_hardware_timing_limits(self, timings: DDR5TimingParameters, modules: List[DetectedRAMModule]) -> float:
        """Check timings against hardware capabilities."""
        compatibility = 1.0
        
        # Check against known module capabilities
        for module in modules:
            if "Kingston" in module.manufacturer:
                # Kingston Fury modules are typically more flexible
                if timings.cl < 28:
                    compatibility *= 0.8
            else:
                # Generic modules are more conservative
                if timings.cl < 32:
                    compatibility *= 0.6
        
        return compatibility
    
    def _predict_timing_stability(self, timings: DDR5TimingParameters, frequency: int) -> float:
        """Predict stability based on timings."""
        # Simple heuristic: aggressive timings reduce stability
        base_stability = 1.0
        
        cl_ratio = timings.cl / (frequency / 100)  # Rough ratio
        if cl_ratio < 0.8:
            base_stability *= 0.7
        
        return max(0.0, min(1.0, base_stability + random.uniform(-0.1, 0.05)))
    
    def _simulate_thermal_load(self, config: DDR5Configuration, modules: List[DetectedRAMModule]) -> float:
        """Simulate thermal load."""
        base_temp = 45.0  # Ambient + base load
        
        # Higher frequency increases heat
        freq_factor = (config.frequency - 3200) / 100 * 2.0
        
        # Higher voltage increases heat exponentially
        voltage_factor = (config.voltage_parameters.vddq - 1.1) * 20.0
        
        total_temp = base_temp + freq_factor + voltage_factor
        return max(30.0, total_temp + random.uniform(-5, 10))
    
    def _predict_system_stability(self, config: DDR5Configuration, modules: List[DetectedRAMModule]) -> float:
        """AI-based stability prediction."""
        # Simplified prediction model
        base_stability = 0.9
        
        # Frequency factor
        if config.frequency > 6000:
            base_stability *= 0.85
        
        # Voltage factor
        if config.voltage_parameters.vddq > 1.25:
            base_stability *= 0.8
        
        # Timing aggressiveness
        if config.timing_parameters.cl < 32:
            base_stability *= 0.9
        
        return max(0.0, min(1.0, base_stability + random.uniform(-0.1, 0.1)))
    
    def _check_bios_rollback_support(self) -> bool:
        """Check if BIOS supports safe rollback."""
        # In real implementation, would check BIOS capabilities
        return random.choice([True, False])  # Simulated
    
    def _check_backup_capability(self) -> float:
        """Check backup/restore capability."""
        return random.uniform(0.6, 1.0)  # Simulated
    
    def _check_bios_compatibility(self, config: DDR5Configuration) -> float:
        """Check BIOS compatibility."""
        return random.uniform(0.7, 1.0)  # Simulated
    
    def _generate_safety_report(
        self, 
        config: DDR5Configuration, 
        modules: List[DetectedRAMModule]
    ) -> LiveTuningSafetyReport:
        """Generate comprehensive safety report."""
        
        # Calculate overall safety
        safety_scores = [result.score for result in self.test_results]
        overall_score = np.mean(safety_scores) if safety_scores else 0.0
        
        # Determine overall safety level
        critical_failures = [r for r in self.test_results if r.safety_level == SafetyLevel.CRITICAL_UNSAFE]
        unsafe_results = [r for r in self.test_results if r.safety_level == SafetyLevel.UNSAFE]
        
        if critical_failures:
            overall_safety = SafetyLevel.CRITICAL_UNSAFE
        elif unsafe_results:
            overall_safety = SafetyLevel.UNSAFE
        elif overall_score >= 0.9:
            overall_safety = SafetyLevel.VERIFIED_SAFE
        elif overall_score >= 0.7:
            overall_safety = SafetyLevel.SAFE
        else:
            overall_safety = SafetyLevel.CAUTION
        
        # Collect warnings and recommendations
        all_warnings = []
        all_recommendations = []
        
        for result in self.test_results:
            all_warnings.extend(result.warnings)
            all_recommendations.extend(result.recommendations)
        
        # Critical warnings
        critical_warnings = [w for r in self.test_results for w in r.warnings 
                           if r.safety_level in [SafetyLevel.CRITICAL_UNSAFE, SafetyLevel.UNSAFE]]
        
        # Generate rollback plan
        rollback_plan = {
            "method": "BIOS manual reset",
            "steps": "1. Power off, 2. Clear CMOS, 3. Reset to defaults",
            "estimated_time": "5-10 minutes",
            "risk_level": "Low"
        }
        
        # Risk assessment
        if overall_safety == SafetyLevel.CRITICAL_UNSAFE:
            risk_level = "CRITICAL - Do not proceed"
        elif overall_safety == SafetyLevel.UNSAFE:
            risk_level = "HIGH - Not recommended"
        elif overall_safety == SafetyLevel.CAUTION:
            risk_level = "MEDIUM - Proceed with caution"
        elif overall_safety == SafetyLevel.SAFE:
            risk_level = "LOW - Generally safe"
        else:
            risk_level = "MINIMAL - Verified safe"
        
        return LiveTuningSafetyReport(
            overall_safety=overall_safety,
            overall_score=overall_score,
            test_results=self.test_results,
            detected_hardware=modules,
            recommended_config=config if overall_safety in [SafetyLevel.SAFE, SafetyLevel.VERIFIED_SAFE] else None,
            critical_warnings=critical_warnings,
            safety_recommendations=list(set(all_recommendations)),
            rollback_plan=rollback_plan,
            estimated_risk_level=risk_level
        )


# Convenience function for quick safety check
def quick_safety_check(config: DDR5Configuration, modules: List[DetectedRAMModule]) -> str:
    """Quick safety assessment for display purposes."""
    validator = LiveTuningSafetyValidator()
    report = validator.run_comprehensive_safety_test(config, modules)
    
    if report.overall_safety == SafetyLevel.VERIFIED_SAFE:
        return "🟢 VERIFIED SAFE"
    elif report.overall_safety == SafetyLevel.SAFE:
        return "🟢 SAFE"
    elif report.overall_safety == SafetyLevel.CAUTION:
        return "🟡 CAUTION"
    elif report.overall_safety == SafetyLevel.UNSAFE:
        return "🔴 UNSAFE"
    else:
        return "⚠️ CRITICAL"
