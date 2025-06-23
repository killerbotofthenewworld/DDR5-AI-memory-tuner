"""
Advanced Hardware Damage Prevention and Predictive Maintenance System
"""
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np


class RiskLevel(Enum):
    """Hardware risk levels"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Alert types for hardware monitoring"""
    VOLTAGE_WARNING = "voltage_warning"
    THERMAL_WARNING = "thermal_warning"
    STABILITY_WARNING = "stability_warning"
    FREQUENCY_WARNING = "frequency_warning"
    TIMING_WARNING = "timing_warning"
    DEGRADATION_WARNING = "degradation_warning"


@dataclass
class HardwareHealth:
    """Hardware health status"""
    component: str
    health_score: float  # 0-100
    risk_level: RiskLevel
    estimated_lifespan_days: int
    degradation_rate: float  # % per day
    last_updated: datetime
    issues: List[str]
    recommendations: List[str]


@dataclass
class SafetyLimits:
    """Safety limits for DDR5 parameters"""
    max_vddq: float = 1.35
    max_vpp: float = 2.0
    max_temperature: float = 85.0
    max_frequency: int = 8400
    min_stability_score: float = 90.0
    max_error_rate: float = 0.001


class HardwareDamagePrevention:
    """Advanced hardware damage prevention system"""
    
    def __init__(self):
        self.safety_limits = SafetyLimits()
        self.health_history: Dict[str, List[HardwareHealth]] = {}
        self.active_alerts: List[Dict] = []
        self.monitoring_enabled = True
        
    def validate_configuration_safety(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive safety validation for DDR5 configuration"""
        
        safety_report = {
            "safe": True,
            "risk_level": RiskLevel.SAFE,
            "violations": [],
            "warnings": [],
            "recommendations": [],
            "estimated_damage_risk": 0.0
        }
        
        # Voltage safety checks
        vddq = config.get('vddq', 1.1)
        vpp = config.get('vpp', 1.8)
        
        if vddq > self.safety_limits.max_vddq:
            safety_report["violations"].append({
                "type": "voltage",
                "parameter": "VDDQ",
                "value": vddq,
                "limit": self.safety_limits.max_vddq,
                "risk": "High voltage can cause permanent damage"
            })
            safety_report["safe"] = False
            safety_report["risk_level"] = RiskLevel.HIGH
        
        if vpp > self.safety_limits.max_vpp:
            safety_report["violations"].append({
                "type": "voltage", 
                "parameter": "VPP",
                "value": vpp,
                "limit": self.safety_limits.max_vpp,
                "risk": "Excessive VPP can damage memory controller"
            })
            safety_report["safe"] = False
            safety_report["risk_level"] = RiskLevel.CRITICAL
        
        # Frequency safety checks
        frequency = config.get('frequency', 4800)
        if frequency > self.safety_limits.max_frequency:
            safety_report["warnings"].append({
                "type": "frequency",
                "message": f"Frequency {frequency} MT/s exceeds typical limits",
                "recommendation": "Ensure adequate cooling and power delivery"
            })
            
        # Timing relationship validation
        timing_issues = self._validate_timing_relationships(config)
        if timing_issues:
            safety_report["violations"].extend(timing_issues)
            safety_report["safe"] = False
            
        # Calculate damage risk score
        safety_report["estimated_damage_risk"] = self._calculate_damage_risk(config)
        
        # Generate recommendations
        safety_report["recommendations"] = self._generate_safety_recommendations(config, safety_report)
        
        return safety_report
    
    def predict_hardware_health(self, 
                               current_metrics: Dict[str, Any],
                               usage_pattern: Dict[str, Any]) -> Dict[str, HardwareHealth]:
        """Predict hardware health and lifespan"""
        
        health_predictions = {}
        
        # Memory modules health prediction
        memory_health = self._predict_memory_health(current_metrics, usage_pattern)
        health_predictions["memory"] = memory_health
        
        # Memory controller health
        controller_health = self._predict_controller_health(current_metrics, usage_pattern)
        health_predictions["controller"] = controller_health
        
        # Motherboard traces health
        trace_health = self._predict_trace_health(current_metrics, usage_pattern)
        health_predictions["traces"] = trace_health
        
        # Update health history
        self._update_health_history(health_predictions)
        
        return health_predictions
    
    def monitor_real_time_health(self, live_data: Dict[str, Any]) -> List[Dict]:
        """Monitor real-time hardware health and generate alerts"""
        
        alerts = []
        
        # Temperature monitoring
        temp = live_data.get('temperature', 0)
        if temp > self.safety_limits.max_temperature:
            alerts.append({
                "type": AlertType.THERMAL_WARNING,
                "severity": RiskLevel.HIGH,
                "message": f"Temperature {temp}°C exceeds safe limit",
                "action": "Reduce frequency or improve cooling",
                "timestamp": datetime.now()
            })
        
        # Stability monitoring
        stability = live_data.get('stability_score', 100)
        if stability < self.safety_limits.min_stability_score:
            alerts.append({
                "type": AlertType.STABILITY_WARNING,
                "severity": RiskLevel.MEDIUM,
                "message": f"Stability score {stability}% below threshold",
                "action": "Adjust timings or reduce frequency",
                "timestamp": datetime.now()
            })
        
        # Error rate monitoring
        error_rate = live_data.get('error_rate', 0)
        if error_rate > self.safety_limits.max_error_rate:
            alerts.append({
                "type": AlertType.STABILITY_WARNING,
                "severity": RiskLevel.HIGH,
                "message": f"Error rate {error_rate} too high",
                "action": "Immediate configuration adjustment needed",
                "timestamp": datetime.now()
            })
        
        # Update active alerts
        self.active_alerts.extend(alerts)
        self._cleanup_old_alerts()
        
        return alerts
    
    def generate_maintenance_schedule(self, 
                                   health_data: Dict[str, HardwareHealth]) -> Dict[str, Any]:
        """Generate predictive maintenance schedule"""
        
        schedule = {
            "immediate_actions": [],
            "weekly_tasks": [],
            "monthly_tasks": [],
            "quarterly_tasks": [],
            "replacement_schedule": []
        }
        
        for component, health in health_data.items():
            # Immediate actions for critical health
            if health.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                schedule["immediate_actions"].append({
                    "component": component,
                    "action": f"Immediate attention required for {component}",
                    "priority": "critical",
                    "estimated_time": "30 minutes"
                })
            
            # Predictive replacement schedule
            if health.estimated_lifespan_days < 365:
                schedule["replacement_schedule"].append({
                    "component": component,
                    "estimated_replacement_date": datetime.now() + timedelta(days=health.estimated_lifespan_days),
                    "confidence": self._calculate_prediction_confidence(health),
                    "budget_estimate": self._estimate_replacement_cost(component)
                })
            
            # Regular maintenance tasks
            if health.health_score < 80:
                schedule["monthly_tasks"].append({
                    "component": component,
                    "task": f"Deep analysis and optimization of {component}",
                    "expected_improvement": f"{10-health.degradation_rate:.1f}% health recovery"
                })
        
        return schedule
    
    def _validate_timing_relationships(self, config: Dict[str, Any]) -> List[Dict]:
        """Validate DDR5 timing relationships"""
        violations = []
        
        cl = config.get('cl', 30)
        trcd = config.get('trcd', 30)
        trp = config.get('trp', 30)
        tras = config.get('tras', 60)
        trc = config.get('trc', 90)
        
        # Basic timing relationships
        if tras < (trcd + cl):
            violations.append({
                "type": "timing",
                "parameter": "tRAS",
                "issue": "tRAS must be >= tRCD + CL",
                "current": tras,
                "minimum": trcd + cl
            })
        
        if trc < (tras + trp):
            violations.append({
                "type": "timing",
                "parameter": "tRC", 
                "issue": "tRC must be >= tRAS + tRP",
                "current": trc,
                "minimum": tras + trp
            })
        
        return violations
    
    def _calculate_damage_risk(self, config: Dict[str, Any]) -> float:
        """Calculate estimated hardware damage risk (0-100%)"""
        risk_factors = []
        
        # Voltage risk
        vddq = config.get('vddq', 1.1)
        vddq_risk = max(0, (vddq - 1.1) / 0.25 * 30)  # 30% max risk from voltage
        risk_factors.append(vddq_risk)
        
        vpp = config.get('vpp', 1.8)
        vpp_risk = max(0, (vpp - 1.8) / 0.2 * 25)  # 25% max risk from VPP
        risk_factors.append(vpp_risk)
        
        # Frequency risk
        frequency = config.get('frequency', 4800)
        freq_risk = max(0, (frequency - 6400) / 2000 * 20)  # 20% max risk from frequency
        risk_factors.append(freq_risk)
        
        # Temperature factor (if available)
        temperature = config.get('temperature', 50)
        temp_risk = max(0, (temperature - 70) / 15 * 25)  # 25% max risk from temperature
        risk_factors.append(temp_risk)
        
        return min(100, sum(risk_factors))
    
    def _predict_memory_health(self, 
                             metrics: Dict[str, Any],
                             usage: Dict[str, Any]) -> HardwareHealth:
        """Predict memory module health"""
        
        # Base health calculation
        voltage_stress = metrics.get('vddq', 1.1) - 1.1
        thermal_stress = max(0, metrics.get('temperature', 50) - 70)
        usage_intensity = usage.get('daily_hours', 8) / 24
        
        # Health score calculation (simplified)
        health_score = 100 - (voltage_stress * 100 + thermal_stress * 2 + usage_intensity * 10)
        health_score = max(0, min(100, health_score))
        
        # Degradation rate (%/day)
        degradation_rate = (voltage_stress * 0.01 + thermal_stress * 0.005 + usage_intensity * 0.002)
        
        # Estimated lifespan
        if degradation_rate > 0:
            lifespan_days = int(health_score / (degradation_rate * 100))
        else:
            lifespan_days = 3650  # 10 years default
        
        # Risk assessment
        if health_score > 90:
            risk = RiskLevel.SAFE
        elif health_score > 75:
            risk = RiskLevel.LOW
        elif health_score > 60:
            risk = RiskLevel.MEDIUM
        elif health_score > 40:
            risk = RiskLevel.HIGH
        else:
            risk = RiskLevel.CRITICAL
        
        return HardwareHealth(
            component="Memory Modules",
            health_score=health_score,
            risk_level=risk,
            estimated_lifespan_days=lifespan_days,
            degradation_rate=degradation_rate,
            last_updated=datetime.now(),
            issues=self._identify_memory_issues(metrics),
            recommendations=self._get_memory_recommendations(health_score, metrics)
        )
    
    def _predict_controller_health(self, 
                                 metrics: Dict[str, Any],
                                 usage: Dict[str, Any]) -> HardwareHealth:
        """Predict memory controller health"""
        
        # Controller-specific health factors
        frequency_stress = max(0, (metrics.get('frequency', 4800) - 5600) / 1000)
        voltage_stress = max(0, metrics.get('vpp', 1.8) - 1.8)
        load_factor = usage.get('memory_utilization', 50) / 100
        
        health_score = 100 - (frequency_stress * 15 + voltage_stress * 200 + load_factor * 10)
        health_score = max(0, min(100, health_score))
        
        # Risk and lifespan calculation
        degradation_rate = frequency_stress * 0.005 + voltage_stress * 0.02
        lifespan_days = int(health_score / max(0.001, degradation_rate * 100))
        
        risk = RiskLevel.SAFE if health_score > 85 else RiskLevel.MEDIUM
        
        return HardwareHealth(
            component="Memory Controller",
            health_score=health_score,
            risk_level=risk,
            estimated_lifespan_days=min(lifespan_days, 7300),  # Cap at 20 years
            degradation_rate=degradation_rate,
            last_updated=datetime.now(),
            issues=[],
            recommendations=[]
        )
    
    def _predict_trace_health(self, 
                            metrics: Dict[str, Any],
                            usage: Dict[str, Any]) -> HardwareHealth:
        """Predict motherboard trace health"""
        
        # Trace degradation factors
        signal_integrity = metrics.get('signal_integrity', 95)
        frequency = metrics.get('frequency', 4800)
        
        # High frequency stress on traces
        freq_stress = max(0, (frequency - 6000) / 2000 * 20)
        integrity_loss = max(0, 100 - signal_integrity)
        
        health_score = 100 - freq_stress - integrity_loss
        health_score = max(0, min(100, health_score))
        
        return HardwareHealth(
            component="PCB Traces",
            health_score=health_score,
            risk_level=RiskLevel.LOW,
            estimated_lifespan_days=5475,  # 15 years typical
            degradation_rate=0.002,
            last_updated=datetime.now(),
            issues=[],
            recommendations=[]
        )
    
    def _identify_memory_issues(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify potential memory issues"""
        issues = []
        
        if metrics.get('error_rate', 0) > 0.0001:
            issues.append("Elevated error rate detected")
        
        if metrics.get('temperature', 50) > 80:
            issues.append("High operating temperature")
        
        if metrics.get('vddq', 1.1) > 1.3:
            issues.append("High VDDQ voltage stress")
        
        return issues
    
    def _get_memory_recommendations(self, health_score: float, metrics: Dict[str, Any]) -> List[str]:
        """Get recommendations for memory health"""
        recommendations = []
        
        if health_score < 80:
            recommendations.append("Consider reducing voltage for longevity")
        
        if metrics.get('temperature', 50) > 70:
            recommendations.append("Improve cooling solution")
        
        if health_score < 60:
            recommendations.append("Plan for memory replacement within 2 years")
        
        return recommendations
    
    def _update_health_history(self, health_data: Dict[str, HardwareHealth]):
        """Update hardware health history"""
        for component, health in health_data.items():
            if component not in self.health_history:
                self.health_history[component] = []
            
            self.health_history[component].append(health)
            
            # Keep only last 30 days of history
            cutoff_date = datetime.now() - timedelta(days=30)
            self.health_history[component] = [
                h for h in self.health_history[component] 
                if h.last_updated > cutoff_date
            ]
    
    def _cleanup_old_alerts(self):
        """Remove old alerts"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.active_alerts = [
            alert for alert in self.active_alerts
            if alert['timestamp'] > cutoff_time
        ]
    
    def _calculate_prediction_confidence(self, health: HardwareHealth) -> float:
        """Calculate confidence in health prediction"""
        # More data points = higher confidence
        history_length = len(self.health_history.get(health.component, []))
        base_confidence = min(0.9, history_length / 30)
        
        # Adjust for health score stability
        if health.degradation_rate < 0.001:
            base_confidence *= 0.8  # Lower confidence for very stable components
        
        return base_confidence
    
    def _estimate_replacement_cost(self, component: str) -> Dict[str, float]:
        """Estimate replacement cost for component"""
        costs = {
            "Memory Modules": {"min": 200, "max": 800, "currency": "USD"},
            "Memory Controller": {"min": 0, "max": 0, "currency": "USD", "note": "Part of CPU"},
            "PCB Traces": {"min": 300, "max": 1500, "currency": "USD", "note": "Motherboard replacement"}
        }
        
        return costs.get(component, {"min": 0, "max": 0, "currency": "USD"})
    
    def _generate_safety_recommendations(self, 
                                       config: Dict[str, Any],
                                       safety_report: Dict[str, Any]) -> List[str]:
        """Generate safety recommendations"""
        recommendations = []
        
        if safety_report["risk_level"] in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append("IMMEDIATE: Reduce voltages to safe levels")
            recommendations.append("Enable conservative safety mode")
        
        if config.get('vddq', 1.1) > 1.25:
            recommendations.append("Consider reducing VDDQ for better longevity")
        
        if config.get('frequency', 4800) > 6400:
            recommendations.append("Ensure adequate power delivery and cooling")
        
        recommendations.append("Monitor temperatures during stress testing")
        recommendations.append("Enable automatic safety shutdowns")
        
        return recommendations


def create_damage_prevention_system():
    """Create hardware damage prevention system"""
    return HardwareDamagePrevention()
