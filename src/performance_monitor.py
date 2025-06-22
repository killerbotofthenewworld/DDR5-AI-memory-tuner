"""
Advanced Performance Monitoring System for DDR5 AI Simulator
Real-time monitoring, analytics, and alerting for DDR5 configurations.
"""

import time
import threading
import queue
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import logging

from ddr5_models import DDR5Configuration
from ddr5_simulator import DDR5Simulator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime
    configuration: DDR5Configuration
    bandwidth: float
    latency: float
    stability: float
    power: float
    temperature: float
    error_rate: float
    throughput: float
    response_time: float
    cpu_usage: float
    memory_usage: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        # Convert configuration to dict
        data['configuration'] = {
            'frequency': self.configuration.frequency,
            'timings': asdict(self.configuration.timings),
            'voltages': asdict(self.configuration.voltages)
        }
        return data


@dataclass
class PerformanceAlert:
    """Performance alert data structure."""
    timestamp: datetime
    alert_type: str
    severity: str  # low, medium, high, critical
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    configuration: DDR5Configuration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['configuration'] = {
            'frequency': self.configuration.frequency,
            'timings': asdict(self.configuration.timings),
            'voltages': asdict(self.configuration.voltages)
        }
        return data


class PerformanceThresholds:
    """Performance thresholds for alerting."""
    
    def __init__(self):
        self.thresholds = {
            'bandwidth_min': 50000,  # Minimum bandwidth (MB/s)
            'latency_max': 200,      # Maximum latency (ns)
            'stability_min': 70,     # Minimum stability (%)
            'power_max': 8000,       # Maximum power (mW)
            'temperature_max': 85,   # Maximum temperature (°C)
            'error_rate_max': 0.1,   # Maximum error rate (%)
            'response_time_max': 1.0,  # Maximum response time (s)
            'cpu_usage_max': 90,     # Maximum CPU usage (%)
            'memory_usage_max': 95   # Maximum memory usage (%)
        }
    
    def check_threshold(self, metric_name: str, value: float) -> Optional[str]:
        """Check if a metric exceeds its threshold."""
        if metric_name.endswith('_min'):
            threshold = self.thresholds.get(metric_name)
            if threshold and value < threshold:
                return f"{metric_name.replace('_min', '')} below minimum threshold"
        
        elif metric_name.endswith('_max'):
            threshold = self.thresholds.get(metric_name)
            if threshold and value > threshold:
                return f"{metric_name.replace('_max', '')} above maximum threshold"
        
        return None
    
    def get_severity(self, metric_name: str, value: float, threshold: float) -> str:
        """Determine alert severity based on how far the value is from threshold."""
        if metric_name.endswith('_min'):
            deviation = (threshold - value) / threshold
        else:
            deviation = (value - threshold) / threshold
        
        if deviation > 0.5:
            return "critical"
        elif deviation > 0.25:
            return "high"
        elif deviation > 0.1:
            return "medium"
        else:
            return "low"


class PerformanceMonitor:
    """Advanced performance monitoring system."""
    
    def __init__(self, simulator: DDR5Simulator = None):
        """Initialize the performance monitor."""
        self.simulator = simulator or DDR5Simulator()
        self.thresholds = PerformanceThresholds()
        
        # Data storage
        self.metrics_history: deque = deque(maxlen=10000)
        self.alerts_history: deque = deque(maxlen=1000)
        self.current_metrics: Optional[PerformanceMetrics] = None
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_interval = 1.0  # seconds
        
        # Event system
        self.event_queue = queue.Queue()
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'alerts_generated': 0,
            'monitoring_uptime': 0,
            'start_time': None
        }
    
    def start_monitoring(self, configuration: DDR5Configuration, interval: float = 1.0):
        """Start continuous performance monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring already in progress")
            return
        
        self.monitoring_interval = interval
        self.is_monitoring = True
        self.stats['start_time'] = datetime.now()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(configuration,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"Started performance monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        # Update stats
        if self.stats['start_time']:
            self.stats['monitoring_uptime'] = (
                datetime.now() - self.stats['start_time']
            ).total_seconds()
        
        logger.info("Stopped performance monitoring")
    
    def _monitoring_loop(self, configuration: DDR5Configuration):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics(configuration)
                
                # Store metrics
                self.metrics_history.append(metrics)
                self.current_metrics = metrics
                self.stats['total_samples'] += 1
                
                # Check for alerts
                alerts = self._check_alerts(metrics)
                for alert in alerts:
                    self.alerts_history.append(alert)
                    self.stats['alerts_generated'] += 1
                    self._trigger_event('alert_generated', alert)
                
                # Trigger metrics update event
                self._trigger_event('metrics_updated', metrics)
                
                # Sleep until next interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)  # Wait before retrying
    
    def _collect_metrics(self, configuration: DDR5Configuration) -> PerformanceMetrics:
        """Collect comprehensive performance metrics."""
        
        # Simulate performance
        sim_result = self.simulator.simulate_performance(configuration)
        
        # Collect system metrics (simulated for now)
        import psutil
        import random
        
        # Get actual system metrics where possible
        try:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory().percent
            
            # Simulate temperature (would come from hardware sensors)
            temperature = random.uniform(45, 85)
            
            # Simulate error rate
            error_rate = random.uniform(0, 0.05)
            
            # Simulate response time
            response_time = random.uniform(0.1, 2.0)
            
        except Exception:
            # Fallback to simulated values
            cpu_usage = random.uniform(20, 80)
            memory_usage = random.uniform(40, 90)
            temperature = random.uniform(45, 85)
            error_rate = random.uniform(0, 0.05)
            response_time = random.uniform(0.1, 2.0)
        
        # Calculate throughput (simplified)
        throughput = sim_result['bandwidth'] * (1 - error_rate/100)
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            configuration=configuration,
            bandwidth=sim_result['bandwidth'],
            latency=sim_result['latency'],
            stability=sim_result['stability'],
            power=sim_result['power'],
            temperature=temperature,
            error_rate=error_rate,
            throughput=throughput,
            response_time=response_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage
        )
    
    def _check_alerts(self, metrics: PerformanceMetrics) -> List[PerformanceAlert]:
        """Check metrics against thresholds and generate alerts."""
        alerts = []
        
        # Check each metric against thresholds
        metric_checks = [
            ('bandwidth_min', metrics.bandwidth),
            ('latency_max', metrics.latency),
            ('stability_min', metrics.stability),
            ('power_max', metrics.power),
            ('temperature_max', metrics.temperature),
            ('error_rate_max', metrics.error_rate),
            ('response_time_max', metrics.response_time),
            ('cpu_usage_max', metrics.cpu_usage),
            ('memory_usage_max', metrics.memory_usage)
        ]
        
        for threshold_name, value in metric_checks:
            violation = self.thresholds.check_threshold(threshold_name, value)
            if violation:
                threshold_value = self.thresholds.thresholds[threshold_name]
                severity = self.thresholds.get_severity(threshold_name, value, threshold_value)
                
                alert = PerformanceAlert(
                    timestamp=datetime.now(),
                    alert_type="threshold_violation",
                    severity=severity,
                    message=f"{violation}: {value:.2f}",
                    metric_name=threshold_name,
                    current_value=value,
                    threshold_value=threshold_value,
                    configuration=metrics.configuration
                )
                alerts.append(alert)
        
        return alerts
    
    def _trigger_event(self, event_type: str, data: Any):
        """Trigger an event for registered handlers."""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent metrics."""
        return self.current_metrics
    
    def get_metrics_history(self, 
                          last_n: Optional[int] = None,
                          time_range: Optional[timedelta] = None) -> List[PerformanceMetrics]:
        """Get historical metrics."""
        metrics = list(self.metrics_history)
        
        # Filter by time range
        if time_range:
            cutoff_time = datetime.now() - time_range
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        # Limit to last N entries
        if last_n:
            metrics = metrics[-last_n:]
        
        return metrics
    
    def get_alerts_history(self, 
                         last_n: Optional[int] = None,
                         severity_filter: Optional[str] = None) -> List[PerformanceAlert]:
        """Get historical alerts."""
        alerts = list(self.alerts_history)
        
        # Filter by severity
        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]
        
        # Limit to last N entries
        if last_n:
            alerts = alerts[-last_n:]
        
        return alerts
    
    def get_performance_summary(self, time_range: timedelta = None) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not time_range:
            time_range = timedelta(hours=1)
        
        metrics = self.get_metrics_history(time_range=time_range)
        
        if not metrics:
            return {"error": "No metrics available for the specified time range"}
        
        # Calculate statistics
        bandwidths = [m.bandwidth for m in metrics]
        latencies = [m.latency for m in metrics]
        stabilities = [m.stability for m in metrics]
        powers = [m.power for m in metrics]
        temperatures = [m.temperature for m in metrics]
        
        summary = {
            "time_range": str(time_range),
            "sample_count": len(metrics),
            "bandwidth": {
                "avg": np.mean(bandwidths),
                "min": np.min(bandwidths),
                "max": np.max(bandwidths),
                "std": np.std(bandwidths)
            },
            "latency": {
                "avg": np.mean(latencies),
                "min": np.min(latencies),
                "max": np.max(latencies),
                "std": np.std(latencies)
            },
            "stability": {
                "avg": np.mean(stabilities),
                "min": np.min(stabilities),
                "max": np.max(stabilities),
                "std": np.std(stabilities)
            },
            "power": {
                "avg": np.mean(powers),
                "min": np.min(powers),
                "max": np.max(powers),
                "std": np.std(powers)
            },
            "temperature": {
                "avg": np.mean(temperatures),
                "min": np.min(temperatures),
                "max": np.max(temperatures),
                "std": np.std(temperatures)
            },
            "alerts": {
                "total": len([a for a in self.alerts_history if a.timestamp >= datetime.now() - time_range]),
                "critical": len([a for a in self.alerts_history if a.severity == "critical" and a.timestamp >= datetime.now() - time_range]),
                "high": len([a for a in self.alerts_history if a.severity == "high" and a.timestamp >= datetime.now() - time_range])
            }
        }
        
        return summary
    
    def export_data(self, output_file: str, format: str = "json"):
        """Export monitoring data to file."""
        data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_samples": len(self.metrics_history),
                "total_alerts": len(self.alerts_history),
                "monitoring_uptime": self.stats['monitoring_uptime']
            },
            "metrics": [m.to_dict() for m in self.metrics_history],
            "alerts": [a.to_dict() for a in self.alerts_history]
        }
        
        if format.lower() == "json":
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported monitoring data to {output_file}")
    
    def get_anomaly_detection(self, sensitivity: float = 2.0) -> List[PerformanceMetrics]:
        """Detect anomalous performance using statistical methods."""
        if len(self.metrics_history) < 10:
            return []
        
        metrics = list(self.metrics_history)
        anomalies = []
        
        # Calculate baseline statistics
        bandwidths = np.array([m.bandwidth for m in metrics])
        latencies = np.array([m.latency for m in metrics])
        stabilities = np.array([m.stability for m in metrics])
        
        # Calculate z-scores
        bandwidth_mean, bandwidth_std = np.mean(bandwidths), np.std(bandwidths)
        latency_mean, latency_std = np.mean(latencies), np.std(latencies)
        stability_mean, stability_std = np.mean(stabilities), np.std(stabilities)
        
        for i, metric in enumerate(metrics):
            # Calculate z-scores for key metrics
            bandwidth_z = abs((metric.bandwidth - bandwidth_mean) / bandwidth_std) if bandwidth_std > 0 else 0
            latency_z = abs((metric.latency - latency_mean) / latency_std) if latency_std > 0 else 0
            stability_z = abs((metric.stability - stability_mean) / stability_std) if stability_std > 0 else 0
            
            # Check if any metric exceeds sensitivity threshold
            if max(bandwidth_z, latency_z, stability_z) > sensitivity:
                anomalies.append(metric)
        
        return anomalies
    
    def get_trend_analysis(self, time_range: timedelta = None) -> Dict[str, str]:
        """Analyze performance trends."""
        if not time_range:
            time_range = timedelta(hours=1)
        
        metrics = self.get_metrics_history(time_range=time_range)
        
        if len(metrics) < 5:
            return {"error": "Insufficient data for trend analysis"}
        
        # Extract time series data
        timestamps = [m.timestamp for m in metrics]
        bandwidths = [m.bandwidth for m in metrics]
        latencies = [m.latency for m in metrics]
        stabilities = [m.stability for m in metrics]
        
        # Calculate trends using linear regression
        def calculate_trend(values):
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            return "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        
        trends = {
            "bandwidth": calculate_trend(bandwidths),
            "latency": calculate_trend(latencies),
            "stability": calculate_trend(stabilities),
            "overall_health": "good" if all(
                trend in ["stable", "increasing"] for trend in [
                    calculate_trend(bandwidths),
                    calculate_trend(stabilities)
                ]
            ) and calculate_trend(latencies) in ["stable", "decreasing"] else "concerning"
        }
        
        return trends
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring system statistics."""
        uptime = 0
        if self.stats['start_time'] and self.is_monitoring:
            uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        return {
            "is_monitoring": self.is_monitoring,
            "monitoring_interval": self.monitoring_interval,
            "uptime_seconds": uptime,
            "total_samples": self.stats['total_samples'],
            "alerts_generated": self.stats['alerts_generated'],
            "metrics_buffer_size": len(self.metrics_history),
            "alerts_buffer_size": len(self.alerts_history),
            "event_handlers": {k: len(v) for k, v in self.event_handlers.items()}
        }
