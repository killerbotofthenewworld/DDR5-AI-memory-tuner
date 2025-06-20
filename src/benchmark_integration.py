"""
Benchmark Integration Module
Correlates DDR5 configurations with real-world benchmark scores
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Real-world benchmark result."""
    benchmark_name: str
    score: float
    units: str
    category: str
    improvement_percent: float


class BenchmarkPredictor:
    """Predict real-world benchmark performance."""
    
    def __init__(self):
        self.benchmarks = self._initialize_benchmarks()
    
    def _initialize_benchmarks(self) -> Dict[str, Dict]:
        """Initialize benchmark database."""
        return {
            "aida64_memory": {
                "read_bandwidth": {"base": 89600, "sensitivity": 0.95},
                "write_bandwidth": {"base": 82400, "sensitivity": 0.92},
                "copy_bandwidth": {"base": 78900, "sensitivity": 0.90},
                "latency": {"base": 65.2, "sensitivity": -0.85}
            },
            "intel_mlc": {
                "loaded_latency": {"base": 68.4, "sensitivity": -0.80},
                "idle_latency": {"base": 62.1, "sensitivity": -0.88}
            },
            "stream_triad": {
                "bandwidth": {"base": 85200, "sensitivity": 0.93}
            },
            "passmark": {
                "memory_mark": {"base": 28500, "sensitivity": 0.75}
            },
            "cinebench": {
                "single_core": {"base": 1650, "sensitivity": 0.12},
                "multi_core": {"base": 24800, "sensitivity": 0.08}
            },
            "blender": {
                "bmw27_render": {"base": 185.2, "sensitivity": -0.15}
            },
            "7zip": {
                "compression": {"base": 98500, "sensitivity": 0.18},
                "decompression": {"base": 102300, "sensitivity": 0.22}
            }
        }
    
    def predict_all_benchmarks(self, config) -> Dict[str, List[BenchmarkResult]]:
        """Predict all benchmark scores."""
        bandwidth_factor = self._calculate_bandwidth_factor(config)
        latency_factor = self._calculate_latency_factor(config)
        
        results = {}
        
        for bench_suite, tests in self.benchmarks.items():
            results[bench_suite] = []
            
            for test_name, test_data in tests.items():
                base_score = test_data["base"]
                sensitivity = test_data["sensitivity"]
                
                if "latency" in test_name:
                    # Lower latency = better score (negative improvement)
                    improvement = (1 - latency_factor) * sensitivity
                    predicted_score = base_score * (1 + improvement)
                else:
                    # Higher bandwidth = better score
                    improvement = (bandwidth_factor - 1) * sensitivity
                    predicted_score = base_score * (1 + improvement)
                
                # Determine units and category
                if "bandwidth" in test_name or "mark" in test_name:
                    units = "MB/s" if "bandwidth" in test_name else "points"
                    category = "Memory"
                elif "latency" in test_name:
                    units = "ns"
                    category = "Memory"
                elif "render" in test_name:
                    units = "seconds"
                    category = "Rendering"
                else:
                    units = "points"
                    category = "CPU"
                
                results[bench_suite].append(BenchmarkResult(
                    benchmark_name=f"{bench_suite.upper()} {test_name.title()}",
                    score=round(predicted_score, 1),
                    units=units,
                    category=category,
                    improvement_percent=round(improvement * 100, 1)
                ))
        
        return results
    
    def _calculate_bandwidth_factor(self, config) -> float:
        """Calculate bandwidth improvement factor."""
        estimated_bandwidth = config.frequency * 8 * 2
        return min(1.4, estimated_bandwidth / 89600)
    
    def _calculate_latency_factor(self, config) -> float:
        """Calculate latency improvement factor."""
        estimated_latency = (config.timings.cl / (config.frequency / 2)) * 1000
        return min(1.25, 65.2 / estimated_latency)
    
    def get_benchmark_score(self, config) -> float:
        """Calculate overall benchmark performance score."""
        results = self.predict_all_benchmarks(config)
        
        weights = {
            "aida64_memory": 0.4,
            "intel_mlc": 0.2,
            "stream_triad": 0.15,
            "passmark": 0.1,
            "cinebench": 0.1,
            "7zip": 0.05
        }
        
        total_improvement = 0
        for suite_name, suite_results in results.items():
            weight = weights.get(suite_name, 0.1)
            suite_avg = sum(r.improvement_percent for r in suite_results) / len(suite_results)
            total_improvement += suite_avg * weight
        
        # Convert to 0-100 score
        return min(100, 85 + total_improvement)
