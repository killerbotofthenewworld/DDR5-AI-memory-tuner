"""
Gaming Performance Predictor for DDR5 AI Sandbox Simulator
Real-world gaming FPS prediction based on memory configuration
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class GameBenchmark:
    """Gaming benchmark result prediction."""
    game_name: str
    fps_1080p: float
    fps_1440p: float
    fps_4k: float
    memory_sensitivity: float  # How much this game benefits from faster RAM
    cpu_bottleneck_threshold: float  # FPS where CPU becomes limiting factor

class GamingPerformancePredictor:
    """Predict real-world gaming performance from DDR5 configuration."""
    
    def __init__(self):
        self.game_database = self._initialize_game_database()
        self.baseline_performance = self._initialize_baselines()
    
    def _initialize_game_database(self) -> Dict[str, GameBenchmark]:
        """Initialize database of games and their memory sensitivity."""
        return {
            "cyberpunk_2077": GameBenchmark(
                game_name="Cyberpunk 2077",
                fps_1080p=85.0,
                fps_1440p=65.0, 
                fps_4k=35.0,
                memory_sensitivity=0.15,  # 15% performance gain from fast RAM
                cpu_bottleneck_threshold=120.0
            ),
            "call_of_duty_warzone": GameBenchmark(
                game_name="Call of Duty: Warzone",
                fps_1080p=110.0,
                fps_1440p=95.0,
                fps_4k=60.0,
                memory_sensitivity=0.12,
                cpu_bottleneck_threshold=165.0
            ),
            "valorant": GameBenchmark(
                game_name="Valorant",
                fps_1080p=280.0,
                fps_1440p=240.0,
                fps_4k=180.0,
                memory_sensitivity=0.08,
                cpu_bottleneck_threshold=400.0
            ),
            "apex_legends": GameBenchmark(
                game_name="Apex Legends",
                fps_1080p=140.0,
                fps_1440p=115.0,
                fps_4k=75.0,
                memory_sensitivity=0.10,
                cpu_bottleneck_threshold=200.0
            ),
            "counter_strike_2": GameBenchmark(
                game_name="Counter-Strike 2",
                fps_1080p=320.0,
                fps_1440p=280.0,
                fps_4k=200.0,
                memory_sensitivity=0.18,  # CS2 loves fast RAM
                cpu_bottleneck_threshold=450.0
            ),
            "fortnite": GameBenchmark(
                game_name="Fortnite",
                fps_1080p=180.0,
                fps_1440p=150.0,
                fps_4k=95.0,
                memory_sensitivity=0.09,
                cpu_bottleneck_threshold=220.0
            ),
            "minecraft_modded": GameBenchmark(
                game_name="Minecraft (Heavily Modded)",
                fps_1080p=120.0,
                fps_1440p=110.0,
                fps_4k=85.0,
                memory_sensitivity=0.22,  # Mods love fast RAM
                cpu_bottleneck_threshold=180.0
            ),
            "gta_v": GameBenchmark(
                game_name="Grand Theft Auto V",
                fps_1080p=95.0,
                fps_1440p=75.0,
                fps_4k=45.0,
                memory_sensitivity=0.11,
                cpu_bottleneck_threshold=140.0
            ),
            "red_dead_redemption_2": GameBenchmark(
                game_name="Red Dead Redemption 2",
                fps_1080p=70.0,
                fps_1440p=55.0,
                fps_4k=32.0,
                memory_sensitivity=0.08,
                cpu_bottleneck_threshold=95.0
            ),
            "world_of_warcraft": GameBenchmark(
                game_name="World of Warcraft",
                fps_1080p=140.0,
                fps_1440p=120.0,
                fps_4k=85.0,
                memory_sensitivity=0.16,  # WoW benefits from low latency
                cpu_bottleneck_threshold=200.0
            ),
            "league_of_legends": GameBenchmark(
                game_name="League of Legends",
                fps_1080p=220.0,
                fps_1440p=200.0,
                fps_4k=160.0,
                memory_sensitivity=0.07,
                cpu_bottleneck_threshold=350.0
            ),
            "overwatch_2": GameBenchmark(
                game_name="Overwatch 2",
                fps_1080p=160.0,
                fps_1440p=135.0,
                fps_4k=85.0,
                memory_sensitivity=0.13,
                cpu_bottleneck_threshold=220.0
            )
        }
    
    def _initialize_baselines(self) -> Dict[str, float]:
        """Initialize baseline performance metrics."""
        return {
            "baseline_bandwidth": 89600,  # MB/s for DDR5-5600 CL36
            "baseline_latency": 65.0,     # ns
            "baseline_frequency": 5600    # MT/s
        }
    
    def predict_gaming_performance(self, config, resolution="1080p") -> Dict[str, Dict]:
        """Predict gaming performance for all games."""
        # Calculate memory performance factors
        bandwidth_factor = self._calculate_bandwidth_factor(config)
        latency_factor = self._calculate_latency_factor(config)
        
        results = {}
        
        for game_id, benchmark in self.game_database.items():
            # Get base FPS for resolution
            if resolution == "1080p":
                base_fps = benchmark.fps_1080p
            elif resolution == "1440p":
                base_fps = benchmark.fps_1440p
            elif resolution == "4k":
                base_fps = benchmark.fps_4k
            else:
                base_fps = benchmark.fps_1080p
            
            # Calculate memory improvement
            memory_improvement = (
                (bandwidth_factor * 0.6 + latency_factor * 0.4) * 
                benchmark.memory_sensitivity
            )
            
            # Apply improvement
            predicted_fps = base_fps * (1 + memory_improvement)
            
            # Apply CPU bottleneck if necessary
            if predicted_fps > benchmark.cpu_bottleneck_threshold:
                predicted_fps = benchmark.cpu_bottleneck_threshold
            
            # Calculate frame time consistency (1% lows)
            one_percent_low = predicted_fps * (0.85 + latency_factor * 0.1)
            frame_time_ms = 1000 / predicted_fps
            
            results[game_id] = {
                "game_name": benchmark.game_name,
                "fps": round(predicted_fps, 1),
                "one_percent_low": round(one_percent_low, 1),
                "frame_time_ms": round(frame_time_ms, 2),
                "memory_improvement_percent": round(memory_improvement * 100, 1),
                "cpu_limited": predicted_fps >= benchmark.cpu_bottleneck_threshold * 0.95
            }
        
        return results
    
    def _calculate_bandwidth_factor(self, config) -> float:
        """Calculate bandwidth improvement factor."""
        # Estimate bandwidth from frequency
        estimated_bandwidth = config.frequency * 8 * 2  # Rough estimate in MB/s
        return min(1.5, estimated_bandwidth / self.baseline_performance["baseline_bandwidth"])
    
    def _calculate_latency_factor(self, config) -> float:
        """Calculate latency improvement factor (lower latency = higher factor)."""
        # Estimate true latency
        estimated_latency = (config.timings.cl / (config.frequency / 2)) * 1000
        latency_ratio = self.baseline_performance["baseline_latency"] / estimated_latency
        return min(1.3, latency_ratio)  # Cap at 30% improvement
    
    def get_gaming_recommendations(self, current_config) -> Dict[str, str]:
        """Get specific gaming optimization recommendations."""
        recommendations = {}
        
        # Analyze current config
        frequency = current_config.frequency
        cl = current_config.timings.cl
        
        if frequency < 5600:
            recommendations["frequency"] = "🚀 Increase frequency to DDR5-5600+ for significant gaming gains"
        elif frequency > 6400:
            recommendations["frequency"] = "⚡ Excellent frequency for gaming!"
        
        if cl > 36:
            recommendations["latency"] = "🎯 Tighten CL timings for better frame time consistency"
        elif cl <= 30:
            recommendations["latency"] = "🏆 Excellent low latency for competitive gaming!"
        
        if frequency >= 6000 and cl <= 32:
            recommendations["competitive"] = "🔥 Perfect setup for competitive esports!"
        
        return recommendations
    
    def calculate_gaming_score(self, config) -> float:
        """Calculate overall gaming performance score (0-100)."""
        performance_results = self.predict_gaming_performance(config, "1080p")
        
        # Weight games by popularity/competitiveness
        game_weights = {
            "valorant": 1.2,
            "counter_strike_2": 1.2,
            "call_of_duty_warzone": 1.1,
            "apex_legends": 1.1,
            "league_of_legends": 1.0,
            "overwatch_2": 1.0,
            "fortnite": 0.9,
            "cyberpunk_2077": 0.8,
            "minecraft_modded": 0.7,
            "gta_v": 0.6,
            "red_dead_redemption_2": 0.5,
            "world_of_warcraft": 0.6
        }
        
        total_score = 0
        total_weight = 0
        
        for game_id, result in performance_results.items():
            weight = game_weights.get(game_id, 1.0)
            # Normalize FPS to score (higher FPS = higher score, but with diminishing returns)
            fps_score = min(100, (result["fps"] / 100) * 50 + 
                          (result["memory_improvement_percent"] / 20) * 50)
            
            total_score += fps_score * weight
            total_weight += weight
        
        return min(100, total_score / total_weight)
