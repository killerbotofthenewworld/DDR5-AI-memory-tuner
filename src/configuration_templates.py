"""
DDR5 Configuration Templates System
Provides pre-built, tested configurations for different use cases.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from src.ddr5_models import DDR5Configuration, DDR5TimingParameters, DDR5VoltageParameters


class UseCase(Enum):
    """Different use cases for DDR5 configurations."""
    GAMING = "gaming"
    PRODUCTIVITY = "productivity"
    CONTENT_CREATION = "content_creation"
    SCIENTIFIC_COMPUTING = "scientific_computing"
    SERVER_WORKLOAD = "server_workload"
    OVERCLOCKING = "overclocking"
    POWER_EFFICIENCY = "power_efficiency"
    STABILITY_FIRST = "stability_first"


class PerformanceLevel(Enum):
    """Performance levels for configurations."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


@dataclass
class ConfigurationTemplate:
    """Template for DDR5 configurations."""
    name: str
    description: str
    use_case: UseCase
    performance_level: PerformanceLevel
    configuration: DDR5Configuration
    estimated_performance: Dict[str, float]
    compatibility_notes: List[str]
    stability_rating: int  # 1-10 scale
    power_consumption: str  # Low, Medium, High
    author: str
    created_date: str
    tags: List[str]


class ConfigurationTemplateManager:
    """Manages DDR5 configuration templates."""
    
    def __init__(self):
        """Initialize the template manager."""
        self.templates: Dict[str, ConfigurationTemplate] = {}
        self.load_built_in_templates()
    
    def load_built_in_templates(self):
        """Load built-in configuration templates."""
        
        # Gaming Templates
        self._add_gaming_templates()
        
        # Productivity Templates
        self._add_productivity_templates()
        
        # Content Creation Templates
        self._add_content_creation_templates()
        
        # Scientific Computing Templates
        self._add_scientific_templates()
        
        # Server Workload Templates
        self._add_server_templates()
        
        # Overclocking Templates
        self._add_overclocking_templates()
        
        # Power Efficiency Templates
        self._add_power_efficiency_templates()
        
        # Stability First Templates
        self._add_stability_templates()
    
    def _add_gaming_templates(self):
        """Add gaming-optimized templates."""
        
        # Conservative Gaming - DDR5-5600
        self.templates["gaming_conservative_5600"] = ConfigurationTemplate(
            name="Gaming Conservative DDR5-5600",
            description="Stable gaming configuration with excellent compatibility",
            use_case=UseCase.GAMING,
            performance_level=PerformanceLevel.CONSERVATIVE,
            configuration=DDR5Configuration(
                frequency=5600,
                timings=DDR5TimingParameters(
                    cl=36, trcd=36, trp=36, tras=76, trc=112
                ),
                voltages=DDR5VoltageParameters(
                    vddq=1.1, vpp=1.8, vddq_tx=1.1, vddq_rx=1.1
                )
            ),
            estimated_performance={
                "bandwidth": 85000,
                "latency": 68,
                "gaming_fps_boost": 8,
                "stability": 95
            },
            compatibility_notes=[
                "Works with most DDR5-5600 kits",
                "Excellent compatibility with Intel 12th gen+",
                "AMD Ryzen 7000 series compatible"
            ],
            stability_rating=9,
            power_consumption="Medium",
            author="DDR5 AI Team",
            created_date="2025-06-22",
            tags=["gaming", "stable", "compatible", "ddr5-5600"]
        )
        
        # Balanced Gaming - DDR5-6000
        self.templates["gaming_balanced_6000"] = ConfigurationTemplate(
            name="Gaming Balanced DDR5-6000",
            description="Optimized for gaming performance with good stability",
            use_case=UseCase.GAMING,
            performance_level=PerformanceLevel.BALANCED,
            configuration=DDR5Configuration(
                frequency=6000,
                timings=DDR5TimingParameters(
                    cl=32, trcd=34, trp=34, tras=68, trc=102
                ),
                voltages=DDR5VoltageParameters(
                    vddq=1.15, vpp=1.85, vddq_tx=1.15, vddq_rx=1.15
                )
            ),
            estimated_performance={
                "bandwidth": 92000,
                "latency": 58,
                "gaming_fps_boost": 12,
                "stability": 88
            },
            compatibility_notes=[
                "Requires quality DDR5-6000+ kit",
                "May need motherboard BIOS tuning",
                "Test stability with your specific setup"
            ],
            stability_rating=8,
            power_consumption="Medium-High",
            author="DDR5 AI Team",
            created_date="2025-06-22",
            tags=["gaming", "performance", "ddr5-6000", "balanced"]
        )
        
        # Aggressive Gaming - DDR5-6400
        self.templates["gaming_aggressive_6400"] = ConfigurationTemplate(
            name="Gaming Aggressive DDR5-6400",
            description="High-performance gaming with tight timings",
            use_case=UseCase.GAMING,
            performance_level=PerformanceLevel.AGGRESSIVE,
            configuration=DDR5Configuration(
                frequency=6400,
                timings=DDR5TimingParameters(
                    cl=30, trcd=32, trp=32, tras=64, trc=96
                ),
                voltages=DDR5VoltageParameters(
                    vddq=1.2, vpp=1.9, vddq_tx=1.2, vddq_rx=1.2
                )
            ),
            estimated_performance={
                "bandwidth": 98000,
                "latency": 52,
                "gaming_fps_boost": 15,
                "stability": 82
            },
            compatibility_notes=[
                "Requires premium DDR5-6400+ kit",
                "May need advanced motherboard tuning",
                "Stability testing recommended"
            ],
            stability_rating=7,
            power_consumption="High",
            author="DDR5 AI Team",
            created_date="2024-12-22",
            tags=["gaming", "aggressive", "performance", "ddr5-6400"]
        )
    
    def _add_productivity_templates(self):
        """Add productivity-optimized templates."""
        
        # Productivity Balanced - DDR5-5200
        self.templates["productivity_balanced_5200"] = ConfigurationTemplate(
            name="Productivity Balanced DDR5-5200",
            description="Optimized for office work and multitasking",
            use_case=UseCase.PRODUCTIVITY,
            performance_level=PerformanceLevel.BALANCED,
            configuration=DDR5Configuration(
                frequency=5200,
                timings=DDR5TimingParameters(
                    cl=38, trcd=38, trp=38, tras=78, trc=116
                ),
                voltages=DDR5VoltageParameters(
                    vddq=1.08, vpp=1.78, vddq_rx=1.08, vddq_tx=1.08
                )
            ),
            estimated_performance={
                "bandwidth": 80000,
                "latency": 72,
                "multitasking_score": 92,
                "stability": 96
            },
            compatibility_notes=[
                "Excellent for business laptops",
                "Low power consumption",
                "High compatibility across platforms"
            ],
            stability_rating=9,
            power_consumption="Low",
            author="DDR5 AI Team",
            created_date="2024-12-22",
            tags=["productivity", "efficiency", "stable", "business"]
        )
    
    def _add_content_creation_templates(self):
        """Add content creation templates."""
        
        # Content Creation Aggressive - DDR5-6000
        self.templates["content_aggressive_6000"] = ConfigurationTemplate(
            name="Content Creation DDR5-6000",
            description="Optimized for video editing and 3D rendering",
            use_case=UseCase.CONTENT_CREATION,
            performance_level=PerformanceLevel.AGGRESSIVE,
            configuration=DDR5Configuration(
                frequency=6000,
                timings=DDR5TimingParameters(
                    cl=30, trcd=32, trp=32, tras=64, trc=96
                ),
                voltages=DDR5VoltageParameters(
                    vddq=1.18, vpp=1.88, vddq_rx=1.18, vddq_tx=1.18
                )
            ),
            estimated_performance={
                "bandwidth": 95000,
                "latency": 55,
                "render_time_improvement": 18,
                "stability": 85
            },
            compatibility_notes=[
                "Ideal for Adobe Creative Suite",
                "Excellent for DaVinci Resolve",
                "May require cooling optimization"
            ],
            stability_rating=8,
            power_consumption="High",
            author="DDR5 AI Team",
            created_date="2024-12-22",
            tags=["content_creation", "rendering", "video_editing", "performance"]
        )
    
    def _add_scientific_templates(self):
        """Add scientific computing templates."""
        
        # Scientific Computing - DDR5-5600
        self.templates["scientific_balanced_5600"] = ConfigurationTemplate(
            name="Scientific Computing DDR5-5600",
            description="Optimized for scientific simulations and data analysis",
            use_case=UseCase.SCIENTIFIC_COMPUTING,
            performance_level=PerformanceLevel.BALANCED,
            configuration=DDR5Configuration(
                frequency=5600,
                timings=DDR5TimingParameters(
                    cl=34, trcd=36, trp=36, tras=72, trc=108
                ),
                voltages=DDR5VoltageParameters(
                    vddq=1.12, vpp=1.82, vddq_rx=1.12, vddq_tx=1.12
                )
            ),
            estimated_performance={
                "bandwidth": 88000,
                "latency": 62,
                "compute_efficiency": 94,
                "stability": 93
            },
            compatibility_notes=[
                "Excellent for MATLAB/Python",
                "Good for machine learning workloads",
                "Stable for long-running simulations"
            ],
            stability_rating=9,
            power_consumption="Medium",
            author="DDR5 AI Team",
            created_date="2024-12-22",
            tags=["scientific", "computing", "simulation", "data_analysis"]
        )
    
    def _add_server_templates(self):
        """Add server workload templates."""
        
        # Server Conservative - DDR5-4800
        self.templates["server_conservative_4800"] = ConfigurationTemplate(
            name="Server Conservative DDR5-4800",
            description="Maximum stability for server workloads",
            use_case=UseCase.SERVER_WORKLOAD,
            performance_level=PerformanceLevel.CONSERVATIVE,
            configuration=DDR5Configuration(
                frequency=4800,
                timings=DDR5TimingParameters(
                    cl=40, trcd=40, trp=40, tras=80, trc=120
                ),
                voltages=DDR5VoltageParameters(
                    vddq=1.06, vpp=1.76, vddq_rx=1.06, vddq_tx=1.06
                )
            ),
            estimated_performance={
                "bandwidth": 75000,
                "latency": 78,
                "uptime_reliability": 99,
                "stability": 98
            },
            compatibility_notes=[
                "Excellent for 24/7 operations",
                "ECC memory compatible",
                "Low power consumption"
            ],
            stability_rating=10,
            power_consumption="Low",
            author="DDR5 AI Team",
            created_date="2024-12-22",
            tags=["server", "stability", "reliability", "24/7"]
        )
    
    def _add_overclocking_templates(self):
        """Add overclocking templates."""
        
        # Extreme Overclocking - DDR5-7200
        self.templates["overclocking_extreme_7200"] = ConfigurationTemplate(
            name="Extreme Overclocking DDR5-7200",
            description="Maximum performance for experienced overclockers",
            use_case=UseCase.OVERCLOCKING,
            performance_level=PerformanceLevel.EXTREME,
            configuration=DDR5Configuration(
                frequency=7200,
                timings=DDR5TimingParameters(
                    cl=32, trcd=34, trp=34, tras=68, trc=102
                ),
                voltages=DDR5VoltageParameters(
                    vddq=1.35, vpp=2.0, vddq_rx=1.35, vddq_tx=1.35
                )
            ),
            estimated_performance={
                "bandwidth": 110000,
                "latency": 48,
                "benchmark_score": 125,
                "stability": 70
            },
            compatibility_notes=[
                "Requires premium cooling",
                "Expert overclocking knowledge needed",
                "May require custom cooling solutions"
            ],
            stability_rating=5,
            power_consumption="Very High",
            author="DDR5 AI Team",
            created_date="2024-12-22",
            tags=["overclocking", "extreme", "performance", "enthusiast"]
        )
    
    def _add_power_efficiency_templates(self):
        """Add power efficiency templates."""
        
        # Power Efficiency - DDR5-4800
        self.templates["power_efficiency_4800"] = ConfigurationTemplate(
            name="Power Efficiency DDR5-4800",
            description="Optimized for battery life and low power consumption",
            use_case=UseCase.POWER_EFFICIENCY,
            performance_level=PerformanceLevel.CONSERVATIVE,
            configuration=DDR5Configuration(
                frequency=4800,
                timings=DDR5TimingParameters(
                    cl=42, trcd=42, trp=42, tras=84, trc=126
                ),
                voltages=DDR5VoltageParameters(
                    vddq=1.05, vpp=1.75, vddq_rx=1.05, vddq_tx=1.05
                )
            ),
            estimated_performance={
                "bandwidth": 72000,
                "latency": 82,
                "power_savings": 25,
                "stability": 97
            },
            compatibility_notes=[
                "Excellent for laptops",
                "Extended battery life",
                "Cool operation"
            ],
            stability_rating=9,
            power_consumption="Very Low",
            author="DDR5 AI Team",
            created_date="2024-12-22",
            tags=["power_efficiency", "battery", "laptop", "cool"]
        )
    
    def _add_stability_templates(self):
        """Add stability-first templates."""
        
        # Stability First - DDR5-5200
        self.templates["stability_first_5200"] = ConfigurationTemplate(
            name="Stability First DDR5-5200",
            description="Maximum stability for critical applications",
            use_case=UseCase.STABILITY_FIRST,
            performance_level=PerformanceLevel.CONSERVATIVE,
            configuration=DDR5Configuration(
                frequency=5200,
                timings=DDR5TimingParameters(
                    cl=40, trcd=40, trp=40, tras=80, trc=120
                ),
                voltages=DDR5VoltageParameters(
                    vddq=1.08, vpp=1.78, vddq_rx=1.08, vddq_tx=1.08
                )
            ),
            estimated_performance={
                "bandwidth": 78000,
                "latency": 75,
                "error_rate": 0.001,
                "stability": 99
            },
            compatibility_notes=[
                "Tested for 24/7 operation",
                "Minimal risk of errors",
                "Wide compatibility"
            ],
            stability_rating=10,
            power_consumption="Low",
            author="DDR5 AI Team",
            created_date="2024-12-22",
            tags=["stability", "reliability", "critical", "error_free"]
        )
    
    def get_templates_by_use_case(self, use_case: UseCase) -> List[ConfigurationTemplate]:
        """Get templates filtered by use case."""
        return [
            template for template in self.templates.values()
            if template.use_case == use_case
        ]
    
    def get_templates_by_performance_level(self, level: PerformanceLevel) -> List[ConfigurationTemplate]:
        """Get templates filtered by performance level."""
        return [
            template for template in self.templates.values()
            if template.performance_level == level
        ]
    
    def get_templates_by_frequency(self, frequency: int) -> List[ConfigurationTemplate]:
        """Get templates filtered by frequency."""
        return [
            template for template in self.templates.values()
            if template.configuration.frequency == frequency
        ]
    
    def get_templates_by_stability_rating(self, min_rating: int) -> List[ConfigurationTemplate]:
        """Get templates with minimum stability rating."""
        return [
            template for template in self.templates.values()
            if template.stability_rating >= min_rating
        ]
    
    def search_templates(self, 
                        use_case: Optional[UseCase] = None,
                        performance_level: Optional[PerformanceLevel] = None,
                        min_stability: Optional[int] = None,
                        max_power: Optional[str] = None,
                        tags: Optional[List[str]] = None) -> List[ConfigurationTemplate]:
        """Search templates with multiple criteria."""
        
        results = list(self.templates.values())
        
        if use_case:
            results = [t for t in results if t.use_case == use_case]
        
        if performance_level:
            results = [t for t in results if t.performance_level == performance_level]
        
        if min_stability:
            results = [t for t in results if t.stability_rating >= min_stability]
        
        if max_power:
            power_levels = ["Very Low", "Low", "Medium", "Medium-High", "High", "Very High"]
            max_index = power_levels.index(max_power)
            results = [
                t for t in results 
                if power_levels.index(t.power_consumption) <= max_index
            ]
        
        if tags:
            results = [
                t for t in results 
                if any(tag in t.tags for tag in tags)
            ]
        
        return results
    
    def get_template(self, template_id: str) -> Optional[ConfigurationTemplate]:
        """Get a specific template by ID."""
        return self.templates.get(template_id)
    
    def add_custom_template(self, template: ConfigurationTemplate, template_id: str):
        """Add a custom template."""
        self.templates[template_id] = template
    
    def export_templates(self, output_file: str):
        """Export templates to JSON file."""
        export_data = {}
        
        for template_id, template in self.templates.items():
            export_data[template_id] = {
                "name": template.name,
                "description": template.description,
                "use_case": template.use_case.value,
                "performance_level": template.performance_level.value,
                "configuration": {
                    "frequency": template.configuration.frequency,
                    "timings": {
                        "cl": template.configuration.timings.cl,
                        "trcd": template.configuration.timings.trcd,
                        "trp": template.configuration.timings.trp,
                        "tras": template.configuration.timings.tras,
                        "trc": template.configuration.timings.trc
                    },
                    "voltages": {
                        "vddq": template.configuration.voltages.vddq,
                        "vpp": template.configuration.voltages.vpp,
                        "vdd1": template.configuration.voltages.vdd1,
                        "vdd2": template.configuration.voltages.vdd2,
                        "vddq_tx": template.configuration.voltages.vddq_tx
                    }
                },
                "estimated_performance": template.estimated_performance,
                "compatibility_notes": template.compatibility_notes,
                "stability_rating": template.stability_rating,
                "power_consumption": template.power_consumption,
                "author": template.author,
                "created_date": template.created_date,
                "tags": template.tags
            }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def import_templates(self, input_file: str):
        """Import templates from JSON file."""
        with open(input_file, 'r') as f:
            import_data = json.load(f)
        
        for template_id, data in import_data.items():
            template = ConfigurationTemplate(
                name=data["name"],
                description=data["description"],
                use_case=UseCase(data["use_case"]),
                performance_level=PerformanceLevel(data["performance_level"]),
                configuration=DDR5Configuration(
                    frequency=data["configuration"]["frequency"],
                    timings=DDR5TimingParameters(**data["configuration"]["timings"]),
                    voltages=DDR5VoltageParameters(**data["configuration"]["voltages"])
                ),
                estimated_performance=data["estimated_performance"],
                compatibility_notes=data["compatibility_notes"],
                stability_rating=data["stability_rating"],
                power_consumption=data["power_consumption"],
                author=data["author"],
                created_date=data["created_date"],
                tags=data["tags"]
            )
            self.templates[template_id] = template
    
    def get_recommendations(self, 
                          user_requirements: Dict[str, Any]) -> List[ConfigurationTemplate]:
        """Get template recommendations based on user requirements."""
        
        # Extract requirements
        use_case_str = user_requirements.get("use_case", "gaming")
        performance_str = user_requirements.get("performance_level", "balanced")
        min_stability = user_requirements.get("min_stability", 7)
        max_power = user_requirements.get("max_power", "High")
        
        # Convert strings to enums
        try:
            use_case = UseCase(use_case_str)
        except ValueError:
            use_case = UseCase.GAMING
        
        try:
            performance_level = PerformanceLevel(performance_str)
        except ValueError:
            performance_level = PerformanceLevel.BALANCED
        
        # Search templates
        recommendations = self.search_templates(
            use_case=use_case,
            performance_level=performance_level,
            min_stability=min_stability,
            max_power=max_power
        )
        
        # Sort by stability rating (descending)
        recommendations.sort(key=lambda x: x.stability_rating, reverse=True)
        
        return recommendations[:5]  # Return top 5 recommendations
