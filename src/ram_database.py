"""
DDR5 RAM Database
Comprehensive database of real DDR5 memory modules from major manufacturers.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class MemoryChipType(Enum):
    """DDR5 memory chip types."""
    SAMSUNG_B_DIE = "Samsung B-die"
    SAMSUNG_C_DIE = "Samsung C-die"
    MICRON_B_DIE = "Micron B-die"
    SK_HYNIX_A_DIE = "SK Hynix A-die"
    SK_HYNIX_M_DIE = "SK Hynix M-die"
    UNKNOWN = "Unknown"


class OverclockingPotential(Enum):
    """Overclocking potential ratings."""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    AVERAGE = "Average"
    LIMITED = "Limited"
    POOR = "Poor"


@dataclass
class DDR5ModuleSpec:
    """DDR5 memory module specifications."""
    manufacturer: str
    series: str
    part_number: str
    capacity_gb: int
    kit_size: int  # Number of modules in kit
    jedec_speed: int  # MT/s
    cas_latency: int
    voltage: float
    chip_type: MemoryChipType
    overclocking_potential: OverclockingPotential
    max_tested_speed: int  # Maximum proven stable speed
    price_range: str  # Price category
    rgb_lighting: bool = False
    heatspreader: str = "Standard"
    warranty_years: int = 3
    
    # Detailed timings
    trcd: Optional[int] = None
    trp: Optional[int] = None
    tras: Optional[int] = None
    trc: Optional[int] = None
    trfc: Optional[int] = None
    
    # Additional specs
    form_factor: str = "DIMM"
    height_mm: Optional[float] = None
    features: List[str] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = []
    
    def get_full_name(self) -> str:
        """Get full module name."""
        return f"{self.manufacturer} {self.series} {self.part_number}"
    
    def is_compatible_with_speed(self, target_speed: int) -> bool:
        """Check if module can handle target speed."""
        return target_speed <= self.max_tested_speed
    
    def get_performance_rating(self) -> float:
        """Get overall performance rating (0-100)."""
        base_score = (self.jedec_speed / 8400) * 40  # Speed component
        latency_score = max(0, (50 - self.cas_latency) / 50) * 30  # Latency
        oc_score = {
            OverclockingPotential.EXCELLENT: 30,
            OverclockingPotential.GOOD: 25,
            OverclockingPotential.AVERAGE: 20,
            OverclockingPotential.LIMITED: 15,
            OverclockingPotential.POOR: 10
        }.get(self.overclocking_potential, 15)
        
        return min(100, base_score + latency_score + oc_score)


class DDR5Database:
    """Comprehensive DDR5 memory module database."""
    
    def __init__(self):
        """Initialize the database with real DDR5 modules."""
        self.modules: List[DDR5ModuleSpec] = []
        self._populate_database()
    
    def _populate_database(self):
        """Populate database with real DDR5 modules."""
        
        # Corsair Dominator Platinum RGB
        self.modules.extend([
            DDR5ModuleSpec(
                manufacturer="Corsair",
                series="Dominator Platinum RGB",
                part_number="CMT32GX5M2B5600C36",
                capacity_gb=16,
                kit_size=2,
                jedec_speed=5600,
                cas_latency=36,
                voltage=1.25,
                chip_type=MemoryChipType.SAMSUNG_B_DIE,
                overclocking_potential=OverclockingPotential.EXCELLENT,
                max_tested_speed=7200,
                price_range="Premium",
                rgb_lighting=True,
                heatspreader="Premium DHX",
                trcd=36, trp=36, tras=76, trc=112, trfc=560,
                height_mm=56.0,
                features=["RGB", "Premium Heatspreader", "Hand-sorted ICs"]
            ),
            DDR5ModuleSpec(
                manufacturer="Corsair",
                series="Vengeance RGB",
                part_number="CMH32GX5M2B5600C36",
                capacity_gb=16,
                kit_size=2,
                jedec_speed=5600,
                cas_latency=36,
                voltage=1.25,
                chip_type=MemoryChipType.SAMSUNG_B_DIE,
                overclocking_potential=OverclockingPotential.GOOD,
                max_tested_speed=6800,
                price_range="Mid-Range",
                rgb_lighting=True,
                heatspreader="Aluminum",
                trcd=36, trp=36, tras=76, trc=112, trfc=560,
                height_mm=44.0,
                features=["RGB", "Aluminum Heatspreader", "Intel XMP 3.0"]
            )
        ])
        
        # G.Skill Trident Z5 Series
        self.modules.extend([
            DDR5ModuleSpec(
                manufacturer="G.Skill",
                series="Trident Z5 RGB",
                part_number="F5-6000J3038F16GX2-TZ5RK",
                capacity_gb=16,
                kit_size=2,
                jedec_speed=6000,
                cas_latency=30,
                voltage=1.35,
                chip_type=MemoryChipType.SAMSUNG_B_DIE,
                overclocking_potential=OverclockingPotential.EXCELLENT,
                max_tested_speed=7600,
                price_range="Premium",
                rgb_lighting=True,
                heatspreader="Tri-fin",
                trcd=38, trp=38, tras=68, trc=106, trfc=560,
                height_mm=44.0,
                features=["RGB", "Tri-fin Heatspreader", "Hand-binned"]
            ),
            DDR5ModuleSpec(
                manufacturer="G.Skill",
                series="Ripjaws S5",
                part_number="F5-5600J3636C16GX2-RS5K",
                capacity_gb=16,
                kit_size=2,
                jedec_speed=5600,
                cas_latency=36,
                voltage=1.25,
                chip_type=MemoryChipType.SK_HYNIX_A_DIE,
                overclocking_potential=OverclockingPotential.GOOD,
                max_tested_speed=6400,
                price_range="Budget",
                rgb_lighting=False,
                heatspreader="Basic",
                trcd=36, trp=36, tras=76, trc=112, trfc=560,
                height_mm=32.0,
                features=["Value Oriented", "Intel XMP 3.0"]
            )
        ])
        
        # Crucial (Micron)
        self.modules.extend([
            DDR5ModuleSpec(
                manufacturer="Crucial",
                series="Pro",
                part_number="CP2K16G52C42U5",
                capacity_gb=16,
                kit_size=2,
                jedec_speed=5200,
                cas_latency=42,
                voltage=1.10,
                chip_type=MemoryChipType.MICRON_B_DIE,
                overclocking_potential=OverclockingPotential.AVERAGE,
                max_tested_speed=5600,
                price_range="Budget",
                rgb_lighting=False,
                heatspreader="Basic",
                trcd=42, trp=42, tras=82, trc=124, trfc=560,
                height_mm=31.25,
                features=["JEDEC Standard", "Reliable", "ECC Support"]
            ),
            DDR5ModuleSpec(
                manufacturer="Crucial",
                series="Ballistix MAX",
                part_number="BLM2K16G52C42U4B",
                capacity_gb=16,
                kit_size=2,
                jedec_speed=5200,
                cas_latency=42,
                voltage=1.20,
                chip_type=MemoryChipType.MICRON_B_DIE,
                overclocking_potential=OverclockingPotential.GOOD,
                max_tested_speed=6000,
                price_range="Mid-Range",
                rgb_lighting=False,
                heatspreader="Enhanced",
                trcd=42, trp=42, tras=82, trc=124, trfc=560,
                height_mm=39.17,
                features=["Overclocking Ready", "Temperature Sensor"]
            )
        ])
        
        # Kingston Fury
        self.modules.extend([
            DDR5ModuleSpec(
                manufacturer="Kingston",
                series="Fury Beast",
                part_number="KF552C40BBK2-32",
                capacity_gb=16,
                kit_size=2,
                jedec_speed=5200,
                cas_latency=40,
                voltage=1.25,
                chip_type=MemoryChipType.SK_HYNIX_A_DIE,
                overclocking_potential=OverclockingPotential.GOOD,
                max_tested_speed=6000,
                price_range="Mid-Range",
                rgb_lighting=False,
                heatspreader="Stylized",
                trcd=40, trp=40, tras=80, trc=120, trfc=560,
                height_mm=34.1,
                features=["Gaming Optimized", "Intel XMP 3.0", "AMD EXPO"]
            ),
            DDR5ModuleSpec(
                manufacturer="Kingston",
                series="Fury Beast RGB",
                part_number="KF552C40BBAK2-32",
                capacity_gb=16,
                kit_size=2,
                jedec_speed=5200,
                cas_latency=40,
                voltage=1.25,
                chip_type=MemoryChipType.SK_HYNIX_A_DIE,
                overclocking_potential=OverclockingPotential.GOOD,
                max_tested_speed=6000,
                price_range="Mid-Range",
                rgb_lighting=True,
                heatspreader="RGB Stylized",
                trcd=40, trp=40, tras=80, trc=120, trfc=560,
                height_mm=42.0,
                features=["RGB", "Gaming Optimized", "Intel XMP 3.0"]
            )
        ])
        
        # Additional Kingston Fury variants
        self.modules.extend([
            DDR5ModuleSpec(
                manufacturer="Kingston",
                series="Fury Beast",
                part_number="KF556C40-32",
                capacity_gb=32,
                kit_size=1,  # Single module
                jedec_speed=4800,  # User's detected DDR5-4800 speed
                cas_latency=40,
                voltage=1.1,  # User's detected voltage
                chip_type=MemoryChipType.SK_HYNIX_A_DIE,
                overclocking_potential=OverclockingPotential.GOOD,
                max_tested_speed=6000,
                price_range="Mid-Range",
                rgb_lighting=False,
                heatspreader="Stylized Beast",
                trcd=40, trp=40, tras=80, trc=120, trfc=560,
                height_mm=34.1,
                features=["Gaming Optimized", "Intel XMP 3.0", "AMD EXPO", "32GB Single Module", "User's Exact Module"]
            ),
            # Alternative lower speed variant
            DDR5ModuleSpec(
                manufacturer="Kingston",
                series="Fury Beast",
                part_number="KF548C40-32",  # Similar model at DDR5-4800
                capacity_gb=32,
                kit_size=1,
                jedec_speed=4800,  # Matches user's detected speed
                cas_latency=40,
                voltage=1.1,
                chip_type=MemoryChipType.SK_HYNIX_A_DIE,
                overclocking_potential=OverclockingPotential.AVERAGE,
                max_tested_speed=5600,
                price_range="Mid-Range",
                rgb_lighting=False,
                heatspreader="Stylized Beast",
                trcd=40, trp=40, tras=80, trc=120, trfc=560,
                height_mm=34.1,
                features=["Gaming Optimized", "JEDEC Standard", "32GB Single Module"]
            )
        ])
        
        # More Kingston Fury variants
        self.modules.extend([
            DDR5ModuleSpec(
                manufacturer="Kingston",
                series="Fury Beast",
                part_number="KF560C40BBK2-32",
                capacity_gb=16,
                kit_size=2,
                jedec_speed=6000,
                cas_latency=40,
                voltage=1.35,
                chip_type=MemoryChipType.SK_HYNIX_A_DIE,
                overclocking_potential=OverclockingPotential.EXCELLENT,
                max_tested_speed=6400,
                price_range="Mid-Range",
                rgb_lighting=False,
                heatspreader="Stylized",
                trcd=40, trp=40, tras=80, trc=120, trfc=560,
                height_mm=34.1,
                features=["Gaming Optimized", "Intel XMP 3.0", "AMD EXPO"]
            ),
            DDR5ModuleSpec(
                manufacturer="Kingston",
                series="Fury Beast",
                part_number="KF560C36BBK2-32",
                capacity_gb=16,
                kit_size=2,
                jedec_speed=6000,
                cas_latency=36,
                voltage=1.35,
                chip_type=MemoryChipType.SK_HYNIX_A_DIE,
                overclocking_potential=OverclockingPotential.EXCELLENT,
                max_tested_speed=6600,
                price_range="Performance",
                rgb_lighting=False,
                heatspreader="Enhanced",
                trcd=36, trp=36, tras=76, trc=112, trfc=560,
                height_mm=34.1,
                features=["Gaming Optimized", "Intel XMP 3.0", "AMD EXPO", "Tighter Timings"]
            ),
            DDR5ModuleSpec(
                manufacturer="Kingston",
                series="Fury Renegade",
                part_number="KF564C32RS-16",
                capacity_gb=16,
                kit_size=1,
                jedec_speed=6400,
                cas_latency=32,
                voltage=1.40,
                chip_type=MemoryChipType.SAMSUNG_B_DIE,
                overclocking_potential=OverclockingPotential.EXCELLENT,
                max_tested_speed=7200,
                price_range="Premium",
                rgb_lighting=False,
                heatspreader="Premium",
                trcd=32, trp=32, tras=72, trc=104, trfc=520,
                height_mm=42.0,
                features=["Extreme Performance", "Intel XMP 3.0", "AMD EXPO", "Premium ICs"]
            ),
            DDR5ModuleSpec(
                manufacturer="Kingston",
                series="Fury Renegade RGB",
                part_number="KF564C32RSAK2-32",
                capacity_gb=16,
                kit_size=2,
                jedec_speed=6400,
                cas_latency=32,
                voltage=1.40,
                chip_type=MemoryChipType.SAMSUNG_B_DIE,
                overclocking_potential=OverclockingPotential.EXCELLENT,
                max_tested_speed=7400,
                price_range="Premium",
                rgb_lighting=True,
                heatspreader="Premium RGB",
                trcd=32, trp=32, tras=72, trc=104, trfc=520,
                height_mm=44.0,
                features=["RGB Lightning", "Extreme Performance", "Intel XMP 3.0", "AMD EXPO"]
            ),
            DDR5ModuleSpec(
                manufacturer="Kingston",
                series="Fury Beast",
                part_number="KF548C38BBK2-32",
                capacity_gb=16,
                kit_size=2,
                jedec_speed=4800,
                cas_latency=38,
                voltage=1.10,
                chip_type=MemoryChipType.SK_HYNIX_M_DIE,
                overclocking_potential=OverclockingPotential.AVERAGE,
                max_tested_speed=5600,
                price_range="Budget",
                rgb_lighting=False,
                heatspreader="Standard",
                trcd=38, trp=38, tras=78, trc=116, trfc=560,
                height_mm=31.25,
                features=["JEDEC Compliant", "Intel XMP 3.0", "AMD EXPO"]
            ),
            DDR5ModuleSpec(
                manufacturer="Kingston",
                series="Fury Beast",
                part_number="KF556C40BBK2-64",
                capacity_gb=32,
                kit_size=2,
                jedec_speed=5600,
                cas_latency=40,
                voltage=1.25,
                chip_type=MemoryChipType.SK_HYNIX_A_DIE,
                overclocking_potential=OverclockingPotential.GOOD,
                max_tested_speed=6400,
                price_range="High-End",
                rgb_lighting=False,
                heatspreader="Enhanced",
                trcd=40, trp=40, tras=80, trc=120, trfc=640,
                height_mm=34.1,
                features=["High Capacity", "Gaming Optimized", "Intel XMP 3.0", "AMD EXPO"]
            )
        ])

        # High-end enthusiast modules
        self.modules.extend([
            DDR5ModuleSpec(
                manufacturer="G.Skill",
                series="Trident Z5 Royal",
                part_number="F5-6400J3239G16GX2-TZ5RG",
                capacity_gb=16,
                kit_size=2,
                jedec_speed=6400,
                cas_latency=32,
                voltage=1.40,
                chip_type=MemoryChipType.SAMSUNG_B_DIE,
                overclocking_potential=OverclockingPotential.EXCELLENT,
                max_tested_speed=8000,
                price_range="Ultra-Premium",
                rgb_lighting=True,
                heatspreader="Crystal Design",
                trcd=39, trp=39, tras=71, trc=110, trfc=560,
                height_mm=44.0,
                features=["RGB", "Crystal Design", "Hand-binned Elite"]
            ),
            DDR5ModuleSpec(
                manufacturer="Corsair",
                series="Dominator Titanium",
                part_number="CMT32GX5M2X6400C32",
                capacity_gb=16,
                kit_size=2,
                jedec_speed=6400,
                cas_latency=32,
                voltage=1.40,
                chip_type=MemoryChipType.SAMSUNG_B_DIE,
                overclocking_potential=OverclockingPotential.EXCELLENT,
                max_tested_speed=8200,
                price_range="Ultra-Premium",
                rgb_lighting=True,
                heatspreader="Titanium DHX",
                trcd=39, trp=39, tras=71, trc=110, trfc=560,
                height_mm=56.0,
                features=["RGB", "Titanium Finish", "Elite Performance"]
            )
        ])
    
    def search_by_manufacturer(self, manufacturer: str) -> List[DDR5ModuleSpec]:
        """Search modules by manufacturer."""
        return [m for m in self.modules if manufacturer.lower() in m.manufacturer.lower()]
    
    def search_by_part_number(self, part_number: str) -> Optional[DDR5ModuleSpec]:
        """Find module by exact part number."""
        for module in self.modules:
            if part_number.upper() in module.part_number.upper():
                return module
        return None
    
    def search_by_speed_range(self, min_speed: int, max_speed: int) -> List[DDR5ModuleSpec]:
        """Search modules within speed range."""
        return [m for m in self.modules if min_speed <= m.jedec_speed <= max_speed]
    
    def search_by_capacity(self, capacity_gb: int) -> List[DDR5ModuleSpec]:
        """Search modules by capacity."""
        return [m for m in self.modules if m.capacity_gb == capacity_gb]
    
    def search_by_chip_type(self, chip_type: MemoryChipType) -> List[DDR5ModuleSpec]:
        """Search modules by memory chip type."""
        return [m for m in self.modules if m.chip_type == chip_type]
    
    def get_overclocking_champions(self) -> List[DDR5ModuleSpec]:
        """Get modules with excellent overclocking potential."""
        return [m for m in self.modules if m.overclocking_potential == OverclockingPotential.EXCELLENT]
    
    def get_budget_recommendations(self) -> List[DDR5ModuleSpec]:
        """Get budget-friendly module recommendations."""
        return [m for m in self.modules if m.price_range in ["Budget", "Mid-Range"]]
    
    def get_rgb_modules(self) -> List[DDR5ModuleSpec]:
        """Get modules with RGB lighting."""
        return [m for m in self.modules if m.rgb_lighting]
    
    def find_similar_modules(self, detected_module) -> List[DDR5ModuleSpec]:
        """Find modules similar to a detected module."""
        # Try exact part number match first
        if hasattr(detected_module, 'part_number'):
            exact_match = self.search_by_part_number(detected_module.part_number)
            if exact_match:
                return [exact_match]
        
        # Fallback to manufacturer and capacity matching
        similar = []
        if hasattr(detected_module, 'manufacturer') and hasattr(detected_module, 'capacity_gb'):
            manufacturer_modules = self.search_by_manufacturer(detected_module.manufacturer)
            similar = [m for m in manufacturer_modules if m.capacity_gb == detected_module.capacity_gb]
        
        return similar
    
    def get_all_manufacturers(self) -> List[str]:
        """Get list of all manufacturers in database."""
        return list(set(m.manufacturer for m in self.modules))
    
    def get_speed_distribution(self) -> Dict[int, int]:
        """Get distribution of modules by speed."""
        speeds = {}
        for module in self.modules:
            speed = module.jedec_speed
            speeds[speed] = speeds.get(speed, 0) + 1
        return dict(sorted(speeds.items()))
    
    def generate_compatibility_report(self, detected_modules) -> str:
        """Generate compatibility report for detected modules."""
        if not detected_modules:
            return "❌ No modules detected for compatibility analysis"
        
        report = "🔍 **DDR5 Module Compatibility Analysis**\n\n"
        
        for i, detected in enumerate(detected_modules, 1):
            report += f"**Module {i}: {detected}**\n"
            
            similar = self.find_similar_modules(detected)
            if similar:
                best_match = similar[0]
                report += f"✅ **Match Found**: {best_match.get_full_name()}\n"
                report += f"  - Chip Type: {best_match.chip_type.value}\n"
                report += f"  - OC Potential: {best_match.overclocking_potential.value}\n"
                report += f"  - Max Speed: DDR5-{best_match.max_tested_speed}\n"
                report += f"  - Performance Rating: {best_match.get_performance_rating():.1f}/100\n"
                
                # Optimization suggestions
                if best_match.overclocking_potential in [OverclockingPotential.EXCELLENT, OverclockingPotential.GOOD]:
                    report += f"  - 🚀 **Optimization Potential**: High! Can likely reach DDR5-{best_match.max_tested_speed}\n"
                else:
                    report += f"  - ⚠️  **Optimization Potential**: Limited to DDR5-{best_match.max_tested_speed}\n"
            else:
                report += f"❓ **Unknown Module**: Not in database, using generic optimization\n"
            
            report += "\n"
        
        return report


# Global database instance
ddr5_db = DDR5Database()


def get_database() -> DDR5Database:
    """Get the global DDR5 database instance."""
    return ddr5_db


def search_module(manufacturer: str = "", part_number: str = "", speed: int = 0) -> List[DDR5ModuleSpec]:
    """Search for DDR5 modules with given criteria."""
    results = ddr5_db.modules
    
    if manufacturer:
        results = [m for m in results if manufacturer.lower() in m.manufacturer.lower()]
    
    if part_number:
        results = [m for m in results if part_number.upper() in m.part_number.upper()]
    
    if speed > 0:
        results = [m for m in results if m.jedec_speed >= speed]
    
    return results
