"""
Motherboard Database
Database of motherboards with DDR5 support, memory topology, and OC capabilities.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class MotherboardVendor(Enum):
    """Motherboard manufacturers."""
    ASUS = "ASUS"
    MSI = "MSI"
    GIGABYTE = "Gigabyte"
    ASROCK = "ASRock"
    EVGA = "EVGA"
    SUPERMICRO = "Supermicro"


class FormFactor(Enum):
    """Motherboard form factors."""
    E_ATX = "E-ATX"
    ATX = "ATX"
    MICRO_ATX = "Micro-ATX"
    MINI_ITX = "Mini-ITX"


class MemoryTopology(Enum):
    """Memory slot topology for signal integrity."""
    DAISY_CHAIN = "Daisy Chain"
    T_TOPOLOGY = "T-Topology"
    HYBRID = "Hybrid"


@dataclass
class MotherboardSpec:
    """Motherboard specifications for DDR5."""
    model: str
    vendor: MotherboardVendor
    socket: str
    chipset: str
    form_factor: FormFactor
    
    # Memory specifications
    memory_slots: int
    max_memory_gb: int
    memory_topology: MemoryTopology
    official_memory_speeds: List[int]  # JEDEC speeds
    max_oc_memory_speed: int
    
    # Features
    supports_ecc: bool = False
    supports_xmp: bool = True
    supports_expo: bool = False  # AMD EXPO
    memory_voltage_max: float = 1.5
    
    # PCB quality indicators
    memory_pcb_layers: int = 4  # 4, 6, 8+ layer PCB
    trace_length_mm: Optional[float] = None
    signal_integrity_rating: int = 7  # 1-10 scale
    
    # Additional specs
    release_year: int = 2022
    price_category: str = "Mid-Range"  # Budget, Mid-Range, High-End, Extreme
    rgb_lighting: bool = False
    wifi_version: str = "Wi-Fi 6E"
    
    # Overclocking features
    memory_oc_profiles: List[str] = None
    advanced_memory_tuning: bool = True
    memory_try_it: bool = False  # MSI Memory Try It
    memory_qvl_size: int = 50  # Number of tested memory kits
    
    def __post_init__(self):
        if self.memory_oc_profiles is None:
            self.memory_oc_profiles = ["XMP", "Manual"]
    
    def get_memory_bandwidth_estimate(self, speed: int) -> float:
        """Estimate memory bandwidth accounting for topology."""
        base_bandwidth = (speed * 8) / 1000  # GB/s
        
        # Apply topology penalty
        if self.memory_topology == MemoryTopology.DAISY_CHAIN:
            efficiency = 0.95  # Better for 2 DIMMs
        elif self.memory_topology == MemoryTopology.T_TOPOLOGY:
            efficiency = 0.92  # Better for 4 DIMMs
        else:
            efficiency = 0.93  # Hybrid
        
        # PCB quality factor
        pcb_factor = min(1.0, 0.85 + (self.memory_pcb_layers / 20))
        
        return base_bandwidth * efficiency * pcb_factor
    
    def is_memory_speed_supported(self, speed: int, overclocked: bool = False) -> bool:
        """Check if motherboard supports memory speed."""
        if overclocked:
            return speed <= self.max_oc_memory_speed
        return speed in self.official_memory_speeds


class MotherboardDatabase:
    """Database of DDR5-compatible motherboards."""
    
    def __init__(self):
        """Initialize motherboard database."""
        self.motherboards: List[MotherboardSpec] = []
        self._populate_database()
    
    def _populate_database(self):
        """Populate database with motherboard specifications."""
        
        # ASUS Z790 Series
        self.motherboards.extend([
            MotherboardSpec(
                model="ROG Maximus Z790 HERO",
                vendor=MotherboardVendor.ASUS,
                socket="LGA1700", chipset="Z790",
                form_factor=FormFactor.ATX,
                memory_slots=4, max_memory_gb=128,
                memory_topology=MemoryTopology.DAISY_CHAIN,
                official_memory_speeds=[4800, 5600, 6000],
                max_oc_memory_speed=8000,
                supports_ecc=False, supports_xmp=True,
                memory_voltage_max=1.5,
                memory_pcb_layers=8,
                signal_integrity_rating=9,
                release_year=2022,
                price_category="High-End",
                rgb_lighting=True,
                memory_oc_profiles=["XMP", "DOCP", "Manual"],
                advanced_memory_tuning=True,
                memory_qvl_size=150
            ),
            MotherboardSpec(
                model="ROG Strix Z790-E Gaming WiFi",
                vendor=MotherboardVendor.ASUS,
                socket="LGA1700", chipset="Z790",
                form_factor=FormFactor.ATX,
                memory_slots=4, max_memory_gb=128,
                memory_topology=MemoryTopology.DAISY_CHAIN,
                official_memory_speeds=[4800, 5600],
                max_oc_memory_speed=7600,
                supports_ecc=False, supports_xmp=True,
                memory_voltage_max=1.45,
                memory_pcb_layers=6,
                signal_integrity_rating=8,
                release_year=2022,
                price_category="High-End",
                rgb_lighting=True,
                memory_oc_profiles=["XMP", "DOCP", "Manual"],
                advanced_memory_tuning=True,
                memory_qvl_size=120
            )
        ])
        
        # MSI Z790 Series
        self.motherboards.extend([
            MotherboardSpec(
                model="MEG Z790 GODLIKE",
                vendor=MotherboardVendor.MSI,
                socket="LGA1700", chipset="Z790",
                form_factor=FormFactor.E_ATX,
                memory_slots=4, max_memory_gb=128,
                memory_topology=MemoryTopology.T_TOPOLOGY,
                official_memory_speeds=[4800, 5600, 6000],
                max_oc_memory_speed=8400,
                supports_ecc=False, supports_xmp=True,
                memory_voltage_max=1.6,
                memory_pcb_layers=10,
                signal_integrity_rating=10,
                release_year=2022,
                price_category="Extreme",
                rgb_lighting=True,
                memory_oc_profiles=["XMP", "Manual", "Memory Try It"],
                advanced_memory_tuning=True,
                memory_try_it=True,
                memory_qvl_size=200
            ),
            MotherboardSpec(
                model="MPG Z790 Carbon WiFi",
                vendor=MotherboardVendor.MSI,
                socket="LGA1700", chipset="Z790",
                form_factor=FormFactor.ATX,
                memory_slots=4, max_memory_gb=128,
                memory_topology=MemoryTopology.DAISY_CHAIN,
                official_memory_speeds=[4800, 5600],
                max_oc_memory_speed=7200,
                supports_ecc=False, supports_xmp=True,
                memory_voltage_max=1.45,
                memory_pcb_layers=6,
                signal_integrity_rating=8,
                release_year=2022,
                price_category="High-End",
                rgb_lighting=True,
                memory_oc_profiles=["XMP", "Manual", "Memory Try It"],
                advanced_memory_tuning=True,
                memory_try_it=True,
                memory_qvl_size=100
            )
        ])
        
        # Gigabyte Z790 Series
        self.motherboards.extend([
            MotherboardSpec(
                model="Z790 AORUS MASTER",
                vendor=MotherboardVendor.GIGABYTE,
                socket="LGA1700", chipset="Z790",
                form_factor=FormFactor.ATX,
                memory_slots=4, max_memory_gb=128,
                memory_topology=MemoryTopology.DAISY_CHAIN,
                official_memory_speeds=[4800, 5600, 6000],
                max_oc_memory_speed=8000,
                supports_ecc=False, supports_xmp=True,
                memory_voltage_max=1.5,
                memory_pcb_layers=8,
                signal_integrity_rating=9,
                release_year=2022,
                price_category="High-End",
                rgb_lighting=True,
                memory_oc_profiles=["XMP", "Manual"],
                advanced_memory_tuning=True,
                memory_qvl_size=120
            )
        ])
        
        # AMD AM5 Motherboards
        self.motherboards.extend([
            MotherboardSpec(
                model="ROG Crosshair X670E HERO",
                vendor=MotherboardVendor.ASUS,
                socket="AM5", chipset="X670E",
                form_factor=FormFactor.ATX,
                memory_slots=4, max_memory_gb=128,
                memory_topology=MemoryTopology.DAISY_CHAIN,
                official_memory_speeds=[4800, 5200, 5600],
                max_oc_memory_speed=7200,
                supports_ecc=True, supports_xmp=True, supports_expo=True,
                memory_voltage_max=1.45,
                memory_pcb_layers=8,
                signal_integrity_rating=9,
                release_year=2022,
                price_category="High-End",
                rgb_lighting=True,
                memory_oc_profiles=["EXPO", "XMP", "DOCP", "Manual"],
                advanced_memory_tuning=True,
                memory_qvl_size=100
            ),
            MotherboardSpec(
                model="MEG X670E GODLIKE",
                vendor=MotherboardVendor.MSI,
                socket="AM5", chipset="X670E",
                form_factor=FormFactor.E_ATX,
                memory_slots=4, max_memory_gb=128,
                memory_topology=MemoryTopology.T_TOPOLOGY,
                official_memory_speeds=[4800, 5200, 5600],
                max_oc_memory_speed=7600,
                supports_ecc=True, supports_xmp=True, supports_expo=True,
                memory_voltage_max=1.5,
                memory_pcb_layers=10,
                signal_integrity_rating=10,
                release_year=2022,
                price_category="Extreme",
                rgb_lighting=True,
                memory_oc_profiles=["EXPO", "XMP", "Manual", "Memory Try It"],
                advanced_memory_tuning=True,
                memory_try_it=True,
                memory_qvl_size=150
            )
        ])
        
        # Budget/Mid-Range Options
        self.motherboards.extend([
            MotherboardSpec(
                model="B650M Pro WiFi",
                vendor=MotherboardVendor.MSI,
                socket="AM5", chipset="B650",
                form_factor=FormFactor.MICRO_ATX,
                memory_slots=4, max_memory_gb=128,
                memory_topology=MemoryTopology.DAISY_CHAIN,
                official_memory_speeds=[4800, 5200],
                max_oc_memory_speed=6000,
                supports_ecc=True, supports_xmp=True, supports_expo=True,
                memory_voltage_max=1.35,
                memory_pcb_layers=4,
                signal_integrity_rating=6,
                release_year=2022,
                price_category="Budget",
                rgb_lighting=False,
                memory_oc_profiles=["EXPO", "XMP"],
                advanced_memory_tuning=False,
                memory_qvl_size=30
            )
        ])
    
    def search_by_model(self, model: str) -> Optional[MotherboardSpec]:
        """Find motherboard by model name."""
        for mb in self.motherboards:
            if model.lower() in mb.model.lower():
                return mb
        return None
    
    def search_by_socket(self, socket: str) -> List[MotherboardSpec]:
        """Get motherboards for specific socket."""
        return [mb for mb in self.motherboards if mb.socket == socket]
    
    def search_by_chipset(self, chipset: str) -> List[MotherboardSpec]:
        """Get motherboards with specific chipset."""
        return [mb for mb in self.motherboards if mb.chipset == chipset]
    
    def find_memory_speed_compatible(
        self, speed: int, overclocked: bool = False
    ) -> List[MotherboardSpec]:
        """Find motherboards compatible with memory speed."""
        return [
            mb for mb in self.motherboards
            if mb.is_memory_speed_supported(speed, overclocked)
        ]
    
    def get_overclocking_champions(self) -> List[MotherboardSpec]:
        """Get motherboards with best memory OC support."""
        return sorted(
            self.motherboards,
            key=lambda mb: (mb.max_oc_memory_speed, mb.signal_integrity_rating),
            reverse=True
        )[:10]
    
    def get_budget_options(self) -> List[MotherboardSpec]:
        """Get budget-friendly motherboards."""
        return [
            mb for mb in self.motherboards
            if mb.price_category in ["Budget", "Mid-Range"]
        ]
    
    def generate_memory_compatibility_report(
        self, motherboard_model: str, target_configs: List[Dict]
    ) -> Dict:
        """Generate memory compatibility report for motherboard."""
        mb = self.search_by_model(motherboard_model)
        if not mb:
            return {"error": f"Motherboard {motherboard_model} not found"}
        
        report = {
            "motherboard": mb.model,
            "chipset": mb.chipset,
            "memory_topology": mb.memory_topology.value,
            "max_oc_speed": mb.max_oc_memory_speed,
            "signal_integrity": mb.signal_integrity_rating,
            "compatibility": []
        }
        
        for config in target_configs:
            speed = config.get("frequency", 0)
            is_official = speed in mb.official_memory_speeds
            is_oc_supported = speed <= mb.max_oc_memory_speed
            
            estimated_bandwidth = mb.get_memory_bandwidth_estimate(speed)
            
            report["compatibility"].append({
                "speed": speed,
                "official_support": is_official,
                "oc_support": is_oc_supported,
                "estimated_bandwidth_gbps": round(estimated_bandwidth, 1),
                "status": "OFFICIAL" if is_official else 
                         "OC" if is_oc_supported else "UNSUPPORTED"
            })
        
        return report


# Global database instance
_motherboard_database = None

def get_motherboard_database() -> MotherboardDatabase:
    """Get global motherboard database instance."""
    global _motherboard_database
    if _motherboard_database is None:
        _motherboard_database = MotherboardDatabase()
    return _motherboard_database
