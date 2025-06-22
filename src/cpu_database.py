"""
CPU Compatibility Database
Database of Intel and AMD CPUs with DDR5 support and memory controller
specifications.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class CPUVendor(Enum):
    """CPU vendors."""
    INTEL = "Intel"
    AMD = "AMD"


class CPUArchitecture(Enum):
    """CPU architectures with DDR5 support."""
    # Intel
    ALDER_LAKE = "Alder Lake"
    RAPTOR_LAKE = "Raptor Lake"
    METEOR_LAKE = "Meteor Lake"
    ARROW_LAKE = "Arrow Lake"
    
    # AMD
    RYZEN_7000 = "Ryzen 7000 (Zen 4)"
    RYZEN_8000 = "Ryzen 8000 (Zen 4 Refresh)"
    RYZEN_9000 = "Ryzen 9000 (Zen 5)"


class MemoryControllerType(Enum):
    """Memory controller types."""
    DUAL_CHANNEL = "Dual Channel"
    QUAD_CHANNEL = "Quad Channel"


@dataclass
class CPUSpec:
    """CPU specifications for DDR5 compatibility."""
    model: str
    vendor: CPUVendor
    architecture: CPUArchitecture
    cores: int
    threads: int
    base_clock_ghz: float
    boost_clock_ghz: float
    
    # Memory controller specs
    memory_controller: MemoryControllerType
    supported_ddr5_speeds: List[int]  # Official JEDEC speeds
    max_ddr5_speed_oc: int  # Maximum overclocked speed
    memory_channels: int
    max_memory_gb: int
    
    # Additional specs
    tdp_watts: int
    socket: str
    release_year: int
    manufacturing_process: str
    
    # DDR5 specific features
    supports_ecc: bool = False
    supports_xmp: bool = True
    supports_expo: bool = False  # AMD EXPO
    memory_voltage_range: tuple = (1.1, 1.35)  # Min, Max voltage
    
    def get_jedec_speed_rating(self) -> int:
        """Get highest official JEDEC speed."""
        if self.supported_ddr5_speeds:
            return max(self.supported_ddr5_speeds)
        return 4800
    
    def is_speed_supported(
        self, speed: int, overclocked: bool = False
    ) -> bool:
        """Check if CPU supports given memory speed."""
        if overclocked:
            return speed <= self.max_ddr5_speed_oc
        return speed in self.supported_ddr5_speeds
    
    def get_memory_bandwidth_theoretical(self, speed: int) -> float:
        """Calculate theoretical memory bandwidth in GB/s."""
        # DDR5 = 64-bit bus width, dual channel = 128-bit
        bus_width_bits = 64 * self.memory_channels
        return (speed * 1_000_000 * bus_width_bits) / (8 * 1_000_000_000)


class CPUDatabase:
    """Database of CPUs with DDR5 support."""
    
    def __init__(self):
        """Initialize CPU database."""
        self.cpus: List[CPUSpec] = []
        self._populate_database()
    
    def _populate_database(self):
        """Populate database with CPU specifications."""
        
        # Intel 12th Gen (Alder Lake)
        self.cpus.extend([
            CPUSpec(
                model="Core i9-12900K",
                vendor=CPUVendor.INTEL,
                architecture=CPUArchitecture.ALDER_LAKE,
                cores=16, threads=24,
                base_clock_ghz=3.2, boost_clock_ghz=5.2,
                memory_controller=MemoryControllerType.DUAL_CHANNEL,
                supported_ddr5_speeds=[4800, 5600],
                max_ddr5_speed_oc=8000,
                memory_channels=2, max_memory_gb=128,
                tdp_watts=125, socket="LGA1700",
                release_year=2021, manufacturing_process="Intel 7",
                supports_ecc=False, supports_xmp=True
            ),
            CPUSpec(
                model="Core i7-12700K",
                vendor=CPUVendor.INTEL,
                architecture=CPUArchitecture.ALDER_LAKE,
                cores=12, threads=20,
                base_clock_ghz=3.6, boost_clock_ghz=5.0,
                memory_controller=MemoryControllerType.DUAL_CHANNEL,
                supported_ddr5_speeds=[4800, 5600],
                max_ddr5_speed_oc=7600,
                memory_channels=2, max_memory_gb=128,
                tdp_watts=125, socket="LGA1700",
                release_year=2021, manufacturing_process="Intel 7",
                supports_ecc=False, supports_xmp=True
            ),
            CPUSpec(
                model="Core i5-12600K",
                vendor=CPUVendor.INTEL,
                architecture=CPUArchitecture.ALDER_LAKE,
                cores=10, threads=16,
                base_clock_ghz=3.7, boost_clock_ghz=4.9,
                memory_controller=MemoryControllerType.DUAL_CHANNEL,
                supported_ddr5_speeds=[4800, 5600],
                max_ddr5_speed_oc=7200,
                memory_channels=2, max_memory_gb=128,
                tdp_watts=125, socket="LGA1700",
                release_year=2021, manufacturing_process="Intel 7",
                supports_ecc=False, supports_xmp=True
            )
        ])
        
        # Intel 13th Gen (Raptor Lake)
        self.cpus.extend([
            CPUSpec(
                model="Core i9-13900K",
                vendor=CPUVendor.INTEL,
                architecture=CPUArchitecture.RAPTOR_LAKE,
                cores=24, threads=32,
                base_clock_ghz=3.0, boost_clock_ghz=5.8,
                memory_controller=MemoryControllerType.DUAL_CHANNEL,
                supported_ddr5_speeds=[4800, 5600, 6000],
                max_ddr5_speed_oc=8400,
                memory_channels=2, max_memory_gb=128,
                tdp_watts=125, socket="LGA1700",
                release_year=2022, manufacturing_process="Intel 7",
                supports_ecc=False, supports_xmp=True
            ),
            CPUSpec(
                model="Core i7-13700K",
                vendor=CPUVendor.INTEL,
                architecture=CPUArchitecture.RAPTOR_LAKE,
                cores=16, threads=24,
                base_clock_ghz=3.4, boost_clock_ghz=5.4,
                memory_controller=MemoryControllerType.DUAL_CHANNEL,
                supported_ddr5_speeds=[4800, 5600, 6000],
                max_ddr5_speed_oc=8000,
                memory_channels=2, max_memory_gb=128,
                tdp_watts=125, socket="LGA1700",
                release_year=2022, manufacturing_process="Intel 7",
                supports_ecc=False, supports_xmp=True
            )
        ])
        
        # Intel 14th Gen (Raptor Lake Refresh)
        self.cpus.extend([
            CPUSpec(
                model="Core i9-14900K",
                vendor=CPUVendor.INTEL,
                architecture=CPUArchitecture.RAPTOR_LAKE,
                cores=24, threads=32,
                base_clock_ghz=3.2, boost_clock_ghz=6.0,
                memory_controller=MemoryControllerType.DUAL_CHANNEL,
                supported_ddr5_speeds=[4800, 5600, 6000, 6400],
                max_ddr5_speed_oc=8600,
                memory_channels=2, max_memory_gb=192,
                tdp_watts=125, socket="LGA1700",
                release_year=2023, manufacturing_process="Intel 7",
                supports_ecc=False, supports_xmp=True
            )
        ])
        
        # AMD Ryzen 7000 Series (Zen 4)
        self.cpus.extend([
            CPUSpec(
                model="Ryzen 9 7950X",
                vendor=CPUVendor.AMD,
                architecture=CPUArchitecture.RYZEN_7000,
                cores=16, threads=32,
                base_clock_ghz=4.5, boost_clock_ghz=5.7,
                memory_controller=MemoryControllerType.DUAL_CHANNEL,
                supported_ddr5_speeds=[4800, 5200, 5600],
                max_ddr5_speed_oc=7200,
                memory_channels=2, max_memory_gb=128,
                tdp_watts=170, socket="AM5",
                release_year=2022, manufacturing_process="TSMC 5nm",
                supports_ecc=True, supports_xmp=True, supports_expo=True
            ),
            CPUSpec(
                model="Ryzen 9 7900X",
                vendor=CPUVendor.AMD,
                architecture=CPUArchitecture.RYZEN_7000,
                cores=12, threads=24,
                base_clock_ghz=4.7, boost_clock_ghz=5.6,
                memory_controller=MemoryControllerType.DUAL_CHANNEL,
                supported_ddr5_speeds=[4800, 5200, 5600],
                max_ddr5_speed_oc=7000,
                memory_channels=2, max_memory_gb=128,
                tdp_watts=170, socket="AM5",
                release_year=2022, manufacturing_process="TSMC 5nm",
                supports_ecc=True, supports_xmp=True, supports_expo=True
            ),
            CPUSpec(
                model="Ryzen 7 7700X",
                vendor=CPUVendor.AMD,
                architecture=CPUArchitecture.RYZEN_7000,
                cores=8, threads=16,
                base_clock_ghz=4.5, boost_clock_ghz=5.4,
                memory_controller=MemoryControllerType.DUAL_CHANNEL,
                supported_ddr5_speeds=[4800, 5200, 5600],
                max_ddr5_speed_oc=6800,
                memory_channels=2, max_memory_gb=128,
                tdp_watts=105, socket="AM5",
                release_year=2022, manufacturing_process="TSMC 5nm",
                supports_ecc=True, supports_xmp=True, supports_expo=True
            ),
            CPUSpec(
                model="Ryzen 5 7600X",
                vendor=CPUVendor.AMD,
                architecture=CPUArchitecture.RYZEN_7000,
                cores=6, threads=12,
                base_clock_ghz=4.7, boost_clock_ghz=5.3,
                memory_controller=MemoryControllerType.DUAL_CHANNEL,
                supported_ddr5_speeds=[4800, 5200, 5600],
                max_ddr5_speed_oc=6400,
                memory_channels=2, max_memory_gb=128,
                tdp_watts=105, socket="AM5",
                release_year=2022, manufacturing_process="TSMC 5nm",
                supports_ecc=True, supports_xmp=True, supports_expo=True
            )
        ])
        
        # AMD Ryzen 9000 Series (Zen 5) - 2024/2025
        self.cpus.extend([
            CPUSpec(
                model="Ryzen 9 9950X",
                vendor=CPUVendor.AMD,
                architecture=CPUArchitecture.RYZEN_9000,
                cores=16, threads=32,
                base_clock_ghz=4.3, boost_clock_ghz=5.7,
                memory_controller=MemoryControllerType.DUAL_CHANNEL,
                supported_ddr5_speeds=[4800, 5200, 5600, 6000],
                max_ddr5_speed_oc=7600,
                memory_channels=2, max_memory_gb=192,
                tdp_watts=170, socket="AM5",
                release_year=2024, manufacturing_process="TSMC 4nm",
                supports_ecc=True, supports_xmp=True, supports_expo=True
            ),
            CPUSpec(
                model="Ryzen 7 9700X",
                vendor=CPUVendor.AMD,
                architecture=CPUArchitecture.RYZEN_9000,
                cores=8, threads=16,
                base_clock_ghz=3.8, boost_clock_ghz=5.5,
                memory_controller=MemoryControllerType.DUAL_CHANNEL,
                supported_ddr5_speeds=[4800, 5200, 5600, 6000],
                max_ddr5_speed_oc=7200,
                memory_channels=2, max_memory_gb=192,
                tdp_watts=65, socket="AM5",
                release_year=2024, manufacturing_process="TSMC 4nm",
                supports_ecc=True, supports_xmp=True, supports_expo=True
            )
        ])
    
    def search_by_model(self, model: str) -> Optional[CPUSpec]:
        """Find CPU by exact model name."""
        for cpu in self.cpus:
            if model.lower() in cpu.model.lower():
                return cpu
        return None
    
    def search_by_vendor(self, vendor: CPUVendor) -> List[CPUSpec]:
        """Get all CPUs from specific vendor."""
        return [cpu for cpu in self.cpus if cpu.vendor == vendor]
    
    def search_by_architecture(self, architecture: CPUArchitecture) -> List[CPUSpec]:
        """Get all CPUs with specific architecture."""
        return [cpu for cpu in self.cpus if cpu.architecture == architecture]
    
    def find_compatible_cpus(self, target_speed: int, overclocked: bool = False) -> List[CPUSpec]:
        """Find CPUs compatible with target memory speed."""
        return [
            cpu for cpu in self.cpus 
            if cpu.is_speed_supported(target_speed, overclocked)
        ]
    
    def get_memory_speed_leaders(self, limit: int = 10) -> List[CPUSpec]:
        """Get CPUs with highest memory speed support."""
        return sorted(
            self.cpus, 
            key=lambda cpu: cpu.max_ddr5_speed_oc, 
            reverse=True
        )[:limit]
    
    def get_latest_cpus(self, year: int = 2024) -> List[CPUSpec]:
        """Get CPUs released in specified year or later."""
        return [cpu for cpu in self.cpus if cpu.release_year >= year]
    
    def generate_compatibility_report(self, cpu_model: str, target_configs: List[Dict]) -> Dict:
        """Generate compatibility report for CPU with memory configurations."""
        cpu = self.search_by_model(cpu_model)
        if not cpu:
            return {"error": f"CPU {cpu_model} not found"}
        
        report = {
            "cpu": cpu.model,
            "architecture": cpu.architecture.value,
            "jedec_speeds": cpu.supported_ddr5_speeds,
            "max_oc_speed": cpu.max_ddr5_speed_oc,
            "compatibility": []
        }
        
        for config in target_configs:
            speed = config.get("frequency", 0)
            is_jedec = speed in cpu.supported_ddr5_speeds
            is_oc_compatible = speed <= cpu.max_ddr5_speed_oc
            
            report["compatibility"].append({
                "speed": speed,
                "jedec_compliant": is_jedec,
                "oc_compatible": is_oc_compatible,
                "status": "JEDEC" if is_jedec else "OC" if is_oc_compatible else "INCOMPATIBLE"
            })
        
        return report


# Global database instance
_cpu_database = None

def get_cpu_database() -> CPUDatabase:
    """Get global CPU database instance."""
    global _cpu_database
    if _cpu_database is None:
        _cpu_database = CPUDatabase()
    return _cpu_database
