"""
Chipset Database
Comprehensive database of DDR5-compatible chipsets with detailed specs.
"""

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum
from datetime import date


class ChipsetVendor(Enum):
    """Chipset vendors."""
    INTEL = "Intel"
    AMD = "AMD"
    QUALCOMM = "Qualcomm"
    MEDIATEK = "MediaTek"


class SocketType(Enum):
    """CPU socket types."""
    LGA1700 = "LGA1700"
    LGA1851 = "LGA1851"
    AM5 = "AM5"
    AM4 = "AM4"
    TR5 = "TR5"
    FCBGA = "FCBGA"


@dataclass
class ChipsetSpec:
    """Chipset specification model."""
    name: str
    vendor: ChipsetVendor
    socket: SocketType
    release_date: date
    max_ddr5_speed_jedec: int  # MT/s
    max_ddr5_speed_oc: int  # MT/s
    memory_channels: int
    max_memory_capacity: int  # GB
    pcie_lanes: int
    pcie_version: str
    usb_ports: Dict[str, int]  # USB version -> count
    sata_ports: int
    m2_slots: int
    features: List[str]
    overclocking_support: bool
    power_efficiency_rating: int  # 1-10 scale
    io_capabilities_score: int  # 1-10 scale
    compatibility_notes: str


class ChipsetDatabase:
    """Database for chipset specifications and compatibility."""

    def __init__(self):
        self.chipsets = self._initialize_database()

    def _initialize_database(self) -> List[ChipsetSpec]:
        """Initialize the chipset database with comprehensive entries."""
        return [
            # Intel LGA1700 Chipsets
            ChipsetSpec(
                name="Z790",
                vendor=ChipsetVendor.INTEL,
                socket=SocketType.LGA1700,
                release_date=date(2022, 10, 20),
                max_ddr5_speed_jedec=5600,
                max_ddr5_speed_oc=8400,
                memory_channels=2,
                max_memory_capacity=128,
                pcie_lanes=20,
                pcie_version="5.0",
                usb_ports={"3.2": 10, "2.0": 4},
                sata_ports=8,
                m2_slots=4,
                features=[
                    "CPU Overclocking", "Memory Overclocking",
                    "XMP 3.0", "Thunderbolt 4"
                ],
                overclocking_support=True,
                power_efficiency_rating=8,
                io_capabilities_score=9,
                compatibility_notes=(
                    "Premium overclocking chipset with excellent DDR5 support"
                )
            ),
            ChipsetSpec(
                name="Z690",
                vendor=ChipsetVendor.INTEL,
                socket=SocketType.LGA1700,
                release_date=date(2021, 11, 4),
                max_ddr5_speed_jedec=4800,
                max_ddr5_speed_oc=7800,
                memory_channels=2,
                max_memory_capacity=128,
                pcie_lanes=20,
                pcie_version="5.0",
                usb_ports={"3.2": 10, "2.0": 4},
                sata_ports=8,
                m2_slots=4,
                features=[
                    "CPU Overclocking", "Memory Overclocking", "XMP 3.0"
                ],
                overclocking_support=True,
                power_efficiency_rating=7,
                io_capabilities_score=8,
                compatibility_notes=(
                    "First-gen DDR5 support, proven overclocking performance"
                )
            ),
            ChipsetSpec(
                name="B760",
                vendor=ChipsetVendor.INTEL,
                socket=SocketType.LGA1700,
                release_date=date(2023, 1, 3),
                max_ddr5_speed_jedec=5600,
                max_ddr5_speed_oc=7200,
                memory_channels=2,
                max_memory_capacity=128,
                pcie_lanes=20,
                pcie_version="4.0",
                usb_ports={"3.2": 8, "2.0": 4},
                sata_ports=6,
                m2_slots=3,
                features=["Memory Overclocking", "XMP 3.0"],
                overclocking_support=False,
                power_efficiency_rating=8,
                io_capabilities_score=7,
                compatibility_notes=(
                    "Mainstream chipset with good DDR5 support, "
                    "no CPU overclocking"
                )
            ),
            ChipsetSpec(
                name="H770",
                vendor=ChipsetVendor.INTEL,
                socket=SocketType.LGA1700,
                release_date=date(2022, 10, 20),
                max_ddr5_speed_jedec=4800,
                max_ddr5_speed_oc=6400,
                memory_channels=2,
                max_memory_capacity=128,
                pcie_lanes=20,
                pcie_version="4.0",
                usb_ports={"3.2": 8, "2.0": 4},
                sata_ports=8,
                m2_slots=3,
                features=["Memory Overclocking", "XMP 3.0", "vPro"],
                overclocking_support=False,
                power_efficiency_rating=9,
                io_capabilities_score=8,
                compatibility_notes="Business-focused with solid DDR5 support"
            ),

            # Intel LGA1851 Chipsets (Arrow Lake)
            ChipsetSpec(
                name="Z890",
                vendor=ChipsetVendor.INTEL,
                socket=SocketType.LGA1851,
                release_date=date(2024, 10, 24),
                max_ddr5_speed_jedec=6400,
                max_ddr5_speed_oc=9600,
                memory_channels=2,
                max_memory_capacity=192,
                pcie_lanes=24,
                pcie_version="5.0",
                usb_ports={"4.0": 2, "3.2": 12, "2.0": 4},
                sata_ports=8,
                m2_slots=5,
                features=[
                    "CPU Overclocking", "Memory Overclocking",
                    "XMP 3.0", "Thunderbolt 5", "WiFi 7"
                ],
                overclocking_support=True,
                power_efficiency_rating=9,
                io_capabilities_score=10,
                compatibility_notes=(
                    "Latest generation with cutting-edge DDR5 support "
                    "and new features"
                )
            ),
            ChipsetSpec(
                name="B860",
                vendor=ChipsetVendor.INTEL,
                socket=SocketType.LGA1851,
                release_date=date(2025, 1, 15),
                max_ddr5_speed_jedec=6400,
                max_ddr5_speed_oc=8000,
                memory_channels=2,
                max_memory_capacity=128,
                pcie_lanes=20,
                pcie_version="4.0",
                usb_ports={"3.2": 10, "2.0": 4},
                sata_ports=6,
                m2_slots=4,
                features=["Memory Overclocking", "XMP 3.0", "WiFi 7"],
                overclocking_support=False,
                power_efficiency_rating=9,
                io_capabilities_score=8,
                compatibility_notes=(
                    "Next-gen mainstream chipset with excellent DDR5 support"
                )
            ),

            # AMD AM5 Chipsets
            ChipsetSpec(
                name="X670E",
                vendor=ChipsetVendor.AMD,
                socket=SocketType.AM5,
                release_date=date(2022, 9, 27),
                max_ddr5_speed_jedec=5200,
                max_ddr5_speed_oc=8000,
                memory_channels=2,
                max_memory_capacity=128,
                pcie_lanes=44,
                pcie_version="5.0",
                usb_ports={"3.2": 12, "2.0": 4},
                sata_ports=8,
                m2_slots=4,
                features=[
                    "CPU Overclocking", "Memory Overclocking",
                    "EXPO", "PCIe 5.0"
                ],
                overclocking_support=True,
                power_efficiency_rating=8,
                io_capabilities_score=10,
                compatibility_notes=(
                    "Flagship AMD chipset with excellent DDR5 EXPO support"
                )
            ),
            ChipsetSpec(
                name="X670",
                vendor=ChipsetVendor.AMD,
                socket=SocketType.AM5,
                release_date=date(2022, 9, 27),
                max_ddr5_speed_jedec=5200,
                max_ddr5_speed_oc=7600,
                memory_channels=2,
                max_memory_capacity=128,
                pcie_lanes=44,
                pcie_version="4.0",
                usb_ports={"3.2": 12, "2.0": 4},
                sata_ports=8,
                m2_slots=4,
                features=["CPU Overclocking", "Memory Overclocking", "EXPO"],
                overclocking_support=True,
                power_efficiency_rating=8,
                io_capabilities_score=9,
                compatibility_notes=(
                    "High-end AMD chipset with strong DDR5 support"
                )
            ),
            ChipsetSpec(
                name="B650E",
                vendor=ChipsetVendor.AMD,
                socket=SocketType.AM5,
                release_date=date(2022, 10, 10),
                max_ddr5_speed_jedec=5200,
                max_ddr5_speed_oc=7200,
                memory_channels=2,
                max_memory_capacity=128,
                pcie_lanes=28,
                pcie_version="5.0",
                usb_ports={"3.2": 8, "2.0": 4},
                sata_ports=6,
                m2_slots=3,
                features=["Memory Overclocking", "EXPO", "PCIe 5.0"],
                overclocking_support=False,
                power_efficiency_rating=9,
                io_capabilities_score=8,
                compatibility_notes=(
                    "Mainstream AMD chipset with good DDR5 EXPO support"
                )
            ),
            ChipsetSpec(
                name="B650",
                vendor=ChipsetVendor.AMD,
                socket=SocketType.AM5,
                release_date=date(2022, 10, 10),
                max_ddr5_speed_jedec=5200,
                max_ddr5_speed_oc=6800,
                memory_channels=2,
                max_memory_capacity=128,
                pcie_lanes=28,
                pcie_version="4.0",
                usb_ports={"3.2": 8, "2.0": 4},
                sata_ports=6,
                m2_slots=3,
                features=["Memory Overclocking", "EXPO"],
                overclocking_support=False,
                power_efficiency_rating=9,
                io_capabilities_score=7,
                compatibility_notes="Entry-level AM5 with solid DDR5 support"
            ),

            # AMD TRX50 (Threadripper)
            ChipsetSpec(
                name="TRX50",
                vendor=ChipsetVendor.AMD,
                socket=SocketType.TR5,
                release_date=date(2023, 10, 19),
                max_ddr5_speed_jedec=5200,
                max_ddr5_speed_oc=6000,
                memory_channels=8,
                max_memory_capacity=1024,
                pcie_lanes=88,
                pcie_version="5.0",
                usb_ports={"3.2": 20, "2.0": 8},
                sata_ports=12,
                m2_slots=8,
                features=[
                    "CPU Overclocking", "Memory Overclocking",
                    "EXPO", "ECC Support"
                ],
                overclocking_support=True,
                power_efficiency_rating=7,
                io_capabilities_score=10,
                compatibility_notes=(
                    "HEDT platform with massive I/O and quad-channel DDR5"
                )
            ),
        ]

    def get_all_chipsets(self) -> List[ChipsetSpec]:
        """Get all chipsets in the database."""
        return self.chipsets.copy()

    def find_by_vendor(self, vendor: ChipsetVendor) -> List[ChipsetSpec]:
        """Find chipsets by vendor."""
        return [chip for chip in self.chipsets if chip.vendor == vendor]

    def find_by_socket(self, socket: SocketType) -> List[ChipsetSpec]:
        """Find chipsets by socket type."""
        return [chip for chip in self.chipsets if chip.socket == socket]

    def find_by_memory_speed(self, min_speed: int,
                             overclocked: bool = False) -> List[ChipsetSpec]:
        """Find chipsets supporting minimum memory speed."""
        if overclocked:
            return [chip for chip in self.chipsets
                    if chip.max_ddr5_speed_oc >= min_speed]
        else:
            return [chip for chip in self.chipsets
                    if chip.max_ddr5_speed_jedec >= min_speed]

    def get_overclocking_capable(self) -> List[ChipsetSpec]:
        """Get chipsets that support CPU overclocking."""
        return [chip for chip in self.chipsets if chip.overclocking_support]

    def get_latest_chipsets(self, year: int) -> List[ChipsetSpec]:
        """Get chipsets released in or after the specified year."""
        return [chip for chip in self.chipsets
                if chip.release_date.year >= year]

    def get_high_performance(self) -> List[ChipsetSpec]:
        """Get high-performance chipsets (top tier)."""
        return [chip for chip in self.chipsets
                if (chip.overclocking_support and
                    chip.max_ddr5_speed_oc >= 7000)]

    def search_by_name(self, name: str) -> List[ChipsetSpec]:
        """Search chipsets by name (case-insensitive)."""
        name_lower = name.lower()
        return [chip for chip in self.chipsets
                if name_lower in chip.name.lower()]

    def get_compatibility_matrix(self) -> Dict[str, List[str]]:
        """Get chipset compatibility matrix by socket."""
        matrix: Dict[str, List[str]] = {}
        for chip in self.chipsets:
            socket_name = chip.socket.value
            if socket_name not in matrix:
                matrix[socket_name] = []
            matrix[socket_name].append(chip.name)
        return matrix


def get_chipset_database() -> ChipsetDatabase:
    """Get the global chipset database instance."""
    return ChipsetDatabase()


if __name__ == "__main__":
    # Demo usage
    db = get_chipset_database()

    print("🔧 Chipset Database Demo")
    print("=" * 40)

    # Show Intel chipsets
    intel_chips = db.find_by_vendor(ChipsetVendor.INTEL)
    print(f"Intel Chipsets: {len(intel_chips)}")
    for chip in intel_chips[:3]:
        print(f"  • {chip.name} - Max OC: {chip.max_ddr5_speed_oc} MT/s")

    # Show AMD chipsets
    amd_chips = db.find_by_vendor(ChipsetVendor.AMD)
    print(f"\nAMD Chipsets: {len(amd_chips)}")
    for chip in amd_chips[:3]:
        print(f"  • {chip.name} - Max OC: {chip.max_ddr5_speed_oc} MT/s")

    # Show overclocking capable
    oc_chips = db.get_overclocking_capable()
    print(f"\nOverclocking Capable: {len(oc_chips)}")
    for chip in oc_chips[:3]:
        print(f"  • {chip.name} ({chip.vendor.value})")
