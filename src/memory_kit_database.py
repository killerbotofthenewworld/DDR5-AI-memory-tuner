"""
Memory Kit Database
Comprehensive database of DDR5 memory kits with specifications and ratings.
"""

from dataclasses import dataclass
from typing import List
from enum import Enum
from datetime import date


class MemoryVendor(Enum):
    """Memory vendors."""
    CORSAIR = "Corsair"
    GSKILL = "G.Skill"
    CRUCIAL = "Crucial"
    KINGSTON = "Kingston"
    TEAMGROUP = "Team Group"
    ADATA = "ADATA"
    PATRIOT = "Patriot"
    MUSHKIN = "Mushkin"
    THERMALTAKE = "Thermaltake"
    OLOY = "OLOY"
    SAMSUNG = "Samsung"
    MICRON = "Micron"
    HYNIX = "SK Hynix"


class MemoryType(Enum):
    """Memory types."""
    DDR5 = "DDR5"
    DDR5_GAMING = "DDR5 Gaming"
    DDR5_OVERCLOCKING = "DDR5 Overclocking"
    DDR5_RGB = "DDR5 RGB"
    DDR5_WORKSTATION = "DDR5 Workstation"
    DDR5_SERVER = "DDR5 Server"


class ICType(Enum):
    """Memory IC types."""
    SAMSUNG_BDIE = "Samsung B-die"
    HYNIX_ADIE = "Hynix A-die"
    HYNIX_MDIE = "Hynix M-die"
    MICRON_BDIE = "Micron B-die"
    SAMSUNG_CDIE = "Samsung C-die"
    UNKNOWN = "Unknown"


@dataclass
class MemoryKit:
    """Memory kit specification model."""
    model: str
    vendor: MemoryVendor
    memory_type: MemoryType
    capacity_gb: int
    stick_count: int
    speed_mt_s: int
    jedec_speed: int
    cl: int
    trcd: int
    trp: int
    tras: int
    trc: int
    trfc: int
    voltage: float
    ic_type: ICType
    heat_spreader: bool
    rgb_lighting: bool
    xmp_support: bool
    expo_support: bool
    overclocking_potential: int  # 1-10 scale
    stability_rating: int  # 1-10 scale
    price_performance_ratio: int  # 1-10 scale
    release_date: date
    warranty_years: int
    special_features: List[str]
    compatibility_notes: str


class MemoryKitDatabase:
    """Database for memory kit specifications and ratings."""

    def __init__(self):
        self.memory_kits = self._initialize_database()

    def _initialize_database(self) -> List[MemoryKit]:
        """Initialize the memory kit database with comprehensive entries."""
        return [
            # High-End Gaming/Overclocking Kits
            MemoryKit(
                model="Dominator Platinum RGB DDR5-6000",
                vendor=MemoryVendor.CORSAIR,
                memory_type=MemoryType.DDR5_RGB,
                capacity_gb=32,
                stick_count=2,
                speed_mt_s=6000,
                jedec_speed=4800,
                cl=30,
                trcd=36,
                trp=36,
                tras=76,
                trc=112,
                trfc=560,
                voltage=1.35,
                ic_type=ICType.SAMSUNG_BDIE,
                heat_spreader=True,
                rgb_lighting=True,
                xmp_support=True,
                expo_support=True,
                overclocking_potential=9,
                stability_rating=9,
                price_performance_ratio=7,
                release_date=date(2024, 3, 15),
                warranty_years=5,
                special_features=[
                    "Capellix RGB LEDs", "Aluminum Heat Spreader",
                    "Hand-sorted ICs", "iCUE Software"
                ],
                compatibility_notes=(
                    "Premium gaming kit with excellent overclocking headroom"
                )
            ),
            MemoryKit(
                model="Trident Z5 RGB DDR5-6400",
                vendor=MemoryVendor.GSKILL,
                memory_type=MemoryType.DDR5_RGB,
                capacity_gb=32,
                stick_count=2,
                speed_mt_s=6400,
                jedec_speed=4800,
                cl=32,
                trcd=39,
                trp=39,
                tras=102,
                trc=141,
                trfc=560,
                voltage=1.40,
                ic_type=ICType.HYNIX_ADIE,
                heat_spreader=True,
                rgb_lighting=True,
                xmp_support=True,
                expo_support=True,
                overclocking_potential=10,
                stability_rating=8,
                price_performance_ratio=8,
                release_date=date(2024, 6, 10),
                warranty_years=5,
                special_features=[
                    "Trident Z5 Design", "Addressable RGB",
                    "Premium ICs", "Overclocking Ready"
                ],
                compatibility_notes=(
                    "Flagship overclocking kit with exceptional potential"
                )
            ),
            MemoryKit(
                model="Fury Beast DDR5-5600",
                vendor=MemoryVendor.KINGSTON,
                memory_type=MemoryType.DDR5_GAMING,
                capacity_gb=32,
                stick_count=2,
                speed_mt_s=5600,
                jedec_speed=4800,
                cl=36,
                trcd=38,
                trp=38,
                tras=80,
                trc=118,
                trfc=560,
                voltage=1.25,
                ic_type=ICType.MICRON_BDIE,
                heat_spreader=True,
                rgb_lighting=False,
                xmp_support=True,
                expo_support=True,
                overclocking_potential=7,
                stability_rating=9,
                price_performance_ratio=9,
                release_date=date(2023, 11, 20),
                warranty_years=3,
                special_features=[
                    "Low Profile Design", "Plug N Play",
                    "Tested Speed", "Reliable Performance"
                ],
                compatibility_notes=(
                    "Excellent value gaming kit with broad compatibility"
                )
            ),
            MemoryKit(
                model="Ballistix MAX DDR5-6000",
                vendor=MemoryVendor.CRUCIAL,
                memory_type=MemoryType.DDR5_OVERCLOCKING,
                capacity_gb=32,
                stick_count=2,
                speed_mt_s=6000,
                jedec_speed=4800,
                cl=40,
                trcd=40,
                trp=40,
                tras=77,
                trc=117,
                trfc=560,
                voltage=1.35,
                ic_type=ICType.MICRON_BDIE,
                heat_spreader=True,
                rgb_lighting=False,
                xmp_support=True,
                expo_support=True,
                overclocking_potential=8,
                stability_rating=10,
                price_performance_ratio=8,
                release_date=date(2024, 1, 8),
                warranty_years=5,
                special_features=[
                    "Micron ICs", "Overclocking Tested",
                    "Temperature Sensor", "Stability Focused"
                ],
                compatibility_notes=(
                    "Micron-based kit with excellent stability"
                )
            ),
            MemoryKit(
                model="T-Force Delta RGB DDR5-6000",
                vendor=MemoryVendor.TEAMGROUP,
                memory_type=MemoryType.DDR5_RGB,
                capacity_gb=32,
                stick_count=2,
                speed_mt_s=6000,
                jedec_speed=4800,
                cl=38,
                trcd=38,
                trp=38,
                tras=78,
                trc=116,
                trfc=560,
                voltage=1.35,
                ic_type=ICType.HYNIX_ADIE,
                heat_spreader=True,
                rgb_lighting=True,
                xmp_support=True,
                expo_support=True,
                overclocking_potential=8,
                stability_rating=8,
                price_performance_ratio=9,
                release_date=date(2024, 4, 12),
                warranty_years=3,
                special_features=[
                    "120° Ultra-Wide Lighting", "Unique Design",
                    "T-Force Blitz Software", "Gaming Optimized"
                ],
                compatibility_notes=(
                    "Great RGB gaming kit with competitive pricing"
                )
            ),

            # Workstation/Professional Kits
            MemoryKit(
                model="Registered ECC DDR5-4800",
                vendor=MemoryVendor.SAMSUNG,
                memory_type=MemoryType.DDR5_SERVER,
                capacity_gb=128,
                stick_count=4,
                speed_mt_s=4800,
                jedec_speed=4800,
                cl=40,
                trcd=40,
                trp=40,
                tras=77,
                trc=117,
                trfc=560,
                voltage=1.10,
                ic_type=ICType.SAMSUNG_BDIE,
                heat_spreader=False,
                rgb_lighting=False,
                xmp_support=False,
                expo_support=False,
                overclocking_potential=3,
                stability_rating=10,
                price_performance_ratio=6,
                release_date=date(2023, 8, 15),
                warranty_years=3,
                special_features=[
                    "ECC Support", "Registered DIMM",
                    "Server Grade", "Enterprise Reliability"
                ],
                compatibility_notes=(
                    "Server/workstation kit with ECC support"
                )
            ),

            # Mainstream Value Kits
            MemoryKit(
                model="Value Select DDR5-4800",
                vendor=MemoryVendor.CORSAIR,
                memory_type=MemoryType.DDR5,
                capacity_gb=16,
                stick_count=2,
                speed_mt_s=4800,
                jedec_speed=4800,
                cl=40,
                trcd=40,
                trp=40,
                tras=77,
                trc=117,
                trfc=560,
                voltage=1.10,
                ic_type=ICType.MICRON_BDIE,
                heat_spreader=False,
                rgb_lighting=False,
                xmp_support=False,
                expo_support=False,
                overclocking_potential=4,
                stability_rating=8,
                price_performance_ratio=10,
                release_date=date(2023, 6, 1),
                warranty_years=3,
                special_features=[
                    "JEDEC Standard", "Plug and Play",
                    "Basic Reliability", "Budget Friendly"
                ],
                compatibility_notes=(
                    "Basic DDR5 kit for budget builds"
                )
            ),

            # High-Speed Extreme Kits
            MemoryKit(
                model="Trident Z5 Royal DDR5-7200",
                vendor=MemoryVendor.GSKILL,
                memory_type=MemoryType.DDR5_OVERCLOCKING,
                capacity_gb=32,
                stick_count=2,
                speed_mt_s=7200,
                jedec_speed=4800,
                cl=34,
                trcd=44,
                trp=44,
                tras=115,
                trc=159,
                trfc=560,
                voltage=1.45,
                ic_type=ICType.SAMSUNG_BDIE,
                heat_spreader=True,
                rgb_lighting=True,
                xmp_support=True,
                expo_support=True,
                overclocking_potential=10,
                stability_rating=7,
                price_performance_ratio=6,
                release_date=date(2024, 9, 20),
                warranty_years=5,
                special_features=[
                    "Royal Series Design", "Crystalline Light Bar",
                    "Hand-Binned ICs", "Extreme Overclocking"
                ],
                compatibility_notes=(
                    "Extreme overclocking kit for enthusiasts"
                )
            ),
            MemoryKit(
                model="Vengeance RGB DDR5-5200",
                vendor=MemoryVendor.CORSAIR,
                memory_type=MemoryType.DDR5_RGB,
                capacity_gb=32,
                stick_count=2,
                speed_mt_s=5200,
                jedec_speed=4800,
                cl=40,
                trcd=40,
                trp=40,
                tras=77,
                trc=117,
                trfc=560,
                voltage=1.25,
                ic_type=ICType.HYNIX_MDIE,
                heat_spreader=True,
                rgb_lighting=True,
                xmp_support=True,
                expo_support=True,
                overclocking_potential=6,
                stability_rating=9,
                price_performance_ratio=8,
                release_date=date(2023, 9, 5),
                warranty_years=5,
                special_features=[
                    "10-zone RGB", "iCUE Compatible",
                    "Anodized Aluminum", "Gaming Optimized"
                ],
                compatibility_notes=(
                    "Popular RGB gaming kit with solid performance"
                )
            ),
        ]

    def get_all_kits(self) -> List[MemoryKit]:
        """Get all memory kits in the database."""
        return self.memory_kits.copy()

    def find_by_vendor(self, vendor: MemoryVendor) -> List[MemoryKit]:
        """Find memory kits by vendor."""
        return [kit for kit in self.memory_kits if kit.vendor == vendor]

    def find_by_capacity(self, capacity_gb: int) -> List[MemoryKit]:
        """Find memory kits by capacity."""
        return [kit for kit in self.memory_kits
                if kit.capacity_gb == capacity_gb]

    def find_by_speed_range(self, min_speed: int,
                            max_speed: int) -> List[MemoryKit]:
        """Find memory kits within speed range."""
        return [kit for kit in self.memory_kits
                if min_speed <= kit.speed_mt_s <= max_speed]

    def find_by_budget(self, max_price_performance: int) -> List[MemoryKit]:
        """Find memory kits within budget (by price-performance ratio)."""
        return [kit for kit in self.memory_kits
                if kit.price_performance_ratio >= max_price_performance]

    def get_overclocking_kits(self, min_potential: int = 7) -> List[MemoryKit]:
        """Get memory kits with high overclocking potential."""
        return [kit for kit in self.memory_kits
                if kit.overclocking_potential >= min_potential]

    def get_stable_kits(self, min_stability: int = 8) -> List[MemoryKit]:
        """Get memory kits with high stability rating."""
        return [kit for kit in self.memory_kits
                if kit.stability_rating >= min_stability]

    def get_rgb_kits(self) -> List[MemoryKit]:
        """Get memory kits with RGB lighting."""
        return [kit for kit in self.memory_kits if kit.rgb_lighting]

    def get_gaming_kits(self) -> List[MemoryKit]:
        """Get memory kits suitable for gaming."""
        return [kit for kit in self.memory_kits
                if kit.memory_type in [MemoryType.DDR5_GAMING,
                                       MemoryType.DDR5_RGB]]

    def get_professional_kits(self) -> List[MemoryKit]:
        """Get memory kits for professional/workstation use."""
        return [kit for kit in self.memory_kits
                if kit.memory_type in [MemoryType.DDR5_WORKSTATION,
                                       MemoryType.DDR5_SERVER]]

    def find_by_ic_type(self, ic_type: ICType) -> List[MemoryKit]:
        """Find memory kits by IC type."""
        return [kit for kit in self.memory_kits if kit.ic_type == ic_type]

    def get_latest_kits(self, year: int) -> List[MemoryKit]:
        """Get memory kits released in or after the specified year."""
        return [kit for kit in self.memory_kits
                if kit.release_date.year >= year]

    def search_by_model(self, model: str) -> List[MemoryKit]:
        """Search memory kits by model name (case-insensitive)."""
        model_lower = model.lower()
        return [kit for kit in self.memory_kits
                if model_lower in kit.model.lower()]

    def get_recommendations(self, use_case: str) -> List[MemoryKit]:
        """Get memory kit recommendations for specific use cases."""
        use_case_lower = use_case.lower()

        if "gaming" in use_case_lower:
            return sorted(self.get_gaming_kits(),
                          key=lambda x: x.price_performance_ratio,
                          reverse=True)[:5]
        elif "overclock" in use_case_lower:
            return sorted(self.get_overclocking_kits(),
                          key=lambda x: x.overclocking_potential,
                          reverse=True)[:5]
        elif "budget" in use_case_lower:
            return sorted(self.memory_kits,
                          key=lambda x: x.price_performance_ratio,
                          reverse=True)[:5]
        elif ("workstation" in use_case_lower or
              "professional" in use_case_lower):
            return sorted(self.get_professional_kits(),
                          key=lambda x: x.stability_rating,
                          reverse=True)[:5]
        else:
            return sorted(self.memory_kits,
                          key=lambda x: (x.price_performance_ratio +
                                         x.stability_rating),
                          reverse=True)[:5]


def get_memory_kit_database() -> MemoryKitDatabase:
    """Get the global memory kit database instance."""
    return MemoryKitDatabase()


if __name__ == "__main__":
    # Demo usage
    db = get_memory_kit_database()

    print("💾 Memory Kit Database Demo")
    print("=" * 40)

    # Show gaming recommendations
    gaming_kits = db.get_recommendations("gaming")
    print("🎮 Gaming Recommendations:")
    for kit in gaming_kits[:3]:
        print(f"  • {kit.model} - {kit.speed_mt_s} MT/s")
        print(f"    Price/Perf: {kit.price_performance_ratio}/10")

    # Show overclocking kits
    oc_kits = db.get_overclocking_kits()
    print(f"\n🚀 Overclocking Kits: {len(oc_kits)}")
    for kit in sorted(oc_kits, key=lambda x: x.overclocking_potential,
                      reverse=True)[:3]:
        print(f"  • {kit.model} - "
              f"OC Potential: {kit.overclocking_potential}/10")

    # Show by vendor
    corsair_kits = db.find_by_vendor(MemoryVendor.CORSAIR)
    print(f"\n🏴 Corsair Kits: {len(corsair_kits)}")
    for kit in corsair_kits:
        print(f"  • {kit.model} - {kit.capacity_gb}GB")
