"""
Vendor Database
Comprehensive database of memory and hardware vendors with their specialties.
"""

from dataclasses import dataclass
from typing import List
from enum import Enum


class VendorType(Enum):
    """Vendor types."""
    MEMORY_MANUFACTURER = "Memory Manufacturer"
    MOTHERBOARD_MANUFACTURER = "Motherboard Manufacturer"
    CPU_MANUFACTURER = "CPU Manufacturer"
    CHIPSET_MANUFACTURER = "Chipset Manufacturer"
    SYSTEM_INTEGRATOR = "System Integrator"
    IC_MANUFACTURER = "IC Manufacturer"


class MarketSegment(Enum):
    """Market segments."""
    GAMING = "Gaming"
    OVERCLOCKING = "Overclocking"
    WORKSTATION = "Workstation"
    SERVER = "Server"
    BUDGET = "Budget"
    MAINSTREAM = "Mainstream"
    PREMIUM = "Premium"
    ENTERPRISE = "Enterprise"


@dataclass
class VendorProfile:
    """Vendor profile model."""
    name: str
    vendor_type: VendorType
    founded_year: int
    headquarters: str
    market_segments: List[MarketSegment]
    specialties: List[str]
    ddr5_support_level: int  # 1-10 scale
    innovation_score: int  # 1-10 scale
    reliability_rating: int  # 1-10 scale
    customer_support_rating: int  # 1-10 scale
    market_share_percentage: float
    notable_products: List[str]
    key_technologies: List[str]
    partnerships: List[str]
    warranty_standard_years: int
    certification_standards: List[str]
    sustainability_rating: int  # 1-10 scale
    price_positioning: str  # "Budget", "Mainstream", "Premium"
    community_reputation: int  # 1-10 scale
    website: str
    description: str


class VendorDatabase:
    """Database for vendor profiles and information."""

    def __init__(self):
        self.vendors = self._initialize_database()

    def _initialize_database(self) -> List[VendorProfile]:
        """Initialize the vendor database with comprehensive entries."""
        return [
            # Memory Manufacturers
            VendorProfile(
                name="Corsair",
                vendor_type=VendorType.MEMORY_MANUFACTURER,
                founded_year=1994,
                headquarters="Fremont, California, USA",
                market_segments=[
                    MarketSegment.GAMING, MarketSegment.OVERCLOCKING,
                    MarketSegment.PREMIUM
                ],
                specialties=[
                    "Gaming Memory", "RGB Lighting", "High-Performance Kits",
                    "Cooling Solutions", "PC Components"
                ],
                ddr5_support_level=9,
                innovation_score=8,
                reliability_rating=9,
                customer_support_rating=8,
                market_share_percentage=15.2,
                notable_products=[
                    "Dominator Platinum RGB", "Vengeance RGB Pro",
                    "Value Select", "Corsair iCUE Software"
                ],
                key_technologies=[
                    "Capellix RGB LEDs", "iCUE Ecosystem",
                    "Custom Heat Spreaders", "Hand-Sorted ICs"
                ],
                partnerships=[
                    "Intel XMP", "AMD EXPO", "Major Motherboard Vendors"
                ],
                warranty_standard_years=5,
                certification_standards=[
                    "JEDEC", "Intel XMP 3.0", "AMD EXPO"
                ],
                sustainability_rating=7,
                price_positioning="Premium",
                community_reputation=9,
                website="https://www.corsair.com",
                description=(
                    "Leading gaming peripheral and component manufacturer "
                    "known for premium RGB memory and cooling solutions."
                )
            ),
            VendorProfile(
                name="G.Skill",
                vendor_type=VendorType.MEMORY_MANUFACTURER,
                founded_year=1989,
                headquarters="Taipei, Taiwan",
                market_segments=[
                    MarketSegment.OVERCLOCKING, MarketSegment.GAMING,
                    MarketSegment.PREMIUM
                ],
                specialties=[
                    "Extreme Overclocking", "High-Speed Memory",
                    "Premium Design", "Memory Innovation"
                ],
                ddr5_support_level=10,
                innovation_score=10,
                reliability_rating=8,
                customer_support_rating=7,
                market_share_percentage=12.8,
                notable_products=[
                    "Trident Z5 RGB", "Trident Z5 Royal", "Ripjaws V",
                    "Flare X5 Series"
                ],
                key_technologies=[
                    "Hand-Binned ICs", "Custom PCB Design",
                    "Advanced Heat Spreaders", "Extreme Speed Bins"
                ],
                partnerships=[
                    "Intel XMP", "AMD EXPO", "Overclocking Community",
                    "Motherboard Vendors"
                ],
                warranty_standard_years=5,
                certification_standards=[
                    "JEDEC", "Intel XMP 3.0", "AMD EXPO"
                ],
                sustainability_rating=6,
                price_positioning="Premium",
                community_reputation=10,
                website="https://www.gskill.com",
                description=(
                    "Taiwan-based memory specialist renowned for "
                    "extreme overclocking performance and innovation."
                )
            ),
            VendorProfile(
                name="Crucial",
                vendor_type=VendorType.MEMORY_MANUFACTURER,
                founded_year=1996,
                headquarters="Boise, Idaho, USA",
                market_segments=[
                    MarketSegment.MAINSTREAM, MarketSegment.WORKSTATION,
                    MarketSegment.BUDGET
                ],
                specialties=[
                    "Micron Memory", "Reliability", "Value Products",
                    "Enterprise Solutions"
                ],
                ddr5_support_level=8,
                innovation_score=7,
                reliability_rating=10,
                customer_support_rating=9,
                market_share_percentage=18.5,
                notable_products=[
                    "Ballistix Gaming", "Ballistix MAX", "Pro Series",
                    "ECC Server Memory"
                ],
                key_technologies=[
                    "Micron 3D NAND", "Temperature Sensors",
                    "Stability Testing", "Enterprise Validation"
                ],
                partnerships=[
                    "Micron Technology", "Intel", "AMD", "Server OEMs"
                ],
                warranty_standard_years=3,
                certification_standards=[
                    "JEDEC", "Intel XMP", "AMD EXPO", "Server Standards"
                ],
                sustainability_rating=8,
                price_positioning="Mainstream",
                community_reputation=8,
                website="https://www.crucial.com",
                description=(
                    "Micron's consumer brand offering reliable memory "
                    "solutions from budget to high-performance."
                )
            ),
            VendorProfile(
                name="Kingston",
                vendor_type=VendorType.MEMORY_MANUFACTURER,
                founded_year=1987,
                headquarters="Fountain Valley, California, USA",
                market_segments=[
                    MarketSegment.MAINSTREAM, MarketSegment.GAMING,
                    MarketSegment.BUDGET, MarketSegment.ENTERPRISE
                ],
                specialties=[
                    "Value Memory", "Gaming Solutions", "Enterprise Memory",
                    "Global Distribution"
                ],
                ddr5_support_level=8,
                innovation_score=7,
                reliability_rating=9,
                customer_support_rating=8,
                market_share_percentage=22.1,
                notable_products=[
                    "Fury Beast", "Fury Renegade", "Value RAM",
                    "Server Premier"
                ],
                key_technologies=[
                    "Plug N Play", "Automatic Overclocking",
                    "Broad Compatibility", "Quality Testing"
                ],
                partnerships=[
                    "Intel", "AMD", "Global OEMs", "System Integrators"
                ],
                warranty_standard_years=3,
                certification_standards=[
                    "JEDEC", "Intel XMP", "AMD EXPO"
                ],
                sustainability_rating=7,
                price_positioning="Mainstream",
                community_reputation=8,
                website="https://www.kingston.com",
                description=(
                    "World's largest independent memory manufacturer "
                    "with focus on reliability and value."
                )
            ),

            # IC Manufacturers
            VendorProfile(
                name="Samsung Semiconductor",
                vendor_type=VendorType.IC_MANUFACTURER,
                founded_year=1969,
                headquarters="Seoul, South Korea",
                market_segments=[
                    MarketSegment.PREMIUM, MarketSegment.SERVER,
                    MarketSegment.ENTERPRISE, MarketSegment.MAINSTREAM
                ],
                specialties=[
                    "Memory ICs", "NAND Flash", "Advanced Process Nodes",
                    "High-Density Memory"
                ],
                ddr5_support_level=10,
                innovation_score=10,
                reliability_rating=9,
                customer_support_rating=7,
                market_share_percentage=42.7,
                notable_products=[
                    "B-die ICs", "ECC Server Memory", "High-Speed GDDR",
                    "Mobile LPDDR"
                ],
                key_technologies=[
                    "14nm Process", "3D V-NAND", "Advanced Packaging",
                    "ECC Technology"
                ],
                partnerships=[
                    "Major Memory Vendors", "CPU Manufacturers",
                    "Server OEMs", "Mobile Device Makers"
                ],
                warranty_standard_years=3,
                certification_standards=[
                    "JEDEC", "Server Standards", "Automotive Standards"
                ],
                sustainability_rating=8,
                price_positioning="Premium",
                community_reputation=9,
                website="https://semiconductor.samsung.com",
                description=(
                    "Leading memory IC manufacturer with cutting-edge "
                    "technology and high-performance solutions."
                )
            ),
            VendorProfile(
                name="SK Hynix",
                vendor_type=VendorType.IC_MANUFACTURER,
                founded_year=1983,
                headquarters="Icheon, South Korea",
                market_segments=[
                    MarketSegment.PREMIUM, MarketSegment.MAINSTREAM,
                    MarketSegment.SERVER
                ],
                specialties=[
                    "Memory ICs", "High-Speed Memory", "Mobile Memory",
                    "Graphics Memory"
                ],
                ddr5_support_level=9,
                innovation_score=9,
                reliability_rating=8,
                customer_support_rating=7,
                market_share_percentage=26.8,
                notable_products=[
                    "A-die ICs", "M-die ICs", "HBM Memory", "GDDR6X"
                ],
                key_technologies=[
                    "1Znm Process", "Advanced Packaging",
                    "High-Bandwidth Memory", "Power Efficiency"
                ],
                partnerships=[
                    "Memory Module Vendors", "Graphics Card Makers",
                    "CPU Manufacturers", "Mobile OEMs"
                ],
                warranty_standard_years=3,
                certification_standards=[
                    "JEDEC", "Graphics Standards", "Mobile Standards"
                ],
                sustainability_rating=7,
                price_positioning="Premium",
                community_reputation=8,
                website="https://www.skhynix.com",
                description=(
                    "Major memory IC manufacturer known for "
                    "high-speed and graphics memory solutions."
                )
            ),

            # CPU Manufacturers
            VendorProfile(
                name="Intel Corporation",
                vendor_type=VendorType.CPU_MANUFACTURER,
                founded_year=1968,
                headquarters="Santa Clara, California, USA",
                market_segments=[
                    MarketSegment.MAINSTREAM, MarketSegment.PREMIUM,
                    MarketSegment.SERVER, MarketSegment.ENTERPRISE
                ],
                specialties=[
                    "x86 Processors", "Chipsets", "Platform Technologies",
                    "AI Acceleration"
                ],
                ddr5_support_level=9,
                innovation_score=9,
                reliability_rating=9,
                customer_support_rating=8,
                market_share_percentage=68.4,
                notable_products=[
                    "Core i9-14900K", "Core i7-14700K", "Xeon Processors",
                    "Arc Graphics"
                ],
                key_technologies=[
                    "Intel 7 Process", "Hybrid Architecture",
                    "XMP 3.0", "Thunderbolt", "WiFi 7"
                ],
                partnerships=[
                    "Memory Vendors", "Motherboard Partners",
                    "OEMs", "Cloud Providers"
                ],
                warranty_standard_years=3,
                certification_standards=[
                    "JEDEC", "PCIe", "USB", "Industry Standards"
                ],
                sustainability_rating=8,
                price_positioning="Premium",
                community_reputation=8,
                website="https://www.intel.com",
                description=(
                    "Leading x86 processor manufacturer with "
                    "comprehensive platform solutions."
                )
            ),
            VendorProfile(
                name="AMD",
                vendor_type=VendorType.CPU_MANUFACTURER,
                founded_year=1969,
                headquarters="Santa Clara, California, USA",
                market_segments=[
                    MarketSegment.GAMING, MarketSegment.WORKSTATION,
                    MarketSegment.SERVER, MarketSegment.MAINSTREAM
                ],
                specialties=[
                    "x86 Processors", "Graphics Cards", "Chipsets",
                    "High-Performance Computing"
                ],
                ddr5_support_level=9,
                innovation_score=9,
                reliability_rating=8,
                customer_support_rating=8,
                market_share_percentage=31.6,
                notable_products=[
                    "Ryzen 9 7950X", "Ryzen 7 7800X3D", "EPYC Processors",
                    "Radeon Graphics"
                ],
                key_technologies=[
                    "Zen 4 Architecture", "3D V-Cache", "EXPO",
                    "Infinity Cache", "RDNA Graphics"
                ],
                partnerships=[
                    "Memory Vendors", "Motherboard Partners",
                    "OEMs", "Game Developers"
                ],
                warranty_standard_years=3,
                certification_standards=[
                    "JEDEC", "PCIe", "Graphics Standards"
                ],
                sustainability_rating=8,
                price_positioning="Mainstream",
                community_reputation=9,
                website="https://www.amd.com",
                description=(
                    "Innovative processor and graphics manufacturer "
                    "with strong gaming and workstation focus."
                )
            ),
        ]

    def get_all_vendors(self) -> List[VendorProfile]:
        """Get all vendors in the database."""
        return self.vendors.copy()

    def find_by_type(self, vendor_type: VendorType) -> List[VendorProfile]:
        """Find vendors by type."""
        return [vendor for vendor in self.vendors
                if vendor.vendor_type == vendor_type]

    def find_by_market_segment(self,
                               segment: MarketSegment) -> List[VendorProfile]:
        """Find vendors by market segment."""
        return [vendor for vendor in self.vendors
                if segment in vendor.market_segments]

    def get_top_vendors_by_market_share(
            self, count: int = 5) -> List[VendorProfile]:
        """Get top vendors by market share."""
        return sorted(self.vendors,
                      key=lambda x: x.market_share_percentage,
                      reverse=True)[:count]

    def get_most_innovative(self, count: int = 5) -> List[VendorProfile]:
        """Get most innovative vendors."""
        return sorted(self.vendors,
                      key=lambda x: x.innovation_score,
                      reverse=True)[:count]

    def get_most_reliable(self, count: int = 5) -> List[VendorProfile]:
        """Get most reliable vendors."""
        return sorted(self.vendors,
                      key=lambda x: x.reliability_rating,
                      reverse=True)[:count]

    def get_best_customer_support(self, count: int = 5) -> List[VendorProfile]:
        """Get vendors with best customer support."""
        return sorted(self.vendors,
                      key=lambda x: x.customer_support_rating,
                      reverse=True)[:count]

    def search_by_name(self, name: str) -> List[VendorProfile]:
        """Search vendors by name (case-insensitive)."""
        name_lower = name.lower()
        return [vendor for vendor in self.vendors
                if name_lower in vendor.name.lower()]

    def get_vendors_with_ddr5_expertise(
            self, min_level: int = 8) -> List[VendorProfile]:
        """Get vendors with high DDR5 support level."""
        return [vendor for vendor in self.vendors
                if vendor.ddr5_support_level >= min_level]

    def get_sustainability_leaders(self,
                                   min_rating: int = 7) -> List[VendorProfile]:
        """Get vendors with high sustainability ratings."""
        return [vendor for vendor in self.vendors
                if vendor.sustainability_rating >= min_rating]


def get_vendor_database() -> VendorDatabase:
    """Get the global vendor database instance."""
    return VendorDatabase()


if __name__ == "__main__":
    # Demo usage
    db = get_vendor_database()

    print("🏢 Vendor Database Demo")
    print("=" * 40)

    # Show memory manufacturers
    memory_vendors = db.find_by_type(VendorType.MEMORY_MANUFACTURER)
    print(f"💾 Memory Manufacturers: {len(memory_vendors)}")
    for vendor in memory_vendors:
        print(f"  • {vendor.name} - "
              f"Market Share: {vendor.market_share_percentage}%")

    # Show top innovative vendors
    innovative = db.get_most_innovative(3)
    print("\n🚀 Most Innovative:")
    for vendor in innovative:
        print(f"  • {vendor.name} - "
              f"Innovation Score: {vendor.innovation_score}/10")

    # Show DDR5 experts
    ddr5_experts = db.get_vendors_with_ddr5_expertise()
    print(f"\n🔥 DDR5 Experts: {len(ddr5_experts)}")
    for vendor in ddr5_experts:
        print(f"  • {vendor.name} - "
              f"DDR5 Level: {vendor.ddr5_support_level}/10")
