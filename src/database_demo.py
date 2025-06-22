"""
Database Integration Demo
Demonstrates all databases in the DDR5 AI Sandbox Simulator.
"""

from cpu_database import get_cpu_database, CPUVendor
from motherboard_database import get_motherboard_database
from benchmark_database import get_benchmark_database, TestPlatform
from oc_profiles_database import get_oc_profile_database
from chipset_database import get_chipset_database, ChipsetVendor
from memory_kit_database import get_memory_kit_database, MemoryVendor
from vendor_database import get_vendor_database, VendorType


def demo_cpu_database():
    """Demonstrate CPU database functionality."""
    print("🖥️  CPU Database Demo")
    print("=" * 50)
    
    cpu_db = get_cpu_database()
    
    # Show latest CPUs
    latest_cpus = cpu_db.get_latest_cpus(2024)
    print(f"📅 Latest CPUs (2024+): {len(latest_cpus)} found")
    for cpu in latest_cpus[:3]:
        print(f"  • {cpu.model} - Max OC: {cpu.max_ddr5_speed_oc} MT/s")
    
    # Find CPUs for high-speed memory
    high_speed_cpus = cpu_db.find_compatible_cpus(7200, overclocked=True)
    print(f"\n🚀 CPUs supporting DDR5-7200 OC: {len(high_speed_cpus)}")
    for cpu in high_speed_cpus[:3]:
        print(f"  • {cpu.model} - {cpu.vendor.value}")
    
    print()


def demo_motherboard_database():
    """Demonstrate motherboard database functionality."""
    print("🏠 Motherboard Database Demo")
    print("=" * 50)
    
    mb_db = get_motherboard_database()
    
    # Show overclocking champions
    oc_champions = mb_db.get_overclocking_champions()
    print("🏆 Top Overclocking Motherboards:")
    for mb in oc_champions[:3]:
        print(f"  • {mb.model} - Max OC: {mb.max_oc_memory_speed} MT/s")
        print(f"    Signal Integrity: {mb.signal_integrity_rating}/10")
    
    # Show compatibility
    z790_boards = mb_db.search_by_chipset("Z790")
    print(f"\n🔧 Z790 Motherboards: {len(z790_boards)} found")
    
    print()


def demo_benchmark_database():
    """Demonstrate benchmark database functionality."""
    print("📊 Benchmark Database Demo")
    print("=" * 50)
    
    bench_db = get_benchmark_database()
    
    # Show gaming benchmarks
    intel_gaming = bench_db.search_by_platform(TestPlatform.INTEL_13TH_GEN)
    gaming_benches = [b for b in intel_gaming if b.fps_avg]
    
    print(f"🎮 Gaming Benchmarks (Intel 13th Gen): {len(gaming_benches)}")
    for bench in gaming_benches[:3]:
        config = bench.memory_config
        print(f"  • {bench.test_name}")
        config_str = f"DDR5-{config['frequency']} CL{config['cl']}"
        print(f"    {config_str}: {bench.fps_avg:.1f} FPS")
    
    # Performance scaling
    scaling_report = bench_db.generate_scaling_report(
        baseline_config={"frequency": 4800, "cl": 40},
        target_configs=[
            {"frequency": 6000, "cl": 36},
            {"frequency": 7200, "cl": 34}
        ],
        platform=TestPlatform.INTEL_13TH_GEN
    )
    
    if "scaling" in scaling_report:
        print("\n📈 Performance Scaling vs DDR5-4800:")
        for data in scaling_report["scaling"]:
            config = data["config"]
            improvement = data["gaming_improvement_percent"]
            freq = config['frequency']
            print(f"  • DDR5-{freq}: +{improvement}% gaming performance")
    
    print()


def demo_oc_profiles_database():
    """Demonstrate OC profiles database functionality."""
    print("⚡ Overclocking Profiles Database Demo")
    print("=" * 50)
    
    oc_db = get_oc_profile_database()
    
    # Show beginner profiles
    beginner_profiles = oc_db.get_beginner_friendly()
    print(f"👶 Beginner-Friendly Profiles: {len(beginner_profiles)}")
    for profile in beginner_profiles:
        print(f"  • {profile.name}")
        print(f"    DDR5-{profile.frequency} - {profile.stability.value}")
        print(f"    Gaming uplift: +{profile.estimated_gaming_uplift}%")
    
    # Get recommendation
    recommendation = oc_db.generate_profile_recommendation({
        "use_case": "Gaming",
        "difficulty": "Intermediate",
        "platform": "Intel Z790",
        "target_frequency": 6400
    })
    
    if "recommended_profile" in recommendation:
        profile_name = recommendation['recommended_profile']
        print(f"\n🎯 Recommended Profile: {profile_name}")
        print(f"   Frequency: DDR5-{recommendation['frequency']}")
        print(f"   Difficulty: {recommendation['difficulty']}")
        improvement = recommendation['estimated_improvement']
        print(f"   Estimated improvement: +{improvement}%")
    
    print()


def demo_chipset_database():
    """Demonstrate chipset database functionality."""
    print("🔧 Chipset Database Demo")
    print("=" * 50)
    
    chipset_db = get_chipset_database()
    
    # Show Intel chipsets
    intel_chips = chipset_db.find_by_vendor(ChipsetVendor.INTEL)
    print(f"Intel Chipsets: {len(intel_chips)}")
    for chip in intel_chips[:3]:
        print(f"  • {chip.name} - Max OC: {chip.max_ddr5_speed_oc} MT/s")
    
    # Show overclocking capable
    oc_chips = chipset_db.get_overclocking_capable()
    print(f"\n🚀 OC-Capable Chipsets: {len(oc_chips)}")
    for chip in oc_chips[:3]:
        print(f"  • {chip.name} ({chip.vendor.value})")
    
    print()


def demo_memory_kit_database():
    """Demonstrate memory kit database functionality."""
    print("💾 Memory Kit Database Demo")
    print("=" * 50)
    
    kit_db = get_memory_kit_database()
    
    # Show gaming recommendations
    gaming_kits = kit_db.get_recommendations("gaming")
    print("🎮 Gaming Kit Recommendations:")
    for kit in gaming_kits[:3]:
        print(f"  • {kit.model} - {kit.speed_mt_s} MT/s")
        print(f"    Price/Performance: {kit.price_performance_ratio}/10")
    
    # Show overclocking kits
    oc_kits = kit_db.get_overclocking_kits()
    print(f"\n🚀 High-OC Kits: {len(oc_kits)}")
    for kit in sorted(oc_kits, key=lambda x: x.overclocking_potential,
                      reverse=True)[:3]:
        print(f"  • {kit.model} - Potential: {kit.overclocking_potential}/10")
    
    print()


def demo_vendor_database():
    """Demonstrate vendor database functionality."""
    print("🏢 Vendor Database Demo")
    print("=" * 50)
    
    vendor_db = get_vendor_database()
    
    # Show memory manufacturers
    memory_vendors = vendor_db.find_by_type(VendorType.MEMORY_MANUFACTURER)
    print(f"Memory Manufacturers: {len(memory_vendors)}")
    for vendor in memory_vendors:
        print(f"  • {vendor.name} - "
              f"Market Share: {vendor.market_share_percentage}%")
    
    # Show DDR5 experts
    ddr5_experts = vendor_db.get_vendors_with_ddr5_expertise()
    print(f"\n🔥 DDR5 Experts: {len(ddr5_experts)}")
    for vendor in ddr5_experts[:3]:
        print(f"  • {vendor.name} - Level: {vendor.ddr5_support_level}/10")
    
    print()


def main():
    """Run all database demos."""
    print("🗄️  DDR5 AI Sandbox - Database Integration Demo")
    print("=" * 60)
    print()
    
    # Run all demos
    demo_cpu_database()
    demo_motherboard_database()
    demo_benchmark_database()
    demo_oc_profiles_database()
    demo_chipset_database()
    demo_memory_kit_database()
    demo_vendor_database()
    
    print("✅ Database integration demo completed!")
    print("All databases are working correctly and ready for integration.")


if __name__ == "__main__":
    main()
