"""
Simulation tab for basic DDR5 configuration and metrics.
"""

import streamlit as st
from typing import Dict, Any

from ..components.charts import create_radar_chart
from src.ddr5_models import DDR5Configuration


def render_simulation_tab(config: DDR5Configuration) -> None:
    """Render the simulation tab."""
    st.header("⚡ DDR5 Performance Simulation")
    
    # Load configuration into simulator
    st.session_state.simulator.load_configuration(config)
    
    # Configuration Summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Current Configuration")
        st.write(f"**Frequency:** DDR5-{config.frequency}")
        st.write(f"**Capacity:** {config.capacity}GB per stick")
        st.write(f"**Rank Count:** {config.rank_count}")
        
        st.write("**Timings:**")
        st.write(f"• CL: {config.timings.cl}")
        st.write(f"• tRCD: {config.timings.trcd}")
        st.write(f"• tRP: {config.timings.trp}")
        st.write(f"• tRAS: {config.timings.tras}")
        st.write(f"• tRC: {config.timings.trc}")
        st.write(f"• tRFC: {config.timings.trfc}")
        
        st.write("**Voltages:**")
        st.write(f"• VDDQ: {config.voltages.vddq}V")
        st.write(f"• VPP: {config.voltages.vpp}V")
    
    with col2:
        st.subheader("📊 Performance Metrics")
        
        # Calculate and display metrics
        config.calculate_performance_metrics()
        
        # Bandwidth simulation
        bandwidth_results = st.session_state.simulator.simulate_bandwidth()
        
        # Latency simulation  
        latency_results = st.session_state.simulator.simulate_latency()
        
        # Power simulation
        power_results = st.session_state.simulator.simulate_power_consumption()
        
        # Display metrics
        st.metric("Theoretical Bandwidth", 
                 f"{config.bandwidth_gbps:.1f} GB/s")
        st.metric("Effective Bandwidth", 
                 f"{bandwidth_results['effective_bandwidth_gbps']:.1f} GB/s")
        st.metric("First Word Latency", 
                 f"{config.latency_ns:.1f} ns")
        st.metric("Effective Latency", 
                 f"{latency_results['effective_latency_ns']:.1f} ns")
        st.metric("Power Consumption", 
                 f"{power_results['total_power_mw']:.0f} mW")
        st.metric("Power Efficiency", 
                 f"{power_results['power_efficiency_mb_per_mw']:.1f} MB/s/mW")
    
    st.divider()
    
    # Performance Radar Chart
    st.subheader("🎯 Performance Profile")
    
    col_chart, col_info = st.columns([2, 1])
    
    with col_chart:
        fig = create_radar_chart(config)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_info:
        st.info("⚙️ Using Sidebar Manual Configuration")
        
        # JEDEC compliance check
        jedec_violations = config.validate_jedec_compliance()
        if any(jedec_violations.values()):
            st.error("❌ Not JEDEC compliant:")
            for category, issues in jedec_violations.items():
                for issue in issues:
                    st.write(f"• {issue}")
        else:
            st.info("🛡️ JEDEC compliant configuration.")
    
    st.divider()
    
    # Advanced Simulation Options
    st.subheader("🔬 Advanced Simulation")
    
    col_sim1, col_sim2, col_sim3 = st.columns(3)
    
    with col_sim1:
        st.write("**Access Pattern Analysis**")
        access_pattern = st.selectbox("Pattern", ["sequential", "random", "mixed"])
        
        if st.button("🔄 Simulate Access Pattern"):
            bandwidth_sim = st.session_state.simulator.simulate_bandwidth(
                access_pattern=access_pattern
            )
            latency_sim = st.session_state.simulator.simulate_latency(
                access_pattern=access_pattern
            )
            
            st.success(f"✅ Pattern: {access_pattern}")
            st.write(f"Bandwidth: {bandwidth_sim['effective_bandwidth_gbps']:.1f} GB/s")
            st.write(f"Latency: {latency_sim['effective_latency_ns']:.1f} ns")
    
    with col_sim2:
        st.write("**Stability Testing**")
        test_duration = st.selectbox("Duration", [15, 30, 60, 120])
        stress_level = st.selectbox("Stress", ["light", "medium", "heavy"])
        
        if st.button("🧪 Run Stability Test"):
            with st.spinner(f"Running {test_duration}min stress test..."):
                stability_results = st.session_state.simulator.run_stability_test(
                    test_duration_minutes=test_duration,
                    stress_level=stress_level
                )
            
            result_color = "success" if stability_results['test_result'] == "EXCELLENT" else "warning"
            st.success(f"Result: {stability_results['test_result']}")
            st.write(f"Stability: {stability_results['final_stability']:.1f}%")
            st.write(f"Error Rate: {stability_results['error_rate']:.3f}%")
    
    with col_sim3:
        st.write("**Thermal Analysis**")
        ambient_temp = st.slider("Ambient Temp (°C)", 20, 40, 25)
        
        if st.button("🌡️ Thermal Simulation"):
            # Simple thermal simulation
            power_dissipation = power_results['total_power_mw'] / 1000  # Convert to W
            thermal_resistance = 5.0  # °C/W (typical for DDR5)
            estimated_temp = ambient_temp + (power_dissipation * thermal_resistance)
            
            st.write(f"Estimated DIMM Temp: {estimated_temp:.1f}°C")
            
            if estimated_temp > 85:
                st.error("⚠️ Temperature too high!")
            elif estimated_temp > 70:
                st.warning("🔥 Monitor temperature closely")
            else:
                st.success("✅ Temperature within safe range")
