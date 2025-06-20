"""
Streamlit Web Interface for DDR5 AI Sandbox Simulator
Interactive dashboard for DDR5 memory optimization.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any

from ddr5_models import (DDR5Configuration, DDR5TimingParameters, 
                         DDR5VoltageParameters)
from ddr5_simulator import DDR5Simulator
from ai_optimizer import AdvancedAIOptimizer


def create_web_interface(simulator: DDR5Simulator, optimizer: AdvancedAIOptimizer):
    """Create and run the Streamlit web interface."""
    
    st.title("🧠 DDR5 AI Sandbox Simulator")
    st.markdown("**Fine-tune DDR5 memory without physical hardware**")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Memory specifications
        st.subheader("Memory Specifications")
        frequency = st.selectbox(
            "Frequency (MT/s)",
            [3200, 3600, 4000, 4400, 4800, 5200, 5600, 6000, 6400, 6800, 7200, 7600, 8000, 8400],
            index=6  # Default to DDR5-5600
        )
        
        capacity = st.selectbox("Capacity per stick (GB)", [8, 16, 32, 64], index=1)
        rank_count = st.selectbox("Rank count", [1, 2], index=0)
        
        # Timing parameters
        st.subheader("Primary Timings")
        
        # Calculate reasonable defaults based on frequency
        base_cl = max(16, int(frequency * 0.0055))
        
        cl = st.slider("CL (CAS Latency)", min_value=16, max_value=50, value=base_cl)
        trcd = st.slider("tRCD", min_value=16, max_value=50, value=base_cl)
        trp = st.slider("tRP", min_value=16, max_value=50, value=base_cl)
        tras = st.slider("tRAS", min_value=30, max_value=80, value=base_cl + 20)
        
        # Secondary timings
        st.subheader("Secondary Timings")
        trc = st.slider("tRC", min_value=50, max_value=120, value=tras + trp)
        trfc = st.slider("tRFC", min_value=200, max_value=400, value=295)
        
        # Voltage parameters
        st.subheader("Voltage Settings")
        vddq = st.slider("VDDQ (V)", min_value=1.05, max_value=1.20, value=1.10, step=0.01)
        vpp = st.slider("VPP (V)", min_value=1.70, max_value=1.90, value=1.80, step=0.01)
    
    # Create configuration
    timings = DDR5TimingParameters(
        cl=cl, trcd=trcd, trp=trp, tras=tras, trc=trc, trfc=trfc
    )
    voltages = DDR5VoltageParameters(vddq=vddq, vpp=vpp)
    config = DDR5Configuration(
        frequency=frequency,
        capacity=capacity,
        rank_count=rank_count,
        timings=timings,
        voltages=voltages
    )
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["Simulation", "AI Optimization", "Analysis", "Export"])
    
    with tab1:
        st.header("Memory Simulation")
        
        # Configuration validation
        violations = config.validate_configuration()
        total_violations = sum(len(v) for v in violations.values())
        
        if total_violations > 0:
            st.warning(f"⚠️ Configuration has {total_violations} validation issues")
            with st.expander("View Issues"):
                for category, issues in violations.items():
                    if issues:
                        st.write(f"**{category.replace('_', ' ').title()}:**")
                        for issue in issues:
                            st.write(f"- {issue}")
        else:
            st.success("✅ Configuration is valid")
        
        # Run simulation
        if st.button("Run Simulation", type="primary"):
            with st.spinner("Running simulation..."):
                simulator.load_configuration(config)
                
                # Performance metrics
                bandwidth_results = simulator.simulate_bandwidth()
                latency_results = simulator.simulate_latency()
                power_results = simulator.simulate_power_consumption()
                stability_results = simulator.run_stability_test()
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Bandwidth",
                        f"{bandwidth_results['effective_bandwidth_gbps']:.1f} GB/s",
                        f"{bandwidth_results['efficiency_percent']:.1f}% efficiency"
                    )
                
                with col2:
                    st.metric(
                        "Latency",
                        f"{latency_results['effective_latency_ns']:.1f} ns",
                        f"{latency_results['effective_latency_ns'] - config.latency_ns:.1f} ns overhead"
                    )
                
                with col3:
                    st.metric(
                        "Power",
                        f"{power_results['total_power_mw']:.0f} mW",
                        f"{power_results['power_efficiency_mb_per_mw']:.1f} MB/s/mW"
                    )
                
                with col4:
                    stability_color = "normal"
                    if stability_results['stability_score'] >= 90:
                        stability_color = "normal"
                    elif stability_results['stability_score'] >= 75:
                        stability_color = "normal"
                    else:
                        stability_color = "inverse"
                    
                    st.metric(
                        "Stability",
                        f"{stability_results['stability_score']:.1f}/100",
                        stability_results['test_result']
                    )
                
                # Detailed results
                st.subheader("Detailed Results")
                
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.write("**Bandwidth Analysis**")
                    st.json(bandwidth_results)
                    
                    st.write("**Power Analysis**")
                    st.json(power_results)
                
                with result_col2:
                    st.write("**Latency Analysis**")
                    st.json(latency_results)
                    
                    st.write("**Stability Analysis**")
                    st.json(stability_results)
    
    with tab2:
        st.header("AI-Powered Optimization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            optimization_goal = st.selectbox(
                "AI Optimization Goal",
                ["ai_balanced", "ai_performance", "ai_stability", "ai_extreme"],
                help="Choose AI optimization strategy"
            )
            
            target_freq = st.selectbox(
                "Target Frequency",
                [3200, 3600, 4000, 4400, 4800, 5200, 5600, 6000, 6400, 6800, 7200],
                index=6
            )
            
            # Advanced AI settings
            st.subheader("AI Targets")
            performance_target = st.slider("Performance Target %", 80, 100, 95)
            stability_target = st.slider("Stability Target %", 70, 100, 85)
        
        with col2:
            st.write("**AI Model Status**")
            if optimizer.is_trained:
                st.success("✅ AI ensemble models are trained and ready")
            else:
                st.warning("⚠️ AI ensemble models need training")
                if st.button("Train AI Models"):
                    with st.spinner("Training AI ensemble... This may take a few minutes."):
                        results = optimizer.train_ensemble_models()
                        st.success("Training complete!")
                        st.json(results)
        
        if st.button("Start AI Optimization", type="primary", disabled=not optimizer.is_trained):
            with st.spinner("Running intelligent AI optimization... This may take several minutes."):
                optimization_results = optimizer.intelligent_optimize(
                    target_frequency=target_freq,
                    optimization_goal=optimization_goal,
                    performance_target=performance_target,
                    stability_target=stability_target
                )
                
                st.success("Optimization complete!")
                
                # Display optimized configuration
                st.subheader("Optimized Configuration")
                optimized_config = optimization_results['optimized_config']
                
                config_col1, config_col2 = st.columns(2)
                
                with config_col1:
                    st.write("**Timings**")
                    st.write(f"CL: {optimized_config.timings.cl}")
                    st.write(f"tRCD: {optimized_config.timings.trcd}")
                    st.write(f"tRP: {optimized_config.timings.trp}")
                    st.write(f"tRAS: {optimized_config.timings.tras}")
                
                with config_col2:
                    st.write("**Voltages**")
                    st.write(f"VDDQ: {optimized_config.voltages.vddq:.3f}V")
                    st.write(f"VPP: {optimized_config.voltages.vpp:.3f}V")
                    st.write(f"Fitness Score: {optimization_results['fitness_score']:.3f}")
                
                # Performance comparison
                st.subheader("Performance Results")
                sim_results = optimization_results['simulation_results']
                
                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                
                with perf_col1:
                    st.metric(
                        "Bandwidth",
                        f"{sim_results['bandwidth']['effective_bandwidth_gbps']:.1f} GB/s"
                    )
                
                with perf_col2:
                    st.metric(
                        "Latency",
                        f"{sim_results['latency']['effective_latency_ns']:.1f} ns"
                    )
                
                with perf_col3:
                    st.metric(
                        "Power",
                        f"{sim_results['power']['total_power_mw']:.0f} mW"
                    )
                
                with perf_col4:
                    st.metric(
                        "Stability",
                        f"{sim_results['stability']['stability_score']:.1f}/100"
                    )
                
                # Fitness evolution chart
                st.subheader("Optimization Progress")
                fitness_df = pd.DataFrame({
                    'Generation': range(len(optimization_results['fitness_history'])),
                    'Fitness': optimization_results['fitness_history']
                })
                
                fig = px.line(fitness_df, x='Generation', y='Fitness', 
                             title='Fitness Evolution During Optimization')
                st.plotly_chart(fig, use_container_width=True)
                
                # AI Insights
                if 'ai_insights' in optimization_results:
                    st.subheader("🧠 AI Insights")
                    for insight in optimization_results['ai_insights']:
                        st.write(f"• {insight}")
                
                # Smart Recommendations
                if 'recommendations' in optimization_results:
                    st.subheader("💡 Smart Recommendations")
                    for rec in optimization_results['recommendations']:
                        st.write(f"• {rec}")
    
    with tab3:
        st.header("Configuration Analysis")
        
        # Performance radar chart
        if st.button("Generate Analysis Charts"):
            simulator.load_configuration(config)
            
            bandwidth_results = simulator.simulate_bandwidth()
            latency_results = simulator.simulate_latency()
            power_results = simulator.simulate_power_consumption()
            stability_results = simulator.run_stability_test()
            
            # Normalize metrics for radar chart
            metrics = {
                'Bandwidth': min(100, bandwidth_results['effective_bandwidth_gbps'] / 100 * 100),
                'Latency': max(0, 100 - latency_results['effective_latency_ns'] / 20),
                'Power Efficiency': min(100, power_results['power_efficiency_mb_per_mw'] / 30 * 100),
                'Stability': stability_results['stability_score'],
                'Timing Efficiency': bandwidth_results['efficiency_percent']
            }
            
            # Create radar chart
            categories = list(metrics.keys())
            values = list(metrics.values())
            values += values[:1]  # Complete the circle
            categories += categories[:1]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Configuration Performance'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Performance Analysis Radar Chart"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Timing relationships chart
            st.subheader("Timing Relationships")
            timing_data = {
                'Parameter': ['CL', 'tRCD', 'tRP', 'tRAS', 'tRC', 'tRFC'],
                'Value': [cl, trcd, trp, tras, trc, trfc],
                'Category': ['Primary', 'Primary', 'Primary', 'Primary', 'Secondary', 'Secondary']
            }
            
            timing_df = pd.DataFrame(timing_data)
            fig2 = px.bar(timing_df, x='Parameter', y='Value', color='Category',
                         title='DDR5 Timing Parameters')
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab4:
        st.header("Export Configuration")
        
        st.subheader("Current Configuration")
        config_dict = {
            'frequency': config.frequency,
            'capacity': config.capacity,
            'rank_count': config.rank_count,
            'timings': {
                'cl': config.timings.cl,
                'trcd': config.timings.trcd,
                'trp': config.timings.trp,
                'tras': config.timings.tras,
                'trc': config.timings.trc,
                'trfc': config.timings.trfc
            },
            'voltages': {
                'vddq': config.voltages.vddq,
                'vpp': config.voltages.vpp
            }
        }
        
        st.json(config_dict)
        
        # Export formats
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export as JSON"):
                st.download_button(
                    label="Download JSON",
                    data=str(config_dict),
                    file_name=f"ddr5_config_{frequency}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("Export as BIOS Settings"):
                bios_settings = generate_bios_format(config)
                st.text_area("BIOS Settings Format", bios_settings, height=300)


def generate_bios_format(config: DDR5Configuration) -> str:
    """Generate BIOS-compatible settings format."""
    return f"""
# DDR5 BIOS Settings
# Generated by DDR5 AI Sandbox Simulator

Memory Frequency: {config.frequency} MT/s
Memory Voltage (VDDQ): {config.voltages.vddq:.3f}V
Memory VPP Voltage: {config.voltages.vpp:.3f}V

# Primary Timings
CAS Latency (CL): {config.timings.cl}
RAS to CAS Delay (tRCD): {config.timings.trcd}
Row Precharge Time (tRP): {config.timings.trp}
Row Active Time (tRAS): {config.timings.tras}

# Secondary Timings
Row Cycle Time (tRC): {config.timings.trc}
Refresh Cycle Time (tRFC): {config.timings.trfc}
Refresh Interval (tREFI): {config.timings.trefi}
Write Recovery Time (tWR): {config.timings.twr}
Read to Precharge (tRTP): {config.timings.trtp}
CAS Write Latency (tCWL): {config.timings.tcwl}

# Sub-timings
Four Bank Activate Window (tFAW): {config.timings.tfaw}
Row Activate to Row Activate Same Bank Group (tRRD_S): {config.timings.trrd_s}
Row Activate to Row Activate Different Bank Group (tRRD_L): {config.timings.trrd_l}
Write to Read Same Bank Group (tWTR_S): {config.timings.twtr_s}
Write to Read Different Bank Group (tWTR_L): {config.timings.twtr_l}

# Note: Test thoroughly before applying to physical hardware
# This configuration was generated by AI simulation
    """.strip()
