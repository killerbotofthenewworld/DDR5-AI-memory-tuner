"""
Analysis tab functionality.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.ddr5_models import (
    DDR5Configuration, DDR5TimingParameters, DDR5VoltageParameters
)


def create_preset_config(frequency: int, capacity: int, rank_count: int):
    """Create a preset configuration for the given frequency."""
    # Simplified preset logic
    if frequency >= 6400:
        cl, trcd, trp, tras, trc, trfc = 32, 39, 39, 32, 76, 295
        vddq, vpp = 1.35, 1.8
    elif frequency >= 5600:
        cl, trcd, trp, tras, trc, trfc = 36, 39, 39, 76, 76, 295
        vddq, vpp = 1.25, 1.8
    else:
        cl, trcd, trp, tras, trc, trfc = 40, 39, 39, 76, 115, 350
        vddq, vpp = 1.1, 1.8
    
    return DDR5Configuration(
        frequency=frequency,
        timings=DDR5TimingParameters(
            cl=cl, trcd=trcd, trp=trp, tras=tras, trc=trc, trfc=trfc
        ),
        voltages=DDR5VoltageParameters(vddq=vddq, vpp=vpp)
    )


def render_analysis_tab(enable_manual=False, frequency=5600, capacity=16,
                        rank_count=1, cl=36, trcd=39, trp=39, tras=76,
                        trc=76, trfc=295, vddq=1.25, vpp=1.8):
    """Render the Analysis tab."""
    st.header("📊 Advanced Analysis")
    
    # Performance comparison
    st.subheader("⚖️ Configuration Comparison")
    
    if st.button("Generate Comparison"):
        # Generate multiple configurations for comparison
        frequencies = [4800, 5600, 6400, 7200]
        comparison_data = []
        
        for freq in frequencies:
            config = create_preset_config(freq, 16, 1)
            st.session_state.simulator.load_configuration(config)
            
            bandwidth = st.session_state.simulator.simulate_bandwidth()
            latency = st.session_state.simulator.simulate_latency()
            power = st.session_state.simulator.simulate_power_consumption()
            stability = st.session_state.simulator.run_stability_test()
            
            comparison_data.append({
                'Frequency': f"DDR5-{freq}",
                'Bandwidth (GB/s)': bandwidth['effective_bandwidth_gbps'],
                'Latency (ns)': latency['effective_latency_ns'],
                'Power (mW)': power['total_power_mw'],
                'Stability (%)': stability['stability_score']
            })
        
        df = pd.DataFrame(comparison_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bandwidth vs Frequency
            fig = px.bar(df, x='Frequency', y='Bandwidth (GB/s)',
                         title='Bandwidth vs Frequency')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Power vs Performance
            fig = px.scatter(df, x='Bandwidth (GB/s)', y='Power (mW)',
                             color='Frequency', title='Power vs Performance')
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df, use_container_width=True)
    
    # Feature importance (if AI is trained)
    if st.session_state.ai_trained:
        st.subheader("🎯 Feature Importance")
        
        feature_importance = st.session_state.optimizer.feature_importance
        if feature_importance:
            importance_df = pd.DataFrame(
                list(feature_importance.items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature',
                         orientation='h',
                         title='Parameter Importance for Performance')
            st.plotly_chart(fig, use_container_width=True)
    
    # Current configuration analysis
    st.subheader("🔍 Current Configuration Analysis")
    
    # Get current configuration
    if (hasattr(st.session_state, 'manual_config') and
            st.session_state.manual_config):
        current_config = st.session_state.manual_config
    elif enable_manual:
        current_config = DDR5Configuration(
            frequency=frequency,
            timings=DDR5TimingParameters(
                cl=cl, trcd=trcd, trp=trp, tras=tras, trc=trc, trfc=trfc
            ),
            voltages=DDR5VoltageParameters(vddq=vddq, vpp=vpp)
        )
    else:
        current_config = create_preset_config(frequency, capacity, rank_count)
    
    # Load and analyze current config
    st.session_state.simulator.load_configuration(current_config)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bandwidth_result = st.session_state.simulator.simulate_bandwidth()
        st.metric("Bandwidth", 
                  f"{bandwidth_result['effective_bandwidth_gbps']:.1f} GB/s",
                  help="Theoretical maximum bandwidth")
    
    with col2:
        latency_result = st.session_state.simulator.simulate_latency()
        st.metric("Latency", 
                  f"{latency_result['effective_latency_ns']:.1f} ns",
                  help="Memory access latency")
    
    with col3:
        power_result = st.session_state.simulator.simulate_power_consumption()
        st.metric("Power", f"{power_result['total_power_mw']:.0f} mW",
                  help="Estimated power consumption")
    
    # Timing relationships analysis
    st.subheader("📏 Timing Relationships")
    
    timing_checks = []
    timings = current_config.timings
    
    # Check critical timing relationships
    if timings.tras >= timings.trcd + timings.cl:
        timing_checks.append("✅ tRAS >= tRCD + CL (Valid)")
    else:
        timing_checks.append("❌ tRAS < tRCD + CL (Invalid - may cause "
                            "instability)")
    
    if timings.trc >= timings.tras + timings.trp:
        timing_checks.append("✅ tRC >= tRAS + tRP (Valid)")
    else:
        timing_checks.append("❌ tRC < tRAS + tRP (Invalid - may cause "
                            "corruption)")
    
    if timings.trfc >= timings.trc * 4:
        timing_checks.append("✅ tRFC timing appears reasonable")
    else:
        timing_checks.append("⚠️ tRFC may be too tight for stability")
    
    for check in timing_checks:
        if "✅" in check:
            st.success(check)
        elif "❌" in check:
            st.error(check)
        else:
            st.warning(check)
    
    # Voltage analysis
    st.subheader("⚡ Voltage Analysis")
    
    voltage_checks = []
    voltages = current_config.voltages
    
    if voltages.vddq <= 1.1:
        voltage_checks.append("🔋 VDDQ: Conservative (Excellent for 24/7)")
    elif voltages.vddq <= 1.25:
        voltage_checks.append("⚖️ VDDQ: Moderate (Good daily driver)")
    elif voltages.vddq <= 1.35:
        voltage_checks.append("⚡ VDDQ: Aggressive (Performance oriented)")
    else:
        voltage_checks.append("🔥 VDDQ: Extreme (Requires excellent cooling)")
    
    if voltages.vpp <= 1.8:
        voltage_checks.append("✅ VPP: Within JEDEC spec")
    else:
        voltage_checks.append("⚠️ VPP: Above JEDEC spec")
    
    for check in voltage_checks:
        st.info(check)

    # Performance vs competitors
    st.subheader("🏁 Performance Comparison")
    
    competitors = [
        {"name": "JEDEC Baseline", "freq": 4800, "cl": 40, "score": 100},
        {"name": "Gaming Sweet Spot", "freq": 5600, "cl": 36, "score": 115},
        {"name": "Enthusiast Choice", "freq": 6000, "cl": 32, "score": 125},
        {"name": "Extreme Overclock", "freq": 7200, "cl": 34, "score": 140}
    ]
    
    # Calculate current config score
    current_score = ((current_config.frequency / 4800) * 100 * 
                     (40 / current_config.timings.cl))
    
    competitor_df = pd.DataFrame(competitors)
    competitor_df.loc[len(competitor_df)] = {
        "name": "Your Config",
        "freq": current_config.frequency,
        "cl": current_config.timings.cl,
        "score": int(current_score)
    }
    
    fig = px.bar(competitor_df, x='name', y='score',
                title='Performance Score Comparison',
                color='score', color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)
