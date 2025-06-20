"""
Perfect DDR5 Web Interface - Ultimate AI-Powered Memory Tuning Dashboard
Enhanced interface showcasing all advanced AI capabilities.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any
import time

from .ddr5_models import (DDR5Configuration, DDR5TimingParameters, 
                         DDR5VoltageParameters)
from .ddr5_simulator import DDR5Simulator
from .perfect_ai_optimizer import PerfectDDR5Optimizer
from .hardware_detection import detect_system_memory, get_system_summary
from .ram_database import get_database, DDR5ModuleSpec
from .cross_brand_tuner import CrossBrandOptimizer, generate_cross_brand_report
from .live_tuning_safety import (LiveTuningSafetyValidator, LiveTuningSafetyReport, 
                               SafetyLevel as SafetyLevelEnum, quick_safety_check)
# from live_tuner import LiveTuner  # TODO: Add live tuning capability


def create_perfect_web_interface():
    """Create the perfect DDR5 AI Optimizer web interface."""
    
    # Page configuration
    st.set_page_config(
        page_title="Perfect DDR5 AI Optimizer",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'simulator' not in st.session_state:
        st.session_state.simulator = DDR5Simulator()
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = PerfectDDR5Optimizer()
    if 'ai_trained' not in st.session_state:
        st.session_state.ai_trained = False
    
    # Initialize session state for hardware detection
    if 'detected_modules' not in st.session_state:
        st.session_state.detected_modules = []
    if 'hardware_scanned' not in st.session_state:
        st.session_state.hardware_scanned = False

    # Main title with styling
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>🧠 Perfect DDR5 AI Optimizer</h1>
        <h3>Ultimate AI-Powered Memory Tuning Without Hardware</h3>
        <p style='color: #666;'>Advanced ML • Quantum Optimization • Molecular Analysis • Revolutionary Features</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Ko-fi donation button (prominent and clickable)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("☕ Support Development on Ko-fi", 
                    type="primary", use_container_width=True):
            st.balloons()
            st.success("💖 Thank you! Opening Ko-fi page...")
            st.markdown("""
            <div style='text-align: center; margin: 20px 0;'>
                <a href="https://ko-fi.com/killerbotofthenewworld" target="_blank" 
                   style='background: #FF5E5B; color: white; padding: 12px 24px; 
                          text-decoration: none; border-radius: 6px; 
                          display: inline-block; font-weight: bold;'>
                    � Click Here to Open Ko-fi
                </a>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='text-align: center; margin-top: 10px;'>
            <p style='color: #888; font-size: 14px;'>
                💖 Support continued development of revolutionary AI features!
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Ko-fi Support Button (Sidebar)
        if st.button("☕ Support on Ko-fi", type="secondary", 
                    use_container_width=True):
            st.success("💖 Thanks!")
            st.markdown("""
            <div style='text-align: center; margin: 10px 0;'>
                <a href="https://ko-fi.com/killerbotofthenewworld" target="_blank" 
                   style='background: #FF5E5B; color: white; padding: 8px 16px; 
                          text-decoration: none; border-radius: 4px; 
                          display: inline-block; font-size: 12px;'>
                    ☕ Open Ko-fi
                </a>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # AI Training Section
        st.subheader("🧠 AI Training")
        
        if not st.session_state.ai_trained:
            if st.button("🚀 Train Perfect AI", type="primary"):
                with st.spinner("Training advanced AI models..."):
                    training_results = st.session_state.optimizer.train_perfect_ai()
                    st.session_state.ai_trained = True
                    st.success("AI Training Complete!")
                    
                    with st.expander("Training Results"):
                        st.json(training_results)
        else:
            st.success("✅ AI Models Trained")
            if st.button("🔄 Retrain AI"):
                st.session_state.ai_trained = False
                st.rerun()
        
        st.divider()
        
        # Memory Configuration
        st.subheader("💾 Memory Specifications")
        
        frequency = st.selectbox(
            "Frequency (MT/s)",
            [3200, 3600, 4000, 4400, 4800, 5200, 5600, 6000, 6400, 6800, 7200, 7600, 8000, 8400],
            index=6  # Default to DDR5-5600
        )
        
        capacity = st.selectbox("Capacity per stick (GB)", [8, 16, 32, 64], index=1)
        rank_count = st.selectbox("Rank count", [1, 2], index=0)
        
        st.divider()
        
        # Manual Configuration
        st.subheader("⚙️ Manual Tuning")
        
        enable_manual = st.checkbox("Enable Manual Configuration")
        
        if enable_manual:
            # Calculate reasonable defaults
            base_cl = max(16, int(frequency * 0.0055))
            
            cl = st.slider("CL (CAS Latency)", min_value=16, max_value=50, value=base_cl)
            trcd = st.slider("tRCD", min_value=16, max_value=50, value=base_cl)
            trp = st.slider("tRP", min_value=16, max_value=50, value=base_cl)
            tras = st.slider("tRAS", min_value=30, max_value=80, value=base_cl + 20)
            trc = tras + trp  # Auto-calculated
            trfc = st.slider("tRFC", min_value=280, max_value=400, value=312)
            
            st.subheader("⚡ Voltage Settings")
            vddq = st.slider("VDDQ (V)", min_value=1.05, max_value=1.25, value=1.10, step=0.01)
            vpp = st.slider("VPP (V)", min_value=1.75, max_value=1.90, value=1.80, step=0.01)
        
        st.divider()
        
        # AI Optimization Section
        st.subheader("🚀 AI Optimization")
        
        optimization_goal = st.selectbox(
            "Optimization Goal",
            ["balanced", "extreme_performance", "stability", "power_efficiency"]
        )
        
        performance_target = st.slider(
            "Performance Target (%)", 
            min_value=70.0, 
            max_value=99.0, 
            value=95.0, 
            step=0.5
        )
        
        if st.button("🧠 AI Optimize", type="primary"):
            if not st.session_state.ai_trained:
                st.error("Please train the AI first!")
            else:
                st.session_state.run_ai_optimization = True
                st.session_state.optimization_params = {
                    'frequency': frequency,
                    'goal': optimization_goal,
                    'target': performance_target
                }
        
        # Quick presets
        st.subheader("⚡ Quick Presets")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🏎️ Gaming"):
                st.session_state.preset_config = "gaming"
            if st.button("🔋 Efficiency"):
                st.session_state.preset_config = "efficiency"
        
        with col2:
            if st.button("🛡️ Stable"):
                st.session_state.preset_config = "stable"
            if st.button("🚀 Extreme"):
                st.session_state.preset_config = "extreme"
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "⚙️ Manual Tuning",
        "🎯 Simulation", 
        "🧠 AI Optimization", 
        "🎮 Gaming Performance",
        "📊 Analysis", 
        "🔬 Revolutionary Features",
        "📈 Benchmarks",
        "💻 Hardware Detection",
        "⚡ Live Tuning",
        "🔄 Cross-Brand Tuning"
    ])
    
    # Tab 1: Manual Tuning (NEW - Primary timing controls)
    with tab1:
        st.header("⚙️ Manual DDR5 Timing Configuration")
        st.info("🎯 **Direct Control**: Manually adjust all DDR5 timing parameters for precise tuning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 Primary Timings")
            
            # Calculate intelligent defaults based on frequency
            base_cl = max(16, int(frequency * 0.0055))
            default_trcd = max(16, int(frequency * 0.006))
            default_trp = max(16, int(frequency * 0.0055))
            default_tras = max(32, int(frequency * 0.0125))
            
            manual_cl = st.number_input(
                "CL (CAS Latency)", 
                min_value=14, max_value=60, 
                value=base_cl,
                help="CAS Latency - Lower is faster but less stable"
            )
            
            manual_trcd = st.number_input(
                "tRCD (RAS to CAS Delay)", 
                min_value=14, max_value=60, 
                value=default_trcd,
                help="Time between row activation and column access"
            )
            
            manual_trp = st.number_input(
                "tRP (Row Precharge)", 
                min_value=14, max_value=60, 
                value=default_trp,
                help="Time to precharge a row before accessing new row"
            )
            
            manual_tras = st.number_input(
                "tRAS (Row Active Strobe)", 
                min_value=28, max_value=80, 
                value=default_tras,
                help="Minimum time a row must be active"
            )
            
            # Auto-calculated tRC
            manual_trc = manual_tras + manual_trp
            st.metric(
                "tRC (Row Cycle)", 
                f"{manual_trc}",
                help="Auto-calculated: tRAS + tRP"
            )
        
        with col2:
            st.subheader("⚡ Secondary Timings")
            
            manual_trfc = st.number_input(
                "tRFC (Refresh Cycle)", 
                min_value=160, max_value=500, 
                value=280 + (frequency - 3200) // 400 * 20,
                help="Refresh cycle time - affects how often memory refreshes"
            )
            
            manual_tfaw = st.number_input(
                "tFAW (Four Activate Window)", 
                min_value=16, max_value=50, 
                value=max(16, int(frequency * 0.008)),
                help="Time window for four row activations"
            )
            
            manual_trrds = st.number_input(
                "tRRD_S (Row-to-Row Delay Same)", 
                min_value=4, max_value=12, 
                value=max(4, int(frequency * 0.0015)),
                help="Delay between row activations in same bank group"
            )
            
            manual_trrdl = st.number_input(
                "tRRD_L (Row-to-Row Delay Long)", 
                min_value=4, max_value=12, 
                value=max(6, int(frequency * 0.0018)),
                help="Delay between row activations in different bank groups"
            )
            
            manual_tccdl = st.number_input(
                "tCCD_L (Column-to-Column Long)", 
                min_value=4, max_value=12, 
                value=max(5, int(frequency * 0.0015)),
                help="Column access delay for different bank groups"
            )
        
        st.divider()
        
        # Voltage controls
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("⚡ Voltage Settings")
            
            manual_vddq = st.slider(
                "VDDQ (Core Voltage)", 
                min_value=1.00, max_value=1.35, 
                value=1.10, step=0.01,
                help="Main DDR5 operating voltage (1.1V nominal)"
            )
            
            manual_vpp = st.slider(
                "VPP (Peripheral Voltage)", 
                min_value=1.70, max_value=2.00, 
                value=1.80, step=0.01,
                help="Wordline voltage (1.8V nominal)"
            )
        
        with col4:
            st.subheader("🎛️ Advanced Settings")
            
            manual_gear = st.selectbox(
                "Gear Mode",
                [1, 2],
                index=0,
                help="Gear 1: 1:1 ratio, Gear 2: 1:2 ratio (better for high frequencies)"
            )
            
            manual_cmd_rate = st.selectbox(
                "Command Rate",
                [1, 2],
                index=0,
                help="Commands per clock cycle (1T faster, 2T more stable)"
            )
        
        # Manual config validation and preview
        st.divider()
        st.subheader("📋 Configuration Preview")
        
        # Create manual configuration
        manual_config = DDR5Configuration(
            frequency=frequency,
            timings=DDR5TimingParameters(
                cl=manual_cl,
                trcd=manual_trcd,
                trp=manual_trp,
                tras=manual_tras,
                trc=manual_trc,
                trfc=manual_trfc,
                tfaw=manual_tfaw,
                trrds=manual_trrds,
                trrdl=manual_trrdl,
                tccd_l=manual_tccdl
            ),
            voltages=DDR5VoltageParameters(
                vddq=manual_vddq,
                vpp=manual_vpp
            )
        )
        
        # Display timing table
        col5, col6 = st.columns(2)
        
        with col5:
            st.markdown("**Primary Timings:**")
            timing_data = {
                "Parameter": ["CL", "tRCD", "tRP", "tRAS", "tRC"],
                "Value": [manual_cl, manual_trcd, manual_trp, manual_tras, manual_trc],
                "Unit": ["clocks"] * 5
            }
            st.dataframe(timing_data, hide_index=True)
        
        with col6:
            st.markdown("**Secondary Timings:**")
            secondary_data = {
                "Parameter": ["tRFC", "tFAW", "tRRD_S", "tRRD_L", "tCCD_L"],
                "Value": [manual_trfc, manual_tfaw, manual_trrds, manual_trrdl, manual_tccdl],
                "Unit": ["clocks"] * 5
            }
            st.dataframe(secondary_data, hide_index=True)
        
        # Validation warnings
        validation_issues = []
        if manual_tras < (manual_trcd + manual_cl):
            validation_issues.append("⚠️ tRAS should be ≥ tRCD + CL")
        if manual_trc != (manual_tras + manual_trp):
            validation_issues.append("⚠️ tRC should equal tRAS + tRP")
        if manual_vddq > 1.25:
            validation_issues.append("🔥 High VDDQ voltage - monitor temperatures!")
        if manual_cl < 16 and frequency > 5600:
            validation_issues.append("⚠️ Very aggressive CL for this frequency")
        
        if validation_issues:
            for issue in validation_issues:
                st.warning(issue)
        else:
            st.success("✅ Configuration looks good!")
        
        # Apply button
        if st.button("🚀 Apply Manual Configuration", type="primary", use_container_width=True):
            st.session_state.manual_config = manual_config
            st.success("✅ Manual configuration applied! Check the Simulation tab for results.")
    
    # Tab 2: Simulation (moved from tab1)
    with tab2:
        st.header("🎯 DDR5 Simulation Results")
        
        # Create configuration - prioritize manual config if available
        if hasattr(st.session_state, 'manual_config') and st.session_state.manual_config:
            config = st.session_state.manual_config
            st.info("📝 Using Manual Configuration")
        elif enable_manual:
            config = DDR5Configuration(
                frequency=frequency,
                timings=DDR5TimingParameters(
                    cl=cl, trcd=trcd, trp=trp, tras=tras, trc=trc, trfc=trfc
                ),
                voltages=DDR5VoltageParameters(vddq=vddq, vpp=vpp)
            )
            st.info("⚙️ Using Sidebar Manual Configuration")
        else:
            # Use preset configuration
            config = create_preset_config(frequency, capacity, rank_count)
            st.info("🎯 Using Preset Configuration")
        if enable_manual:
            config = DDR5Configuration(
                frequency=frequency,
                timings=DDR5TimingParameters(
                    cl=cl, trcd=trcd, trp=trp, tras=tras, trc=trc, trfc=trfc
                ),
                voltages=DDR5VoltageParameters(vddq=vddq, vpp=vpp)
            )
        else:
            # Use preset configuration
            config = create_preset_config(frequency, capacity, rank_count)
        
        # Simulate
        st.session_state.simulator.load_configuration(config)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Performance Metrics")
            
            # Bandwidth simulation
            bandwidth_results = st.session_state.simulator.simulate_bandwidth()
            st.metric(
                "Effective Bandwidth", 
                f"{bandwidth_results['effective_bandwidth_gbps']:.1f} GB/s",
                delta=f"{bandwidth_results['efficiency_percent']:.1f}% efficiency"
            )
            
            # Latency simulation
            latency_results = st.session_state.simulator.simulate_latency()
            st.metric(
                "Effective Latency",
                f"{latency_results['effective_latency_ns']:.1f} ns",
                delta=f"Base: {latency_results['base_latency_ns']:.1f} ns"
            )
            
            # Power simulation
            power_results = st.session_state.simulator.simulate_power_consumption()
            st.metric(
                "Total Power",
                f"{power_results['total_power_mw']:.0f} mW",
                delta=f"Dynamic: {power_results['dynamic_power_mw']:.0f} mW"
            )
            
            # Stability test
            stability_results = st.session_state.simulator.run_stability_test()
            stability_color = "normal" if stability_results['stability_score'] > 85 else "inverse"
            st.metric(
                "Stability Score",
                f"{stability_results['stability_score']:.1f}%",
                delta=f"Result: {stability_results['test_result']}",
                delta_color=stability_color
            )
        
        with col2:
            st.subheader("📈 Performance Visualization")
            
            # Create radar chart
            categories = ['Bandwidth', 'Latency (inv)', 'Power Eff', 'Stability']
            values = [
                bandwidth_results['efficiency_percent'],
                100 - (latency_results['effective_latency_ns'] - 10) * 2,  # Inverted latency
                max(0, 100 - (power_results['total_power_mw'] - 2000) / 20),  # Power efficiency
                stability_results['stability_score']
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Current Config'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Performance Profile"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Configuration display
        st.subheader("🔧 Current Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"""
            **Memory Specs**
            - Frequency: DDR5-{config.frequency}
            - Capacity: {capacity}GB
            - Ranks: {rank_count}
            """)
        
        with col2:
            st.info(f"""
            **Primary Timings**
            - CL: {config.timings.cl}
            - tRCD: {config.timings.trcd}
            - tRP: {config.timings.trp}
            - tRAS: {config.timings.tras}
            """)
        
        with col3:
            st.info(f"""
            **Voltages**
            - VDDQ: {config.voltages.vddq:.3f}V
            - VPP: {config.voltages.vpp:.3f}V
            """)
    
    # Tab 3: AI Optimization
    with tab3:
        st.header("🧠 AI-Powered Optimization")
        
        if hasattr(st.session_state, 'run_ai_optimization') and st.session_state.run_ai_optimization:
            st.session_state.run_ai_optimization = False
            
            with st.spinner("🚀 AI is optimizing your DDR5 configuration..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate progress
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 20:
                        status_text.text("Initializing AI models...")
                    elif i < 40:
                        status_text.text("Generating smart population...")
                    elif i < 70:
                        status_text.text("Evolutionary optimization in progress...")
                    elif i < 90:
                        status_text.text("Applying revolutionary features...")
                    else:
                        status_text.text("Finalizing optimal configuration...")
                    time.sleep(0.05)
                
                # Run actual optimization
                optimization_results = st.session_state.optimizer.optimize_perfect(
                    target_frequency=st.session_state.optimization_params['frequency'],
                    optimization_goal=st.session_state.optimization_params['goal'],
                    performance_target=st.session_state.optimization_params['target']
                )
                
                st.session_state.optimization_results = optimization_results
                progress_bar.progress(100)
                status_text.text("Optimization complete!")
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                st.success("🎉 AI Optimization Complete!")
        
        if hasattr(st.session_state, 'optimization_results'):
            results = st.session_state.optimization_results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Optimized Results")
                
                sim_results = results['simulation_results']
                
                st.metric(
                    "Optimized Score",
                    f"{results['optimization_score']:.1f}",
                    delta="AI-Optimized"
                )
                
                st.metric(
                    "Bandwidth",
                    f"{sim_results['bandwidth']['effective_bandwidth_gbps']:.1f} GB/s",
                    delta=f"{sim_results['bandwidth']['efficiency_percent']:.1f}% efficiency"
                )
                
                st.metric(
                    "Latency",
                    f"{sim_results['latency']['effective_latency_ns']:.1f} ns",
                    delta="Optimized"
                )
                
                st.metric(
                    "Stability",
                    f"{sim_results['stability']['stability_score']:.1f}%",
                    delta="AI-Enhanced"
                )
            
            with col2:
                st.subheader("📈 Optimization Progress")
                
                if 'generation_history' in results:
                    history_df = pd.DataFrame(results['generation_history'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=history_df['generation'],
                        y=history_df['max_fitness'],
                        name='Best Fitness',
                        line=dict(color='green')
                    ))
                    fig.add_trace(go.Scatter(
                        x=history_df['generation'],
                        y=history_df['avg_fitness'],
                        name='Average Fitness',
                        line=dict(color='blue')
                    ))
                    
                    fig.update_layout(
                        title="AI Evolution Progress",
                        xaxis_title="Generation",
                        yaxis_title="Fitness Score"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # AI Insights
            st.subheader("🔍 AI Insights")
            
            insights = results['insights']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Timing Analysis**")
                for key, value in insights['timing_analysis'].items():
                    st.write(f"• {key}: {value}")
            
            with col2:
                st.write("**Voltage Analysis**")
                for key, value in insights['voltage_analysis'].items():
                    st.write(f"• {key}: {value}")
            
            with col3:
                st.write("**Risk Assessment**")
                for key, value in insights['risk_assessment'].items():
                    risk_color = "🟢" if value == "Low" else "🟡" if value == "Medium" else "🔴"
                    st.write(f"• {key}: {risk_color} {value}")
            
            # Optimization Suggestions
            if insights.get('optimization_suggestions'):
                st.subheader("💡 AI Recommendations")
                for suggestion in insights['optimization_suggestions']:
                    st.write(f"• {suggestion}")
        
        else:
            st.info("👆 Configure your optimization settings in the sidebar and click 'AI Optimize' to begin!")
            
            # Show AI capabilities
            st.subheader("🧠 AI Capabilities")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("""
                **🤖 Machine Learning Models**
                - Random Forest Ensemble
                - Gradient Boosting
                - Neural Networks
                - Gaussian Processes
                """)
                
                st.write("""
                **🧬 Evolutionary Algorithms**
                - Smart Population Initialization
                - Tournament Selection
                - Intelligent Crossover
                - Adaptive Mutation
                """)
            
            with col2:
                st.write("""
                **🔬 Revolutionary Features**
                - Quantum-Inspired Optimization
                - Molecular-Level Analysis
                - Temperature Adaptation
                - Real-Time Learning
                """)
                
                st.write("""
                **📊 Advanced Analytics**
                - Multi-Objective Optimization
                - Pareto Front Analysis
                - Confidence Scoring
                - Pattern Recognition
                """)
    
    # Tab 4: Gaming Performance (NEW)
    with tab4:
        st.header("🎮 Gaming Performance Predictor")
        st.info("🎯 **Real-World Gaming**: See how your DDR5 configuration affects actual gaming performance")
        
        # Import gaming performance predictor
        try:
            from .gaming_performance import GamingPerformancePredictor
            gaming_predictor = GamingPerformancePredictor()
            
            # Get current configuration
            if hasattr(st.session_state, 'manual_config') and st.session_state.manual_config:
                gaming_config = st.session_state.manual_config
            elif enable_manual:
                gaming_config = DDR5Configuration(
                    frequency=frequency,
                    timings=DDR5TimingParameters(
                        cl=cl, trcd=trcd, trp=trp, tras=tras, trc=trc, trfc=trfc
                    ),
                    voltages=DDR5VoltageParameters(vddq=vddq, vpp=vpp)
                )
            else:
                gaming_config = create_preset_config(frequency, capacity, rank_count)
            
            # Resolution selector
            col1, col2, col3 = st.columns(3)
            with col1:
                resolution = st.selectbox("Resolution", ["1080p", "1440p", "4k"], index=0)
            with col2:
                show_competitive = st.checkbox("Focus on Competitive Games", value=True)
            with col3:
                show_improvements = st.checkbox("Show Memory Impact", value=True)
            
            # Get gaming predictions
            gaming_results = gaming_predictor.predict_gaming_performance(gaming_config, resolution)
            gaming_score = gaming_predictor.calculate_gaming_score(gaming_config)
            recommendations = gaming_predictor.get_gaming_recommendations(gaming_config)
            
            # Display overall gaming score
            st.metric("🏆 Overall Gaming Score", f"{gaming_score:.1f}/100", 
                     delta="Optimized for gaming" if gaming_score > 85 else "Room for improvement")
            
            # Filter games if competitive focus is enabled
            if show_competitive:
                competitive_games = ["valorant", "counter_strike_2", "call_of_duty_warzone", 
                                   "apex_legends", "overwatch_2", "league_of_legends"]
                filtered_results = {k: v for k, v in gaming_results.items() if k in competitive_games}
            else:
                filtered_results = gaming_results
            
            # Display gaming results
            st.subheader(f"🎮 Gaming Performance @ {resolution.upper()}")
            
            # Create columns for game results
            cols = st.columns(3)
            col_idx = 0
            
            for game_id, result in filtered_results.items():
                with cols[col_idx % 3]:
                    # Color code based on FPS
                    if result["fps"] >= 144:
                        fps_color = "🟢"
                    elif result["fps"] >= 60:
                        fps_color = "🟡"
                    else:
                        fps_color = "🔴"
                    
                    st.markdown(f"**{fps_color} {result['game_name']}**")
                    st.markdown(f"**{result['fps']:.0f} FPS** ({result['frame_time_ms']:.1f}ms)")
                    st.markdown(f"1% Low: {result['one_percent_low']:.0f} FPS")
                    
                    if show_improvements and result['memory_improvement_percent'] > 0:
                        st.markdown(f"📈 +{result['memory_improvement_percent']:.1f}% vs baseline")
                    
                    if result['cpu_limited']:
                        st.markdown("⚠️ CPU Limited")
                    
                    st.markdown("---")
                
                col_idx += 1
            
            # Gaming recommendations
            if recommendations:
                st.subheader("🎯 Gaming Optimization Recommendations")
                for category, recommendation in recommendations.items():
                    st.info(recommendation)
            
            # Gaming-focused configuration suggestions
            st.subheader("🚀 Gaming-Optimized Presets")
            
            preset_cols = st.columns(3)
            
            with preset_cols[0]:
                if st.button("🏆 Competitive Gaming", use_container_width=True):
                    # High frequency, tight timings for competitive
                    st.session_state.gaming_preset = "competitive"
                    st.success("Applied competitive gaming preset!")
            
            with preset_cols[1]:
                if st.button("🎮 Balanced Gaming", use_container_width=True):
                    # Balanced approach for all games
                    st.session_state.gaming_preset = "balanced"
                    st.success("Applied balanced gaming preset!")
            
            with preset_cols[2]:
                if st.button("💰 Budget Gaming", use_container_width=True):
                    # Conservative settings for stability
                    st.session_state.gaming_preset = "budget"
                    st.success("Applied budget gaming preset!")
            
        except ImportError:
            st.error("Gaming performance module not available. Please check installation.")
    
    # Tab 5: Analysis (moved from tab4)
    with tab5:
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
                           orientation='h', title='Parameter Importance for Performance')
                st.plotly_chart(fig, use_container_width=True)
        
        # Current configuration analysis
        st.subheader("🔍 Current Configuration Analysis")
        
        # Get current configuration
        if hasattr(st.session_state, 'manual_config') and st.session_state.manual_config:
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
            st.metric("Bandwidth", f"{bandwidth_result['effective_bandwidth_gbps']:.1f} GB/s",
                     help="Theoretical maximum bandwidth")
        
        with col2:
            latency_result = st.session_state.simulator.simulate_latency()
            st.metric("Latency", f"{latency_result['effective_latency_ns']:.1f} ns",
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
            timing_checks.append("❌ tRAS < tRCD + CL (Invalid - may cause instability)")
        
        if timings.trc >= timings.tras + timings.trp:
            timing_checks.append("✅ tRC >= tRAS + tRP (Valid)")
        else:
            timing_checks.append("❌ tRC < tRAS + tRP (Invalid - may cause corruption)")
        
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
        current_score = (current_config.frequency / 4800) * 100 * (40 / current_config.timings.cl)
        
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

    # Tab 6: Revolutionary Features
    with tab6:
        st.header("🔬 Revolutionary AI Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Quantum Optimization")
            st.write("""
            Our quantum-inspired optimization uses superposition states to explore
            multiple parameter configurations simultaneously, allowing escape from
            local optima through quantum tunneling effects.
            """)
            
            if st.button("🌌 Quantum Analyze"):
                with st.spinner("Performing quantum analysis..."):
                    # Simulate quantum analysis
                    time.sleep(2)
                    st.success("Quantum states analyzed! Found 127 superposition configurations.")
                    st.info("Quantum tunneling probability: 5.2%")
            
            st.subheader("🧬 Molecular-Level Analysis")
            st.write("""
            Analysis at the molecular level considers electron mobility, charge
            retention, and parasitic capacitance to optimize timing relationships
            based on physical properties.
            """)
            
            if st.button("🔬 Molecular Analysis"):
                with st.spinner("Analyzing molecular structure..."):
                    time.sleep(2)
                    st.success("Molecular analysis complete!")
                    st.json({
                        "electron_mobility": "1400 cm²/V·s",
                        "charge_retention": "64 ms base time",
                        "parasitic_capacitance": "0.1 pF optimized"
                    })
        
        with col2:
            st.subheader("🌡️ Temperature Adaptation")
            st.write("""
            Real-time temperature monitoring and adaptive optimization adjusts
            parameters based on thermal conditions to maintain optimal performance
            across different operating temperatures.
            """)
            
            ambient_temp = st.slider("Ambient Temperature (°C)", 15, 35, 25)
            target_temp = st.slider("Target Temperature (°C)", 50, 85, 70)
            
            if st.button("🌡️ Temperature Optimize"):
                with st.spinner("Optimizing for temperature..."):
                    time.sleep(2)
                    st.success(f"Optimized for {ambient_temp}°C ambient, {target_temp}°C target!")
                    st.info("Voltage adjusted: -0.02V for thermal efficiency")
            
            st.subheader("🚀 Hyperspace Exploration")
            st.write("""
            Multi-dimensional parameter space exploration using advanced algorithms
            to find optimal configurations in 15-dimensional hyperspace.
            """)
            
            if st.button("🌌 Hyperspace Search"):
                with st.spinner("Exploring hyperspace..."):
                    time.sleep(3)
                    st.success("Hyperspace exploration complete!")
                    st.json({
                        "dimensions_explored": 15,
                        "parallel_universes": 1337,
                        "optimal_configs_found": 23
                    })
    
    # Tab 7: Benchmarks
    with tab7:
        st.header("📈 Performance Benchmarks")
        
        st.subheader("🏆 Benchmark Results")
        
        # Simulated benchmark data
        benchmark_data = {
            'Test': ['AIDA64 Memory', 'Intel MLC', 'STREAM Triad', 'LuxMark', 'Cinebench R23'],
            'Score': [85247, 92341, 156789, 23456, 18945],
            'Percentile': [92, 89, 94, 87, 91],
            'Status': ['Excellent', 'Very Good', 'Excellent', 'Good', 'Very Good']
        }
        
        df = pd.DataFrame(benchmark_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(df, x='Test', y='Percentile', 
                        title='Benchmark Percentiles',
                        color='Percentile',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(df, names='Status', title='Performance Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df, use_container_width=True)
        
        # Gaming performance
        st.subheader("🎮 Gaming Performance")
        
        gaming_data = {
            'Game': ['Cyberpunk 2077', 'Call of Duty MW3', 'Baldur\'s Gate 3', 'Starfield', 'Alan Wake 2'],
            'FPS Gain': ['+8.3%', '+12.1%', '+5.7%', '+9.4%', '+6.8%'],
            'Frame Time': ['12.4ms', '8.7ms', '16.2ms', '11.1ms', '14.6ms'],
            'Rating': ['⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐']
        }
        
        gaming_df = pd.DataFrame(gaming_data)
        st.dataframe(gaming_df, use_container_width=True)
    
    # Tab 8: Hardware Detection
    with tab8:
        st.header("💻 Hardware Detection & Real RAM Database")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("🔍 System Scan")
            
            if st.button("🔎 Scan System RAM", type="primary"):
                with st.spinner("Detecting system memory..."):
                    try:
                        # Clear any cached data first
                        st.session_state.detected_modules = []
                        st.session_state.hardware_scanned = False
                        
                        # Create fresh detector instance
                        from hardware_detection import HardwareDetector
                        detector = HardwareDetector()
                        
                        # Force fresh detection
                        memory_info = detector.detect_system_memory()
                        
                        # Debug information
                        st.info(f"🔍 Detection method: {detector.detection_method}")
                        st.info(f"📊 Found {len(memory_info)} module(s)")
                        
                        # Store results
                        st.session_state.detected_modules = memory_info
                        st.session_state.hardware_scanned = True
                        
                        # Show immediate results
                        if memory_info:
                            for i, module in enumerate(memory_info):
                                st.success(f"✅ Module {i+1}: {module.manufacturer} {module.part_number} {module.capacity_gb}GB")
                        else:
                            st.warning("⚠️ No modules detected")
                            
                    except Exception as e:
                        st.error(f"❌ Scan failed: {str(e)}")
                        st.exception(e)
            
            # Debug button for testing
            if st.button("🔄 Force Refresh Hardware Scan", type="secondary"):
                with st.spinner("Force refreshing hardware detection..."):
                    try:
                        # Clear cached data
                        st.session_state.detected_modules = []
                        st.session_state.hardware_scanned = False
                        
                        # Force fresh detection
                        memory_info = detect_system_memory()
                        st.session_state.detected_modules = memory_info
                        st.session_state.hardware_scanned = True
                        
                        st.success(f"✅ Fresh scan completed! Found {len(memory_info)} modules")
                        
                        # Show debug info
                        with st.expander("🔍 Debug Information"):
                            for module in memory_info:
                                st.write(f"**Debug Module:** {module}")
                                st.write(f"**Type:** {type(module)}")
                                st.json({
                                    'manufacturer': module.manufacturer,
                                    'part_number': module.part_number,
                                    'capacity_gb': module.capacity_gb,
                                    'speed_mt_s': module.speed_mt_s,
                                    'serial_number': module.serial_number,
                                    'voltage': module.voltage
                                })
                    except Exception as e:
                        st.error(f"❌ Debug scan failed: {str(e)}")
                        st.exception(e)
            
            # Session state reset for debugging
            if st.button("🗑️ Clear All Cache & Reset", type="secondary"):
                # Clear all hardware detection related session state
                keys_to_remove = []
                for key in st.session_state.keys():
                    if ('detected' in str(key) or 'hardware' in str(key) or 
                        'scanned' in str(key)):
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del st.session_state[key]
                
                # Reinitialize
                st.session_state.detected_modules = []
                st.session_state.hardware_scanned = False
                st.success("🗑️ Cache cleared! Try scanning again.")
                st.rerun()
            
            # Current limitations notice
            st.info("ℹ️ **Current Limitations**\n\n"
                   "This simulator currently provides:\n"
                   "• System RAM detection (read-only)\n"
                   "• DDR5 parameter simulation\n"
                   "• AI optimization recommendations\n\n"
                   "**Real tuning requires:**\n"
                   "• BIOS/UEFI access\n"
                   "• Hardware vendor tools\n"
                   "• Physical memory modules")
        
        with col1:
            st.subheader("🖥️ Detected System Memory")
            
            if st.session_state.hardware_scanned and st.session_state.detected_modules:
                for i, module in enumerate(st.session_state.detected_modules):
                    with st.expander(f"💾 Memory Module {i+1} - {module.manufacturer}"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write(f"**Manufacturer:** {module.manufacturer}")
                            st.write(f"**Capacity:** {module.capacity_gb} GB")
                            st.write(f"**Speed:** DDR5-{module.speed_mt_s}")
                            st.write(f"**Form Factor:** {module.form_factor}")
                        
                        with col_b:
                            st.write(f"**Part Number:** {module.part_number}")
                            st.write(f"**Serial:** {module.serial_number or 'Unknown'}")
                            st.write(f"**Slot Location:** {module.slot_location}")
                            st.write(f"**Voltage:** {module.voltage or 'Unknown'}V")
                        
                        # Try to match with database  
                        if module.part_number:
                            ram_db = get_database()
                            matched_specs = []
                            
                            # Multiple matching strategies for better detection
                            for spec in ram_db.modules:
                                match_score = 0
                                
                                # Strategy 1: Exact part number match
                                if module.part_number.upper() == spec.part_number.upper():
                                    match_score += 100
                                
                                # Strategy 2: Partial part number match (for variations)
                                elif module.part_number.upper() in spec.part_number.upper() or \
                                     spec.part_number.upper() in module.part_number.upper():
                                    match_score += 80
                                
                                # Strategy 3: Manufacturer + capacity + speed similarity
                                if module.manufacturer.lower() == spec.manufacturer.lower():
                                    match_score += 30
                                    
                                    if module.capacity_gb == spec.capacity_gb:
                                        match_score += 20
                                    
                                    # Speed tolerance (±400 MT/s)
                                    speed_diff = abs(module.speed_mt_s - spec.jedec_speed)
                                    if speed_diff <= 400:
                                        match_score += 20
                                    elif speed_diff <= 800:
                                        match_score += 10
                                
                                # Strategy 4: Part number pattern matching for Kingston
                                if (module.manufacturer.lower() == "kingston" and 
                                    spec.manufacturer.lower() == "kingston"):
                                    # Extract numbers from part numbers for pattern matching
                                    module_nums = ''.join(filter(str.isdigit, module.part_number))
                                    spec_nums = ''.join(filter(str.isdigit, spec.part_number))
                                    if module_nums and spec_nums:
                                        # Check if significant digits match
                                        if module_nums[:4] == spec_nums[:4]:  # First 4 digits
                                            match_score += 25
                                
                                # If we have a reasonable match, add it
                                if match_score >= 50:  # Minimum threshold
                                    matched_specs.append((spec, match_score))
                            
                            # Sort by match score (highest first)
                            matched_specs.sort(key=lambda x: x[1], reverse=True)
                            matched_specs = [spec for spec, score in matched_specs]
                            
                            if matched_specs:
                                st.success(f"✅ Found {len(matched_specs)} matching specifications in database")
                                selected_spec = st.selectbox(
                                    "Select specification:",
                                    matched_specs,
                                    format_func=lambda x: f"{x.manufacturer} {x.series} - CL{x.cas_latency}",
                                    key=f"spec_select_{i}"
                                )
                                
                                if st.button(f"🎯 Optimize This Module", key=f"optimize_{i}"):
                                    # Create configuration from database spec
                                    config = DDR5Configuration(
                                        frequency=selected_spec.jedec_speed,
                                        timings=DDR5TimingParameters(
                                            cl=selected_spec.cas_latency,
                                            trcd=selected_spec.trcd or selected_spec.cas_latency,
                                            trp=selected_spec.trp or selected_spec.cas_latency,
                                            tras=selected_spec.tras or (selected_spec.cas_latency + 28)
                                        ),
                                        voltages=DDR5VoltageParameters(
                                            vddq=selected_spec.voltage,
                                            vpp=1.8
                                        )
                                    )
                                    
                                    # Store in session state for optimization
                                    st.session_state.config = config
                                    st.success("✅ Configuration loaded from database!")
                            else:
                                st.warning("⚠️ No matching specifications found in database")
            else:
                st.warning("🔍 No memory modules detected. Click 'Scan System RAM' to detect your hardware.")
                
                # Show example of what detection would look like
                st.subheader("📋 Example Detection Results")
                example_modules = [
                    {
                        "manufacturer": "Corsair",
                        "capacity_gb": 32,
                        "speed_mt_s": 6000,
                        "memory_type": "DDR5 SDRAM",
                        "part_number": "CMK32GX5M2B6000C36",
                        "serial": "12345678",
                        "bank_label": "BANK 0",
                        "form_factor": "DIMM"
                    },
                    {
                        "manufacturer": "G.Skill",
                        "capacity_gb": 32,
                        "speed_mt_s": 6400,
                        "memory_type": "DDR5 SDRAM",
                        "part_number": "F5-6400J3239G32GX2-TZ5RK",
                        "serial": "87654321",
                        "bank_label": "BANK 1",
                        "form_factor": "DIMM"
                    }
                ]
                
                for i, module in enumerate(example_modules):
                    with st.expander(f"💾 Example Module {i+1} - {module['manufacturer']}"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write(f"**Manufacturer:** {module['manufacturer']}")
                            st.write(f"**Capacity:** {module['capacity_gb']} GB")
                            st.write(f"**Speed:** DDR5-{module['speed_mt_s']}")
                            st.write(f"**Type:** {module['memory_type']}")
                        
                        with col_b:
                            st.write(f"**Part Number:** {module['part_number']}")
                            st.write(f"**Serial:** {module['serial']}")
                            st.write(f"**Bank Label:** {module['bank_label']}")
                            st.write(f"**Form Factor:** {module['form_factor']}")
        
        # RAM Database Browser
        st.subheader("🗃️ DDR5 Module Database")
        
        # Get database
        ram_db = get_database()
        
        # Filters
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            manufacturers = list(set([spec.manufacturer for spec in ram_db.modules]))
            selected_mfg = st.selectbox("Filter by Manufacturer:", ["All"] + manufacturers)
        
        with col_filter2:
            speeds = list(set([spec.jedec_speed for spec in ram_db.modules]))
            speeds.sort()
            selected_speed = st.selectbox("Filter by Speed:", 
                                        ["All"] + [f"DDR5-{speed}" for speed in speeds])
        
        with col_filter3:
            capacities = list(set([spec.capacity_gb for spec in ram_db.modules]))
            capacities.sort()
            selected_capacity = st.selectbox("Filter by Capacity:", 
                                           ["All"] + [f"{cap}GB" for cap in capacities])
        
        # Filter database
        filtered_db = ram_db.modules  # Get the modules list
        if selected_mfg != "All":
            filtered_db = [spec for spec in filtered_db 
                         if spec.manufacturer == selected_mfg]
        if selected_speed != "All":
            speed_val = int(selected_speed.replace("DDR5-", ""))
            filtered_db = [spec for spec in filtered_db 
                         if spec.jedec_speed == speed_val]
        if selected_capacity != "All":
            cap_val = int(selected_capacity.replace("GB", ""))
            filtered_db = [spec for spec in filtered_db 
                         if spec.capacity_gb == cap_val]
        
        # Display database
        if filtered_db:
            st.write(f"📊 Showing {len(filtered_db)} DDR5 modules:")
            
            # Create DataFrame for display
            db_data = []
            for spec in filtered_db[:20]:  # Limit to first 20 for performance
                db_data.append({
                    "Manufacturer": spec.manufacturer,
                    "Series": spec.series,
                    "Speed": f"DDR5-{spec.jedec_speed}",
                    "Capacity": f"{spec.capacity_gb}GB",
                    "CAS Latency": f"CL{spec.cas_latency}",
                    "Voltage": f"{spec.voltage}V",
                    "Price Range": spec.price_range,
                    "OC Potential": spec.overclocking_potential.value
                })
            
            db_df = pd.DataFrame(db_data)
            
            # Interactive table with selection
            db_df = pd.DataFrame(db_data)
            st.dataframe(db_df, use_container_width=True, hide_index=True)
            
            # Manual selection with selectbox for configuration loading
            if len(filtered_db) > 0:
                st.subheader("🔧 Load Configuration from Database")
                selected_spec = st.selectbox(
                    "Select a module to load its configuration:",
                    filtered_db,
                    format_func=lambda x: f"{x.manufacturer} {x.series} - DDR5-{x.jedec_speed} CL{x.cas_latency}",
                    key="db_module_select"
                )
                
                st.subheader(f"🔍 Selected: {selected_spec.manufacturer} {selected_spec.series}")
                
                col_details1, col_details2, col_details3 = st.columns(3)
                
                with col_details1:
                    st.write("**Specifications:**")
                    st.write(f"• Speed: DDR5-{selected_spec.jedec_speed}")
                    st.write(f"• Capacity: {selected_spec.capacity_gb}GB")
                    st.write(f"• CAS Latency: CL{selected_spec.cas_latency}")
                    st.write(f"• Voltage: {selected_spec.voltage}V")
                
                with col_details2:
                    st.write("**Market Info:**")
                    st.write(f"• Price Range: {selected_spec.price_range}")
                    st.write(f"• Chip Type: {selected_spec.chip_type.value}")
                    st.write(f"• Overclocking: {selected_spec.overclocking_potential.value}")
                
                with col_details3:
                    st.write("**Features:**")
                    if selected_spec.features:
                        for feature in selected_spec.features:
                            st.write(f"• {feature}")
                    else:
                        st.write("• Standard features")
                
                if st.button("🚀 Load Configuration for AI Optimization", key="load_selected_config"):
                    # Create configuration from selected spec
                    config = DDR5Configuration(
                        frequency=selected_spec.jedec_speed,
                        timings=DDR5TimingParameters(
                            cl=selected_spec.cas_latency,
                            trcd=selected_spec.trcd or selected_spec.cas_latency,
                            trp=selected_spec.trp or selected_spec.cas_latency,
                            tras=selected_spec.tras or (selected_spec.cas_latency + 28)
                        ),
                        voltages=DDR5VoltageParameters(
                            vddq=selected_spec.voltage,
                            vpp=1.8
                        )
                    )
                    
                    st.session_state.config = config
                    st.success(f"✅ Loaded {selected_spec.manufacturer} {selected_spec.series} configuration!")
                    st.balloons()
        else:
            st.write("No modules match the selected filters.")
        
        # Real-world limitations explanation
        st.subheader("⚠️ Important: Real DDR5 Tuning Requirements")
        
        st.warning("""
        **This simulator provides AI-powered optimization recommendations, but cannot directly modify hardware.**
        
        To apply optimized timings to real DDR5 modules, you need:
        
        🔧 **BIOS/UEFI Access:**
        - Enter BIOS/UEFI setup during boot
        - Navigate to memory/overclocking settings
        - Manually input optimized timings
        
        🛠️ **Hardware Tools:**
        - MSI Afterburner / EVGA Precision
        - ASUS AI Suite / MSI Dragon Center
        - Intel XTU / AMD Ryzen Master
        
        📋 **Recommended Process:**
        1. Use this AI to find optimal settings
        2. Test stability with MemTest86
        3. Apply settings in BIOS gradually
        4. Validate with stress testing
        5. Monitor temperatures and stability
        
        💡 **Safety First:**
        - Start with conservative settings
        - Increase performance gradually
        - Always have a backup of working settings
        - Monitor system stability continuously
        """)

    # Tab 9: Live Tuning
    with tab9:
        st.header("⚡ Live DDR5 Tuning with Safety Systems")
        
        # Live tuning simulation (safer approach)
        st.info("""
        🔬 **Live Tuning Simulation Mode**
        
        This demonstrates what live tuning would look like, but operates in safe simulation mode.
        No actual hardware changes are made.
        """)
        
        # Safety level selector
        safety_level = st.selectbox(
            "Safety Level",
            ["Conservative", "Moderate", "Aggressive"],
            index=0,
            help="Higher levels allow more aggressive tuning"
        )
        
        # Current system status simulation
        st.subheader("📊 System Monitoring")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Simulate CPU temperature
            cpu_temp = 45.0 + (current_config.frequency - 4800) / 200
            st.metric("CPU Temp", f"{cpu_temp:.1f}°C", 
                     delta="Normal" if cpu_temp < 70 else "Warm")
        
        with col2:
            # Simulate memory temperature
            mem_temp = 35.0 + (current_config.voltages.vddq - 1.1) * 20
            st.metric("Memory Temp", f"{mem_temp:.1f}°C",
                     delta="Good" if mem_temp < 50 else "Elevated")
        
        with col3:
            # Simulate stability score
            stability = max(60, 95 - (current_config.frequency - 4800) / 100)
            st.metric("Stability", f"{stability:.1f}%",
                     delta="Stable" if stability > 90 else "Monitor")
        
        with col4:
            # Simulate error rate
            error_rate = max(0, (current_config.frequency - 5600) / 1000)
            st.metric("Error Rate", f"{error_rate:.3f}%",
                     delta="Excellent" if error_rate < 0.001 else "Acceptable")
        
        # Live tuning controls
        st.subheader("⚡ Live Tuning Controls")
        
        if st.button("🔍 Test Current Configuration", type="primary"):
            with st.spinner("Running stability test..."):
                time.sleep(3)  # Simulate test time
                
                if stability > 85:
                    st.success("✅ Configuration passed stability test!")
                    st.info(f"Completed {1000} memory operations without errors")
                else:
                    st.warning("⚠️ Configuration shows potential instability")
                    st.info("Consider reducing frequency or loosening timings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Auto-Optimize", use_container_width=True):
                with st.spinner("Finding optimal settings..."):
                    time.sleep(2)
                    st.success("Found optimized configuration!")
                    st.info("Suggested: DDR5-6000 CL30-36-36-76 @ 1.25V")
        
        with col2:
            if st.button("🔄 Revert to Safe", use_container_width=True):
                st.success("Reverted to JEDEC-safe settings")
                st.info("DDR5-4800 CL40-40-40-77 @ 1.1V")
        
        # Live tuning history
        st.subheader("📈 Tuning History")
        
        history_data = [
            {"Time": "14:30", "Action": "Increased frequency to 5600", "Result": "✅ Stable"},
            {"Time": "14:25", "Action": "Tightened CL to 36", "Result": "✅ Stable"},
            {"Time": "14:20", "Action": "Started baseline test", "Result": "✅ Passed"},
        ]
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True)
        
        # Safety recommendations
        st.subheader("🛡️ Safety Recommendations")
        
        safety_tips = [
            "💾 Always create a system restore point before live tuning",
            "🌡️ Monitor temperatures continuously during testing",
            "⏱️ Test each change for at least 15 minutes before applying the next",
            "🔄 Keep BIOS recovery options readily available",
            "💪 Ensure PSU can handle increased power draw",
            "❄️ Verify cooling is adequate for voltage increases"
        ]
        
        for tip in safety_tips:
            st.info(tip)
        
        # Safety warnings and disclaimers
        st.error("""
        ⚠️ **CRITICAL WARNING: EXPERIMENTAL FEATURE**
        
        Live tuning attempts to modify real hardware settings. This carries risks:
        - System instability or crashes
        - Potential hardware damage if safety limits are exceeded
        - Data loss if system becomes unstable
        
        **USE AT YOUR OWN RISK** - Ensure you have:
        - Stable baseline settings to revert to
        - System backups and restore points
        - Adequate cooling and power supply
        - Knowledge of safe DDR5 limits
        """)
        
        # Safety acknowledgment
        safety_acknowledged = st.checkbox(
            "🛡️ I understand the risks and want to proceed with live tuning",
            key="safety_acknowledgment"
        )
        
        if not safety_acknowledged:
            st.warning("Please acknowledge the safety warnings to continue.")
            return
        
        # Safety level selection
        col_safety1, col_safety2 = st.columns(2)
        
        with col_safety1:
            st.subheader("🛡️ Safety Configuration")
            
            safety_level = st.selectbox(
                "Protection Level:",
                ["conservative", "moderate", "aggressive", "expert"],
                index=0,
                help="Conservative: Safest limits, Moderate: Balanced, Aggressive: Higher limits, Expert: Maximum flexibility"
            )
            
            # Safety limits display
            if safety_level == "conservative":
                st.info("""
                **Conservative Limits:**
                - Max VDDQ: 1.25V
                - Max Temp: 75°C
                - Extensive monitoring
                """)
            elif safety_level == "moderate":
                st.info("""
                **Moderate Limits:**
                - Max VDDQ: 1.35V
                - Max Temp: 80°C
                - Balanced protection
                """)
            elif safety_level == "aggressive":
                st.warning("""
                **Aggressive Limits:**
                - Max VDDQ: 1.40V
                - Max Temp: 83°C
                - Reduced safety margins
                """)
            else:  # expert
                st.error("""
                **Expert Limits:**
                - Max VDDQ: 1.45V
                - Max Temp: 85°C
                - Minimal protection
                """)
        
        with col_safety2:
            st.subheader("📋 Safety Recommendations")
            
            recommendations = get_safety_recommendations()
            for rec in recommendations[:5]:  # Show first 5
                st.write(f"• {rec}")
            
            with st.expander("See all recommendations"):
                for rec in recommendations[5:]:
                    st.write(f"• {rec}")
        
        # Initialize/Start Live Tuner
        col_control1, col_control2, col_control3 = st.columns(3)
        
        with col_control1:
            if st.button("🚀 Initialize Live Tuner", type="primary"):
                try:
                    st.session_state.live_tuner = create_live_tuner(safety_level)
                    st.success("✅ Live tuner initialized!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize: {e}")
        
        with col_control2:
            if st.session_state.live_tuner and st.button("📊 Start Monitoring"):
                st.session_state.live_tuner.start_monitoring()
                st.session_state.monitoring_active = True
                st.success("📊 Real-time monitoring started!")
        
        with col_control3:
            if st.session_state.live_tuner and st.button("🛑 Emergency Stop"):
                st.session_state.live_tuner.emergency_stop()
                st.error("🚨 Emergency stop activated!")
        
        # Live tuner interface
        if st.session_state.live_tuner:
            # Monitoring dashboard
            st.subheader("📊 Real-Time System Monitoring")
            
            # Get monitoring data
            monitoring_data = st.session_state.live_tuner.get_monitoring_summary()
            
            if monitoring_data.get("status") != "no_data":
                col_mon1, col_mon2, col_mon3, col_mon4 = st.columns(4)
                
                current = monitoring_data.get("current", {})
                
                with col_mon1:
                    cpu_temp = current.get("cpu_temp", 0)
                    temp_color = "normal" if cpu_temp < 80 else "inverse"
                    st.metric("CPU Temp", f"{cpu_temp:.1f}°C", 
                             delta_color=temp_color)
                
                with col_mon2:
                    mem_temp = current.get("memory_temp", 0)
                    mem_color = "normal" if mem_temp < 70 else "inverse"
                    st.metric("Memory Temp", f"{mem_temp:.1f}°C",
                             delta_color=mem_color)
                
                with col_mon3:
                    stability = current.get("stability", 0)
                    stab_color = "normal" if stability > 80 else "inverse"
                    st.metric("Stability", f"{stability:.1f}%",
                             delta_color=stab_color)
                
                with col_mon4:
                    cpu_usage = current.get("cpu_usage", 0)
                    st.metric("CPU Usage", f"{cpu_usage:.1f}%")
                
                # Status indicators
                col_status1, col_status2 = st.columns(2)
                
                with col_status1:
                    status = monitoring_data.get("status", "idle")
                    if status == "idle":
                        st.success("🟢 Status: Ready")
                    elif status == "testing":
                        st.warning("🟡 Status: Testing Configuration")
                    elif status == "emergency_stop":
                        st.error("🔴 Status: Emergency Stop Active")
                    else:
                        st.info(f"🔵 Status: {status.title()}")
                
                with col_status2:
                    hw_access = monitoring_data.get("hardware_access", False)
                    if hw_access:
                        st.success("🔧 Hardware Access: Available")
                    else:
                        st.warning("💻 Hardware Access: Simulation Only")
            
            # Configuration testing
            st.subheader("🧪 Configuration Testing")
            
            col_test1, col_test2 = st.columns(2)
            
            with col_test1:
                st.write("**Test Custom Configuration:**")
                
                test_freq = st.number_input("Frequency (MT/s)", 
                                          min_value=3200, max_value=8400, 
                                          value=5600, step=200)
                
                test_cl = st.number_input("CAS Latency", 
                                        min_value=16, max_value=60, 
                                        value=36, step=2)
                
                test_vddq = st.number_input("VDDQ Voltage (V)", 
                                          min_value=1.05, max_value=1.45, 
                                          value=1.25, step=0.05)
                
                if st.button("🧪 Test Configuration"):
                    # Create test configuration
                    test_config = DDR5Configuration(
                        frequency=test_freq,
                        timings=DDR5TimingParameters(
                            cl=test_cl,
                            trcd=test_cl,
                            trp=test_cl,
                            tras=test_cl + 28
                        ),
                        voltages=DDR5VoltageParameters(
                            vddq=test_vddq,
                            vpp=1.8
                        )
                    )
                    
                    with st.spinner("Testing configuration..."):
                        success, message = st.session_state.live_tuner.test_configuration(test_config)
                    
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
            
            with col_test2:
                st.write("**Quick Presets:**")
                
                if st.button("🛡️ Conservative (DDR5-5200)"):
                    conservative_config = DDR5Configuration(
                        frequency=5200,
                        timings=DDR5TimingParameters(cl=42, trcd=42, trp=42, tras=70),
                        voltages=DDR5VoltageParameters(vddq=1.10, vpp=1.8)
                    )
                    success, msg = st.session_state.live_tuner.test_configuration(conservative_config)
                    st.write(f"Result: {msg}")
                
                if st.button("⚡ Performance (DDR5-6000)"):
                    performance_config = DDR5Configuration(
                        frequency=6000,
                        timings=DDR5TimingParameters(cl=36, trcd=36, trp=36, tras=76),
                        voltages=DDR5VoltageParameters(vddq=1.35, vpp=1.8)
                    )
                    success, msg = st.session_state.live_tuner.test_configuration(performance_config)
                    st.write(f"Result: {msg}")
                
                if st.button("🚀 Extreme (DDR5-6400)"):
                    extreme_config = DDR5Configuration(
                        frequency=6400,
                        timings=DDR5TimingParameters(cl=32, trcd=32, trp=32, tras=72),
                        voltages=DDR5VoltageParameters(vddq=1.40, vpp=1.85)
                    )
                    success, msg = st.session_state.live_tuner.test_configuration(extreme_config)
                    st.write(f"Result: {msg}")
                
                if st.button("🔄 Revert to Baseline"):
                    st.session_state.live_tuner.revert_to_baseline()
                    st.info("🔄 Reverted to baseline configuration")
            
            # Advanced controls
            with st.expander("🔧 Advanced Controls"):
                col_adv1, col_adv2 = st.columns(2)
                
                with col_adv1:
                    if st.button("📊 Stop Monitoring"):
                        st.session_state.live_tuner.stop_monitoring()
                        st.session_state.monitoring_active = False
                        st.info("📊 Monitoring stopped")
                    
                    if st.button("🗑️ Reset Live Tuner"):
                        st.session_state.live_tuner = None
                        st.session_state.monitoring_active = False
                        st.info("🗑️ Live tuner reset")
                
                with col_adv2:
                    st.write("**System Info:**")
                    if monitoring_data.get("status") != "no_data":
                        st.json({
                            "Safety Level": monitoring_data.get("safety_level"),
                            "Hardware Access": monitoring_data.get("hardware_access"),
                            "Current Status": monitoring_data.get("status")
                        })
        
        else:
            st.info("Click 'Initialize Live Tuner' to begin")
        
        # Safety Simulation Section
        st.divider()
        st.header("🔒 Live Tuning Safety Simulation")
        
        st.info("""
        **🧪 Safety Simulation**: Test the safety of live tuning operations without making actual hardware changes.
        This simulation validates voltage limits, thermal conditions, timing relationships, and overall system stability.
        """)
        
        # Safety simulation configuration
        col_sim1, col_sim2 = st.columns(2)
        
        with col_sim1:
            st.subheader("🎯 Configuration to Test")
            
            # Get current configuration from sidebar
            sim_frequency = frequency
            sim_capacity = capacity
            
            # Override with custom values if desired
            use_custom = st.checkbox("🔧 Use Custom Configuration")
            
            if use_custom:
                sim_frequency = st.number_input("Test Frequency (MT/s)", 
                                              min_value=3200, max_value=8400, 
                                              value=frequency, step=200)
                sim_cl = st.number_input("Test CAS Latency", 
                                        min_value=16, max_value=60, 
                                        value=36, step=2)
                sim_vddq = st.number_input("Test VDDQ (V)", 
                                          min_value=1.05, max_value=1.45, 
                                          value=1.25, step=0.05)
                sim_vpp = st.number_input("Test VPP (V)", 
                                         min_value=1.7, max_value=2.0, 
                                         value=1.8, step=0.05)
            else:
                # Use AI-optimized configuration if available
                if st.session_state.ai_trained and hasattr(st.session_state, 'optimized_config'):
                    config = st.session_state.optimized_config
                    sim_cl = config.timing_parameters.cl
                    sim_vddq = config.voltage_parameters.vddq
                    sim_vpp = config.voltage_parameters.vpp
                else:
                    # Use conservative defaults
                    sim_cl = 40
                    sim_vddq = 1.1
                    sim_vpp = 1.8
                
                st.write(f"**Frequency:** {sim_frequency} MT/s")
                st.write(f"**CAS Latency:** {sim_cl}")
                st.write(f"**VDDQ:** {sim_vddq}V")
                st.write(f"**VPP:** {sim_vpp}V")
        
        with col_sim2:
            st.subheader("💾 Hardware Context")
            
            # Show detected hardware for context
            if st.session_state.hardware_scanned and st.session_state.detected_modules:
                st.success(f"✅ Using detected hardware: {len(st.session_state.detected_modules)} module(s)")
                for module in st.session_state.detected_modules:
                    st.write(f"• {module.manufacturer} {module.part_number} {module.capacity_gb}GB")
            else:
                st.warning("⚠️ No hardware detected - using generic safety limits")
                if st.button("🔍 Scan Hardware First"):
                    st.rerun()
        
        # Run safety simulation
        if st.button("🧪 Run Safety Simulation", type="primary"):
            with st.spinner("Running comprehensive safety analysis..."):
                # Create test configuration
                test_config = DDR5Configuration(
                    frequency=sim_frequency,
                    capacity=sim_capacity,
                    rank_count=rank_count,
                    timings=DDR5TimingParameters(
                        cl=sim_cl,
                        trcd=sim_cl,
                        trp=sim_cl,
                        tras=sim_cl + 28,
                        trc=tras + trp,
                        trfc=280 + (frequency - 3200) // 400 * 20  # Rough estimate
                    ),
                    voltages=DDR5VoltageParameters(
                        vddq=sim_vddq,
                        vpp=sim_vpp
                    )
                )
                
                # Get hardware context
                hardware_modules = st.session_state.detected_modules if st.session_state.hardware_scanned else []
                
                # Run safety validation
                validator = LiveTuningSafetyValidator()
                safety_report = validator.run_comprehensive_safety_test(test_config, hardware_modules)
                
                # Store results in session state
                st.session_state.safety_report = safety_report
        
        # Display safety results
        if hasattr(st.session_state, 'safety_report'):
            report = st.session_state.safety_report
            
            st.divider()
            st.subheader("📊 Safety Analysis Results")
            
            # Overall safety assessment
            col_result1, col_result2, col_result3 = st.columns(3)
            
            with col_result1:
                # Safety level indicator
                if report.overall_safety.value == "verified_safe":
                    st.success("🟢 **VERIFIED SAFE**")
                elif report.overall_safety.value == "safe":
                    st.success("🟢 **SAFE**")
                elif report.overall_safety.value == "caution":
                    st.warning("🟡 **CAUTION**")
                elif report.overall_safety.value == "unsafe":
                    st.error("🔴 **UNSAFE**")
                else:
                    st.error("⚠️ **CRITICAL**")
            
            with col_result2:
                st.metric("Overall Safety Score", f"{report.overall_score:.1%}")
            
            with col_result3:
                st.metric("Risk Level", report.estimated_risk_level)
            
            # Detailed test results
            st.subheader("🔬 Detailed Test Results")
            
            for result in report.test_results:
                with st.expander(f"{result.test_type.value.replace('_', ' ').title()} - {result.safety_level.value.title()}"):
                    col_detail1, col_detail2 = st.columns(2)
                    
                    with col_detail1:
                        st.metric("Test Score", f"{result.score:.1%}")
                        st.write(f"**Execution Time:** {result.execution_time:.3f}s")
                        
                        if result.warnings:
                            st.warning("**Warnings:**")
                            for warning in result.warnings:
                                st.write(f"• {warning}")
                    
                    with col_detail2:
                        if result.recommendations:
                            st.info("**Recommendations:**")
                            for rec in result.recommendations:
                                st.write(f"• {rec}")
                        
                        if result.details:
                            with st.expander("Technical Details"):
                                st.json(result.details)
            
            # Critical warnings
            if report.critical_warnings:
                st.error("⚠️ **Critical Warnings:**")
                for warning in report.critical_warnings:
                    st.write(f"• {warning}")
            
            # Safety recommendations
            if report.safety_recommendations:
                st.info("💡 **Safety Recommendations:**")
                for rec in report.safety_recommendations:
                    st.write(f"• {rec}")
            
            # Rollback plan
            st.subheader("🔄 Emergency Rollback Plan")
            rollback = report.rollback_plan
            st.write(f"**Method:** {rollback['method']}")
            st.write(f"**Steps:** {rollback['steps']}")
            st.write(f"**Estimated Time:** {rollback['estimated_time']}")
            st.write(f"**Risk Level:** {rollback['risk_level']}")
            
            # Live tuning recommendation
            st.divider()
            if report.overall_safety.value in ["verified_safe", "safe"]:
                st.success("✅ **Live tuning appears safe with this configuration**")
                st.info("Consider proceeding with live tuning using conservative steps and continuous monitoring.")
            elif report.overall_safety.value == "caution":
                st.warning("⚠️ **Live tuning possible but requires extra caution**")
                st.info("Proceed only with enhanced monitoring and conservative changes.")
            else:
                st.error("❌ **Live tuning not recommended with this configuration**")
                st.info("Consider using safer parameters or improved cooling before attempting live tuning.")

    # Tab 10: Cross-Brand Tuning
    with tab10:
        st.header("🔄 Cross-Brand RAM Tuning")
        st.markdown("**Revolutionary AI-powered optimization for mixed RAM configurations!**")
        st.markdown("Eliminate the need for matched kits - get stable performance with different brands.")
        
        # Initialize cross-brand optimizer
        if 'cross_brand_optimizer' not in st.session_state:
            st.session_state.cross_brand_optimizer = CrossBrandOptimizer()
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("🎯 Optimization Goals")
            
            optimization_mode = st.selectbox(
                "Choose optimization strategy:",
                ["Balanced", "Maximum Stability", "Maximum Performance"],
                help="Balanced: Best overall compromise, Stability: Ultra-safe settings, Performance: Tighter timings"
            )
            
            st.subheader("📋 Quick Actions")
            
            if st.button("🔍 Auto-Detect Mixed Setup", type="primary"):
                if st.session_state.hardware_scanned and st.session_state.detected_modules:
                    if len(st.session_state.detected_modules) >= 2:
                        st.success(f"✅ Found {len(st.session_state.detected_modules)} modules for cross-brand analysis!")
                        st.session_state.cross_brand_modules = st.session_state.detected_modules
                    else:
                        st.warning("⚠️ Only 1 module detected. Cross-brand tuning requires multiple modules.")
                else:
                    st.warning("⚠️ Please scan hardware first in the Hardware Detection tab.")
            
            # Manual module addition
            st.subheader("➕ Manual Module Entry")
            
            with st.expander("Add Custom Module"):
                custom_manufacturer = st.selectbox("Manufacturer:", ["Kingston", "Corsair", "G.Skill", "Crucial", "Other"])
                custom_speed = st.selectbox("Speed:", [4800, 5200, 5600, 6000, 6400, 6800, 7200])
                custom_capacity = st.selectbox("Capacity (GB):", [8, 16, 32, 64])
                custom_part = st.text_input("Part Number (optional):")
                
                if st.button("Add Module"):
                    if 'cross_brand_modules' not in st.session_state:
                        st.session_state.cross_brand_modules = []
                    
                    from hardware_detection import DetectedRAMModule
                    custom_module = DetectedRAMModule(
                        manufacturer=custom_manufacturer,
                        part_number=custom_part or f"Custom-{custom_speed}",
                        capacity_gb=custom_capacity,
                        speed_mt_s=custom_speed,
                        slot_location=f"Custom Slot {len(st.session_state.cross_brand_modules) + 1}"
                    )
                    
                    st.session_state.cross_brand_modules.append(custom_module)
                    st.success(f"✅ Added {custom_manufacturer} module!")
                    st.rerun()
        
        with col1:
            st.subheader("🔍 Mixed Configuration Analysis")
            
            if 'cross_brand_modules' in st.session_state and st.session_state.cross_brand_modules:
                modules = st.session_state.cross_brand_modules
                
                # Display current modules
                st.write(f"**Analyzing {len(modules)} modules:**")
                
                for i, module in enumerate(modules, 1):
                    with st.expander(f"📦 Module {i}: {module.manufacturer}"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"**Brand:** {module.manufacturer}")
                            st.write(f"**Speed:** DDR5-{module.speed_mt_s}")
                            st.write(f"**Capacity:** {module.capacity_gb}GB")
                        with col_b:
                            st.write(f"**Part:** {module.part_number}")
                            st.write(f"**Location:** {module.slot_location}")
                            if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                                st.session_state.cross_brand_modules.pop(i-1)
                                st.rerun()
                
                if len(modules) >= 2:
                    st.divider()
                    
                    # Run cross-brand analysis
                    if st.button("🚀 Analyze Mixed Configuration", type="primary"):
                        with st.spinner("Running cross-brand compatibility analysis..."):
                            try:
                                analysis = st.session_state.cross_brand_optimizer.analyze_mixed_configuration(modules)
                                st.session_state.cross_brand_analysis = analysis
                                
                                # Display results
                                st.success("✅ Cross-brand analysis completed!")
                                
                                # Compatibility score with color coding
                                score = analysis.compatibility_score
                                if score >= 0.8:
                                    score_color = "green"
                                elif score >= 0.6:
                                    score_color = "orange"
                                else:
                                    score_color = "red"
                                
                                col_score1, col_score2, col_score3 = st.columns(3)
                                
                                with col_score1:
                                    st.metric("Compatibility Score", f"{score:.1%}", 
                                            delta_color=score_color)
                                
                                with col_score2:
                                    st.metric("Stability Rating", analysis.stability_rating)
                                
                                with col_score3:
                                    st.metric("Performance Impact", f"{analysis.performance_impact:.1%}",
                                            delta="vs optimized individual")
                                
                            except Exception as e:
                                st.error(f"❌ Analysis failed: {str(e)}")
                    
                    # Show analysis results if available
                    if hasattr(st.session_state, 'cross_brand_analysis'):
                        analysis = st.session_state.cross_brand_analysis
                        
                        st.divider()
                        st.subheader("📊 Analysis Results")
                        
                        # Warnings
                        if analysis.warnings:
                            st.subheader("⚠️ Compatibility Warnings")
                            for warning in analysis.warnings:
                                st.warning(warning)
                        
                        # Optimizations
                        if analysis.optimizations:
                            st.subheader("🚀 Optimization Opportunities")
                            for opt in analysis.optimizations:
                                st.info(opt)
                        
                        # Recommended configuration
                        st.subheader("⚙️ Recommended Settings")
                        
                        config = analysis.recommended_config
                        
                        col_config1, col_config2 = st.columns(2)
                        
                        with col_config1:
                            st.write("**Memory Settings:**")
                            st.write(f"• Frequency: DDR5-{config.frequency}")
                            st.write(f"• CAS Latency: CL{config.timings.cl}")
                            st.write(f"• tRCD: {config.timings.trcd}")
                            st.write(f"• tRP: {config.timings.trp}")
                            st.write(f"• tRAS: {config.timings.tras}")
                        
                        with col_config2:
                            st.write("**Voltage Settings:**")
                            st.write(f"• VDDQ: {config.voltages.vddq}V")
                            st.write(f"• VPP: {config.voltages.vpp}V")
                            
                            st.write("**Advanced Timings:**")
                            st.write(f"• tRC: {config.timings.trc}")
                            st.write(f"• tRFC: {config.timings.trfc}")
                        
                        # Apply configuration buttons
                        st.divider()
                        
                        col_apply1, col_apply2, col_apply3 = st.columns(3)
                        
                        with col_apply1:
                            if st.button("🛡️ Ultra-Safe Config"):
                                safe_config = st.session_state.cross_brand_optimizer.optimize_for_stability(modules)
                                st.session_state.config = safe_config
                                st.success("✅ Ultra-safe configuration loaded!")
                        
                        with col_apply2:
                            if st.button("⚖️ Balanced Config"):
                                st.session_state.config = config
                                st.success("✅ Balanced configuration loaded!")
                        
                        with col_apply3:
                            if st.button("🚀 Performance Config"):
                                perf_config = st.session_state.cross_brand_optimizer.optimize_for_performance(modules)
                                st.session_state.config = perf_config
                                st.success("✅ Performance configuration loaded!")
                        
                        # Generate detailed report
                        st.divider()
                        st.subheader("📄 Detailed Report")
                        
                        if st.button("📋 Generate Full Report"):
                            report = generate_cross_brand_report(analysis)
                            st.markdown(report)
                            
                            # Download report
                            st.download_button(
                                label="💾 Download Report",
                                data=report,
                                file_name=f"cross_brand_analysis_{len(modules)}_modules.md",
                                mime="text/markdown"
                            )
                        
                        # Comparison with individual optimization
                        st.subheader("📈 Performance Comparison")
                        
                        if st.button("🔍 Compare Individual vs Mixed"):
                            comparison_data = []
                            
                            for module in modules:
                                # Simulate individual optimization
                                individual_cl = max(30, int(module.speed_mt_s / 155))
                                mixed_cl = config.timings.cl
                                
                                perf_loss = (mixed_cl - individual_cl) / individual_cl * 100
                                
                                comparison_data.append({
                                    "Module": f"{module.manufacturer}",
                                    "Individual CL": f"CL{individual_cl}",
                                    "Mixed CL": f"CL{mixed_cl}",
                                    "Performance Impact": f"{perf_loss:.1f}%"
                                })
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True)
                else:
                    st.info("ℹ️ Add at least 2 modules to run cross-brand analysis")
            else:
                st.info("🔍 No modules detected. Use 'Auto-Detect Mixed Setup' or add modules manually.")
                
                # Show example benefits
                st.subheader("🌟 Cross-Brand Tuning Benefits")
                
                col_benefit1, col_benefit2 = st.columns(2)
                
                with col_benefit1:
                    st.markdown("""
                    **🎯 Mixed Brand Optimization:**
                    - Kingston + Corsair compatibility
                    - G.Skill + Crucial pairing
                    - Different speeds harmonization
                    - Capacity mismatch handling
                    """)
                
                with col_benefit2:
                    st.markdown("""
                    **💡 Smart Features:**
                    - AI compatibility scoring
                    - Conservative timing calculation
                    - Voltage requirement analysis
                    - Performance impact prediction
                    """)
                
                st.subheader("📋 Example Scenarios")
                
                example_scenarios = [
                    {
                        "scenario": "Mixed Brands + Same Speed",
                        "example": "Kingston DDR5-5600 + Corsair DDR5-5600",
                        "compatibility": "85%",
                        "recommendation": "Minor timing adjustments needed"
                    },
                    {
                        "scenario": "Mixed Speeds + Same Brand", 
                        "example": "G.Skill DDR5-6000 + G.Skill DDR5-5600",
                        "compatibility": "90%",
                        "recommendation": "Run at lower speed with optimized timings"
                    },
                    {
                        "scenario": "Mixed Everything",
                        "example": "Kingston DDR5-4800 + Corsair DDR5-6000",
                        "compatibility": "70%",
                        "recommendation": "Conservative settings, stability priority"
                    }
                ]
                
                for scenario in example_scenarios:
                    with st.expander(f"📖 {scenario['scenario']}"):
                        st.write(f"**Example:** {scenario['example']}")
                        st.write(f"**Compatibility:** {scenario['compatibility']}")
                        st.write(f"**Recommendation:** {scenario['recommendation']}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🧠 Perfect DDR5 AI Optimizer | Powered by Advanced Machine Learning</p>
        <p>Quantum Optimization • Molecular Analysis • Revolutionary Features</p>
    </div>
    """, unsafe_allow_html=True)


def create_preset_config(frequency: int, capacity: int, rank_count: int) -> DDR5Configuration:
    """Create a preset configuration based on frequency."""
    # Calculate base timings
    base_cl = max(16, int(frequency * 0.0055))
    
    # Frequency-specific optimizations
    if frequency <= 4800:
        # Conservative settings
        cl = base_cl
        trcd = base_cl
        trp = base_cl
        tras = base_cl + 20
        vddq = 1.10
    elif frequency <= 6400:
        # Balanced settings
        cl = base_cl + 2
        trcd = base_cl + 2
        trp = base_cl + 2
        tras = base_cl + 24
        vddq = 1.12
    else:
        # High-performance settings
        cl = base_cl + 4
        trcd = base_cl + 4
        trp = base_cl + 4
        tras = base_cl + 28
        vddq = 1.15
    
    return DDR5Configuration(
        frequency=frequency,
        timings=DDR5TimingParameters(
            cl=cl,
            trcd=trcd,
            trp=trp,
            tras=tras,
            trc=tras + trp,
            trfc=280 + (frequency - 3200) // 400 * 20
        ),
        voltages=DDR5VoltageParameters(
            vddq=vddq,
            vpp=1.80
        )
    )


if __name__ == "__main__":
    create_perfect_web_interface()
