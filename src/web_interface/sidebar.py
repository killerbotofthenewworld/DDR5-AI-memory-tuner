"""
Sidebar configuration component.
"""

import streamlit as st
from typing import Tuple

from src.ddr5_models import (
    DDR5Configuration, DDR5TimingParameters, DDR5VoltageParameters
)


def render_sidebar() -> Tuple[DDR5Configuration, dict]:
    """
    Render the sidebar configuration panel.
    
    Returns:
        Tuple of (configuration, optimization_settings)
    """
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Ko-fi Support Button (Sidebar)
        if st.button("☕ Support on Ko-fi", type="secondary",
                     use_container_width=True):
            st.success("💖 Thanks!")
            st.markdown("""
            <div style='text-align: center; margin: 10px 0;'>
                <a href="https://ko-fi.com/killerbotofthenewworld" 
                   target="_blank" 
                   style='background: #FF5E5B; color: white; 
                          padding: 8px 16px; text-decoration: none; 
                          border-radius: 4px; display: inline-block; 
                          font-size: 12px;'>
                    ☕ Open Ko-fi
                </a>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Safety Mode
        st.subheader("🛡️ Safety Mode")
        safe_mode = st.checkbox("Enable Safe Mode (recommended)", value=True)
        st.session_state["safe_mode"] = safe_mode
        if safe_mode:
            st.info("Safe Mode requires a Dry-Run preflight to pass before live tuning.")
        
        st.divider()
        
        # AI Training Section
        st.subheader("🧠 AI Training")
        
        if not st.session_state.ai_trained:
            if st.button("🚀 Train Perfect AI", type="primary"):
                with st.spinner("Training advanced AI models..."):
                    training_results = (
                        st.session_state.optimizer.train_perfect_ai()
                    )
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
            [4000, 4400, 4800, 5200, 5600, 6000, 6400, 
             6800, 7200, 7600, 8000, 8400],
            index=4  # Default to DDR5-5600
        )
        
        capacity = st.selectbox(
            "Capacity per stick (GB)", [8, 16, 32, 64], index=1
        )
        rank_count = st.selectbox("Rank count", [1, 2], index=0)
        
        st.divider()
        
        # Manual Configuration
        st.subheader("⚙️ Manual Tuning")
        
        enable_manual = st.checkbox("Enable Manual Configuration")
        
        if enable_manual:
            # Calculate reasonable defaults
            base_cl = max(16, int(frequency * 0.0055))
            
            cl = st.slider(
                "CL (CAS Latency)", min_value=16, max_value=50, value=base_cl
            )
            trcd = st.slider("tRCD", min_value=16, max_value=50, value=base_cl)
            trp = st.slider("tRP", min_value=16, max_value=50, value=base_cl)
            tras = st.slider(
                "tRAS", min_value=30, max_value=80, value=base_cl + 20
            )
            trc = tras + trp  # Auto-calculated
            trfc = st.slider("tRFC", min_value=280, max_value=400, value=312)
            
            st.subheader("⚡ Voltage Settings")
            vddq = st.slider(
                "VDDQ (V)", min_value=1.05, max_value=1.25, 
                value=1.10, step=0.01
            )
            vpp = st.slider(
                "VPP (V)", min_value=1.75, max_value=1.90, 
                value=1.80, step=0.01
            )
        else:
            # Use intelligent defaults based on frequency
            base_cl = max(16, int(frequency * 0.0055))
            cl = base_cl
            trcd = base_cl
            trp = base_cl
            tras = base_cl + 20
            trc = tras + trp
            trfc = 280 + (frequency - 4000) // 400 * 20
            vddq = 1.10
            vpp = 1.80
        
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

        # Build configuration
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
        
        optimization_settings = {
            'goal': optimization_goal,
            'performance_target': performance_target
        }
        
        return config, optimization_settings
