"""
Gaming Performance tab functionality.
"""

import streamlit as st
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


def render_gaming_tab(config=None, enable_manual=False, frequency=5600,
                      capacity=16, rank_count=1, cl=36, trcd=39, trp=39,
                      tras=76, trc=76, trfc=295, vddq=1.25, vpp=1.8):
    """Render the Gaming Performance tab."""
    st.header("🎮 Gaming Performance Predictor")
    st.info("🎯 **Real-World Gaming**: See how your DDR5 configuration affects "
            "actual gaming performance")
    
    # Import gaming performance predictor
    try:
        from src.gaming_performance import GamingPerformancePredictor
        gaming_predictor = GamingPerformancePredictor()
        
        # Get current configuration
        if (hasattr(st.session_state, 'manual_config') and
                st.session_state.manual_config):
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
            gaming_config = create_preset_config(
                frequency, capacity, rank_count)
        
        # Resolution selector
        col1, col2, col3 = st.columns(3)
        with col1:
            resolution = st.selectbox(
                "Resolution", ["1080p", "1440p", "4k"], index=0)
        with col2:
            show_competitive = st.checkbox(
                "Focus on Competitive Games", value=True)
        with col3:
            show_improvements = st.checkbox("Show Memory Impact", value=True)
        
        # Get gaming predictions
        gaming_results = gaming_predictor.predict_gaming_performance(
            gaming_config, resolution)
        gaming_score = gaming_predictor.calculate_gaming_score(gaming_config)
        recommendations = gaming_predictor.get_gaming_recommendations(
            gaming_config)
        
        # Display overall gaming score
        delta_text = ("Optimized for gaming" if gaming_score > 85 else
                      "Room for improvement")
        st.metric("🏆 Overall Gaming Score", f"{gaming_score:.1f}/100",
                  delta=delta_text)
        
        # Filter games if competitive focus is enabled
        if show_competitive:
            competitive_games = [
                "valorant", "counter_strike_2", "call_of_duty_warzone",
                "apex_legends", "overwatch_2", "league_of_legends"
            ]
            filtered_results = {k: v for k, v in gaming_results.items()
                                if k in competitive_games}
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
                st.markdown(f"**{result['fps']:.0f} FPS** "
                            f"({result['frame_time_ms']:.1f}ms)")
                st.markdown(f"1% Low: {result['one_percent_low']:.0f} FPS")
                
                if (show_improvements and
                        result['memory_improvement_percent'] > 0):
                    improvement_pct = result['memory_improvement_percent']
                    st.markdown(f"📈 +{improvement_pct:.1f}% vs baseline")
                
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
        st.error("Gaming performance module not available. "
                 "Please check installation.")
        
        # Show placeholder info when module is not available
        st.subheader("🎮 Gaming Performance Preview")
        st.info("Gaming performance prediction will be available when the "
                "gaming_performance module is properly installed.")
        
        # Show sample gaming data
        st.subheader("📊 Sample Gaming Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🟢 Valorant", "240 FPS", delta="Competitive Ready")
            st.metric("🟢 CS2", "180 FPS", delta="High Performance")
        
        with col2:
            st.metric("🟡 Warzone", "120 FPS", delta="Good for 1440p")
            st.metric("🟢 Apex Legends", "144 FPS", delta="Smooth Gaming")
        
        with col3:
            st.metric("🟢 Overwatch 2", "165 FPS", delta="High Refresh")
            st.metric("🟢 League of Legends", "300+ FPS", delta="Maximum")
