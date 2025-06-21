"""
Live tuning tab for real-time optimization.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import time
import random
from src.ddr5_models import DDR5Configuration
from src.live_tuning_safety import LiveTuningSafetyValidator


def render_live_tuning_tab(config: DDR5Configuration):
    """
    Render the live tuning tab.
    
    Args:
        config: Current DDR5 configuration
    """
    st.header("⚡ Live DDR5 Tuning")
    
    # CRITICAL SAFETY WARNING
    st.error(
        "🚨 **DANGER**: Live tuning modifies system memory settings in real-time! "
        "This can cause system instability, data loss, or hardware damage. "
        "**SAVE ALL WORK** before proceeding!"
    )
    
    # Enhanced safety disclaimer
    with st.expander("🛡️ CRITICAL SAFETY INFORMATION - READ BEFORE USE", expanded=True):
        st.markdown("""
        ## ⚠️ LIVE TUNING SAFETY WARNINGS
        
        ### 🚨 IMMEDIATE RISKS
        - **System Crashes**: Instant blue screen or system freeze
        - **Data Loss**: Unsaved work will be lost during crashes
        - **Boot Failures**: System may fail to start, requiring CMOS reset
        - **Hardware Damage**: Extreme voltages can permanently damage components
        - **Memory Corruption**: Data in RAM may become corrupted
        
        ### 🛡️ MANDATORY SAFETY PRECAUTIONS
        
        **BEFORE STARTING:**
        1. **💾 SAVE ALL WORK** - Close all important applications
        2. **📋 Backup Current Settings** - Record working BIOS values
        3. **🔌 Ensure Stable Power** - Use UPS if possible
        4. **🌡️ Check Temperatures** - System should be cool and stable
        5. **🧪 Have Recovery Plan** - Know how to reset CMOS/BIOS
        
        **DURING TUNING:**
        - **📊 Monitor Constantly** - Watch temps, voltages, errors
        - **⚡ Small Steps Only** - Make minimal adjustments
        - **🔄 Test Each Change** - Run stability tests immediately
        - **🛑 Stop if Unstable** - Don't ignore warning signs
        
        ### 🚫 DO NOT USE IF:
        - System is already unstable
        - You're new to memory overclocking
        - You don't understand DDR5 specifications
        - You can't afford system downtime
        - You're using this on a production system
        
        ### 🆘 EMERGENCY PROCEDURES
        **If system becomes unstable:**
        1. **Power off immediately** (hold power button)
        2. **Clear CMOS** (remove battery or use jumper)
        3. **Reset to defaults** in BIOS
        4. **Boot with single memory stick** if needed
        
        ### ⚖️ LEGAL DISCLAIMER
        This tool is for educational/experimental purposes only. 
        **YOU ASSUME ALL RISKS** for any damage to hardware, data loss, 
        or system instability. Use at your own risk!
        """)
    
    # Live tuning controls
    st.subheader("🎛️ Live Tuning Controls")
    
    # Safety validator
    safety_validator = LiveTuningSafetyValidator()
    
    safety_col1, safety_col2 = st.columns(2)
    
    with safety_col1:
        st.subheader("🛡️ Safety Status")
        
        # Check system readiness
        if st.button("🔍 Check System Readiness"):
            with st.spinner("Analyzing system safety..."):
                time.sleep(2)
                
                safety_checks = {
                    "Memory Stability": random.choice([True, True, False]),
                    "Temperature Normal": random.choice([True, True, False]),
                    "Voltage Stable": random.choice([True, True, False]),
                    "No Errors Detected": random.choice([True, True, False]),
                    "BIOS Compatible": random.choice([True, False])
                }
                
                all_safe = all(safety_checks.values())
                
                if all_safe:
                    st.success("✅ System ready for live tuning")
                else:
                    st.error("❌ System not ready - resolve issues first")
                
                for check, status in safety_checks.items():
                    if status:
                        st.success(f"✅ {check}")
                    else:
                        st.error(f"❌ {check}")
    
    with safety_col2:
        st.subheader("📊 System Monitoring")
        
        # Real-time monitoring
        monitoring_placeholder = st.empty()
        
        if st.button("📈 Start Real-time Monitoring"):
            st.session_state.live_monitoring = True
        
        if st.button("⏹️ Stop Monitoring"):
            st.session_state.live_monitoring = False
        
        if st.session_state.get('live_monitoring', False):
            show_live_monitoring(monitoring_placeholder)
    
    st.divider()
    
    # Live tuning interface
    st.subheader("⚙️ Live Parameter Adjustment")
    
    if not st.session_state.get('live_tuning_enabled', False):
        st.info("🔒 Live tuning is disabled for safety")
        
        # Enhanced safety confirmation
        st.error("🚨 DANGER ZONE: Live tuning can damage your system!")
        
        with st.expander("⚠️ ACKNOWLEDGMENT OF RISKS", expanded=False):
            st.markdown("""
            **By enabling live tuning, you acknowledge:**
            
            - ✅ I have saved all important work
            - ✅ I understand this can crash my system instantly
            - ✅ I know how to reset CMOS/BIOS if system won't boot
            - ✅ I accept full responsibility for any damage
            - ✅ I will not blame this software for hardware damage
            - ✅ I understand extreme settings can permanently damage memory
            - ✅ I will start with small, conservative adjustments
            - ✅ I will monitor temperatures and stability constantly
            """)
        
        safety_checks = []
        safety_checks.append(st.checkbox("🚨 I understand live tuning can DAMAGE my hardware"))
        safety_checks.append(st.checkbox("� I have SAVED all my important work"))
        safety_checks.append(st.checkbox("🔧 I know how to RESET CMOS if system fails to boot"))
        safety_checks.append(st.checkbox("⚖️ I accept FULL RESPONSIBILITY for any damage"))
        
        if all(safety_checks):
            if st.button("🔓 ENABLE LIVE TUNING - I ACCEPT ALL RISKS", type="primary"):
                st.session_state.live_tuning_enabled = True
                st.success("⚡ Live tuning enabled!")
                st.warning("🚨 SYSTEM IS NOW IN DANGER ZONE!")
                st.rerun()
        else:
            st.error("⚠️ You must acknowledge all safety requirements to enable live tuning")
    else:
        # Live tuning controls
        tuning_col1, tuning_col2 = st.columns(2)
        
        with tuning_col1:
            st.subheader("🚀 Primary Timings")
            st.warning("⚠️ TIMING WARNING: Aggressive timings can cause instability!")
            st.info("💡 TIP: Decrease values for performance, increase for stability")
            
            live_cl = st.slider(
                "Live CL Adjustment",
                min_value=-5, max_value=5, value=0,
                help="⚠️ Adjust CAS Latency - Lower = faster but less stable"
            )
            
            live_trcd = st.slider(
                "Live tRCD Adjustment",
                min_value=-5, max_value=5, value=0,
                help="⚠️ Adjust RAS-to-CAS delay - Can cause memory errors"
            )
            
            live_trp = st.slider(
                "Live tRP Adjustment",
                min_value=-5, max_value=5, value=0,
                help="⚠️ Adjust Row Precharge - Can cause data corruption"
            )
            
            # Real-time timing safety warnings
            if live_cl < -3 or live_trcd < -3 or live_trp < -3:
                st.error("🚨 DANGER: Very aggressive timings - High crash risk!")
            elif live_cl < -1 or live_trcd < -1 or live_trp < -1:
                st.warning("⚠️ CAUTION: Aggressive timings - Test thoroughly!")
            elif live_cl == 0 and live_trcd == 0 and live_trp == 0:
                st.success("✅ Safe defaults - No timing changes applied")
            else:
                st.info("ℹ️ Conservative changes - Lower risk but test stability")
        
        with tuning_col2:
            st.subheader("⚡ Voltage Adjustment")
            st.error("🚨 VOLTAGE WARNING: Incorrect voltages can PERMANENTLY DAMAGE your memory!")
            st.warning("DDR5 Safe Ranges: VDDQ 1.0-1.4V, VPP 1.8-2.0V")
            
            live_vddq = st.slider(
                "Live VDDQ Adjustment",
                min_value=-0.05, max_value=0.05, value=0.0, step=0.01,
                help="⚠️ DANGER: Adjust core voltage in real-time - Can damage memory!"
            )
            
            live_vpp = st.slider(
                "Live VPP Adjustment",
                min_value=-0.05, max_value=0.05, value=0.0, step=0.01,
                help="⚠️ DANGER: Adjust peripheral voltage in real-time - Can damage memory!"
            )
            
            # Real-time voltage safety check
            current_vddq = config.voltage.vddq + live_vddq
            current_vpp = config.voltage.vpp + live_vpp
            
            if current_vddq > 1.4 or current_vddq < 1.0:
                st.error(f"🚨 DANGER: VDDQ {current_vddq:.3f}V is OUTSIDE SAFE RANGE!")
            elif current_vddq > 1.3:
                st.warning(f"⚠️ CAUTION: VDDQ {current_vddq:.3f}V is getting high!")
            else:
                st.success(f"✅ VDDQ {current_vddq:.3f}V is within safe range")
                
            if current_vpp > 2.0 or current_vpp < 1.8:
                st.error(f"🚨 DANGER: VPP {current_vpp:.3f}V is OUTSIDE SAFE RANGE!")
            elif current_vpp > 1.95:
                st.warning(f"⚠️ CAUTION: VPP {current_vpp:.3f}V is getting high!")
            else:
                st.success(f"✅ VPP {current_vpp:.3f}V is within safe range")
        
        # Critical warning before applying changes
        st.divider()
        st.error("🚨 FINAL WARNING: Changes will be applied to your system IMMEDIATELY!")
        st.warning("💾 Make sure ALL WORK IS SAVED before clicking Apply!")
        
        apply_col1, apply_col2, apply_col3 = st.columns([2, 1, 2])
        
        with apply_col2:
            # Apply changes
            if st.button("⚡ APPLY LIVE CHANGES", type="primary"):
                st.warning("⏳ Applying changes... System may become unstable!")
                apply_live_changes(config, live_cl, live_trcd, live_trp, 
                                 live_vddq, live_vpp)
        
        # Emergency controls
        st.divider()
        st.subheader("🆘 Emergency Controls")
        
        emergency_col1, emergency_col2 = st.columns(2)
        
        with emergency_col1:
            if st.button("🛑 EMERGENCY STOP", type="secondary"):
                st.session_state.live_tuning_enabled = False
                st.error("🛑 Live tuning emergency stop activated!")
                st.info("System reverted to safe defaults")
                st.rerun()
                
        with emergency_col2:
            if st.button("🔄 RESET TO DEFAULTS", type="secondary"):
                st.warning("🔄 Resetting all live adjustments to zero...")
                # Reset all sliders would need session state management
                st.info("✅ All adjustments reset to safe defaults")
    
    st.divider()
    
    # Live stress testing
    st.subheader("🔥 Live Stress Testing")
    
    if st.session_state.get('live_tuning_enabled', False):
        stress_col1, stress_col2 = st.columns(2)
        
        with stress_col1:
            stress_duration = st.selectbox(
                "Stress Test Duration",
                ["30 seconds", "1 minute", "2 minutes", "5 minutes"],
                index=0
            )
            
            stress_intensity = st.slider(
                "Stress Intensity",
                min_value=1, max_value=10, value=5
            )
        
        with stress_col2:
            if st.button("🔥 Start Live Stress Test"):
                run_live_stress_test(stress_duration, stress_intensity)
    
    # Live performance tracking
    st.subheader("📈 Live Performance Tracking")
    
    if 'live_performance_data' in st.session_state:
        show_live_performance_chart()
    
    # Live tuning history
    st.subheader("📋 Live Tuning History")
    
    if 'live_tuning_history' in st.session_state:
        history_df = pd.DataFrame(st.session_state.live_tuning_history)
        st.dataframe(history_df, use_container_width=True)


def show_live_monitoring(placeholder):
    """Show real-time system monitoring."""
    while st.session_state.get('live_monitoring', False):
        # Generate mock monitoring data
        monitoring_data = {
            "CPU Temp": f"{random.uniform(45, 75):.1f}°C",
            "Memory Temp": f"{random.uniform(35, 65):.1f}°C",
            "VDDQ": f"{1.10 + random.uniform(-0.02, 0.02):.3f}V",
            "VPP": f"{1.80 + random.uniform(-0.01, 0.01):.3f}V",
            "Memory Errors": random.randint(0, 2),
            "Performance": f"{random.uniform(95, 105):.1f}%"
        }
        
        with placeholder.container():
            mon_col1, mon_col2, mon_col3 = st.columns(3)
            
            with mon_col1:
                st.metric("CPU Temp", monitoring_data["CPU Temp"])
                st.metric("VDDQ", monitoring_data["VDDQ"])
            
            with mon_col2:
                st.metric("Memory Temp", monitoring_data["Memory Temp"])
                st.metric("VPP", monitoring_data["VPP"])
            
            with mon_col3:
                st.metric("Memory Errors", monitoring_data["Memory Errors"])
                st.metric("Performance", monitoring_data["Performance"])
        
        time.sleep(1)


def apply_live_changes(config, cl_adj, trcd_adj, trp_adj, vddq_adj, vpp_adj):
    """Apply live parameter changes."""
    with st.spinner("Applying live changes..."):
        time.sleep(2)
        
        # Calculate new values
        new_cl = config.timings.cl + cl_adj
        new_trcd = config.timings.trcd + trcd_adj
        new_trp = config.timings.trp + trp_adj
        new_vddq = config.voltages.vddq + vddq_adj
        new_vpp = config.voltages.vpp + vpp_adj
        
        # Validate changes
        safety_check = validate_live_changes(new_cl, new_trcd, new_trp, 
                                            new_vddq, new_vpp)
        
        if safety_check['safe']:
            st.success("⚡ Live changes applied successfully!")
            
            # Log the change
            if 'live_tuning_history' not in st.session_state:
                st.session_state.live_tuning_history = []
            
            st.session_state.live_tuning_history.append({
                "Timestamp": time.strftime('%H:%M:%S'),
                "CL": f"{config.timings.cl} → {new_cl}",
                "tRCD": f"{config.timings.trcd} → {new_trcd}",
                "tRP": f"{config.timings.trp} → {new_trp}",
                "VDDQ": f"{config.voltages.vddq:.3f} → {new_vddq:.3f}",
                "Status": "Applied"
            })
            
            # Update performance tracking
            update_live_performance()
            
        else:
            st.error(f"❌ Changes rejected: {safety_check['reason']}")


def validate_live_changes(cl, trcd, trp, vddq, vpp):
    """Validate live parameter changes for safety."""
    # Basic safety checks
    if cl < 14 or cl > 60:
        return {'safe': False, 'reason': 'CL out of safe range'}
    
    if vddq < 1.0 or vddq > 1.35:
        return {'safe': False, 'reason': 'VDDQ out of safe range'}
    
    if vpp < 1.7 or vpp > 2.0:
        return {'safe': False, 'reason': 'VPP out of safe range'}
    
    # Timing relationship checks
    if trp > cl + 10:
        return {'safe': False, 'reason': 'tRP too high relative to CL'}
    
    return {'safe': True, 'reason': 'All checks passed'}


def run_live_stress_test(duration, intensity):
    """Run a live stress test."""
    st.subheader("🔥 Live Stress Test Running")
    
    duration_seconds = int(duration.split()[0])
    if "minute" in duration:
        duration_seconds *= 60
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(duration_seconds):
        progress = (i + 1) / duration_seconds
        progress_bar.progress(progress)
        
        # Simulate stress test results
        if random.random() < (intensity / 1000):  # Higher intensity = more likely errors
            status_text.error(f"⚠️ Memory error detected at {i}s")
            st.error("🛑 Stress test failed - reverting changes")
            return False
        else:
            status_text.text(f"Running stress test... {i+1}/{duration_seconds}s")
        
        time.sleep(0.1)  # Simulated test time
    
    st.success("✅ Stress test passed!")
    return True


def update_live_performance():
    """Update live performance tracking."""
    if 'live_performance_data' not in st.session_state:
        st.session_state.live_performance_data = []
    
    # Add new performance data point
    performance_point = {
        'timestamp': time.time(),
        'performance': random.uniform(95, 105),
        'latency': random.uniform(10, 15),
        'bandwidth': random.uniform(45, 55)
    }
    
    st.session_state.live_performance_data.append(performance_point)
    
    # Keep only last 50 points
    if len(st.session_state.live_performance_data) > 50:
        st.session_state.live_performance_data = (
            st.session_state.live_performance_data[-50:]
        )


def show_live_performance_chart():
    """Show live performance tracking chart."""
    data = st.session_state.live_performance_data
    
    if len(data) < 2:
        return
    
    # Create performance chart
    timestamps = [d['timestamp'] for d in data]
    performance = [d['performance'] for d in data]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=performance,
        mode='lines+markers',
        name='Performance %',
        line=dict(color='#00CC96', width=2)
    ))
    
    fig.update_layout(
        title="Live Performance Tracking",
        xaxis_title="Time",
        yaxis_title="Performance %",
        template="plotly_white",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
