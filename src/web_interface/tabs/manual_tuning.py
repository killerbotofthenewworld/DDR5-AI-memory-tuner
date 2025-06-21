"""
Manual tuning tab for DDR5 parameters.
"""

import streamlit as st
from src.ddr5_models import (
    DDR5Configuration, DDR5TimingParameters, DDR5VoltageParameters
)


def render_manual_tuning_tab(config: DDR5Configuration) -> DDR5Configuration:
    """
    Render the manual tuning tab.
    
    Args:
        config: Current DDR5 configuration
        
    Returns:
        Updated DDR5 configuration
    """
    st.header("⚙️ Manual DDR5 Timing Configuration")
    st.info("🎯 **Direct Control**: Manually adjust all DDR5 timing parameters for precise tuning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Primary Timings")
        
        # Calculate intelligent defaults based on frequency
        base_cl = max(16, int(config.frequency * 0.0055))
        default_trcd = max(16, int(config.frequency * 0.006))
        default_trp = max(16, int(config.frequency * 0.0055))
        default_tras = max(32, int(config.frequency * 0.0125))
        
        manual_cl = st.number_input(
            "CL (CAS Latency)", 
            min_value=14, max_value=60, 
            value=config.timings.cl,
            help="CAS Latency - Lower is faster but less stable"
        )
        
        manual_trcd = st.number_input(
            "tRCD (RAS to CAS Delay)", 
            min_value=14, max_value=60, 
            value=config.timings.trcd,
            help="Time between row activation and column access"
        )
        
        manual_trp = st.number_input(
            "tRP (Row Precharge)", 
            min_value=14, max_value=60, 
            value=config.timings.trp,
            help="Time to precharge a row before accessing new row"
        )
        
        manual_tras = st.number_input(
            "tRAS (Row Active Strobe)", 
            min_value=28, max_value=80, 
            value=config.timings.tras,
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
            value=config.timings.trfc,
            help="Refresh cycle time - affects how often memory refreshes"
        )
        
        manual_tfaw = st.number_input(
            "tFAW (Four Activate Window)", 
            min_value=16, max_value=50, 
            value=max(16, int(config.frequency * 0.008)),
            help="Time window for four row activations"
        )
        
        manual_trrds = st.number_input(
            "tRRD_S (Row-to-Row Delay Same)", 
            min_value=4, max_value=12, 
            value=max(4, int(config.frequency * 0.0015)),
            help="Delay between row activations in same bank group"
        )
        
        manual_trrdl = st.number_input(
            "tRRD_L (Row-to-Row Delay Long)", 
            min_value=4, max_value=12, 
            value=max(6, int(config.frequency * 0.0018)),
            help="Delay between row activations in different bank groups"
        )
        
        manual_tccdl = st.number_input(
            "tCCD_L (Column-to-Column Long)", 
            min_value=4, max_value=12, 
            value=max(5, int(config.frequency * 0.0015)),
            help="Column access delay for different bank groups"
        )
    
    # Voltage Settings
    st.subheader("⚡ Voltage Configuration")
    vol_col1, vol_col2, vol_col3 = st.columns(3)
    
    with vol_col1:
        manual_vddq = st.number_input(
            "VDDQ (Core Voltage)",
            min_value=1.05, max_value=1.35, 
            value=config.voltages.vddq, step=0.01,
            help="Core voltage - affects stability and power consumption"
        )
    
    with vol_col2:
        manual_vpp = st.number_input(
            "VPP (Peripheral Voltage)",
            min_value=1.70, max_value=1.95, 
            value=config.voltages.vpp, step=0.01,
            help="Peripheral voltage - affects I/O operations"
        )
    
    with vol_col3:
        # Show calculated power consumption
        power_estimate = (
            (manual_vddq * 8.0) + (manual_vpp * 2.0) +
            (config.frequency / 1000.0 * 1.5)
        )
        st.metric(
            "Est. Power (W)",
            f"{power_estimate:.1f}",
            help="Estimated power consumption"
        )
    
    # Validation and Warnings
    st.subheader("⚠️ Configuration Validation")
    
    warnings = []
    errors = []
    
    # Check timing relationships
    if manual_tras < (manual_trcd + manual_cl):
        errors.append("tRAS must be >= tRCD + CL")
    
    if manual_trc < manual_tras:
        errors.append("tRC must be >= tRAS")
    
    # Check voltage limits
    if manual_vddq > 1.25:
        warnings.append("VDDQ > 1.25V may reduce memory lifespan")
    
    if manual_vpp > 1.85:
        warnings.append("VPP > 1.85V may cause instability")
    
    # Check frequency vs timing ratios
    cl_ns = (manual_cl / config.frequency) * 2000
    if cl_ns < 13.0:
        warnings.append(f"CL timing very aggressive ({cl_ns:.1f}ns)")
    
    # Display validation results
    if errors:
        for error in errors:
            st.error(f"❌ {error}")
    
    if warnings:
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
    
    if not errors and not warnings:
        st.success("✅ Configuration validated successfully")
    
    # Memory timing comparison
    st.subheader("📊 Timing Analysis")
    
    timing_col1, timing_col2 = st.columns(2)
    
    with timing_col1:
        st.metric(
            "Real CL (ns)",
            f"{(manual_cl / config.frequency) * 2000:.2f}",
            help="Actual CAS latency in nanoseconds"
        )
        
        st.metric(
            "Real tRCD (ns)",
            f"{(manual_trcd / config.frequency) * 2000:.2f}",
            help="Actual RAS-to-CAS delay in nanoseconds"
        )
    
    with timing_col2:
        st.metric(
            "Real tRP (ns)",
            f"{(manual_trp / config.frequency) * 2000:.2f}",
            help="Actual row precharge time in nanoseconds"
        )
        
        st.metric(
            "Real tRAS (ns)",
            f"{(manual_tras / config.frequency) * 2000:.2f}",
            help="Actual row active time in nanoseconds"
        )
    
    # Apply manual settings button
    if st.button("✅ Apply Manual Settings", type="primary"):
        # Create updated configuration
        timings = DDR5TimingParameters(
            cl=manual_cl,
            trcd=manual_trcd,
            trp=manual_trp,
            tras=manual_tras,
            trc=manual_trc,
            trfc=manual_trfc
        )
        
        voltages = DDR5VoltageParameters(
            vddq=manual_vddq,
            vpp=manual_vpp
        )
        
        updated_config = DDR5Configuration(
            frequency=config.frequency,
            capacity=config.capacity,
            rank_count=config.rank_count,
            timings=timings,
            voltages=voltages
        )
        
        if not errors:
            st.session_state.manual_config = updated_config
            st.success("🎯 Manual configuration applied!")
            return updated_config
        else:
            st.error("Cannot apply configuration with errors")
            return config
    
    return config
