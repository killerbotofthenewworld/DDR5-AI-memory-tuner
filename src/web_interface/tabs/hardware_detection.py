"""
Hardware Detection tab functionality.
"""

import streamlit as st


def render_hardware_detection_tab():
    """Render the Hardware Detection tab."""
    st.header("💻 Hardware Detection")
    
    # Hardware detection button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Scan System Hardware", type="primary",
                     use_container_width=True):
            with st.spinner("Scanning system hardware..."):
                _perform_hardware_scan()
    
    # Display detected hardware
    if (hasattr(st.session_state, 'hardware_scanned') and
            st.session_state.hardware_scanned):
        
        if (hasattr(st.session_state, 'detected_modules') and
                st.session_state.detected_modules):
            
            st.subheader("🔧 Detected Memory Modules")
            _display_detected_modules()
        
        else:
            _display_system_info()
    
    else:
        _display_instructions()


def _perform_hardware_scan():
    """Perform hardware detection scan."""
    try:
        from src.hardware_detection import detect_system_memory
        
        # Perform hardware detection
        detected_modules = detect_system_memory()
        st.session_state.detected_modules = detected_modules
        st.session_state.hardware_scanned = True
        
        if detected_modules:
            st.success(f"✅ Detected {len(detected_modules)} "
                       f"memory module(s)!")
        else:
            st.warning("⚠️ No compatible DDR5 modules detected")
            
    except ImportError:
        st.error("Hardware detection module not available")
        st.session_state.hardware_scanned = False
    except Exception as e:
        st.error(f"Hardware detection failed: {str(e)}")
        st.session_state.hardware_scanned = False


def _display_detected_modules():
    """Display detected memory modules."""
    for i, module in enumerate(st.session_state.detected_modules):
        module_name = f"{module.manufacturer} {module.part_number}"
        with st.expander(f"📋 Module {i+1}: {module_name}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Information**")
                st.write(f"• **Manufacturer**: {module.manufacturer}")
                st.write(f"• **Part Number**: {module.part_number}")
                st.write(f"• **Capacity**: {module.capacity_gb}GB")
                st.write(f"• **Speed**: DDR5-{module.speed_mt_s}")
                st.write(f"• **Slot**: {module.slot_location}")
            
            with col2:
                st.write("**Technical Details**")
                st.write(f"• **Serial**: {module.serial_number or 'N/A'}")
                st.write(f"• **Voltage**: {module.voltage or 'N/A'}V")
                st.write(f"• **Form Factor**: {module.form_factor}")
                st.write(f"• **Chip Type**: {module.chip_type or 'N/A'}")
            
            # Performance prediction
            st.write("**Performance Estimation**")
            bandwidth = (module.speed_mt_s * 64) / 8 / 1000  # GB/s per module
            st.metric("Theoretical Bandwidth", f"{bandwidth:.1f} GB/s")
            
            # Optimization suggestions
            st.write("**Optimization Suggestions**")
            if module.speed_mt_s >= 6000:
                st.success("🚀 High-performance module - great for overclocking")
            elif module.speed_mt_s >= 5000:
                st.info("⚡ Good performance - suitable for most applications")
            else:
                st.warning("🔋 Basic performance - consider upgrade")
            
            # Load configuration button
            if st.button(f"🔄 Load Configuration from Module {i+1}",
                         key=f"load_config_{i}"):
                _load_module_config(module, i)


def _load_module_config(module, module_index):
    """Load configuration from detected module."""
    try:
        # Create configuration from detected module
        from src.ddr5_models import (
            DDR5Configuration, DDR5TimingParameters,
            DDR5VoltageParameters
        )
        
        # Calculate reasonable defaults based on detected speed
        base_cl = max(16, int(module.speed_mt_s * 0.0055))
        
        config = DDR5Configuration(
            frequency=module.speed_mt_s,
            capacity=module.capacity_gb,
            rank_count=1,  # Default, as we can't detect this easily
            timings=DDR5TimingParameters(
                cl=base_cl,
                trcd=base_cl,
                trp=base_cl,
                tras=base_cl + 20,
                trc=base_cl * 2 + 20,
                trfc=280 + (module.speed_mt_s - 4000) // 400 * 20
            ),
            voltages=DDR5VoltageParameters(
                vddq=module.voltage or 1.10,
                vpp=1.80
            )
        )
        
        st.session_state.manual_config = config
        st.success(f"✅ Loaded configuration from Module {module_index+1}")
        
        # Show loaded configuration details
        with st.expander("Loaded Configuration Details"):
            st.json({
                "Frequency": f"{config.frequency} MT/s",
                "Capacity": f"{config.capacity}GB",
                "Timings": f"{config.timings.cl}-{config.timings.trcd}-{config.timings.trp}-{config.timings.tras}",
                "Voltages": f"VDDQ: {config.voltages.vddq}V, VPP: {config.voltages.vpp}V"
            })
        
    except Exception as e:
        st.error(f"Failed to load configuration: {str(e)}")
    except Exception as e:
        st.error(f"Failed to load configuration: {str(e)}")


def _display_system_info():
    """Display system information when no DDR5 modules are detected."""
    st.info("ℹ️ No DDR5 modules detected in this system")
    
    # Show system information instead
    st.subheader("💻 System Information")
    
    try:
        import platform
        import psutil
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**System Details**")
            st.write(f"• **OS**: {platform.system()} {platform.release()}")
            st.write(f"• **Architecture**: {platform.machine()}")
            st.write(f"• **Processor**: {platform.processor()}")
            
        with col2:
            st.write("**Memory Information**")
            memory = psutil.virtual_memory()
            total_gb = memory.total // (1024**3)
            available_gb = memory.available // (1024**3)
            st.write(f"• **Total RAM**: {total_gb} GB")
            st.write(f"• **Available**: {available_gb} GB")
            st.write(f"• **Usage**: {memory.percent}%")
            
    except ImportError:
        st.info("System information not available")


def _display_instructions():
    """Display instructions when no hardware scan has been performed."""
    st.subheader("🔍 Hardware Detection Instructions")
    
    st.info("""
    **How Hardware Detection Works:**
    
    1. **Click 'Scan System Hardware'** to detect your DDR5 modules
    2. **View detected modules** with their specifications
    3. **Load configurations** directly from your hardware
    4. **Simulate performance** with real-world settings
    """)
    
    st.subheader("📋 Supported Hardware")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        **✅ Supported Memory Types**
        - DDR5 Desktop Modules
        - DDR5 SO-DIMM (Laptops)
        - Registered DDR5 (Servers)
        - ECC DDR5 Modules
        """)
    
    with col2:
        st.write("""
        **🔧 Detection Methods**
        - SPD (Serial Presence Detect)
        - System BIOS/UEFI
        - Memory Controller Registers
        - Operating System APIs
        """)
    
    st.subheader("⚠️ Important Notes")
    st.warning("""
    - Hardware detection requires appropriate permissions
    - Some systems may not expose all memory information
    - Virtual machines may show limited or no hardware details
    - The simulator works without hardware detection
    """)
    
    # Placeholder hardware examples
    st.subheader("📊 Example Hardware Profiles")
    
    example_modules = [
        {
            "name": "Corsair Dominator Platinum RGB",
            "speed": "DDR5-6000",
            "capacity": "32GB (2x16GB)",
            "timings": "CL36-39-39-76",
            "voltage": "1.35V"
        },
        {
            "name": "G.Skill Trident Z5 RGB",
            "speed": "DDR5-5600",
            "capacity": "32GB (2x16GB)",
            "timings": "CL36-36-36-76",
            "voltage": "1.25V"
        },
        {
            "name": "Kingston Fury Beast",
            "speed": "DDR5-4800",
            "capacity": "16GB (2x8GB)",
            "timings": "CL40-40-40-77",
            "voltage": "1.1V"
        }
    ]
    
    for module in example_modules:
        with st.expander(f"📋 {module['name']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Speed**: {module['speed']}")
                st.write(f"**Capacity**: {module['capacity']}")
            
            with col2:
                st.write(f"**Timings**: {module['timings']}")
                st.write(f"**Voltage**: {module['voltage']}")
