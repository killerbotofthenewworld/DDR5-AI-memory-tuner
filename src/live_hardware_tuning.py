"""
Live Hardware Tuning Integration
Connects the hardware interface with the live tuning tab for real hardware control.
"""

import streamlit as st
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from src.hardware_interface import hardware_manager, SafetyState
from src.ddr5_models import DDR5Configuration
from src.hardware_detection import HardwareDetector, DetectedRAMModule


@dataclass
class LiveTuningSession:
    """Represents an active live tuning session."""
    session_id: str
    start_time: float
    original_config: DDR5Configuration
    current_config: DDR5Configuration
    changes_applied: int = 0
    safety_violations: int = 0
    emergency_stops: int = 0
    is_active: bool = True


class LiveHardwareTuner:
    """Manages live hardware tuning operations with safety controls."""
    
    def __init__(self):
        self._detector = None
        self._cached_modules = None
        self.session = None
        self.hardware_initialized = False
        self.capabilities = None
        # Lazy-initialized hardware detector and cache for detected modules
        
    def initialize_hardware(self) -> bool:
        """Initialize hardware interface and check capabilities."""
        try:
            success = hardware_manager.initialize()
        except (RuntimeError, OSError, ValueError) as e:
            st.error(f"❌ Hardware initialization failed: {e}")
            return False

        if success:
            self.capabilities = hardware_manager.capabilities
            self.hardware_initialized = True
            return True
        return False

    def detect_modules(self) -> List[DetectedRAMModule]:
        """Detect installed RAM modules using the cross-platform HardwareDetector.

        Returns:
            List of detected RAM modules (may be demo data if detection unsupported)
        """
        try:
            if self._cached_modules is not None:
                return self._cached_modules

            if self._detector is None:
                self._detector = HardwareDetector()

            modules = self._detector.detect_system_memory()
            # Cache results for the session to avoid repeated probes
            self._cached_modules = modules
            return modules
        except (RuntimeError, OSError, ValueError) as e:
            logging.getLogger(__name__).warning("Hardware detection failed: %s", e)
            return []
    
    def get_hardware_status(self) -> Dict[str, Any]:
        """Get current hardware status for display."""
        if not self.hardware_initialized:
            return {"status": "not_initialized", "message": "Hardware not initialized"}

        try:
            status = hardware_manager.get_hardware_status()
            safety_state = hardware_manager.interface.monitor_stability()
        except (RuntimeError, OSError, ValueError) as e:
            return {"status": "error", "message": f"Status check failed: {e}"}

        return {
            "status": "ready" if self._is_system_safe(safety_state) else "unsafe",
            "capabilities": status.get("capabilities", {}),
            "safety": getattr(safety_state, "__dict__", {}),
            "platform": status.get("platform", {}),
            "message": self._get_status_message(safety_state),
        }
    
    def _is_system_safe(self, safety_state: SafetyState) -> bool:
        """Check if system is safe for live tuning."""
        return (safety_state.temperature_safe and 
                safety_state.memory_stable and 
                safety_state.power_stable and
                safety_state.backup_created)
    
    def _get_status_message(self, safety_state: SafetyState) -> str:
        """Generate human-readable status message."""
        if not safety_state.temperature_safe:
            return "🌡️ Temperature too high - system not safe for tuning"
        elif not safety_state.memory_stable:
            return "💾 Memory instability detected - system not safe"
        elif not safety_state.power_stable:
            return "⚡ Power instability detected - system not safe"
        elif not safety_state.backup_created:
            return "💾 No safety backup created - create backup first"
        else:
            return "✅ System safe and ready for live tuning"
    
    def start_live_session(self, config: DDR5Configuration) -> bool:
        """Start a new live tuning session."""
        if not self.hardware_initialized:
            st.error("❌ Hardware not initialized")
            return False
        
        # Check system safety
        safety_state = hardware_manager.interface.monitor_stability()
        if not self._is_system_safe(safety_state):
            st.error("❌ System not safe for live tuning")
            return False
        
        # Create safety backup
        if not hardware_manager.create_safety_backup():
            st.error("❌ Could not create safety backup")
            return False
        
        # Create session
        session_id = f"live_session_{int(time.time())}"
        self.session = LiveTuningSession(
            session_id=session_id,
            start_time=time.time(),
            original_config=config.model_copy(),
            current_config=config.model_copy()
        )
        
        st.success(f"✅ Live tuning session started: {session_id}")
        return True
    
    def apply_live_adjustment(self, parameter: str, value: Any) -> bool:
        """Apply a single parameter adjustment to hardware."""
        if not self.session or not self.session.is_active:
            st.error("❌ No active live tuning session")
            return False
        
        try:
            # Update configuration
            self._update_config_parameter(parameter, value)

            # Convert to hardware settings format
            hardware_settings = self._config_to_hardware_settings(self.session.current_config)

            # Apply to hardware
            success = hardware_manager.apply_ddr5_settings(hardware_settings)

            if success:
                self.session.changes_applied += 1
                st.success(f"✅ Applied {parameter} = {value}")

                # Monitor for stability after change
                time.sleep(1)  # Brief delay for hardware to settle
                safety_state = hardware_manager.interface.monitor_stability()

                if not self._is_system_safe(safety_state):
                    st.warning("⚠️ System became unstable after change")
                    self.session.safety_violations += 1
                    return False

                return True
            else:
                st.error(f"❌ Failed to apply {parameter}")
                return False
        except (RuntimeError, OSError, ValueError) as e:
            st.error(f"❌ Error applying adjustment: {e}")
            return False
    
    def _update_config_parameter(self, parameter: str, value: Any):
        """Update configuration parameter."""
        if not self.session:
            raise RuntimeError("No active live tuning session")
        config = self.session.current_config
        
        # Map parameter names to config attributes
        param_mapping = {
            "cl": ("timings", "cl"),
            "trcd": ("timings", "trcd"),
            "trp": ("timings", "trp"),
            "tras": ("timings", "tras"),
            "vddq": ("voltages", "vddq"),
            "vpp": ("voltages", "vpp"),
            "frequency": ("frequency",)
        }
        
        if parameter in param_mapping:
            path = param_mapping[parameter]
            if len(path) == 1:
                setattr(config, path[0], value)
            elif len(path) == 2:
                obj = getattr(config, path[0])
                setattr(obj, path[1], value)
    
    def _config_to_hardware_settings(self, config: DDR5Configuration) -> Dict[str, Any]:
        """Convert DDR5Configuration to hardware settings format."""
        return {
            "frequency": config.frequency,
            "cl": config.timings.cl,
            "trcd": config.timings.trcd,
            "trp": config.timings.trp,
            "tras": config.timings.tras,
            "vddq": config.voltages.vddq,
            "vpp": config.voltages.vpp,
            # Add more parameters as needed
        }
    
    def emergency_stop(self) -> bool:
        """Emergency stop and restore original settings."""
        if not self.session:
            return False
        
        try:
            st.error("🚨 EMERGENCY STOP ACTIVATED")

            # Mark session as inactive
            self.session.is_active = False
            self.session.emergency_stops += 1

            # Restore from backup
            success = hardware_manager.emergency_restore()

            if success:
                st.success("✅ Emergency restore completed")
                return True
            else:
                st.error("❌ Emergency restore failed - manual intervention required")
                return False
        except (RuntimeError, OSError, ValueError) as e:
            st.error(f"❌ Emergency stop failed: {e}")
            return False
    
    def stop_session(self) -> bool:
        """Gracefully stop the live tuning session."""
        if not self.session:
            return False
        
        self.session.is_active = False
        
        # Display session summary
        duration = time.time() - self.session.start_time
        st.info(f"""
        📊 Live Tuning Session Summary:
        - Duration: {duration:.1f} seconds
        - Changes Applied: {self.session.changes_applied}
        - Safety Violations: {self.session.safety_violations}
        - Emergency Stops: {self.session.emergency_stops}
        """)
        
        self.session = None
        return True
    
    def get_session_status(self) -> Optional[Dict[str, Any]]:
        """Get current session status."""
        if not self.session:
            return None
        
        return {
            "session_id": self.session.session_id,
            "active": self.session.is_active,
            "duration": time.time() - self.session.start_time,
            "changes_applied": self.session.changes_applied,
            "safety_violations": self.session.safety_violations,
            "emergency_stops": self.session.emergency_stops
        }


# Global live tuner instance
live_hardware_tuner = LiveHardwareTuner()


def render_hardware_status_panel():
    """Render hardware status panel for live tuning."""
    st.subheader("🖥️ Hardware Status")
    
    # Initialize hardware if not done
    if not live_hardware_tuner.hardware_initialized:
        if st.button("🔌 Initialize Hardware Interface"):
            with st.spinner("Initializing hardware interface..."):
                success = live_hardware_tuner.initialize_hardware()
                if success:
                    st.success("✅ Hardware interface initialized")
                    st.rerun()
                else:
                    st.error("❌ Hardware initialization failed")
        return
    
    # Get hardware status
    status = live_hardware_tuner.get_hardware_status()
    
    # Display status
    if status["status"] == "ready":
        st.success(status["message"])
    elif status["status"] == "unsafe":
        st.error(status["message"])
    else:
        st.warning(status["message"])
    
    # Display capabilities
    with st.expander("🔧 Hardware Capabilities", expanded=False):
        caps = status.get("capabilities", {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Platform Capabilities:**")
            st.write(f"🖥️ Platform: {caps.get('platform', 'Unknown')}")
            st.write(f"👑 Admin Required: {'Yes' if caps.get('admin_required') else 'No'}")
            st.write(f"🔧 UEFI Variables: {'Yes' if caps.get('uefi_vars') else 'No'}")
        
        with col2:
            st.write("**Control Capabilities:**")
            st.write(f"💾 Memory Controller: {'Yes' if caps.get('memory_controller') else 'No'}")
            st.write(f"🏭 Vendor Tools: {'Yes' if caps.get('vendor_tools') else 'No'}")
            st.write(f"💾 Backup/Restore: {'Yes' if caps.get('backup_restore') else 'No'}")
    
    # Display safety status
    with st.expander("🛡️ Safety Status", expanded=True):
        safety = status.get("safety", {})
        
        col1, col2 = st.columns(2)
        with col1:
            temp_safe = safety.get("temperature_safe", False)
            st.write(f"🌡️ Temperature: {'✅ Safe' if temp_safe else '❌ Unsafe'}")
            
            mem_safe = safety.get("memory_stable", False)
            st.write(f"💾 Memory: {'✅ Stable' if mem_safe else '❌ Unstable'}")
        
        with col2:
            power_safe = safety.get("power_stable", False)
            st.write(f"⚡ Power: {'✅ Stable' if power_safe else '❌ Unstable'}")
            
            backup_exists = safety.get("backup_created", False)
            st.write(f"💾 Backup: {'✅ Created' if backup_exists else '❌ Missing'}")


def render_live_session_controls(config: DDR5Configuration):
    """Render live session controls."""
    st.subheader("⚡ Live Session Controls")
    
    session = live_hardware_tuner.get_session_status()
    
    if not session:
        # No active session - show start controls
        if st.button("🚀 Start Live Tuning Session", type="primary"):
            if live_hardware_tuner.start_live_session(config):
                st.rerun()
    else:
        # Active session - show session info and controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Session Duration", f"{session['duration']:.1f}s")
        with col2:
            st.metric("Changes Applied", session['changes_applied'])
        with col3:
            st.metric("Safety Violations", session['safety_violations'])
        
        # Emergency controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🛑 EMERGENCY STOP", type="secondary"):
                live_hardware_tuner.emergency_stop()
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop Session", type="secondary"):
                live_hardware_tuner.stop_session()
                st.rerun()


def apply_live_hardware_change(parameter: str, value: Any) -> bool:
    """Apply a live hardware change with safety checks."""
    return live_hardware_tuner.apply_live_adjustment(parameter, value)
