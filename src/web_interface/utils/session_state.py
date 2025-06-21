"""
Session state management utilities for the web interface.
"""

import streamlit as st
from src.ddr5_simulator import DDR5Simulator
from src.perfect_ai_optimizer import PerfectDDR5Optimizer


def initialize_session_state():
    """Initialize all session state variables."""
    # Core components
    if 'simulator' not in st.session_state:
        st.session_state.simulator = DDR5Simulator()
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = PerfectDDR5Optimizer()
    if 'ai_trained' not in st.session_state:
        st.session_state.ai_trained = False

    # Hardware detection
    if 'detected_modules' not in st.session_state:
        st.session_state.detected_modules = []
    if 'hardware_scanned' not in st.session_state:
        st.session_state.hardware_scanned = False

    # Live tuning
    if 'live_tuner' not in st.session_state:
        st.session_state.live_tuner = None
    if 'monitoring_active' not in st.session_state:
        st.session_state.monitoring_active = False

    # Cross-brand optimization
    if 'cross_brand_optimizer' not in st.session_state:
        st.session_state.cross_brand_optimizer = None
