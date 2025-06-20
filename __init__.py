"""
DDR5 AI Sandbox Simulator

The Ultimate AI-Powered DDR5 Memory Tuning Simulator Without Hardware Requirements
"""

__version__ = "3.0.0"
__author__ = "DDR5 AI Sandbox Simulator Team"
__license__ = "MIT"

from .src import ddr5_models, ddr5_simulator, ai_optimizer

__all__ = [
    "ddr5_models",
    "ddr5_simulator", 
    "ai_optimizer",
]
