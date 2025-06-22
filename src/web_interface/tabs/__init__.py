"""
Tabs package for web interface tab components.
"""

try:
    from .simulation import render_simulation_tab
except ImportError:
    from simulation import render_simulation_tab

__all__ = ['render_simulation_tab']
