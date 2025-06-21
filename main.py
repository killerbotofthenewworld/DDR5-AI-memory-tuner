#!/usr/bin/env python3
"""
Perfect DDR5 AI Sandbox Simulator
The ultimate AI-powered DDR5 memory tuning simulator.
"""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from src.web_interface import create_perfect_web_interface


def main():
    """Main entry point for the Perfect DDR5 AI Simulator."""
    create_perfect_web_interface()


if __name__ == "__main__":
    main()
