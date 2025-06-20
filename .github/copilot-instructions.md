<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# DDR5 AI Sandbox Simulator - Copilot Instructions

This project is an AI-powered simulator for fine-tuning DDR5 memory parameters. When working on this project:

## Project Context
- This is a machine learning application focused on DDR5 memory optimization
- The simulator models DDR5 JEDEC specifications and uses AI to find optimal timings
- Users should be able to tune memory without physical hardware

## Code Style Guidelines
- Use type hints for all Python functions
- Follow PEP 8 conventions with black formatter
- Document all classes and functions with docstrings
- Use dataclasses or Pydantic models for structured data

## DDR5 Technical Context
- DDR5 operates at frequencies from 3200 MT/s to 8400+ MT/s
- Key timings include: CL, tRCD, tRP, tRAS, tRC, tRFC
- Voltage parameters: VDDQ (1.1V nominal), VPP (1.8V nominal)
- Memory channels: Dual-channel, with sub-channels per DIMM

## AI/ML Considerations
- Use regression models for performance prediction
- Implement reinforcement learning for timing optimization
- Consider genetic algorithms for parameter exploration
- Validate models against known DDR5 performance data

## Safety and Validation
- Always validate timing relationships (e.g., tRAS >= tRCD + tCL)
- Implement bounds checking for all parameters
- Include stability scoring for parameter combinations
- Warn users about potentially unstable configurations
