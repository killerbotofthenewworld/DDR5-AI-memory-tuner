# DDR5 AI Sandbox Simulator – Windows Installation

This provides a real Windows installer (no mocks).

What it does:

- Installs to `%LOCALAPPDATA%\DDR5-AI-Sandbox-Simulator`
- Creates a Python virtual environment and installs dependencies from `requirements.txt`
- Adds Start Menu and Desktop shortcuts to launch the app
- Registers an uninstaller entry (per-user) in Add/Remove Programs

## Prerequisites

- Windows 10/11
- Python 3.9+ on PATH (install from <https://www.python.org/downloads/>)

## Install

- Double-click `windows/install.bat` (or right-click `install.ps1` > Run with PowerShell)
- Follow console output; no admin required

## Run

- Use the Desktop shortcut: "DDR5 AI Sandbox Simulator"
- Or Start Menu: Programs > DDR5 AI Sandbox Simulator
- Or run `%LOCALAPPDATA%\DDR5-AI-Sandbox-Simulator\run_ddr5_simulator.bat`

The app opens at: <http://localhost:8521>

## Uninstall

- Open “Add or Remove Programs”, find "DDR5 AI Sandbox Simulator" and click Uninstall
- Or run `%LOCALAPPDATA%\DDR5-AI-Sandbox-Simulator\uninstall.ps1` with PowerShell

## Notes

- First run may take a while while dependencies install.
- If your company policy blocks PowerShell scripts, run `install.bat`.
- If Streamlit’s default port is busy, edit `run_ddr5_simulator.bat` and change `--server.port`.
