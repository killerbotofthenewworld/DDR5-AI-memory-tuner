# Windows EXE Installer (Inno Setup)

This folder contains an Inno Setup script (`ddr5-ai-sandbox-simulator.iss`) to build a per-user Windows installer.

Steps:

1. Install Inno Setup 6
2. Open the `.iss` file and build the installer
3. Run the generated EXE; it will place files in `%LOCALAPPDATA%\DDR5-AI-Memory-Tuner` and call `windows/install.ps1`

Notes:

- You can code sign the EXE post-build using `signtool.exe` if available
- Consider bundling wheels (offline cache) for heavy dependencies in a future iteration
