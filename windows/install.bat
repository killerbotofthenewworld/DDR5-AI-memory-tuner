@echo off
REM Windows helper to run the PowerShell installer with bypass policy
powershell -ExecutionPolicy Bypass -File "%~dp0install.ps1"
