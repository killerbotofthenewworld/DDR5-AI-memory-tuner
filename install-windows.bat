@echo off
REM One-click Windows installer wrapper
pushd %~dp0
powershell -ExecutionPolicy Bypass -File "%~dp0windows\install.ps1"
popd
