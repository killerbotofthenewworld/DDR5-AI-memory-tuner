<#
DDR5 AI Memory Tuner - Windows Installer (Real, no mocks)

What this does:
- Copies the app to %LOCALAPPDATA%\DDR5-AI-Memory-Tuner
- Creates a Python virtual environment and installs requirements
- Adds Start Menu and Desktop shortcuts
- Registers an uninstaller entry in Add/Remove Programs (per-user)

Requirements:
- Windows 10/11 with PowerShell 5.1+
- Python 3.9+ available on PATH (python or py launcher)

Run (no admin required):
  Right-click > Run with PowerShell
  or in a PowerShell prompt:
    powershell -ExecutionPolicy Bypass -File .\windows\install.ps1
#>

[CmdletBinding(SupportsShouldProcess=$true)]
param(
    [string]$InstallDir = "$env:LOCALAPPDATA\DDR5-AI-Memory-Tuner",
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info($msg){ Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Ok($msg){ Write-Host "[ OK ] $msg" -ForegroundColor Green }
function Write-Warn($msg){ Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg){ Write-Host "[ERR ] $msg" -ForegroundColor Red }

# Resolve repository root (this script lives in repo\windows)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir '..') | Select-Object -ExpandProperty Path

Write-Host "🚀 DDR5 AI Memory Tuner - Windows Installer" -ForegroundColor Magenta
Write-Host "====================================================="
Write-Info "Repo: $RepoRoot"
Write-Info "Install dir: $InstallDir"

# 1) Check Python
Write-Info "Checking Python availability..."
$pythonCmd = $null
try {
    $ver = & python --version 2>$null
    if ($LASTEXITCODE -eq 0) { $pythonCmd = 'python' }
} catch {}
if (-not $pythonCmd) {
    try {
        $ver = & py -3 --version 2>$null
        if ($LASTEXITCODE -eq 0) { $pythonCmd = 'py -3' }
    } catch {}
}
if (-not $pythonCmd) {
    Write-Err "Python 3.9+ not found on PATH. Please install from https://www.python.org/downloads/ and re-run."
    exit 1
}
Write-Ok "Using $pythonCmd ($ver)"

# 2) Copy files to InstallDir
Write-Info "Copying application files..."
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

# Prefer robocopy for speed and reliability; fallback to Copy-Item if it fails
Write-Info "Using robocopy to mirror files..."
& robocopy $RepoRoot $InstallDir /MIR /XD .git venv .pytest_cache __pycache__ screenshots .azure dist build windows\out ddr5-simulator-installer /XF *.pyc *.pyo | Out-Null
$rcExit = $LASTEXITCODE
if ($rcExit -ge 8) {
    Write-Warn "robocopy failed with exit $rcExit. Falling back to Copy-Item..."
    try {
        # Basic copy
        Copy-Item -Path (Join-Path $RepoRoot '*') -Destination $InstallDir -Recurse -Force -ErrorAction Stop
        # Remove excluded directories from destination
        $excludeDirs = @('.git','venv','.pytest_cache','__pycache__','screenshots','.azure','dist','build','windows\\out','ddr5-simulator-installer')
        foreach ($d in $excludeDirs) {
            $dirs = Get-ChildItem -Path $InstallDir -Recurse -Directory -Filter $d -ErrorAction SilentlyContinue
            foreach ($dir in $dirs) { Remove-Item -LiteralPath $dir.FullName -Recurse -Force -ErrorAction SilentlyContinue }
        }
        # Remove excluded files
        Get-ChildItem -Path $InstallDir -Recurse -Include *.pyc,*.pyo -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    }
    catch {
        Write-Err "File copy fallback failed: $($_.Exception.Message)"
        exit 1
    }
}
Write-Ok "Files copied"

# 3) Create venv and install deps
$devCleanup = @('.venv','tests','test_models','screenshots','ddr5-simulator-installer','windows\\installer')
foreach ($dc in $devCleanup) {
    $p = Join-Path $InstallDir $dc
    if (Test-Path $p) {
        Write-Warn "Removing development artifact: ${dc}"
        try {
            Remove-Item -Recurse -Force -LiteralPath $p -ErrorAction Stop
        } catch {
            $errMsg = "Could not remove ${dc}: " + $_.Exception.Message
            Write-Warn $errMsg
        }
    }
}

$VenvDir = Join-Path $InstallDir 'venv'
if ((Test-Path $VenvDir) -and $Force) {
    Write-Warn "Removing existing venv (Force)"; Remove-Item -Recurse -Force $VenvDir
}
if (!(Test-Path $VenvDir)) {
    Write-Info "Creating virtual environment..."
    & $pythonCmd -m venv "$VenvDir"
    if ($LASTEXITCODE -ne 0) { Write-Err "Failed to create venv"; exit 1 }
    Write-Ok "venv created"
}

$PyExe = Join-Path $VenvDir 'Scripts\python.exe'
$PipExe = Join-Path $VenvDir 'Scripts\pip.exe'

Write-Info "Upgrading pip..."
& $PyExe -m pip install --upgrade pip setuptools wheel

Write-Info "Installing Python dependencies (this may take a while)..."
& $PipExe install -r (Join-Path $InstallDir 'requirements.txt')
if ($LASTEXITCODE -ne 0) {
    Write-Err "pip install failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# Quick import check
Write-Info "Verifying core imports..."
& $PyExe -c "import streamlit, numpy, pandas, plotly; print('Core libs OK')" | Out-Null
Write-Ok "Dependencies installed"

# 4) Create run script
$RunBat = Join-Path $InstallDir 'run_ddr5_simulator.bat'
$batContent = @'
@echo off
setlocal enableextensions enabledelayedexpansion
echo Launching DDR5 AI Memory Tuner...
call "%~dp0venv\Scripts\activate.bat"
echo Access at: http://localhost:8521
streamlit run "%~dp0main.py" --server.port 8521
'@
Set-Content -Path $RunBat -Value $batContent -Encoding ASCII
Write-Ok "Launcher created: $RunBat"

# 5) Shortcuts (Start Menu + Desktop)
function New-Shortcut {
    param(
        [Parameter(Mandatory)] [string]$ShortcutPath,
        [Parameter(Mandatory)] [string]$TargetPath,
        [string]$Arguments,
        [string]$WorkingDirectory
    )
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut($ShortcutPath)
    $Shortcut.TargetPath = $TargetPath
    if ($Arguments) { $Shortcut.Arguments = $Arguments }
    if ($WorkingDirectory) { $Shortcut.WorkingDirectory = $WorkingDirectory }
    $Shortcut.IconLocation = $TargetPath
    $Shortcut.Save()
}

$StartMenuDir = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\DDR5 AI Memory Tuner'
New-Item -ItemType Directory -Force -Path $StartMenuDir | Out-Null

$DesktopLnk = Join-Path $env:USERPROFILE 'Desktop\DDR5 AI Memory Tuner.lnk'
$StartLnk   = Join-Path $StartMenuDir 'DDR5 AI Memory Tuner.lnk'

New-Shortcut -ShortcutPath $DesktopLnk -TargetPath $RunBat -WorkingDirectory $InstallDir
New-Shortcut -ShortcutPath $StartLnk   -TargetPath $RunBat -WorkingDirectory $InstallDir
Write-Ok "Shortcuts created (Desktop + Start Menu)"

# 6) Uninstall script and registry entry
$UninstallPs1 = Join-Path $InstallDir 'uninstall.ps1'
$unContent = @'
param([switch]`$Silent)
`$ErrorActionPreference = 'Continue'
function Remove-Shortcut([string]`$path){ if (Test-Path `"`$path`") { Remove-Item -Force `"`$path`" } }
Write-Host 'Uninstalling DDR5 AI Memory Tuner...'
try {
  # Remove Start Menu folder
    `$startDir = Join-Path `$env:APPDATA 'Microsoft\Windows\Start Menu\Programs\DDR5 AI Memory Tuner'
  if (Test-Path `$startDir) { Remove-Item -Recurse -Force `$startDir }
  # Remove Desktop shortcut
    Remove-Shortcut (Join-Path `$env:USERPROFILE 'Desktop\DDR5 AI Memory Tuner.lnk')
  # Remove registry uninstall entry
    Remove-Item -Recurse -Force 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\DDR5-AI-Memory-Tuner' -ErrorAction SilentlyContinue
  # Remove install dir
  `$root = Split-Path -Parent $MyInvocation.MyCommand.Path
  Set-Location `$env:TEMP
  Start-Sleep -Milliseconds 200
  Remove-Item -Recurse -Force `$root
  Write-Host 'Uninstall complete.'
} catch {
  Write-Host "Uninstall encountered issues: `$($_.Exception.Message)" -ForegroundColor Yellow
}
'@
Set-Content -Path $UninstallPs1 -Value $unContent -Encoding UTF8

# Per-user ARP entry
$UninstallRegPath = 'HKCU:Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\DDR5-AI-Memory-Tuner'
New-Item -Path $UninstallRegPath -Force | Out-Null
New-ItemProperty -Path $UninstallRegPath -Name 'DisplayName'     -Value 'DDR5 AI Memory Tuner' -PropertyType String -Force | Out-Null
New-ItemProperty -Path $UninstallRegPath -Name 'Publisher'        -Value 'killerbotofthenewworld'    -PropertyType String -Force | Out-Null
New-ItemProperty -Path $UninstallRegPath -Name 'DisplayVersion'   -Value '6.0.0'                     -PropertyType String -Force | Out-Null
New-ItemProperty -Path $UninstallRegPath -Name 'InstallLocation'  -Value $InstallDir                  -PropertyType String -Force | Out-Null
New-ItemProperty -Path $UninstallRegPath -Name 'NoModify'         -Value 1                            -PropertyType DWord  -Force | Out-Null
New-ItemProperty -Path $UninstallRegPath -Name 'NoRepair'         -Value 1                            -PropertyType DWord  -Force | Out-Null
New-ItemProperty -Path $UninstallRegPath -Name 'UninstallString'  -Value "powershell.exe -ExecutionPolicy Bypass -File `"$UninstallPs1`"" -PropertyType String -Force | Out-Null
Write-Ok "Registered uninstaller (per-user)"

# Create Start Menu Uninstall shortcut
$UninstallLnk = Join-Path $StartMenuDir 'Uninstall DDR5 AI Memory Tuner.lnk'
$pwshExe = (Get-Command powershell.exe).Source
New-Shortcut -ShortcutPath $UninstallLnk -TargetPath $pwshExe -Arguments "-ExecutionPolicy Bypass -File `"$UninstallPs1`"" -WorkingDirectory $InstallDir
Write-Ok "Uninstall shortcut created"

Write-Host "`n🎉 Installation complete!" -ForegroundColor Green
Write-Host "Start from: Desktop or Start Menu > DDR5 AI Memory Tuner"
Write-Host "Or run: $RunBat"
