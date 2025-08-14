<#
Build the Windows installer EXE using Inno Setup (ISCC).
- Looks for ISCC in PATH and common install locations
- Builds the installer defined in windows/installer/ddr5-ai-sandbox-simulator.iss
- Outputs the EXE to windows/installer/ (per the .iss OutputDir)
#>

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

function Write-Info($m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Write-Ok($m){ Write-Host "[ OK ] $m" -ForegroundColor Green }
function Write-Warn($m){ Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Write-Err($m){ Write-Host "[ERR ] $m" -ForegroundColor Red }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$IssPath   = Join-Path $ScriptDir 'ddr5-ai-sandbox-simulator.iss'

if (-not (Test-Path $IssPath)) {
  Write-Err "ISS script not found: $IssPath"
  exit 1
}

function Find-ISCC {
  # 1) PATH
  $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  # 2) Common install locations
  $candidates = @(
    'C:\Program Files (x86)\Inno Setup 6\ISCC.exe',
    'C:\Program Files\Inno Setup 6\ISCC.exe'
  )
  foreach ($p in $candidates) { if (Test-Path $p) { return $p } }
  return $null
}

Write-Info "Locating Inno Setup Compiler (ISCC.exe)..."
$iscc = Find-ISCC
if (-not $iscc) {
  Write-Err "ISCC.exe not found. Please install Inno Setup 6: https://jrsoftware.org/isinfo.php"
  Write-Info "Alternatively on Windows with Chocolatey: choco install innosetup -y"
  exit 127
}
Write-Ok "Found ISCC: $iscc"

# Build
Push-Location $ScriptDir
try {
  Write-Info "Building installer from $IssPath ..."
  & "$iscc" "$IssPath"
  if ($LASTEXITCODE -ne 0) {
    Write-Err "ISCC exited with code $LASTEXITCODE"
    exit $LASTEXITCODE
  }
  $out = Get-ChildItem -Path $ScriptDir -Filter 'DDR5-AI-Memory-Tuner-Setup*.exe' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($out) {
    Write-Ok "Built: $($out.FullName)"
  } else {
    Write-Warn "Build completed but output EXE not found in $ScriptDir"
  }
}
finally {
  Pop-Location
}
