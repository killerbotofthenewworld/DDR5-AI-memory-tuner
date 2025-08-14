; Inno Setup Script for DDR5 AI Sandbox Simulator
; Builds a per-user installer that extracts files to %LOCALAPPDATA% and runs the PowerShell installer

#define MyAppName "DDR5 AI Sandbox Simulator"
#define MyAppVersion "6.0.0"
#define MyAppPublisher "killerbotofthenewworld"
#define MyAppExeName "DDR5 AI Sandbox Simulator"

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\DDR5-AI-Sandbox-Simulator
DisableDirPage=yes
DisableProgramGroupPage=yes
OutputDir=.
OutputBaseFilename=DDR5-AI-Sandbox-Simulator-Setup
Compression=lzma
SolidCompression=yes
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
; Explicit includes to keep installer lean
; Core application source
Source: "..\\..\\src\\*"; DestDir: "{app}\\src"; Flags: recursesubdirs createallsubdirs overwritereadonly ignoreversion
; Entry points and configs
Source: "..\\..\\main.py"; DestDir: "{app}"; Flags: ignoreversion overwritereadonly
Source: "..\\..\\launch_ddr5.py"; DestDir: "{app}"; Flags: ignoreversion overwritereadonly
Source: "..\\..\\ai_config.json"; DestDir: "{app}"; Flags: ignoreversion overwritereadonly
Source: "..\\..\\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion overwritereadonly
Source: "..\\..\\LICENSE"; DestDir: "{app}"; Flags: ignoreversion overwritereadonly
; Assets optionally used by UI/docs
Source: "..\\..\\ddr5-simulator.png"; DestDir: "{app}"; Flags: ignoreversion overwritereadonly
; Windows helper scripts (post-install runs install.ps1)
Source: "..\\..\\windows\\install.ps1"; DestDir: "{app}\\windows"; Flags: ignoreversion overwritereadonly
Source: "..\\..\\windows\\install.bat"; DestDir: "{app}\\windows"; Flags: ignoreversion overwritereadonly
Source: "..\\..\\windows\\README-windows.md"; DestDir: "{app}\\windows"; Flags: ignoreversion overwritereadonly

[Run]
Filename: "powershell.exe"; Parameters: "-ExecutionPolicy Bypass -File ""{app}\windows\install.ps1"""; StatusMsg: "Finalizing installation..."; Flags: runhidden

[Icons]
Name: "{userdesktop}\{#MyAppExeName}"; Filename: "{localappdata}\DDR5-AI-Sandbox-Simulator\run_ddr5_simulator.bat"
Name: "{userprograms}\{#MyAppExeName}\{#MyAppExeName}"; Filename: "{localappdata}\DDR5-AI-Sandbox-Simulator\run_ddr5_simulator.bat"
Name: "{userprograms}\{#MyAppExeName}\Uninstall {#MyAppExeName}"; Filename: "powershell.exe"; Parameters: "-ExecutionPolicy Bypass -File ""{localappdata}\DDR5-AI-Sandbox-Simulator\uninstall.ps1"""; IconFilename: "{sys}\imageres.dll"; IconIndex: 27
