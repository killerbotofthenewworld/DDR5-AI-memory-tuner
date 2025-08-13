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
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "..\..\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs overwritereadonly ignoreversion
; Exclude unnecessary directories and files
Source: "..\..\*"; DestDir: "{app}"; Excludes: ".git\*;__pycache__\*;build\*;dist\*;*.pyc;*.pyo;*.log;*.tmp;test_models\*;tests\*;screenshots\*"; Flags: recursesubdirs createallsubdirs overwritereadonly ignoreversion

[Run]
Filename: "powershell.exe"; Parameters: "-ExecutionPolicy Bypass -File \"{app}\\windows\\install.ps1\""; StatusMsg: "Finalizing installation..."; Flags: runhidden

[Icons]
Name: "{userdesktop}\{#MyAppExeName}"; Filename: "{localappdata}\\DDR5-AI-Sandbox-Simulator\\run_ddr5_simulator.bat"
Name: "{userprograms}\{#MyAppExeName}\{#MyAppExeName}"; Filename: "{localappdata}\\DDR5-AI-Sandbox-Simulator\\run_ddr5_simulator.bat"
Name: "{userprograms}\{#MyAppExeName}\Uninstall {#MyAppExeName}"; Filename: "powershell.exe"; Parameters: "-ExecutionPolicy Bypass -File \"{localappdata}\\DDR5-AI-Sandbox-Simulator\\uninstall.ps1\""; IconFilename: "{sys}\\imageres.dll"; IconIndex: 27
