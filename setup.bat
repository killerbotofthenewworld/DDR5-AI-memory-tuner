@echo off
REM DDR5 AI Sandbox Simulator v6.0.0 - Easy Setup Script (Windows)
REM Enhanced Features Edition with Real-Time Hardware Integration

echo 🚀 DDR5 AI Sandbox Simulator v6.0.0 - Enhanced Features Setup
echo ================================================================
echo.

REM Check Python version
echo 🐍 Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org and try again.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo ✅ Python %python_version% detected

REM Check if virtual environment exists
if not exist "venv" (
    echo.
    echo 📦 Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
)

REM Activate virtual environment
echo.
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✅ Virtual environment activated

REM Upgrade pip
echo.
echo 📈 Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

REM Install requirements
echo.
echo 📚 Installing dependencies (this may take a few minutes)...
echo    Installing core dependencies...
pip install -q numpy pandas scikit-learn matplotlib seaborn streamlit plotly pydantic pytest

echo    Installing AI/ML libraries...
pip install -q torch optuna xgboost lightgbm transformers

echo    Installing enhanced features dependencies...
pip install -q opencv-python pillow psutil websockets openai anthropic requests kaleido lxml

echo    Installing remaining requirements...
pip install -q -r requirements.txt

echo ✅ Dependencies installed

REM Test installation
echo.
echo 🧪 Testing installation...
python -c "import streamlit, numpy, pandas, plotly; print('✅ Core libraries working')" 2>nul
if errorlevel 1 (
    echo ⚠️  Some core libraries may not be working correctly
) else (
    echo ✅ Installation test passed
)

REM Create launcher script
echo.
echo 🎯 Creating launcher script...
(
echo @echo off
echo echo 🚀 Starting DDR5 AI Sandbox Simulator v6.0.0...
echo echo Enhanced Features Edition
echo echo.
echo.
echo REM Activate virtual environment
echo call venv\Scripts\activate.bat
echo.
echo REM Check if streamlit is available
echo streamlit --version ^>nul 2^>^&1
echo if errorlevel 1 ^(
echo     echo ❌ Streamlit not found. Please run setup.bat first.
echo     pause
echo     exit /b 1
echo ^)
echo.
echo REM Start the simulator
echo echo 🌐 Starting web interface...
echo echo Access at: http://localhost:8521
echo echo.
echo echo 🎨 Enhanced Features Available:
echo echo   • Dark/Light Theme ^& Custom CSS
echo echo   • 3D Performance Charts
echo echo   • WebSocket Real-Time Monitoring
echo echo   • Optional LLM Integration
echo echo   • Hardware Damage Prevention
echo echo   • AutoML Pipeline
echo echo   • Popular Tool Imports ^(ASUS, MSI, Intel^)
echo echo.
echo echo Press Ctrl+C to stop the simulator
echo echo.
echo.
echo streamlit run main.py --server.port 8521
) > run_ddr5_simulator.bat

echo ✅ Launcher script created: run_ddr5_simulator.bat

REM Create quick test script
echo.
echo 🔬 Creating test script...
(
echo import sys
echo import importlib
echo.
echo def test_import^(module_name, feature_name^):
echo     try:
echo         importlib.import_module^(module_name^)
echo         print^(f"✅ {feature_name}"^)
echo         return True
echo     except ImportError as e:
echo         print^(f"❌ {feature_name}: {e}"^)
echo         return False
echo.
echo def main^(^):
echo     print^("🧪 DDR5 AI Sandbox Simulator v6.0.0 - Feature Test"^)
echo     print^("=" * 55^)
echo     
echo     total_tests = 0
echo     passed_tests = 0
echo     
echo     tests = [
echo         ^("src.ddr5_models", "DDR5 Models"^),
echo         ^("src.ddr5_simulator", "DDR5 Simulator"^),
echo         ^("src.ai_optimizer", "AI Optimizer"^),
echo         ^("src.web_interface.main", "Web Interface"^),
echo         ^("src.web_interface.components.enhanced_ui", "Enhanced UI"^),
echo         ^("src.web_interface.components.charts_3d", "3D Charts"^),
echo         ^("src.web_interface.components.websocket_client", "WebSocket Client"^),
echo         ^("src.web_interface.components.llm_integration", "LLM Integration"^),
echo         ^("src.web_interface.components.damage_prevention", "Damage Prevention"^),
echo         ^("src.web_interface.components.tool_imports", "Tool Imports"^),
echo         ^("src.web_interface.tabs.enhanced_features_v2", "Enhanced Features V2"^),
echo     ]
echo     
echo     for module, name in tests:
echo         total_tests += 1
echo         if test_import^(module, name^):
echo             passed_tests += 1
echo     
echo     print^(^)
echo     print^(f"📊 Results: {passed_tests}/{total_tests} features working"^)
echo     
echo     if passed_tests == total_tests:
echo         print^("🎉 All features are working correctly!"^)
echo         print^("Ready to run: run_ddr5_simulator.bat"^)
echo     else:
echo         print^("⚠️  Some features may not work correctly."^)
echo         print^("This is normal - optional features may need additional setup."^)
echo     
echo     return passed_tests == total_tests
echo.
echo if __name__ == "__main__":
echo     success = main^(^)
echo     sys.exit^(0 if success else 1^)
) > test_ddr5_features.py

echo ✅ Test script created: test_ddr5_features.py

REM Final instructions
echo.
echo 🎉 Setup Complete!
echo ====================
echo.
echo 🚀 To start the DDR5 AI Sandbox Simulator:
echo    run_ddr5_simulator.bat
echo.
echo 🧪 To test features:
echo    python test_ddr5_features.py
echo.
echo 🌐 Web Interface will be available at:
echo    http://localhost:8521
echo.
echo 🌟 Enhanced Features Available:
echo    • 🎨 Dark/Light Theme ^& Custom CSS
echo    • 📊 3D Performance Charts
echo    • ⚡ WebSocket Real-Time Monitoring
echo    • 🤖 Optional LLM Integration
echo    • 🛡️ Hardware Damage Prevention
echo    • 🔧 AutoML Pipeline
echo    • 🔄 Popular Tool Imports
echo.
echo 📖 For more information, see README.md
echo 💖 Support development: https://ko-fi.com/killerbotofthenewworld
echo.
echo Ready to optimize your DDR5 memory! 🧠⚡
echo.
pause
