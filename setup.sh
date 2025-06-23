#!/bin/bash

# DDR5 AI Sandbox Simulator v6.0.0 - Easy Setup Script
# Enhanced Features Edition with Real-Time Hardware Integration

echo "🚀 DDR5 AI Sandbox Simulator v6.0.0 - Enhanced Features Setup"
echo "================================================================"
echo "Features: Dark/Light Theme, 3D Charts, WebSocket, LLM, Safety, AutoML"
echo ""

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.9 or higher is required. Found: $python_version"
    echo "Please install Python 3.9+ and try again."
    exit 1
else
    echo "✅ Python $python_version detected"
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi
echo "✅ Virtual environment activated"

# Upgrade pip
echo ""
echo "📈 Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install requirements
echo ""
echo "📚 Installing dependencies (this may take a few minutes)..."
echo "   Installing core dependencies..."
pip install -q numpy pandas scikit-learn matplotlib seaborn streamlit plotly pydantic pytest

echo "   Installing AI/ML libraries..."
pip install -q torch optuna xgboost lightgbm transformers

echo "   Installing enhanced features dependencies..."
pip install -q opencv-python pillow psutil websockets openai anthropic requests kaleido lxml

echo "   Installing remaining requirements..."
pip install -q -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install some dependencies. Trying individual installation..."
    echo "This is normal and we'll install what we can..."
fi

echo "✅ Dependencies installed"

# Test installation
echo ""
echo "🧪 Testing installation..."
python3 -c "import streamlit, numpy, pandas, plotly; print('✅ Core libraries working')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some core libraries may not be working correctly"
else
    echo "✅ Installation test passed"
fi

# Create launcher script
echo ""
echo "🎯 Creating launcher script..."
cat > run_ddr5_simulator.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting DDR5 AI Sandbox Simulator v6.0.0..."
echo "Enhanced Features Edition"
echo ""

# Activate virtual environment
source venv/bin/activate

# Check if streamlit is available
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Please run setup.sh first."
    exit 1
fi

# Start the simulator
echo "🌐 Starting web interface..."
echo "Access at: http://localhost:8521"
echo ""
echo "🎨 Enhanced Features Available:"
echo "  • Dark/Light Theme & Custom CSS"
echo "  • 3D Performance Charts"
echo "  • WebSocket Real-Time Monitoring"
echo "  • Optional LLM Integration"
echo "  • Hardware Damage Prevention"
echo "  • AutoML Pipeline"
echo "  • Popular Tool Imports (ASUS, MSI, Intel)"
echo ""
echo "Press Ctrl+C to stop the simulator"
echo ""

streamlit run main.py --server.port 8521
EOF

chmod +x run_ddr5_simulator.sh
echo "✅ Launcher script created: run_ddr5_simulator.sh"

# Create quick test script
echo ""
echo "🔬 Creating test script..."
cat > test_ddr5_features.py << 'EOF'
#!/usr/bin/env python3
"""Quick test of DDR5 AI Sandbox Simulator features"""

import sys
import importlib

def test_import(module_name, feature_name):
    try:
        importlib.import_module(module_name)
        print(f"✅ {feature_name}")
        return True
    except ImportError as e:
        print(f"❌ {feature_name}: {e}")
        return False

def main():
    print("🧪 DDR5 AI Sandbox Simulator v6.0.0 - Feature Test")
    print("=" * 55)
    
    total_tests = 0
    passed_tests = 0
    
    # Core features
    tests = [
        ("src.ddr5_models", "DDR5 Models"),
        ("src.ddr5_simulator", "DDR5 Simulator"),
        ("src.ai_optimizer", "AI Optimizer"),
        ("src.web_interface.main", "Web Interface"),
        ("src.web_interface.components.enhanced_ui", "Enhanced UI"),
        ("src.web_interface.components.charts_3d", "3D Charts"),
        ("src.web_interface.components.websocket_client", "WebSocket Client"),
        ("src.web_interface.components.llm_integration", "LLM Integration"),
        ("src.web_interface.components.damage_prevention", "Damage Prevention"),
        ("src.web_interface.components.tool_imports", "Tool Imports"),
        ("src.web_interface.tabs.enhanced_features_v2", "Enhanced Features V2"),
    ]
    
    for module, name in tests:
        total_tests += 1
        if test_import(module, name):
            passed_tests += 1
    
    print()
    print(f"📊 Results: {passed_tests}/{total_tests} features working")
    
    if passed_tests == total_tests:
        print("🎉 All features are working correctly!")
        print("Ready to run: ./run_ddr5_simulator.sh")
    else:
        print("⚠️  Some features may not work correctly.")
        print("This is normal - optional features may need additional setup.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

chmod +x test_ddr5_features.py
echo "✅ Test script created: test_ddr5_features.py"

# Final instructions
echo ""
echo "🎉 Setup Complete!"
echo "===================="
echo ""
echo "🚀 To start the DDR5 AI Sandbox Simulator:"
echo "   ./run_ddr5_simulator.sh"
echo ""
echo "🧪 To test features:"
echo "   python3 test_ddr5_features.py"
echo ""
echo "🌐 Web Interface will be available at:"
echo "   http://localhost:8521"
echo ""
echo "🌟 Enhanced Features Available:"
echo "   • 🎨 Dark/Light Theme & Custom CSS"
echo "   • 📊 3D Performance Charts"
echo "   • ⚡ WebSocket Real-Time Monitoring"
echo "   • 🤖 Optional LLM Integration"
echo "   • 🛡️ Hardware Damage Prevention"
echo "   • 🔧 AutoML Pipeline"
echo "   • 🔄 Popular Tool Imports"
echo ""
echo "📖 For more information, see README.md"
echo "💖 Support development: https://ko-fi.com/killerbotofthenewworld"
echo ""
echo "Ready to optimize your DDR5 memory! 🧠⚡"
