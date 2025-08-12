# Project Setup and Development Guide

## Installation

1. **Clone or download the project**
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Option 1: Streamlit Web Interface (Recommended)
```bash
streamlit run main.py
```
Then open your browser to `http://localhost:8501`

### Windows Installer (Easy)

For a real per-user install on Windows with Start Menu/Desktop shortcuts:

```cmd
windows\install.bat
```

This will install to `%LOCALAPPDATA%\DDR5-AI-Sandbox-Simulator` and create a `run_ddr5_simulator.bat` launcher.

### Option 2: Python Script Mode
```bash
python main.py
```

## Project Structure

```
ddr5/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── setup.py               # Development setup guide
├── src/                   # Core source code
│   ├── ddr5_models.py     # DDR5 configuration models
│   ├── ddr5_simulator.py  # Memory simulation engine
│   ├── ai_optimizer.py    # AI optimization algorithms
│   └── web_interface.py   # Streamlit web interface
├── tests/                 # Unit tests
│   └── test_ddr5_models.py
├── data/                  # Training data and presets
└── .github/               # GitHub configuration
    └── copilot-instructions.md
```

## Features

### 1. DDR5 Memory Simulation
- Accurate modeling of DDR5 JEDEC specifications
- Bandwidth and latency simulation
- Power consumption estimation
- Stability analysis and validation

### 2. AI-Powered Optimization
- Machine learning models for performance prediction
- Genetic algorithm optimization
- Multiple optimization goals (performance, stability, power efficiency)
- Automatic parameter tuning

### 3. Interactive Web Interface
- Real-time configuration editing
- Visual performance analysis
- Export capabilities (JSON, BIOS format)
- Comprehensive validation and warnings

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Formatting
```bash
black src/ tests/
```

### Type Checking
```bash
mypy src/
```

## Usage Examples

### Basic Configuration Testing
1. Open the web interface
2. Adjust memory frequency and timings in the sidebar
3. Click "Run Simulation" to see performance metrics
4. Review validation warnings if any

### AI Optimization
1. Go to the "AI Optimization" tab
2. Select optimization goal (balanced, performance, stability, etc.)
3. Click "Train AI Models" if not already trained
4. Click "Start AI Optimization" to find optimal settings

### Exporting Results
1. Configure your desired settings
2. Go to the "Export" tab
3. Download JSON configuration or copy BIOS settings

## Safety Notes

⚠️ **Important**: This is a simulation tool. Always test configurations carefully on physical hardware before applying them permanently. Start with conservative settings and gradually optimize.

## Troubleshooting

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Performance Issues
- Reduce the number of optimization generations
- Use smaller training datasets for AI models
- Close other applications to free up system resources

### Web Interface Not Loading
- Check that port 8501 is available
- Try running: `streamlit run main.py --server.port 8502`

## Contributing

1. Follow PEP 8 coding standards
2. Add type hints to all functions
3. Include docstrings for classes and methods
4. Write unit tests for new features
5. Update documentation as needed
