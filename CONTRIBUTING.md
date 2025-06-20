# Contributing to DDR5 AI Sandbox Simulator

We love your input! We want to make contributing to DDR5 AI Sandbox Simulator as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Code Style

* Follow PEP 8 Python style guidelines
* Use type hints for all function parameters and return values
* Include docstrings for all classes and functions
* Keep line length to 79 characters
* Use meaningful variable and function names

## Testing

* Write unit tests for new features
* Ensure all tests pass before submitting PR
* Maintain test coverage above 80%

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Report Bugs

We use GitHub issues to track public bugs. Report a bug by opening a new issue.

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Feature Requests

We welcome feature requests! Please:

1. Check if the feature has already been requested
2. Provide a clear description of the feature
3. Explain why this feature would be useful
4. Consider offering to implement the feature yourself

## Areas for Contribution

### 🧠 AI & Machine Learning
- New optimization algorithms
- Improved model architectures
- Enhanced training procedures
- Performance optimizations

### 🔬 Hardware Support
- Additional RAM manufacturer support
- New motherboard compatibility
- Enhanced hardware detection
- Expanded databases

### 🌐 User Interface
- Mobile responsiveness improvements
- New visualization components
- Accessibility enhancements
- Multi-language support

### 🔒 Safety & Validation
- Enhanced safety checks
- Improved risk assessment
- Better error handling
- Comprehensive testing

### 📚 Documentation
- Tutorial improvements
- API documentation
- Code examples
- Video guides

## Getting Started

1. **Setup Development Environment**
   ```bash
   git clone https://github.com/killerbotofthenewworld/ddr5-ai-sandbox-simulator.git
   cd ddr5-ai-sandbox-simulator
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Run Tests**
   ```bash
   python -m pytest tests/
   ```

3. **Start Development Server**
   ```bash
   streamlit run main.py --server.runOnSave true
   ```

## Code Review Process

The core team looks at Pull Requests on a regular basis. We will:

1. Review code for style and functionality
2. Test the changes locally
3. Provide feedback if needed
4. Merge when ready

## Community

* Be respectful and inclusive
* Help newcomers get started
* Share knowledge and experiences
* Provide constructive feedback

## Recognition

Contributors will be:
- Added to the contributors list
- Mentioned in release notes
- Invited to join the core team (for significant contributions)

Thank you for contributing to DDR5 AI Sandbox Simulator! 🚀
