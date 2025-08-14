# 🧪 Testing Guide

This project ships with a lightweight but effective test and quality setup. Use this guide to run tests, coverage, lint, types, and security scans locally.

## Prerequisites

- Python 3.9–3.12
- Recommended: create/activate a virtual environment
- Install dev and runtime deps:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## Run tests

```bash
pytest -q
# or verbose + coverage
pytest -v --cov=src/ --cov-report=term-missing --cov-report=xml
```

Notes:

- Tests include JEDEC invariant checks and optimizer determinism (seeded) where applicable.
- Heavy ML libs run on CPU; GPU is optional.

## Linting & format

```bash
black src/ tests/ main.py
flake8 src/ tests/ main.py
isort src/ tests/
```

## Type checking

```bash
mypy src/
```

Configuration lives in `mypy.ini` and `.flake8`. CI uses continue-on-error today; you can tighten locally.

## Security

```bash
bandit -r src/ -f txt
safety check
```

## Tips

- Seeded tests: prefer fixed seeds for stochastic optimizers to make failures reproducible.
- Performance: if tests get slow, use `-k` to target modules, or `-q` to reduce output noise.
- Coverage HTML is written to `htmlcov/` when enabled.
