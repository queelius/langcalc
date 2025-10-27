# Contributing to LangCalc

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your changes
4. Make your changes
5. Run tests
6. Submit a pull request

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/langcalc.git
cd langcalc
pip install -e .[dev]
```

## Running Tests

```bash
pytest tests/
pytest tests/ --cov=langcalc --cov-report=html
```

## Code Standards

- Follow PEP 8
- Use type hints
- Write docstrings
- Add tests for new features

## Pull Request Process

1. Update documentation
2. Add tests
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Submit PR with clear description

_See CLAUDE.md in the repository for detailed guidance._
