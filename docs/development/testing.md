# Testing Guide

Running and writing tests for LangCalc.

## Test Organization

```
tests/
├── test_unit/         # 262 unit tests
└── test_integration/  # 37 integration tests
```

## Running Tests

```bash
# All tests
pytest tests/

# Specific category
pytest tests/test_unit/
pytest tests/test_integration/

# With coverage
pytest tests/ --cov=langcalc --cov-report=html
```

## Writing Tests

See `tests/conftest.py` for shared fixtures.

## Test Coverage

Current: 95% on core modules.

_For complete testing documentation, see tests/README.md in the repository._
