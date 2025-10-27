# Test Suite Documentation

## Overview

Comprehensive pytest-based test suite for **LangCalc** (Language Calculus), following TDD best practices. Tests cover the algebraic framework, infinigrams, and model composition.

## Structure

```
tests/
├── pytest.ini              # Configuration (80% coverage target)
├── conftest.py            # Shared fixtures
├── test_unit/             # Unit tests (228 tests)
│   ├── test_model_algebra_core.py       # Core algebra (53 tests)
│   ├── test_model_algebra_additional.py # Extended operations (28 tests)
│   ├── test_model_algebra_coverage.py   # Coverage tests (17 tests) ⭐NEW
│   ├── test_algebraic_operations.py     # Algebraic properties (27 tests)
│   ├── test_ngram_model.py              # N-gram models (36 tests)
│   ├── test_projections.py              # Projections (40 tests)
│   └── test_suffix_array.py             # Suffix arrays (33 tests)
└── test_integration/      # Integration tests (35 tests)
    ├── test_lightweight_grounding.py     # Grounding system (18 tests)
    ├── test_model_composition.py         # Composition workflows (19 tests)
    └── test_ollama_integration.py        # LLM integration
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run with coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Run specific test categories
```bash
pytest tests/ -m unit          # Unit tests only
pytest tests/ -m integration   # Integration tests only
pytest tests/ -m "not slow"    # Skip slow tests
```

### Run specific test files
```bash
pytest tests/test_unit/test_suffix_array.py
pytest tests/test_integration/test_lightweight_grounding.py
```

## Test Statistics

- **Total Tests**: 263 ✅ (All passing)
- **Unit Tests**: 228
- **Integration Tests**: 35
- **Overall Coverage**: 35%
- **Core Module Coverage**: model_algebra.py at **95%** ⭐
- **Execution Time**: ~8 seconds

## Test Categories

### Unit Tests

#### Algebraic Operations (27 tests)
- Model addition and composition
- Weight normalization
- Algebraic properties (associativity, commutativity)
- Interface compliance

#### N-gram Models (61 tests)
- Initialization with various parameters
- Probability computation
- Sampling configurations
- Edge cases (empty corpus, large orders)

#### Projections (40 tests)
- Recency projection
- Semantic projection
- Projection composition
- Error handling

#### Suffix Arrays (18 tests)
- Construction and validation
- Pattern matching
- Deterministic behavior
- Various input sizes

### Integration Tests

#### Lightweight Grounding (17 tests)
- System initialization
- Weight mixing
- Factual accuracy
- Performance testing

#### Model Composition (19 tests)
- Complex workflows
- Projection integration
- Data flow validation
- Real-world scenarios

## Key Features

### TDD Best Practices
- ✅ Test behavior, not implementation
- ✅ Clear test names (Given-When-Then)
- ✅ Comprehensive fixtures
- ✅ Parametrized tests
- ✅ Edge case coverage

### Test Quality
- **Fast execution**: Unit tests < 1 second
- **Clear failures**: Descriptive assertions
- **Isolated tests**: No interdependencies
- **Realistic scenarios**: Production-like testing

## Coverage Report

### Recent Improvements (October 2025)
- ✅ **Fixed all 15 failing tests** - 100% pass rate achieved
- ✅ **model_algebra.py: 89% → 95%** - Added 17 comprehensive tests
- ✅ **263 tests total** - Up from 165 tests (+98 tests)
- 📊 **Overall: 35%** - Core modules well-covered, demos excluded

### Current Coverage by Module

**Excellent Coverage (≥90%)**
```
Module                          Stmts   Miss  Cover  Status
------------------------------------------------------------
model_algebra.py                 550     29    95%   ⭐ Excellent
ngram_projections/__init__.py      8      0   100%   ✅ Complete
```

**Good Coverage (70-89%)**
```
Module                          Stmts   Miss  Cover
----------------------------------------------------
suffix_array.py                   99     18    82%
models/ngram.py                  111     33    70%
models/base.py                   117     37    68%
```

**Moderate Coverage (50-69%)**
```
Module                          Stmts   Miss  Cover
----------------------------------------------------
models/mixture.py                 80     32    60%
projections/semantic.py           73     33    55%
```

**Needs Improvement (<50%)**
```
Module                          Stmts   Miss  Cover  Priority
--------------------------------------------------------------
lightweight_grounding.py         282    177    37%   High
models/llm.py                    120     69    42%   High
projections/base.py               91     49    46%   High
projections/recency.py            44     25    43%   High
projections/edit_distance.py      75     64    15%   Medium
```

### Path to 80% Coverage

See `TEST_COVERAGE_SUMMARY.md` and `COVERAGE_ANALYSIS.md` for detailed roadmap:
- **Estimated effort**: 100-120 additional tests
- **Time required**: 8-12 hours
- **Phase 1 priorities**: models/base.py, models/ngram.py edge cases
- **Phase 2 priorities**: lightweight_grounding.py selected classes
- **Phase 3 priorities**: projections modules, LLM integration

## Fixtures (conftest.py)

### Core Fixtures
- `sample_corpus`: Text data for testing
- `sample_vocab`: Vocabulary set
- `ngram_model`: Pre-configured n-gram model
- `suffix_array`: Test suffix array instance
- `mock_llm`: Mock LLM for testing

### Helper Functions
- `assert_valid_distribution`: Validate probability distributions
- `assert_model_interface`: Check model compliance
- `create_test_corpus`: Generate test data

## Test Markers

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Tests taking >1 second
- `@pytest.mark.network`: Tests requiring network
- `@pytest.mark.parametrized`: Parametrized variations

## Contributing

1. Write tests before implementation (TDD)
2. Follow existing test patterns
3. Maintain >80% coverage for new code
4. Use descriptive test names
5. Add appropriate markers

## Continuous Integration

The test suite is designed for CI/CD integration:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -e .[dev]
    pytest tests/ --cov=src --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v2
```