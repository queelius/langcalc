# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LangCalc** is an algebraic framework for compositional language modeling. The key innovation is treating language models as first-class mathematical objects that can be composed using algebraic operators. The project demonstrates that combining LLMs with suffix array-based pattern matching (infinigrams) using just 5% weight can achieve 70% perplexity reduction.

## Core Architecture

### Three-Layer Design

1. **Algebraic Framework** (`src/model_algebra.py`): Complete operator algebra supporting 10+ operators
2. **Lightweight Grounding** (`src/lightweight_grounding.py`): Suffix array integration for factual grounding
3. **N-gram Projections** (`src/ngram_projections/`): Modular package for model composition

### Key Mathematical Abstraction

Models are algebraic objects supporting:
- **Arithmetic**: `+`, `*`, `-`, `/`, `**` (temperature)
- **Set operations**: `|` (max), `&` (min), `^` (symmetric difference)
- **Transformations**: `<<` (apply context transform), `>>` (apply function)

Example:
```python
model = (0.7 * llm + 0.2 * (wiki << LongestSuffix(sa)) + 0.1 * ngram) ** 0.9
```

### Suffix Arrays vs N-grams

The project uses suffix arrays for efficient pattern matching:
- **34x more memory efficient** than hash-based n-grams
- **O(m log n) query time** with binary search
- **Variable-length patterns** without pre-computing n
- Implementation in `src/suffix_array_demo.py`

## Development Commands

### Testing

```bash
# Run all tests (165 total: 128 unit + 36 integration)
pytest tests/

# Run with coverage (target: 80%)
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest tests/ -m unit          # Unit tests only
pytest tests/ -m integration   # Integration tests only
pytest tests/ -m "not slow"    # Skip slow tests

# Run specific test file
pytest tests/test_unit/test_suffix_array.py
pytest tests/test_integration/test_lightweight_grounding.py

# Run single test
pytest tests/test_unit/test_model_algebra_core.py::test_addition_creates_sum_model
```

### Installation

```bash
# Install package in development mode
pip install -e .

# Install with dev dependencies
pip install -e .[dev]

# Install with experiment dependencies (matplotlib, jupyter, requests)
pip install -e .[experiments]
```

### Running Examples

```bash
# Core algebraic examples
python examples/algebra_examples.py

# Comprehensive experiments
python examples/comprehensive_experiments.py

# Lightweight grounding demo
python examples/lightweight_experiments.py

# Ollama integration (requires Ollama running at 192.168.0.225)
python examples/ollama_integration_example.py
```

### Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Recommended learning order:
# 1. notebooks/explore_algebra.ipynb          (45 min - foundations)
# 2. notebooks/lightweight_grounding_demo.ipynb  (60 min - practical)
# 3. notebooks/unified_algebra.ipynb          (60 min - advanced theory)
```

## Critical Implementation Details

### Model Interface Contract

All models must implement `LanguageModel` interface:
```python
def predict(context: List[str], top_k: int = 50) -> Dict[str, float]:
    """Return probability distribution over next tokens."""
```

Key implementations:
- `NGramModel` (`src/ngram_projections/models/ngram.py`)
- `MockLLM` (`src/ngram_projections/models/llm.py`)
- `MixtureModel` (`src/lightweight_grounding.py`)

### Context Transformations

Transform context before model prediction (`src/model_algebra.py`):
- `LongestSuffixTransform`: Find longest matching suffix in corpus
- `MaxKWordsTransform`: Limit context to k most recent words
- `RecencyWeightTransform`: Apply exponential decay to older tokens
- `FocusTransform`: Filter to specific word types

Compose with `|` (sequential) or `&` (parallel):
```python
transform = LongestSuffixTransform(sa) | RecencyWeightTransform(0.9)
model << transform
```

### Algebraic Properties

The framework maintains mathematical consistency:
- **Associativity**: `(a + b) + c = a + (b + c)`
- **Distributivity**: `α*(a + b) = α*a + α*b`
- **Identity**: `a + 0*b = a`

Tests verify these properties in `tests/test_unit/test_algebraic_operations.py`.

## Test Infrastructure

### Fixtures (`tests/conftest.py`)

Core shared fixtures:
- `sample_corpus`: Deterministic corpus for reproducible tests
- `ngram_model`: Pre-configured n-gram model
- `mock_llm`: Mock LLM with controllable behavior
- `test_context`: Standard context for model evaluation

Helper assertions:
- `assert_valid_logprobs`: Validate log probability arrays
- `assert_valid_samples`: Check token samples
- `assert_valid_probabilities`: Verify probability distributions

### Test Organization

```
tests/
├── pytest.ini              # Configuration (80% coverage requirement)
├── conftest.py            # Shared fixtures
├── test_unit/             # 128 tests
│   ├── test_model_algebra_core.py       # Core algebra (38 tests)
│   ├── test_model_algebra_additional.py # Extended operations (17 tests)
│   ├── test_ngram_model.py              # N-gram models (61 tests)
│   ├── test_projections.py              # Projections (40 tests)
│   └── test_suffix_array.py             # Suffix arrays (18 tests)
└── test_integration/      # 36 tests
    ├── test_lightweight_grounding.py     # Grounding system (17 tests)
    ├── test_model_composition.py         # Workflows (19 tests)
    └── test_ollama_integration.py        # LLM integration
```

## Performance Characteristics

### Lightweight Grounding Results
- **Optimal mixing**: 95% LLM + 5% suffix array
- **Perplexity reduction**: 70% improvement
- **Query latency**: 0.03ms for suffix lookups
- **LLM overhead**: Only 6.5% (2.66ms) with real models

### Memory Efficiency
- **Suffix arrays**: 1GB for 1B token corpus
- **N-gram hash tables**: 34GB for same corpus
- **Speedup**: 34x memory reduction

## Package Structure

```
src/ngram_projections/
├── __init__.py            # Package exports
├── models/
│   ├── base.py           # LanguageModel abstract base
│   ├── ngram.py          # N-gram implementation
│   ├── llm.py            # Mock LLM for testing
│   └── mixture.py        # Model mixing
├── projections/
│   ├── base.py           # Projection abstract base
│   ├── recency.py        # Recency-based projection
│   ├── edit_distance.py  # Edit distance projection
│   └── semantic.py       # Semantic similarity projection
├── data/
│   └── suffix_array.py   # Suffix array utilities
└── algebra/
    └── combinators.py    # Higher-order composition
```

## Python Version

Requires Python >=3.8 (currently testing with 3.12.3).

## External Dependencies

### Core (production)
- `numpy>=1.19.0`: Numerical operations
- `scipy>=1.5.0`: Statistical functions

### Development
- `pytest>=6.0`: Testing framework
- `pytest-cov`: Coverage reporting
- `black`: Code formatting
- `flake8`: Linting

### Experiments
- `matplotlib>=3.3.0`: Visualization
- `jupyter>=1.0.0`: Interactive notebooks
- `requests>=2.25.0`: HTTP (for Ollama integration)

## Important Notes

### Ollama Integration
The project supports Ollama LLM integration with server at `192.168.0.225`. Integration tests in `tests/test_integration/test_ollama_integration.py`.

### Legacy Code
The `legacy/` directory contains experimental code from earlier iterations. Focus development on `src/` and `tests/`.

### Coverage Target
Tests enforce 80% coverage minimum (configured in `pytest.ini`). Current coverage is 49% - contributions should improve this.

### Algebraic Laws
When adding new operators, verify they satisfy algebraic properties (associativity, commutativity where applicable). Add tests to `test_algebraic_operations.py`.
