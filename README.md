# LangCalc: A Calculus for Language Models

An elegant mathematical framework for composing language models through algebraic operations, featuring efficient suffix array-based grounding (infinigrams) and lightweight model mixing.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-299%20passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-95%25-green.svg)](TEST_COVERAGE_SUMMARY.md)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🎯 Overview

LangCalc introduces a comprehensive algebraic framework for language model composition that treats models as first-class mathematical objects. The key innovation is **lightweight grounding**: combining Large Language Models (LLMs) with suffix array-based pattern matching (infinigrams) using just 5% weight to achieve 70% perplexity reduction.

```python
# Express sophisticated models as elegant algebra
from langcalc import Infinigram, create_infinigram
from langcalc.models import NGramModel, HuggingFaceModel

# Create infinigram from Wikipedia
wiki = Infinigram(wikipedia_corpus, max_length=20)

# Compose with LLM
model = 0.95 * llm + 0.05 * wiki

# Or use algebra module
from langcalc.algebra import LongestSuffixTransform
grounded = llm + (wiki << LongestSuffixTransform(sa))
```

## 📁 Project Structure

```
langcalc/                # Main package (NEW in v0.4.0)
├── __init__.py         # Public API
├── infinigram.py       # Variable-length n-grams
├── algebra.py          # Algebraic framework with 10+ operators
├── grounding.py        # Lightweight grounding system
├── models/             # Language model implementations
│   ├── base.py        # LanguageModel interface
│   ├── ngram.py       # N-gram models
│   ├── llm.py         # LLM wrappers
│   └── mixture.py     # Model composition
├── projections/        # Context transformations
│   ├── recency.py     # Recency-based projection
│   ├── semantic.py    # Semantic similarity
│   └── edit_distance.py # Edit distance projection
└── data/               # Data structures
    ├── suffix_array.py # Efficient suffix arrays
    └── incremental.py  # Incremental updates

examples/               # Usage examples
├── algebra_examples.py
├── comprehensive_experiments.py
└── lightweight_experiments.py

tests/                  # Test suite (299 tests)
├── test_unit/         # 262 unit tests
└── test_integration/  # 37 integration tests

notebooks/              # Jupyter demos
├── explore_algebra.ipynb
├── lightweight_grounding_demo.ipynb
└── unified_algebra.ipynb

papers/                 # Academic paper (LaTeX)
└── paper.pdf
```

## 🚀 Quick Start

### Installation

```bash
# Install from source (development mode)
git clone https://github.com/queelius/langcalc.git
cd langcalc
pip install -e .

# Or with development dependencies
pip install -e .[dev]

# Or with experiment dependencies
pip install -e .[experiments]
```

### Basic Usage

```python
# Import the package
from langcalc import Infinigram, NGramModel, create_infinigram

# Create an infinigram model
corpus = [1, 2, 3, 4, 2, 3, 5, 6, 2, 3, 4]
model = Infinigram(corpus, max_length=10)

# Predict next token
context = [2, 3]
probs = model.predict(context)  # Variable-length suffix matching
print(f"Predictions: {probs}")

# Compose models
ngram = NGramModel(corpus, n=3)
mixture = 0.7 * model + 0.3 * ngram
```

### Run Examples

```bash
# Run algebraic examples
python examples/algebra_examples.py

# Run comprehensive experiments
python examples/comprehensive_experiments.py

# Run lightweight grounding experiments
python examples/lightweight_experiments.py

# Interactive notebooks
jupyter notebook notebooks/explore_algebra.ipynb
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=langcalc --cov-report=html

# Run specific test categories
pytest tests/test_unit/          # Unit tests only
pytest tests/test_integration/   # Integration tests only
```

## 🔑 Key Features

### Algebraic Operators
- **10+ operators**: `+`, `*`, `|`, `&`, `^`, `**`, `>>`, `<<`, `~`
- **Context transforms**: LongestSuffix, MaxKWords, RecencyWeight
- **Mathematical consistency**: Associativity, distributivity, composability

### Suffix Arrays
- **34x more memory efficient** than n-gram hash tables
- **O(m log n) query time** with binary search
- **Variable-length patterns** without pre-computing n

### Infinigrams (NEW in v0.4.0)
- **Variable-length n-grams** with dynamic pattern matching
- **Automatic suffix array construction** for efficient queries
- **O(m log n) query time** for longest suffix matching
- **Incremental updates** for streaming data
- **36 comprehensive tests** covering all functionality

### Production Ready
- **299 comprehensive tests** with 100% pass rate ✅
- **95% coverage** on core algebraic framework ⭐
- **Tested with Ollama** integration
- **Only 6.5% overhead** (2.66ms) with real LLMs
- **70% perplexity reduction** with 5% grounding weight

## 📊 Results

| Metric | Value |
|--------|-------|
| **Package Version** | **0.4.0** (Beta) 🎉 |
| **Test Suite** | **299 tests** (all passing) ✅ |
| **Test Coverage** | **95%** on langcalc.algebra ⭐ |
| **Infinigram Tests** | **36 tests** (NEW) ✨ |
| Memory Efficiency | 34x better (1GB vs 34GB) |
| Query Latency | 0.03ms (suffix arrays) |
| Perplexity Reduction | 70% |
| LLM Overhead | 6.5% |
| Optimal Weight | 95% LLM + 5% suffix |

## 📚 Documentation

### Complete Documentation

**[📖 Read the Full Documentation →](https://langcalc.readthedocs.io/)**

The complete LangCalc documentation includes:

- **[Getting Started](docs/getting-started/index.md)** - Installation, quick start, core concepts
- **[Projection System](docs/projection-system/index.md)** - Mathematical formalism and implementation
- **[User Guide](docs/user-guide/index.md)** - Comprehensive guides and examples
- **[API Reference](docs/api/index.md)** - Detailed API documentation
- **[Advanced Topics](docs/advanced/index.md)** - Suffix arrays, grounding, performance
- **[Development Guide](docs/development/index.md)** - Contributing, testing, code style

### Quick Links

- **[Algebraic Design](docs/ALGEBRA_DESIGN.md)** - Complete API reference
- **[Test Suite](tests/README.md)** - Comprehensive test documentation and coverage
- **[Academic Paper](papers/paper.pdf)** - Formal treatment
- **[Examples](examples/algebra_examples.py)** - Practical usage
- **[Results Analysis](docs/experiment_analysis.md)** - Experimental findings
- **[Test Coverage Reports](TEST_COVERAGE_SUMMARY.md)** - Detailed coverage analysis

## 🔬 Research Contributions

1. **Unified Algebraic Framework**: Treating language models as algebraic objects
2. **Lightweight Grounding**: Minimal weight (5%) for maximum benefit
3. **Suffix Array Integration**: Scalable alternative to n-grams
4. **Context Transformations**: Sophisticated operators for model composition

## 📖 Citation

```bibtex
@article{langcalc-2025,
  title={LangCalc: A Calculus for Compositional Language Modeling with Infinigram Grounding},
  year={2025}
}
```

## 🚦 Future Work

- Learnable operator weights
- Automatic composition search  
- GPU acceleration
- Distributed suffix arrays

---

**LangCalc** - A calculus for language models. Built with mathematical elegance and engineering pragmatism.
