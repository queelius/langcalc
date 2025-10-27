#!/usr/bin/env python3
"""Generate stub documentation pages for MkDocs."""

import os
from pathlib import Path

DOCS_DIR = Path("/home/spinoza/github/beta/langcalc/docs")

# Define stub content for each page
STUBS = {
    "user-guide/models.md": """# Language Models

Working with different types of language models in LangCalc.

## Infinigram Models

Variable-length n-gram models using suffix arrays.

## N-Gram Models

Traditional fixed-length n-gram models.

## LLM Integration

Working with Large Language Models via Ollama.

## Mock Models

Testing models for development.

_This page is under construction. See the [API Reference](../api/models.md) for detailed information._
""",

    "user-guide/algebra.md": """# Algebraic Operations

Complete reference for algebraic operations on language models.

## Arithmetic Operations

- Addition (+)
- Multiplication (*)
- Subtraction (-)
- Division (/)

## Set Operations

- Maximum (|)
- Minimum (&)
- Symmetric Difference (^)

## Transformations

- Temperature Scaling (**)
- Context Transform (<<)
- Function Application (>>)
- Negation (~)

_This page is under construction. See [Core Concepts](../getting-started/concepts.md) for examples._
""",

    "user-guide/transformations.md": """# Context Transformations

Transforming context before model prediction.

## Built-in Transformations

- Longest Suffix Transform
- Max K Words Transform
- Recency Weight Transform
- Focus Transform

## Composing Transformations

Sequential and parallel composition.

_This page is under construction. See the [Projection System](../projection-system/index.md) for related concepts._
""",

    "user-guide/examples.md": """# Examples & Patterns

Real-world usage patterns and complete examples.

## Basic Examples

See [Quick Start](../getting-started/quickstart.md) for introductory examples.

## Advanced Examples

Check the `/home/spinoza/github/beta/langcalc/examples/` directory for complete examples.

## Interactive Notebooks

Explore Jupyter notebooks in `/home/spinoza/github/beta/langcalc/notebooks/`.

_This page is under construction._
""",

    "user-guide/best-practices.md": """# Best Practices

Tips and recommendations for using LangCalc effectively.

## Performance

- Use augmentations for simple transformations
- Profile before optimizing
- Choose appropriate max_length for infinigrams

## Composition

- Follow canonical projection ordering
- Test both projection and augmentation approaches
- Start simple and add complexity incrementally

## Testing

- Verify projections with small test cases
- Check model composition properties
- Measure query latency

_This page is under construction._
""",

    "api/index.md": """# API Reference

Detailed API documentation for LangCalc.

## Modules

- **[Core](core.md)** - Core classes and interfaces
- **[Models](models.md)** - Language model implementations
- **[Projections](projections.md)** - Context projection classes
- **[Augmentations](augmentations.md)** - Corpus augmentation classes
- **[Algebra](algebra.md)** - Algebraic operations

## Quick Links

- [Full Package Index](https://github.com/queelius/langcalc/tree/master/langcalc)
- [Test Suite](https://github.com/queelius/langcalc/tree/master/tests)
- [Examples](https://github.com/queelius/langcalc/tree/master/examples)

_This page is under construction. For now, please refer to the source code and tests for API details._
""",

    "api/core.md": """# Core API

Core classes and interfaces.

## LanguageModel

Abstract base class for all language models.

## Infinigram

Variable-length n-gram model using suffix arrays.

## Functions

- `create_infinigram()` - Create infinigram model
- `predict()` - Get probability distribution
- `sample()` - Generate tokens

_This page is under construction._
""",

    "api/models.md": """# Models API

Language model implementations.

## InfinigramModel

## NGramModel

## OllamaModel

## MockLLM

## ProjectedModel

## MultiProjectionModel

_This page is under construction. See source code at `/home/spinoza/github/beta/langcalc/langcalc/models/`._
""",

    "api/projections.md": """# Projections API

Context projection classes.

## Base Classes

- `Projection` - Abstract base class

## Basic Projections

- `IdentityProjection`
- `RecencyProjection`
- `TruncationProjection`
- `LowercaseProjection`
- `UppercaseProjection`
- `WhitespaceProjection`
- `UnicodeNormalizationProjection`

## Advanced Projections

- `EditDistanceProjection`
- `LongestSuffixProjection`
- `SynonymProjection`

_This page is under construction. See [Reference Implementation](../projection-system/implementation.md)._
""",

    "api/augmentations.md": """# Augmentations API

Corpus augmentation classes.

## Base Classes

- `Augmentation` - Abstract base class

## Basic Augmentations

- `LowercaseAugmentation`
- `CaseAugmentation`
- `WhitespaceAugmentation`
- `NFCAugmentation`
- `StandardAugmentation`

## Advanced Augmentations

- `UnicodeAugmentation`
- `PunctuationAugmentation`
- `ASCIIFoldingAugmentation`

_This page is under construction. See [Canonical Augmentations](../projection-system/augmentations.md)._
""",

    "api/algebra.md": """# Algebra API

Algebraic operations and transformations.

## Operators

Implemented as operator overloads on LanguageModel.

## Context Transforms

- `LongestSuffixTransform`
- `MaxKWordsTransform`
- `RecencyWeightTransform`
- `FocusTransform`

_This page is under construction._
""",

    "advanced/index.md": """# Advanced Topics

Advanced concepts and optimization techniques.

## Contents

- **[Suffix Arrays](suffix-arrays.md)** - Understanding suffix array implementation
- **[Lightweight Grounding](grounding.md)** - LLM grounding with minimal overhead
- **[Performance Optimization](performance.md)** - Making LangCalc faster
- **[Extending LangCalc](extending.md)** - Creating custom models and projections

## Topics

### Suffix Arrays

How LangCalc achieves 34× memory efficiency over n-grams.

### Lightweight Grounding

Achieving 70% perplexity reduction with just 5% weight.

### Performance

Optimizing query latency and memory usage.

### Extending

Building custom models, projections, and augmentations.
""",

    "advanced/suffix-arrays.md": """# Suffix Arrays

Understanding LangCalc's efficient pattern matching.

## What are Suffix Arrays?

Data structure for efficient substring search.

## Implementation

See `langcalc/data/suffix_array.py`.

## Performance

- Memory: O(n)
- Query: O(m log n)
- 34× more efficient than hash-based n-grams

_This page is under construction. See source code for implementation details._
""",

    "advanced/grounding.md": """# Lightweight Grounding

Grounding LLMs with minimal overhead.

## Research Results

- Optimal weight: 95% LLM + 5% suffix array
- Perplexity reduction: 70%
- Overhead: Only 6.5% (2.66ms)

## Implementation

See `examples/lightweight_experiments.py`.

_This page is under construction._
""",

    "advanced/performance.md": """# Performance Optimization

Making LangCalc faster and more memory efficient.

## Memory Optimization

- Choose appropriate max_length
- Use augmentation vs projection wisely
- Stream large corpora

## Query Optimization

- Cache suffix arrays
- Batch predictions
- Use appropriate top_k values

## Profiling

Use pytest benchmarks and cProfile.

_This page is under construction._
""",

    "advanced/extending.md": """# Extending LangCalc

Creating custom components.

## Custom Models

Implement the `LanguageModel` interface.

## Custom Projections

Subclass `Projection` abstract base class.

## Custom Augmentations

Subclass `Augmentation` abstract base class.

## Custom Transforms

Create transformation functions for `<<` operator.

_This page is under construction. See [Development Guide](../development/contributing.md)._
""",

    "development/index.md": """# Development Guide

Contributing to LangCalc.

## Contents

- **[Contributing](contributing.md)** - How to contribute
- **[Testing](testing.md)** - Running and writing tests
- **[Code Style](style.md)** - Coding standards
- **[Release Process](releases.md)** - How releases are made

## Quick Start for Contributors

1. Fork the repository
2. Install dev dependencies: `pip install -e .[dev]`
3. Make changes
4. Run tests: `pytest tests/`
5. Submit pull request

See [Contributing Guide](contributing.md) for details.
""",

    "development/contributing.md": """# Contributing to LangCalc

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
""",

    "development/testing.md": """# Testing Guide

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
""",

    "development/style.md": """# Code Style Guide

Coding standards for LangCalc.

## Python Style

- Follow PEP 8
- Use Black for formatting
- Use flake8 for linting
- Use mypy for type checking

## Documentation

- Write clear docstrings
- Include examples in docstrings
- Document parameters and return values
- Add type hints

## Testing

- Write tests for new features
- Maintain high coverage (>80%)
- Use descriptive test names

## Tools

```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

_This page is under construction._
""",

    "development/releases.md": """# Release Process

How LangCalc releases are made.

## Versioning

LangCalc follows semantic versioning (SemVer).

## Release Checklist

1. Update version number
2. Update CHANGELOG.md
3. Run full test suite
4. Build documentation
5. Create git tag
6. Push to GitHub
7. Create GitHub release

_This page is under construction._
""",

    "about/index.md": """# About LangCalc

Learn more about the project.

## Contents

- **[License](license.md)** - MIT License
- **[Changelog](changelog.md)** - Version history
- **[Academic Paper](paper.md)** - Research paper
- **[Citation](citation.md)** - How to cite

## Project Information

- **Version**: 0.4.0 (Beta)
- **License**: MIT
- **Repository**: [github.com/queelius/langcalc](https://github.com/queelius/langcalc)

## Community

- **GitHub Discussions**: Ask questions and share ideas
- **GitHub Issues**: Report bugs and request features
- **Pull Requests**: Contribute code and documentation
""",

    "about/license.md": """# License

LangCalc is released under the MIT License.

## MIT License

Copyright (c) 2025 LangCalc Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Third-Party Licenses

LangCalc depends on:

- NumPy (BSD License)
- SciPy (BSD License)
- infinigram (MIT License)

See individual packages for their license terms.
""",

    "about/changelog.md": """# Changelog

Version history for LangCalc.

## [0.4.0] - 2025-01-29

### Added
- Infinigram variable-length n-gram support
- Comprehensive projection system formalism
- 36 new infinigram tests
- MkDocs documentation

### Changed
- Rebranded from "Model Algebra" to "LangCalc"
- Improved test coverage to 95%
- Updated package structure

## [0.3.0] - Previous versions

See git history for earlier changes.

_This changelog follows [Keep a Changelog](https://keepachangelog.com/) format._
""",

    "about/paper.md": """# Academic Paper

Research paper on LangCalc.

## Title

**LangCalc: A Calculus for Compositional Language Modeling with Infinigram Grounding**

## Abstract

This paper introduces LangCalc, an algebraic framework for compositional language modeling.
The key innovation is lightweight grounding: combining LLMs with suffix array-based pattern
matching using just 5% weight to achieve 70% perplexity reduction.

## PDF

The full paper is available at: `/home/spinoza/github/beta/langcalc/papers/paper.pdf`

## Key Results

- 70% perplexity reduction with 5% grounding weight
- 34× memory efficiency over n-gram hash tables
- O(m log n) query time with suffix arrays
- Minimal overhead (6.5%, 2.66ms) with real LLMs

## Citation

See [How to Cite](citation.md) for BibTeX.
""",

    "about/citation.md": """# How to Cite

If you use LangCalc in your research, please cite:

## BibTeX

```bibtex
@article{langcalc-2025,
  title={LangCalc: A Calculus for Compositional Language Modeling with Infinigram Grounding},
  year={2025}
}
```

## APA Style

LangCalc Project. (2025). LangCalc: A Calculus for Compositional Language Modeling with Infinigram Grounding.

## MLA Style

LangCalc Project. "LangCalc: A Calculus for Compositional Language Modeling with Infinigram Grounding." 2025.

## In Text

When referring to LangCalc in text:

> "We used LangCalc (LangCalc Project, 2025), an algebraic framework for compositional language modeling..."

## Software Citation

If citing the software specifically:

```bibtex
@software{langcalc-software,
  title={LangCalc: A Calculus for Language Models},
  author={LangCalc Project},
  year={2025},
  url={https://github.com/queelius/langcalc},
  version={0.4.0}
}
```
""",
}

def create_stubs():
    """Create all stub documentation pages."""
    for path, content in STUBS.items():
        file_path = DOCS_DIR / path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Creating {path}...")
        with open(file_path, 'w') as f:
            f.write(content)

    print(f"\n✓ Created {len(STUBS)} documentation stub pages")

if __name__ == "__main__":
    create_stubs()
