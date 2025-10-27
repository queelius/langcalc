# LangCalc: A Calculus for Language Models

<div align="center">

**An elegant mathematical framework for compositional language modeling**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-299%20passing-brightgreen.svg)](https://github.com/queelius/langcalc/tree/master/tests)
[![Coverage](https://img.shields.io/badge/coverage-95%25-green.svg)](https://github.com/queelius/langcalc)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/queelius/langcalc/blob/master/LICENSE)

</div>

---

## What is LangCalc?

LangCalc introduces a comprehensive **algebraic framework** for language model composition that treats models as first-class mathematical objects. The key innovation is **lightweight grounding**: combining Large Language Models (LLMs) with suffix array-based pattern matching (infinigrams) using just 5% weight to achieve 70% perplexity reduction.

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

---

## Key Features

### Mathematical Elegance

- **10+ algebraic operators**: `+`, `*`, `|`, `&`, `^`, `**`, `>>`, `<<`, `~`
- **Context transformations**: LongestSuffix, MaxKWords, RecencyWeight
- **Proven properties**: Associativity, distributivity, composability

### Production Ready

- **299 comprehensive tests** with 100% pass rate
- **95% code coverage** on core algebraic framework
- **Tested with real LLMs** (Ollama integration)
- **Minimal overhead**: Only 6.5% (2.66ms) with real models

### Efficient Pattern Matching

- **34x more memory efficient** than n-gram hash tables
- **O(m log n) query time** with binary search
- **Variable-length patterns** without pre-computing n
- **Incremental updates** for streaming data

### Projection System (NEW)

A rigorous mathematical framework for context transformation and corpus augmentation:

- **Projections**: Query-time context transformations ($\pi: \Sigma^* \times 2^{\Sigma^*} \to \Sigma^*$)
- **Augmentations**: Training-time corpus expansion ($\alpha: 2^{\Sigma^*} \to 2^{\Sigma^*}$)
- **Duality Theorem**: Trade space for time via augmentation
- **Composition algebra**: Build complex pipelines from simple parts

---

## Quick Start

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

### Your First Model

```python
from langcalc import Infinigram, NGramModel

# Create an infinigram model
corpus = [1, 2, 3, 4, 2, 3, 5, 6, 2, 3, 4]
model = Infinigram(corpus, max_length=10)

# Predict next token
context = [2, 3]
probs = model.predict(context)  # Variable-length suffix matching

# Compose models
ngram = NGramModel(corpus, n=3)
mixture = 0.7 * model + 0.3 * ngram
```

See the [Quick Start Guide](getting-started/quickstart.md) for more examples.

---

## Core Concepts

### Algebraic Framework

LangCalc treats language models as algebraic objects supporting mathematical operations:

```python
# Arithmetic combinations
ensemble = 0.7 * llm + 0.2 * wiki + 0.1 * ngram

# Set operations
best_of = llm | wiki  # max probability
conservative = llm & wiki  # min probability

# Temperature scaling
creative = model ** 1.5  # Higher temperature
focused = model ** 0.5   # Lower temperature
```

### Context Transformations

Transform context before model prediction:

```python
from langcalc.algebra import LongestSuffixTransform, RecencyWeightTransform

# Find longest matching suffix in corpus
grounded = model << LongestSuffixTransform(suffix_array)

# Apply recency weighting
recent = model << RecencyWeightTransform(decay=0.9)

# Chain transformations
pipeline = model << (LongestSuffixTransform(sa) | RecencyWeightTransform(0.9))
```

### Projection System

The projection system enables flexible context transformation:

```python
from langcalc.projections import LowercaseProjection, WhitespaceProjection
from langcalc.models.projected import ProjectedModel

# Create projection pipeline
projection = (
    WhitespaceProjection() >>  # Normalize whitespace
    LowercaseProjection() >>   # Case-insensitive
    RecencyProjection(100)     # Keep recent tokens
)

# Apply to model
projected_model = ProjectedModel(base_model, projection, corpus)
```

See [Core Concepts](getting-started/concepts.md) for detailed explanations.

---

## Results

| Metric | Value |
|--------|-------|
| **Perplexity Reduction** | **70%** |
| **Optimal Weight** | 95% LLM + 5% suffix |
| **Memory Efficiency** | 34x better (1GB vs 34GB) |
| **Query Latency** | 0.03ms (suffix arrays) |
| **LLM Overhead** | 6.5% (2.66ms) |
| **Test Coverage** | 95% on core modules |

---

## Documentation Structure

<div class="grid cards" markdown>

-   :material-rocket-launch: **Getting Started**

    ---

    Installation, quick start, and core concepts

    [:octicons-arrow-right-24: Get Started](getting-started/index.md)

-   :material-function: **Projection System**

    ---

    Mathematical formalism for context transformation

    [:octicons-arrow-right-24: Learn More](projection-system/index.md)

-   :material-book-open: **User Guide**

    ---

    Comprehensive guides and examples

    [:octicons-arrow-right-24: Read Guide](user-guide/index.md)

-   :material-api: **API Reference**

    ---

    Detailed API documentation

    [:octicons-arrow-right-24: Browse API](api/index.md)

-   :material-cog: **Advanced Topics**

    ---

    Suffix arrays, grounding, performance

    [:octicons-arrow-right-24: Explore](advanced/index.md)

-   :material-code-braces: **Development**

    ---

    Contributing, testing, code style

    [:octicons-arrow-right-24: Contribute](development/index.md)

</div>

---

## Research Contributions

1. **Unified Algebraic Framework**: Treating language models as algebraic objects with mathematical operations
2. **Lightweight Grounding**: Minimal weight (5%) suffix array integration for maximum benefit (70% perplexity reduction)
3. **Suffix Array Integration**: Scalable alternative to n-grams with 34x memory efficiency
4. **Context Transformations**: Sophisticated operators for model composition
5. **Projection-Augmentation Duality**: Novel theorem enabling space-time tradeoffs

See our [Academic Paper](about/paper.md) for the formal treatment.

---

## Example: Complete Pipeline

Here's a complete example showing LangCalc's power:

```python
from langcalc import Infinigram
from langcalc.models import OllamaModel
from langcalc.projections import (
    EditDistanceProjection,
    LowercaseProjection,
    WhitespaceProjection,
    RecencyProjection
)
from langcalc.models.projected import ProjectedModel

# Load corpus
with open('wikipedia.txt', 'rb') as f:
    corpus = list(f.read())

# Create suffix array model
wiki = Infinigram(corpus, max_length=20)

# Create LLM
llm = OllamaModel(model_name='llama2', base_url='http://localhost:11434')

# Create projection pipeline
projection = (
    EditDistanceProjection(max_distance=1) >>  # Fix typos
    WhitespaceProjection() >>                   # Normalize whitespace
    LowercaseProjection() >>                    # Case-insensitive
    RecencyProjection(max_length=100)           # Keep recent context
)

# Compose models
projected_wiki = ProjectedModel(wiki, projection, corpus)
final_model = 0.95 * llm + 0.05 * projected_wiki

# Use the model
context = list("The capital of France is".encode('utf-8'))
probs = final_model.predict(context)
print(f"Next token probabilities: {probs}")
```

This example demonstrates:

- **Infinigram** for efficient pattern matching
- **Projection pipeline** for robust context transformation
- **Model composition** using algebraic operators
- **Lightweight grounding** with optimal mixing weights

---

## Community

- **GitHub**: [github.com/queelius/langcalc](https://github.com/queelius/langcalc)
- **Issues**: [Report bugs or request features](https://github.com/queelius/langcalc/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/queelius/langcalc/discussions)

---

## Citation

If you use LangCalc in your research, please cite:

```bibtex
@article{langcalc-2025,
  title={LangCalc: A Calculus for Compositional Language Modeling with Infinigram Grounding},
  year={2025}
}
```

---

## License

LangCalc is released under the [MIT License](about/license.md).

---

**Ready to get started?** Head to the [Quick Start Guide](getting-started/quickstart.md) or explore the [Projection System](projection-system/index.md)!
