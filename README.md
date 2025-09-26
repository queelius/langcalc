# Algebraic Language Model Composition

An elegant mathematical framework for composing language models through algebraic operations, featuring efficient suffix array-based grounding and lightweight model mixing.

## 🎯 Overview

This project introduces a comprehensive algebraic framework for language model composition that treats models as first-class mathematical objects. The key innovation is **lightweight grounding**: combining Large Language Models (LLMs) with suffix array-based pattern matching using just 5% weight to achieve 70% perplexity reduction.

```python
# Express sophisticated models as elegant algebra
model = (0.7 * llm + 0.2 * (wiki << LongestSuffix(sa)) + 0.1 * ngram) ** 0.9
```

## 📁 Project Structure

```
.
├── src/                 # Core implementation
│   ├── model_algebra.py        # Algebraic framework with 10+ operators
│   ├── lightweight_grounding.py # Main grounding system
│   ├── suffix_array_demo.py    # Efficient suffix arrays
│   └── ngram_projections/      # Package modules
├── tests/              # Test suites and experiments
├── docs/               # Documentation and results
├── notebooks/          # Jupyter demos
├── papers/             # Academic paper (LaTeX)
├── examples/           # Usage examples
├── data/              # Experiment results
├── scripts/           # Utility scripts
└── wikipedia_data/    # Wikipedia n-gram models

```

## 🚀 Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib

# Run basic demo
python src/suffix_array_demo.py

# Run full experiments
python tests/lightweight_experiments.py

# Interactive notebook
jupyter notebook notebooks/lightweight_grounding_demo.ipynb
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

### Production Ready
- **Tested with Ollama** at 192.168.0.225
- **Only 6.5% overhead** (2.66ms) with real LLMs
- **70% perplexity reduction** with 5% grounding weight

## 📊 Results

| Metric | Value |
|--------|-------|
| Memory Efficiency | 34x better (1GB vs 34GB) |
| Query Latency | 0.03ms |
| Perplexity Reduction | 70% |
| LLM Overhead | 6.5% |
| Optimal Weight | 95% LLM + 5% suffix |

## 📚 Documentation

- **[Algebraic Design](docs/ALGEBRA_DESIGN.md)** - Complete API reference
- **[Academic Paper](papers/paper.pdf)** - Formal treatment
- **[Examples](examples/algebra_examples.py)** - Practical usage
- **[Results Analysis](docs/experiment_analysis.md)** - Experimental findings

## 🔬 Research Contributions

1. **Unified Algebraic Framework**: Treating language models as algebraic objects
2. **Lightweight Grounding**: Minimal weight (5%) for maximum benefit
3. **Suffix Array Integration**: Scalable alternative to n-grams
4. **Context Transformations**: Sophisticated operators for model composition

## 📖 Citation

```bibtex
@article{algebraic-lm-2025,
  title={An Algebraic Framework for Language Model Composition with Efficient Suffix-Based Grounding},
  year={2025}
}
```

## 🚦 Future Work

- Learnable operator weights
- Automatic composition search  
- GPU acceleration
- Distributed suffix arrays

---

Built with mathematical elegance and engineering pragmatism.
