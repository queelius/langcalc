# LangCalc: A Calculus for Language Models

An elegant Python package providing an algebraic API for composing language models using mathematical operators. LangCalc enables sophisticated model composition including infinigrams, n-grams, and neural models, following the Unix philosophy of composable, single-purpose tools.

## Key Features

### 1. **Algebraic Model Composition**
Combine language models using intuitive mathematical operators:

```python
# Addition creates mixtures
model = ngram_model + llm

# Weighted combinations
model = 0.3 * ngram_model + 0.7 * gpt_model

# Sequential composition
model = ngram >> transformer >> llm

# Fallback/ensemble
model = fast_model | accurate_model

# Apply projections
model = ngram @ recency_projection
```

### 2. **Composable Projections**
First-class projection functions that transform contexts:

```python
# Compose projections
projection = recency >> semantic >> attention

# Union of projections
projection = recency | edit_distance

# Apply to models
model = language_model @ projection
```

### 3. **Efficient N-Gram Operations**
Built on suffix arrays for O(m log n) pattern matching:

```python
# Create n-gram model with efficient retrieval
ngram = NGramModel(corpus, n=3)

# Find longest matching suffix
position, length = ngram.suffix_array.find_longest_suffix(query)
```

### 4. **Functional Combinators**
Higher-order functions for sophisticated composition:

```python
# Compose multiple models
model = compose(ngram, transformer, llm)

# Create ensembles
model = ensemble([model1, model2, model3])

# Conditional selection
model = choose(condition, if_true=ngram, if_false=llm)
```

## Installation

```bash
pip install ngram-projections
```

Or install from source:

```bash
git clone https://github.com/yourusername/ngram-projections.git
cd ngram-projections
pip install -e .
```

## Quick Example

```python
from ngram_projections import NGramModel, MockLLM
from ngram_projections.projections import RecencyProjection, SemanticProjection

# Create models
ngram = NGramModel(corpus, n=3)
llm = MockLLM(vocab_size=1000)

# Create projections
recency = RecencyProjection(corpus)
semantic = SemanticProjection()

# Compose using algebra
model = 0.3 * (ngram @ recency) + 0.7 * (llm @ semantic)

# Use the composed model
logprobs = model.logprobs(tokens=[1, 2, 3], context=[4, 5, 6])
generated = model.sample(context=[7, 8, 9], max_tokens=10)
```

## Architecture

```
ngram_projections/
├── models/
│   ├── base.py          # LanguageModel ABC with operators
│   ├── ngram.py         # N-gram model implementation
│   ├── llm.py           # LLM wrappers (HuggingFace, etc.)
│   └── mixture.py       # Mixture and ensemble models
├── projections/
│   ├── base.py          # Projection ABC
│   ├── recency.py       # Longest suffix matching
│   ├── edit_distance.py # Edit distance projection
│   └── semantic.py      # Embedding-based projection
├── data/
│   └── suffix_array.py  # Efficient suffix array
└── algebra/
    └── combinators.py   # Functional combinators
```

## Core Abstractions

### LanguageModel
Base class supporting algebraic operations:
- `+` : Create mixture models
- `*` : Weight models
- `>>` : Sequential composition
- `|` : Fallback/union
- `@` : Apply projections

### Projection
Transform contexts for retrieval:
- `RecencyProjection`: Longest suffix matching
- `EditDistanceProjection`: Approximate matching
- `SemanticProjection`: Embedding similarity
- `AttentionProjection`: Attention-based selection

### Combinators
Functional programming patterns:
- `compose`: Sequential composition
- `ensemble`: Model ensembles
- `cascade`: Confidence-based fallback
- `choose`: Conditional selection
- `memoize`: Add caching

## Philosophy

This package embodies:
- **Unix Philosophy**: Do one thing well, compose freely
- **Mathematical Elegance**: Express complex ideas simply
- **Functional Programming**: Pure functions, composability
- **Zero-Cost Abstractions**: Performance without compromise

## Complex Composition Example

```python
# Build a sophisticated hybrid model
hybrid = memoize(
    0.3 * (
        0.6 * (ngram_3 @ recency) +
        0.4 * (ngram_5 @ semantic)
    ) +
    0.7 * (fast_llm | (accurate_llm @ attention))
)
```

This creates a memoized model that:
1. Combines two n-gram models with different projections (60/40 split)
2. Falls back from fast to accurate LLM with attention
3. Mixes n-gram and LLM components (30/70 split)
4. Caches results for efficiency

## Performance

- Suffix array construction: O(n log n)
- Pattern search: O(m log n)
- Projection application: O(k) for k-length context
- Model composition: Minimal overhead

## Testing

```bash
# Run tests
python test_algebra.py

# Run comprehensive demo
cd examples
python demo_algebra.py
```

## Contributing

Contributions welcome! The codebase emphasizes:
- Clean, composable abstractions
- Type hints throughout
- Minimal dependencies
- Comprehensive tests

## License

MIT License - See LICENSE file for details.

## Citation

If you use this package in research, please cite:
```bibtex
@software{ngram_projections,
  title={NGram Projections: Algebraic Language Model Composition},
  author={NGram Projections Contributors},
  year={2025},
  url={https://github.com/yourusername/ngram-projections}
}
```

## Acknowledgments

Inspired by:
- The n-gram projection paper and InfiniGram
- Alex Stepanov's generic programming
- Rich Hickey's simplicity philosophy
- Category theory and functional programming