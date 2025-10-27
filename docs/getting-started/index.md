# Getting Started with LangCalc

Welcome to LangCalc! This guide will help you get up and running with the algebraic framework for compositional language modeling.

## What You'll Learn

This section covers everything you need to start using LangCalc:

1. **[Installation](installation.md)** - How to install LangCalc and its dependencies
2. **[Quick Start](quickstart.md)** - A 5-minute tutorial to create your first model
3. **[Core Concepts](concepts.md)** - Understanding projections, augmentations, and algebraic operations

## Prerequisites

Before starting, you should have:

- **Python 3.8 or higher** installed
- Basic understanding of **language models** and probability distributions
- Familiarity with **NumPy** (helpful but not required)

## Learning Path

### For Beginners

If you're new to LangCalc:

1. Start with [Installation](installation.md) to set up your environment
2. Follow the [Quick Start](quickstart.md) for hands-on examples
3. Read [Core Concepts](concepts.md) to understand the fundamentals
4. Explore the [User Guide](../user-guide/index.md) for practical applications

### For Researchers

If you're interested in the mathematical foundations:

1. Install LangCalc following the [Installation](installation.md) guide
2. Read the [Mathematical Formalism](../projection-system/formalism.md)
3. Study the [Projection-Augmentation Duality](../projection-system/augmentations.md)
4. Review the [Academic Paper](../about/paper.md)

### For Developers

If you want to extend or contribute to LangCalc:

1. Install with dev dependencies: `pip install -e .[dev]`
2. Review the [Reference Implementation](../projection-system/implementation.md)
3. Check the [Contributing Guide](../development/contributing.md)
4. Explore the [Testing Documentation](../development/testing.md)

## Quick Example

Here's a taste of what you can do with LangCalc:

```python
from langcalc import Infinigram, NGramModel

# Create models
corpus = [1, 2, 3, 4, 2, 3, 5, 6, 2, 3, 4]
infini = Infinigram(corpus, max_length=10)
ngram = NGramModel(corpus, n=3)

# Compose using algebra
model = 0.7 * infini + 0.3 * ngram

# Make predictions
context = [2, 3]
probs = model.predict(context)
```

## Next Steps

Ready to dive in? Start with [Installation](installation.md)!

## Getting Help

If you get stuck:

- Check the [User Guide](../user-guide/index.md) for detailed examples
- Browse the [API Reference](../api/index.md) for technical details
- Ask questions in [GitHub Discussions](https://github.com/queelius/langcalc/discussions)
- Report bugs in [GitHub Issues](https://github.com/queelius/langcalc/issues)

## Related Resources

- [Examples Directory](https://github.com/queelius/langcalc/tree/master/examples) - Complete working examples
- [Jupyter Notebooks](https://github.com/queelius/langcalc/tree/master/notebooks) - Interactive tutorials
- [Test Suite](https://github.com/queelius/langcalc/tree/master/tests) - Reference implementations
