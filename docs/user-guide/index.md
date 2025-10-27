# User Guide

Comprehensive guides for using LangCalc in practice.

## Contents

- **[Language Models](models.md)** - Working with different model types
- **[Algebraic Operations](algebra.md)** - Composing models mathematically
- **[Context Transformations](transformations.md)** - Transform context before prediction
- **[Examples & Patterns](examples.md)** - Real-world usage patterns
- **[Best Practices](best-practices.md)** - Tips and recommendations

## Quick Navigation

### By Use Case

- **Text generation** → [Models](models.md) + [Algebra](algebra.md)
- **Pattern matching** → [Transformations](transformations.md)
- **Model mixing** → [Algebra](algebra.md) + [Examples](examples.md)
- **Performance optimization** → [Best Practices](best-practices.md)

### By Experience Level

- **Beginners** → Start with [Models](models.md), then [Examples](examples.md)
- **Intermediate** → Focus on [Algebra](algebra.md) and [Transformations](transformations.md)
- **Advanced** → See [Best Practices](best-practices.md) and [Advanced Topics](../advanced/index.md)

## Common Workflows

### Workflow 1: Building a Text Predictor

1. Load corpus and create infinigram model
2. Optionally compose with LLM
3. Add projections for robustness
4. Optimize based on use case

See [Examples](examples.md#building-text-predictor) for details.

### Workflow 2: Grounding an LLM

1. Create infinigram from knowledge base
2. Create or load LLM
3. Mix with optimal weights (95% LLM + 5% infinigram)
4. Evaluate perplexity improvement

See [Examples](examples.md#grounding-llm) for details.

### Workflow 3: Custom Model Composition

1. Define component models
2. Choose algebraic operations
3. Tune weights experimentally
4. Add temperature scaling if needed

See [Algebra](algebra.md) for complete operator reference.
