# Algebraic Framework for Language Model Composition

## Overview

This framework provides a comprehensive algebraic API for composing language models using mathematical operators and functional transformations. It follows the Unix philosophy of "do one thing well" and enables elegant, composable model architectures through intuitive mathematical notation.

## Core Design Principles

### 1. Models as Algebraic Objects

Language models are treated as first-class algebraic objects that support mathematical operations:

```python
# Basic arithmetic
grounded = 0.95 * llm + 0.03 * ngram + 0.02 * suffix

# Set operations
optimistic = llm | wiki  # Union (max probability)
conservative = llm & wiki # Intersection (min probability)

# Temperature control
sharp = model ** 0.5      # Sharpen distribution
smooth = model ** 2.0     # Smooth distribution

# Context transformations
transformed = model << LongestSuffixTransform(sa)

# Function application
boosted = model >> boost_technical_terms
```

### 2. Composability

All components are designed to compose naturally:

```python
# Complex composition in one line
sophisticated = (
    0.7 * (llm << MaxKWordsTransform(10)) +
    0.2 * (ngram << LongestSuffixTransform(sa)) +
    0.1 * (suffix << RecencyWeightTransform(0.9))
) ** 0.85  # Apply temperature
```

### 3. Mathematical Consistency

The framework respects mathematical laws:

- **Associativity**: `(a + b) + c = a + (b + c)`
- **Commutativity**: `a + b = b + a` (for commutative operations)
- **Distributivity**: `α*(a + b) = α*a + α*b`
- **Identity elements**: `a + 0*b = a`

## Operator Reference

### Arithmetic Operators

| Operator | Operation | Example | Description |
|----------|-----------|---------|-------------|
| `+` | Addition/Mixture | `model1 + model2` | Linear mixture with equal weights |
| `*` | Scalar multiplication | `0.7 * model` | Weight a model |
| `-` | Subtraction | `model1 - model2` | Subtract probabilities |
| `/` | Division/Normalization | `model / 2.0` | Scale down probabilities |
| `**` | Power/Temperature | `model ** 0.8` | Apply temperature scaling |

### Set Operators

| Operator | Operation | Example | Description |
|----------|-----------|---------|-------------|
| `\|` | Union/Max | `model1 \| model2` | Take maximum probability |
| `&` | Intersection/Min | `model1 & model2` | Take minimum probability |
| `^` | Symmetric difference | `model1 ^ model2` | Tokens in one but not both |

### Transformation Operators

| Operator | Operation | Example | Description |
|----------|-----------|---------|-------------|
| `<<` | Apply transform | `model << transform` | Transform context before prediction |
| `>>` | Apply function | `model >> func` | Transform predictions after computation |

## Context Transformations

### Built-in Transforms

```python
# Longest suffix matching
model << LongestSuffixTransform(suffix_array, max_length=10)

# Limit context window
model << MaxKWordsTransform(5)

# Apply recency weighting
model << RecencyWeightTransform(decay_rate=0.9)

# Focus on specific word types
model << FocusTransform('content')  # content words only

# Pattern matching
model << PatternMatchTransform(suffix_array)

# Sliding window
model << SlidingWindowTransform(window_size=10, stride=2)
```

### Transform Composition

```python
# Sequential composition
transform = MaxKWordsTransform(10) | LongestSuffixTransform(sa)

# Parallel composition
transform = FocusTransform('content') & RecencyWeightTransform()
```

## Advanced Models

### Adaptive Models

```python
# Adaptive suffix matching
adaptive = AdaptiveSuffixModel(suffix_array, min_n=2, max_n=10)

# Recency-biased predictions
recency = RecencyBiasedModel(base_model, decay_rate=0.95)

# Caching for temporal coherence
cached = CacheModel(base_model, cache_size=100, cache_weight=0.1)
```

### Ensemble Methods

```python
# Simple average
ensemble = (model1 + model2 + model3) / 3

# Weighted ensemble
ensemble = 0.5 * model1 + 0.3 * model2 + 0.2 * model3

# Max voting
ensemble = model1 | model2 | model3

# Conservative (min)
ensemble = model1 & model2 & model3

# Geometric mean
ensemble = create_ensemble([m1, m2, m3], method='product')
```

## Builder Pattern

For complex compositions, use the fluent builder:

```python
model = (ModelBuilder()
    .with_base(llm)
    .multiply(0.7)
    .add(ngram, 0.2)
    .add(suffix, 0.1)
    .transform(MaxKWordsTransform(10))
    .temperature(0.9)
    .filter(lambda t: len(t) > 2)
    .threshold(0.001)
    .top_k(50)
    .build())
```

## Practical Examples

### 1. Basic Grounding

```python
# The fundamental grounding equation
grounded = 0.95 * llm + 0.03 * ngram + 0.02 * suffix
```

### 2. Sophisticated Grounding

```python
grounded = (
    0.95 * llm +
    0.03 * (ngram << LongestSuffixTransform(sa)) +
    0.02 * (suffix << MaxKWordsTransform(5))
) ** 0.9  # Temperature
```

### 3. Context-Sensitive Model

```python
class ContextSensitiveModel(AlgebraicModel):
    def predict(self, context, top_k=50):
        if is_technical(context):
            return technical_model.predict(context, top_k)
        elif is_conversational(context):
            return chat_model.predict(context, top_k)
        else:
            return balanced_model.predict(context, top_k)
```

### 4. Multi-Scale Model

```python
multi_scale = (
    0.05 * (char_model << MaxKWordsTransform(2)) +
    0.50 * (word_model << MaxKWordsTransform(5)) +
    0.30 * (phrase_model << PatternMatchTransform(sa)) +
    0.15 * (sentence_model << RecencyWeightTransform(0.9))
)
```

### 5. Production System

```python
production = (ModelBuilder()
    .with_base(llm * 0.70)
    .add(wiki_grounded, 0.10)
    .add(news_recent, 0.05)
    .add(exact_filtered, 0.08)
    .add(adaptive, 0.05)
    .add(multi_scale, 0.02)
    .temperature(0.85)
    .threshold(0.001)
    .top_k(100)
    .build())

# Add caching and recency
final = RecencyBiasedModel(
    CacheModel(production, cache_size=50, cache_weight=0.05),
    decay_rate=0.95
)
```

## Key Benefits

1. **Intuitive Notation**: Mathematical operators make compositions readable
2. **Composability**: Components combine naturally like mathematical functions
3. **Flexibility**: Easy to experiment with different architectures
4. **Extensibility**: Simple to add new operators and transforms
5. **Production Ready**: Includes caching, filtering, and optimization features

## Architecture Components

### Core Classes

- `AlgebraicModel`: Base class with operator overloading
- `ContextTransform`: Base class for context transformations
- `ModelOperator`: Base class for custom operators

### Model Types

- `MixtureModel`: Weighted linear combination
- `MaxModel`/`MinModel`: Set operations
- `TransformedModel`: Models with context transformation
- `TemperatureModel`: Temperature-adjusted predictions
- `FilteredModel`: Filtered predictions
- `ThresholdModel`: Probability threshold
- `CacheModel`: Temporal coherence via caching
- `RecencyBiasedModel`: Recency-weighted predictions

### Transforms

- `LongestSuffixTransform`: Find longest matching suffix
- `MaxKWordsTransform`: Limit context length
- `RecencyWeightTransform`: Exponential decay weighting
- `FocusTransform`: Focus on specific word types
- `PatternMatchTransform`: Match corpus patterns
- `ComposedTransform`: Sequential composition
- `ParallelTransform`: Parallel composition

## Integration with Existing Systems

The framework seamlessly integrates with:
- Suffix arrays for efficient pattern matching
- N-gram models for statistical grounding
- Any model with a `predict(context, top_k)` method via `AlgebraicModelWrapper`

## Summary

This algebraic framework transforms language model composition from complex engineering to elegant mathematical expression. By treating models as algebraic objects, we can:

1. Express sophisticated architectures in simple equations
2. Compose models using intuitive mathematical operators
3. Build production systems from simple, testable components
4. Maintain mathematical consistency and properties
5. Create adaptive, context-aware systems with minimal code

The result is a powerful, flexible system that makes complex model compositions as simple as writing mathematical equations.