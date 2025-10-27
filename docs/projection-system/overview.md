# Projection System Overview

The LangCalc projection system provides a rigorous mathematical framework for context transformation and corpus augmentation in language modeling.

## What are Projections?

A **projection** is a transformation that maps a query context onto a corpus, enabling flexible pattern matching and generalization.

**Mathematical Definition:**

$$\pi: \Sigma^* \times 2^{\Sigma^*} \to \Sigma^*$$

A projection $\pi$ takes:

- **Input**: Query context $x \in \Sigma^*$ and corpus $C \subseteq 2^{\Sigma^*}$
- **Output**: Transformed context $\pi(x, C) \in \Sigma^*$

**Example:**

```python
from langcalc.projections import LowercaseProjection

projection = LowercaseProjection()

# Transform "HELLO" to "hello" before matching
context = list("HELLO".encode('utf-8'))
transformed = projection.project(context, corpus)
# Result: list("hello".encode('utf-8'))
```

## What are Augmentations?

An **augmentation** expands the corpus by adding transformed variants.

**Mathematical Definition:**

$$\alpha: 2^{\Sigma^*} \to 2^{\Sigma^*}$$

An augmentation $\alpha$ takes:

- **Input**: Corpus $C$
- **Output**: Augmented corpus $\alpha(C)$ containing $C$ plus variants

**Example:**

```python
from langcalc.augmentations import LowercaseAugmentation

augmentation = LowercaseAugmentation()

# Add lowercase variant to corpus
corpus = list("Hello World".encode('utf-8'))
augmented = augmentation.augment(corpus)
# Result: corpus + list("hello world".encode('utf-8'))
```

## Key Innovation: Projection-Augmentation Duality

**Theorem (Duality):** For certain transformations, projections and augmentations are equivalent:

$$\text{LMS}(\pi(x, C), C) = \text{LMS}(x, \alpha(C))$$

This means:

- **Projecting the query** onto the original corpus
- **Augmenting the corpus** and using the original query

...produce the same matching results!

**Practical Implication:** Choose the more efficient approach:

- **Simple transformations** (case, whitespace) → Use **augmentation** (pay space, save time)
- **Complex transformations** (edit distance, semantic) → Use **projection** (save space, pay time)

## System Components

### 1. Mathematical Formalism

[Read the full formalism →](formalism.md)

- Formal definitions of projections and augmentations
- Projection algebra (composition operations)
- Complexity analysis
- Projected language models

### 2. Canonical Augmentations

[Browse the catalog →](augmentations.md)

Standard corpus augmentations:

- **Case normalization**: lowercase, uppercase, titlecase
- **Whitespace normalization**: collapsing, stripping
- **Unicode normalization**: NFC, NFD, NFKC, NFKD
- **Punctuation handling**: removal, normalization
- **Composite augmentations**: standard, aggressive

### 3. Ordering Principles

[Learn about ordering →](ordering.md)

Projections are **non-commutative** - order matters!

Canonical pipeline:

```
EditDistance >> Normalize >> Synonym >> LongestSuffix >> Recency
     ↓             ↓           ↓            ↓              ↓
Fix typos    Standardize   Expand     Find patterns   Focus context
```

### 4. Reference Implementation

[See the code →](implementation.md)

Complete Python implementation:

- Abstract base classes (`Projection`, `Augmentation`)
- Composition operators (`>>`, `|`, `+`)
- Basic and advanced projections
- Model integration (`ProjectedModel`, `MultiProjectionModel`)

## Quick Examples

### Example 1: Case-Insensitive Matching

**Using Projection:**

```python
from langcalc.projections import LowercaseProjection
from langcalc.models.projected import ProjectedModel

projection = LowercaseProjection()
model = ProjectedModel(base_model, projection, corpus)
```

**Using Augmentation:**

```python
from langcalc.augmentations import LowercaseAugmentation

augmented_corpus = LowercaseAugmentation().augment(corpus)
model = Infinigram(augmented_corpus)
```

Both achieve case-insensitive matching!

### Example 2: Robust Text Matching

```python
from langcalc.projections import (
    WhitespaceProjection,
    LowercaseProjection,
    RecencyProjection
)

# Chain projections
projection = (
    WhitespaceProjection() >>  # Normalize whitespace
    LowercaseProjection() >>   # Case-insensitive
    RecencyProjection(100)     # Keep recent 100 tokens
)

model = ProjectedModel(base_model, projection, corpus)
```

### Example 3: Standard Augmentation

```python
from langcalc.augmentations import StandardAugmentation

# Case + whitespace + Unicode NFC (≈8× corpus)
augmented = StandardAugmentation().augment(corpus)
model = Infinigram(augmented)
```

## When to Use What

| Goal | Approach | Example |
|------|----------|---------|
| Case-insensitive | Augmentation | `LowercaseAugmentation()` |
| Format robustness | Augmentation | `WhitespaceAugmentation()` |
| Typo correction | Projection | `EditDistanceProjection(max_distance=2)` |
| Synonym expansion | Projection | `SynonymProjection()` |
| Context truncation | Projection | `RecencyProjection(max_length=100)` |
| Unicode compatibility | Augmentation | `NFCAugmentation()` |

**Decision Tree:**

```
Can the transformation be precomputed?
├─ YES → How much memory available?
│  ├─ Plenty (2-10× corpus) → Use AUGMENTATION
│  └─ Limited → Use PROJECTION
└─ NO (context-dependent) → Use PROJECTION
```

## Space-Time Tradeoffs

| Augmentation | Space Multiplier | Query Time Saved | When to Use |
|--------------|------------------|------------------|-------------|
| Lowercase | 2× | Significant | Almost always |
| Full Case | 4× | Significant | Case-insensitive search |
| Whitespace | 2× | Moderate | Mixed formatting |
| Unicode NFC | 2× | Significant | International text |
| Full Unicode | 5× | Significant | Maximum compatibility |
| Standard | ≈8× | High | General purpose |
| Aggressive | ≈20× | Very High | Large corpora only |

**Rule of thumb:** If you have memory for $k\times$ corpus expansion, use augmentation. Otherwise, use projection.

## Implementation Roadmap

### Phase 1: Core Infrastructure ✓

- [x] `Projection` abstract base class
- [x] `Augmentation` abstract base class
- [x] Composition operators (`>>`, `|`, `+`)

### Phase 2: Basic Projections ✓

- [x] `IdentityProjection`
- [x] `RecencyProjection`
- [x] `LowercaseProjection`
- [x] `WhitespaceProjection`
- [x] `UnicodeNormalizationProjection`

### Phase 3: Basic Augmentations ✓

- [x] `LowercaseAugmentation`
- [x] `CaseAugmentation`
- [x] `WhitespaceAugmentation`
- [x] `NFCAugmentation`
- [x] `StandardAugmentation`

### Phase 4: Model Integration 🚧

- [ ] `ProjectedModel(base_model, projection, corpus)`
- [ ] `MultiProjectionModel(base_model, weighted_projections, corpus)`
- [ ] Update `InfinigramModel` to accept projections/augmentations

### Phase 5: Advanced Projections (Future)

- [ ] `EditDistanceProjection`
- [ ] `LongestSuffixProjection`
- [ ] `SynonymProjection`

### Phase 6: Presets and Utilities (Future)

- [ ] `StandardTextProjection` pipeline
- [ ] `CodeCompletionProjection` pipeline
- [ ] Validation utilities

## Documentation Structure

1. **[Mathematical Formalism](formalism.md)** - Rigorous definitions and theorems
2. **[Canonical Augmentations](augmentations.md)** - Catalog of standard transformations
3. **[Ordering Principles](ordering.md)** - Non-commutativity and canonical pipelines
4. **[Reference Implementation](implementation.md)** - Complete code reference
5. **[Index](index.md)** - Complete roadmap and API summary

## Research Contributions

The projection system makes several novel contributions:

1. **Unified Framework**: Treating projections and augmentations within a single mathematical formalism
2. **Duality Theorem**: Proving equivalence between query-time projection and training-time augmentation
3. **Non-Commutativity Analysis**: Establishing ordering principles for projection composition
4. **Canonical Augmentations**: Comprehensive catalog of standard transformations
5. **Space-Time Tradeoffs**: Quantitative analysis of augmentation costs

## Next Steps

Explore the complete documentation:

- **New to projections?** Start with [Mathematical Formalism](formalism.md)
- **Want to implement?** See [Reference Implementation](implementation.md)
- **Building pipelines?** Read [Ordering Principles](ordering.md)
- **Looking for augmentations?** Browse [Canonical Augmentations](augmentations.md)
- **Complete reference?** Check the [Index](index.md)
