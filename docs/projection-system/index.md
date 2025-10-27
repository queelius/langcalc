# Projection System - Complete Documentation Index

## Overview

This index provides a roadmap to the complete projection system documentation for LangCalc. The projection system enables flexible context transformation and corpus augmentation for improved pattern matching in language models.

---

## Documentation Structure

### 1. Core Theory

#### [formalism.md](formalism.md)
**Mathematical foundations of the projection system**

**Contents:**
- Basic definitions (corpus, context, language model)
- Projection theory (projections as corpus-aware transformations)
- Corpus augmentation (normal forms)
- Projection-augmentation duality theorem
- Projection algebra (composition operations)
- Canonical augmentations (case, whitespace, Unicode)
- Complexity analysis (space-time tradeoffs)
- Projected language models
- Future directions (learnable, semantic, adaptive projections)

**Key Concepts:**
- Projection: $\pi: \Sigma^* \times 2^{\Sigma^*} \to \Sigma^*$
- Augmentation: $\alpha: 2^{\Sigma^*} \to 2^{\Sigma^*}$
- Duality: $\text{LMS}(\pi(x, C), C) = \text{LMS}(x, \alpha(C))$
- Composition: $\pi_1 \circ \pi_2$, $\pi_1 \sqcup \pi_2$

**Read this:** For mathematical understanding and theoretical foundations.

---

### 2. Canonical Augmentations

#### [augmentations.md](augmentations.md)
**Catalog of standard corpus augmentations (normal forms)**

**Contents:**
- Case normalization (lowercase, uppercase, titlecase)
- Whitespace normalization (collapsing, stripping)
- Unicode normalization (NFC, NFD, NFKC, NFKD)
- Punctuation handling (removal, normalization)
- Composite augmentations (standard, aggressive)
- Language-specific augmentations (ASCII folding)
- Augmentation composition (sequential, parallel)
- Recommended augmentation sets
- Space-time tradeoffs
- Implementation checklist
- Testing strategy

**Key Augmentations:**
- `LowercaseAugmentation` - Case-insensitive matching (2× corpus)
- `CaseAugmentation` - Full case coverage (4× corpus)
- `WhitespaceAugmentation` - Format robustness (2× corpus)
- `NFCAugmentation` - Unicode handling (2× corpus)
- `StandardAugmentation` - Recommended default (≈8× corpus)

**Read this:** For practical augmentation implementation guide.

---

### 3. Implementation Reference

#### [implementation.md](implementation.md)
**Complete reference implementation of the projection system**

**Contents:**
- Core abstractions (`Projection`, `Augmentation` base classes)
- Composition implementations (sequential, parallel, weighted)
- Basic projections (identity, recency, truncation)
- Normalization projections (case, whitespace, Unicode)
- Advanced projections (edit distance, longest suffix)
- Basic augmentations (case, whitespace)
- Model integration (`ProjectedModel`, `MultiProjectionModel`)
- Usage examples
- Testing strategy

**Key Classes:**
```python
class Projection(ABC):
    def project(self, context: List[int], corpus: List[int]) -> List[int]
    def project_multi(self, context: List[int], corpus: List[int]) -> Set[Tuple[int, ...]]
    def __rshift__(self, other: 'Projection') -> 'Projection'  # >>
    def __or__(self, other: 'Projection') -> 'Projection'      # |

class Augmentation(ABC):
    def augment(self, corpus: List[int]) -> List[int]
    def __add__(self, other: 'Augmentation') -> 'Augmentation'
```

**Read this:** For implementation details and code examples.

---

### 4. Ordering and Composition

#### [ordering.md](ordering.md)
**Non-commutativity and ordering principles for projection composition**

**Contents:**
- Non-commutativity of projections (proof and examples)
- When order matters (dependency analysis)
- Classes of dependencies (lossy→lossless, correction→transformation, etc.)
- Canonical ordering principles
- Commutativity classes (identifying commutative pairs)
- Practical guidelines (decision tree, testing)
- Multi-path projections (exploring multiple orders)
- Mathematical properties (associativity, partial ordering)
- Case studies (search, code completion, chat)

**Key Principles:**
1. **Error correction first** - Fix typos before semantic transformations
2. **Normalization before expansion** - Normalize, then expand synonyms
3. **Semantic before structural** - Expand semantic space before matching
4. **Lossy operations last** - Preserve information as long as possible

**Canonical Pipeline:**
```
EditDistance >> Normalize >> Synonym >> LongestSuffix >> Recency
```

**Read this:** To understand projection ordering and avoid common mistakes.

---

## Quick Reference

### When to Use What

| Goal | Use | Example |
|------|-----|---------|
| Case-insensitive matching | Augmentation | `LowercaseAugmentation()` |
| Format robustness | Augmentation | `WhitespaceAugmentation()` |
| Typo correction | Projection | `EditDistanceProjection(max_distance=2)` |
| Synonym expansion | Projection | `SynonymProjection()` |
| Context truncation | Projection | `RecencyProjection(max_length=100)` |
| Unicode compatibility | Augmentation | `NFCAugmentation()` |
| General text matching | Preset pipeline | `StandardTextProjection` |

### Decision Tree: Projection vs Augmentation

```
Can the transformation be precomputed?
├─ YES → How expensive is it?
│  ├─ Cheap (case, whitespace) → Use AUGMENTATION
│  └─ Expensive (all variants) → Use PROJECTION
└─ NO (context-dependent) → Use PROJECTION
```

**Examples:**
- **Augmentation:** Lowercase (precompute all case variants)
- **Projection:** Edit distance (too many variants to precompute)
- **Projection:** Recency (depends on query context length)

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (Priority 1)

**Files to create:**
- `langcalc/projections/base.py` - `Projection` abstract base class
- `langcalc/augmentations/base.py` - `Augmentation` abstract base class
- `langcalc/projections/composition.py` - `SequentialProjection`, `ParallelProjection`

**Implement:**
- [ ] `Projection` base class with composition operators
- [ ] `Augmentation` base class with composition
- [ ] `IdentityProjection`
- [ ] Unit tests for composition

### Phase 2: Basic Projections (Priority 1)

**File:** `langcalc/projections/basic.py`

**Implement:**
- [ ] `RecencyProjection(max_length)`
- [ ] `TruncationProjection(max_length)`
- [ ] `LowercaseProjection()`
- [ ] `UppercaseProjection()`
- [ ] `WhitespaceProjection()`
- [ ] `UnicodeNormalizationProjection(form='NFC')`
- [ ] Unit tests for each

### Phase 3: Basic Augmentations (Priority 1)

**File:** `langcalc/augmentations/basic.py`

**Implement:**
- [ ] `LowercaseAugmentation()`
- [ ] `CaseAugmentation()`
- [ ] `WhitespaceAugmentation()`
- [ ] `NFCAugmentation()`
- [ ] `StandardAugmentation()` (preset)
- [ ] Unit tests for each

### Phase 4: Model Integration (Priority 1)

**File:** `langcalc/models/projected.py`

**Implement:**
- [ ] `ProjectedModel(base_model, projection, corpus)`
- [ ] `MultiProjectionModel(base_model, weighted_projections, corpus)`
- [ ] Update `InfinigramModel` to accept projection/augmentation
- [ ] Integration tests

### Phase 5: Advanced Projections (Priority 2)

**File:** `langcalc/projections/advanced.py`

**Implement:**
- [ ] `EditDistanceProjection(max_distance)`
- [ ] `LongestSuffixProjection(min_length)`
- [ ] `SynonymProjection()` (requires WordNet/embedding)
- [ ] Unit tests

### Phase 6: Advanced Augmentations (Priority 2)

**File:** `langcalc/augmentations/advanced.py`

**Implement:**
- [ ] `UnicodeAugmentation()` (all forms)
- [ ] `PunctuationAugmentation()`
- [ ] `NoPunctuationAugmentation()`
- [ ] `ASCIIFoldingAugmentation()`
- [ ] Unit tests

### Phase 7: Presets and Utilities (Priority 3)

**Files:**
- `langcalc/projections/presets.py`
- `langcalc/augmentations/presets.py`

**Implement:**
- [ ] `StandardTextProjection` pipeline
- [ ] `CodeCompletionProjection` pipeline
- [ ] `ChatMessageProjection` pipeline
- [ ] `MinimalAugmentation` preset
- [ ] `AggressiveAugmentation` preset
- [ ] Validation utilities
- [ ] Commutativity testing utilities

---

## Testing Strategy

### Unit Tests

**Test each projection individually:**
```python
def test_lowercase_projection():
    proj = LowercaseProjection()
    context = list("Hello".encode('utf-8'))
    result = proj.project(context, corpus=[])
    assert bytes(result).decode('utf-8') == "hello"
```

**Test each augmentation individually:**
```python
def test_lowercase_augmentation():
    aug = LowercaseAugmentation()
    corpus = list("Hello".encode('utf-8'))
    result = aug.augment(corpus)
    text = bytes(result).decode('utf-8')
    assert "Hello" in text and "hello" in text
```

### Integration Tests

**Test projection-augmentation duality:**
```python
def test_duality():
    corpus = list("Hello World".encode('utf-8'))
    context = list("HELLO".encode('utf-8'))

    # Approach 1: Project query
    proj_model = ProjectedModel(
        InfinigramModel(corpus),
        LowercaseProjection(),
        corpus
    )

    # Approach 2: Augment corpus
    aug_corpus = LowercaseAugmentation().augment(corpus)
    aug_model = InfinigramModel(aug_corpus)

    # Should produce similar results
    tokens = list(range(256))
    probs1 = proj_model.logprobs(tokens, context)
    probs2 = aug_model.logprobs(tokens, context)

    assert np.allclose(probs1, probs2, atol=0.1)
```

**Test composition:**
```python
def test_composition():
    pipeline = (
        WhitespaceProjection() >>
        LowercaseProjection() >>
        RecencyProjection(10)
    )

    context = list("  HELLO  WORLD  ".encode('utf-8'))
    result = pipeline.project(context, corpus=[])

    # Should normalize, lowercase, then truncate
    text = bytes(result).decode('utf-8')
    assert text == "hello world"  # normalized and lowercased
    assert len(result) <= 10 * 4  # truncated (UTF-8 max 4 bytes/char)
```

### Property-Based Tests

```python
from hypothesis import given, strategies as st

@given(st.lists(st.integers(0, 255)))
def test_identity_projection_is_identity(context):
    proj = IdentityProjection()
    assert proj.project(context, corpus=[]) == context

@given(st.lists(st.integers(0, 255)))
def test_augmentation_includes_original(corpus):
    aug = LowercaseAugmentation()
    result = aug.augment(corpus)
    assert corpus == result[:len(corpus)]  # Original is prefix
```

---

## Usage Examples

### Example 1: Simple Case-Insensitive Model

```python
from langcalc.models import InfinigramModel
from langcalc.augmentations import LowercaseAugmentation

# Create corpus with lowercase augmentation
corpus = list("Hello World".encode('utf-8'))
augmented = LowercaseAugmentation().augment(corpus)

# Create model
model = InfinigramModel(augmented)

# Query with uppercase - will match
context = list("HELLO".encode('utf-8'))
probs = model.logprobs(list(range(256)), context)
```

### Example 2: Projection Pipeline

```python
from langcalc.models import InfinigramModel
from langcalc.projections import (
    WhitespaceProjection, LowercaseProjection, RecencyProjection
)
from langcalc.models.projected import ProjectedModel

# Define pipeline
projection = (
    WhitespaceProjection() >>
    LowercaseProjection() >>
    RecencyProjection(max_length=100)
)

# Create model
corpus = list("the cat sat on the mat".encode('utf-8'))
base_model = InfinigramModel(corpus)
model = ProjectedModel(base_model, projection, corpus)

# Query with messy input
context = list("  THE  CAT  ".encode('utf-8'))
samples = model.sample(context, max_tokens=10)
```

### Example 3: Multi-Projection Model

```python
from langcalc.projections import IdentityProjection, LowercaseProjection
from langcalc.models.projected import MultiProjectionModel

# Try multiple projections with weights
projections = [
    (IdentityProjection(), 0.7),      # Original: 70%
    (LowercaseProjection(), 0.3),     # Lowercase: 30%
]

model = MultiProjectionModel(
    InfinigramModel(corpus),
    projections,
    corpus
)

# Model tries both projections, weighted mixture
probs = model.logprobs(tokens, context)
```

### Example 4: Standard Text Pipeline

```python
from langcalc.projections.presets import StandardTextProjection
from langcalc.models.projected import ProjectedModel

# Use preset pipeline
projection = StandardTextProjection()

model = ProjectedModel(
    InfinigramModel(corpus),
    projection,
    corpus
)
```

---

## Migration Guide

### From NGramModel with Projections

**Old code:**
```python
from langcalc.models.ngram import NGramModel
from langcalc.projections import RecencyProjection

model = NGramModel(corpus, n=3, projection=RecencyProjection(decay=0.9))
```

**New code:**
```python
from langcalc.models import InfinigramModel
from langcalc.projections import RecencyProjection
from langcalc.models.projected import ProjectedModel

base_model = InfinigramModel(corpus, max_length=3)
projection = RecencyProjection(max_length=10)
model = ProjectedModel(base_model, projection, corpus)
```

### From Infinigram Augmentations

**Infinigram augmentations** (corpus-level):
```python
# Infinigram's augmentation (training-time)
from infinigram import augment
augmented_corpus = augment(corpus, ['lowercase', 'uppercase'])
```

**LangCalc augmentations** (equivalent):
```python
# LangCalc's augmentation (training-time)
from langcalc.augmentations import CaseAugmentation

augmented_corpus = CaseAugmentation().augment(corpus)
```

**Infinigram recursive transformers** (query-time):
```python
# Infinigram's recursive transformer (query-time)
from infinigram.recursive import SynonymTransformer

model = RecursiveInfinigram(corpus, transformers=[SynonymTransformer()])
```

**LangCalc projections** (equivalent):
```python
# LangCalc's projection (query-time)
from langcalc.projections import SynonymProjection
from langcalc.models.projected import ProjectedModel

projection = SynonymProjection()
model = ProjectedModel(InfinigramModel(corpus), projection, corpus)
```

---

## API Summary

### Core Classes

```python
# Abstract base classes
from langcalc.projections import Projection
from langcalc.augmentations import Augmentation

# Basic projections
from langcalc.projections import (
    IdentityProjection,
    RecencyProjection,
    TruncationProjection,
    LowercaseProjection,
    WhitespaceProjection,
    UnicodeNormalizationProjection,
)

# Advanced projections
from langcalc.projections.advanced import (
    EditDistanceProjection,
    LongestSuffixProjection,
    SynonymProjection,
)

# Basic augmentations
from langcalc.augmentations import (
    LowercaseAugmentation,
    CaseAugmentation,
    WhitespaceAugmentation,
    NFCAugmentation,
    StandardAugmentation,
)

# Advanced augmentations
from langcalc.augmentations.advanced import (
    UnicodeAugmentation,
    PunctuationAugmentation,
    ASCIIFoldingAugmentation,
)

# Model integration
from langcalc.models.projected import (
    ProjectedModel,
    MultiProjectionModel,
)

# Presets
from langcalc.projections.presets import StandardTextProjection
from langcalc.augmentations.presets import (
    MinimalAugmentation,
    AggressiveAugmentation,
)
```

### Composition Operators

```python
# Sequential composition (left-to-right)
pipeline = proj1 >> proj2 >> proj3

# Parallel composition (union)
multi = proj1 | proj2 | proj3

# Augmentation composition
augmentation = aug1 + aug2 + aug3
```

---

## References

### Related Documentation

- `PROJECTIONS_COMPARISON.md` - Comparison with Infinigram's concepts
- `OLLAMA_NGRAM_SUMMARY.md` - NGramModel removal plan
- `CURRENT_STATUS.md` - Current implementation status

### External Resources

- Infinigram paper: [Infinigram: Scaling Unbounded n-gram Language Models to a Trillion Tokens](https://arxiv.org/abs/2401.17377)
- Suffix arrays: [Suffix Array - Wikipedia](https://en.wikipedia.org/wiki/Suffix_array)
- Unicode normalization: [UAX #15: Unicode Normalization Forms](https://unicode.org/reports/tr15/)
- WordNet: [WordNet - Princeton](https://wordnet.princeton.edu/)

---

## Contributing

When adding new projections or augmentations:

1. **Document ordering constraints** in the docstring
2. **Add unit tests** for correctness
3. **Test commutativity** with existing projections
4. **Update this index** with your addition
5. **Add examples** to the reference implementation

### Projection Template

```python
class MyProjection(Projection):
    """
    Brief description.

    Ordering constraints:
    - AFTER: [projections that should come before]
    - BEFORE: [projections that should come after]
    - COMMUTES WITH: [projections that commute with this]

    Args:
        param: Parameter description

    Example:
        >>> proj = MyProjection(param=value)
        >>> result = proj.project(context, corpus)
    """

    def __init__(self, param):
        self.param = param

    def project(self, context: List[int], corpus: List[int]) -> List[int]:
        # Implementation
        pass

    def __repr__(self) -> str:
        return f"MyProjection(param={self.param})"
```

---

## Summary

The projection system provides:

1. **Flexible context transformation** - Project queries onto corpus
2. **Efficient corpus augmentation** - Precompute common transformations
3. **Composable algebra** - Build complex pipelines from simple parts
4. **Principled ordering** - Guidelines for non-commutative composition
5. **Duality theorem** - Trade space for time via augmentation

**Start here:**
- **Theory:** Read `formalism.md`
- **Practice:** Read `augmentations.md`
- **Code:** Read `implementation.md`
- **Composition:** Read `ordering.md`

**Next steps:**
- Implement Phase 1 (core infrastructure)
- Implement Phase 2 (basic projections)
- Implement Phase 3 (basic augmentations)
- Update `InfinigramModel` to support projections/augmentations
- Remove `NGramModel` (once projection system is complete)
