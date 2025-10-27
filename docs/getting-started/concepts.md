# Core Concepts

This page explains the fundamental concepts in LangCalc: projections, augmentations, and algebraic operations.

## Language Models as Algebraic Objects

In LangCalc, language models are treated as mathematical objects that support algebraic operations. A language model is any object that can:

1. Compute probability distributions over next tokens given context
2. Be combined with other models using algebraic operators
3. Be transformed using context projections

```python
# All of these are valid language models:
infinigram = Infinigram(corpus)
ngram = NGramModel(corpus, n=3)
llm = OllamaModel(model_name='llama2')

# They can all be composed algebraically:
ensemble = 0.5 * infinigram + 0.3 * ngram + 0.2 * llm
```

---

## Projections vs Augmentations

Understanding the difference between projections and augmentations is crucial.

### Projections (Query-Time)

**Definition:** A projection $\pi$ transforms the query context before matching:

$$\pi: \Sigma^* \times 2^{\Sigma^*} \to \Sigma^*$$

```python
from langcalc.projections import LowercaseProjection

# Project query to lowercase at prediction time
projection = LowercaseProjection()
model = ProjectedModel(base_model, projection, corpus)

# Query: "HELLO" -> projection -> "hello" -> match in corpus
```

**Characteristics:**

- Applied at **query time** (every prediction)
- Flexible (can depend on corpus or context)
- Lower memory usage
- Slightly slower queries

**Use when:**

- Transformation is context-dependent (edit distance, recency)
- Cannot precompute all variants (too many possibilities)
- Memory is limited

### Augmentations (Training-Time)

**Definition:** An augmentation $\alpha$ expands the corpus with variants:

$$\alpha: 2^{\Sigma^*} \to 2^{\Sigma^*}$$

```python
from langcalc.augmentations import LowercaseAugmentation

# Augment corpus once with lowercase variant
augmentation = LowercaseAugmentation()
augmented_corpus = augmentation.augment(corpus)  # corpus + lowercase(corpus)
model = Infinigram(augmented_corpus)

# Corpus now contains: "Hello" and "hello"
```

**Characteristics:**

- Applied at **training time** (once)
- Fast queries (no transformation needed)
- Higher memory usage (stores variants)
- Predictable behavior

**Use when:**

- Transformation is simple (case, whitespace, Unicode)
- Can afford extra memory (2-10x corpus size)
- Want fastest possible queries

### Projection-Augmentation Duality

**Theorem:** For certain transformations, projection and augmentation are equivalent:

$$\text{LMS}(\pi(x, C), C) = \text{LMS}(x, \alpha(C))$$

This means you can choose either approach for the same semantic effect!

**Example:**

```python
# Approach 1: Projection (query-time)
projection = LowercaseProjection()
model1 = ProjectedModel(Infinigram(corpus), projection, corpus)

# Approach 2: Augmentation (training-time)
augmented = LowercaseAugmentation().augment(corpus)
model2 = Infinigram(augmented)

# Both give same results for case-insensitive matching!
```

**Decision Guide:**

```
Can transformation be precomputed?
├─ YES → How expensive is storage?
│  ├─ Cheap (2-4x) → Use AUGMENTATION
│  └─ Expensive (>10x) → Use PROJECTION
└─ NO (context-dependent) → Use PROJECTION
```

---

## Algebraic Operations

LangCalc supports a rich algebra of operations on language models.

### Arithmetic Operations

#### Weighted Mixture (+, *)

Combine models with weights:

```python
# Weighted sum: 0.7*m1 + 0.3*m2
ensemble = 0.7 * model1 + 0.3 * model2

# Probability: p(token) = 0.7 * p1(token) + 0.3 * p2(token)
```

**Use cases:**

- Ensemble different models
- Balance fluency (LLM) and factuality (infinigram)
- Combine complementary strengths

#### Subtraction (-)

```python
# What model1 learned beyond model2
residual = model1 - model2
```

**Use cases:**

- Analyze model differences
- Remove biases
- Experimental feature

#### Division (/)

```python
# Ratio of probabilities
ratio_model = model1 / model2
```

**Use cases:**

- Importance weighting
- Contrast estimation
- Advanced research

### Set Operations

#### Maximum (|)

Take maximum probability:

```python
# max(p1(token), p2(token))
best_of_both = model1 | model2
```

**Use cases:**

- Fallback behavior (if model1 unsure, try model2)
- Combining specialized models

#### Minimum (&)

Take minimum probability:

```python
# min(p1(token), p2(token))
conservative = model1 & model2
```

**Use cases:**

- Conservative predictions
- Agreement between models

#### Symmetric Difference (^)

Highlight disagreement:

```python
# Where models disagree
disagreement = model1 ^ model2
```

**Use cases:**

- Uncertainty estimation
- Model comparison

### Transformations

#### Temperature Scaling (**)

```python
# Higher temperature = more diversity
creative = model ** 1.5

# Lower temperature = more focused
focused = model ** 0.5
```

**How it works:**

$$p(token) \propto p_{\text{original}}(token)^{1/T}$$

where $T$ is temperature.

#### Context Transformation (<<)

Apply transformation before prediction:

```python
from langcalc.algebra import RecencyWeightTransform

# Apply recency weighting to context
transformed = model << RecencyWeightTransform(decay=0.9)
```

#### Function Application (>>)

Apply function to outputs:

```python
# Custom transformation of predictions
processed = model >> custom_function
```

#### Negation (~)

Complement probability:

```python
# 1 - p(token)
anti_model = ~model
```

**Use cases:**

- Negative sampling
- Contrast learning

---

## Context Transformations

Beyond projections, LangCalc supports sophisticated context transformations.

### Built-in Transformations

#### Longest Suffix Transform

Find longest matching suffix in corpus:

```python
from langcalc.algebra import LongestSuffixTransform

transform = LongestSuffixTransform(suffix_array)
grounded = model << transform
```

#### Max K Words Transform

Keep only recent k words:

```python
from langcalc.algebra import MaxKWordsTransform

transform = MaxKWordsTransform(k=10)
recent_context = model << transform
```

#### Recency Weight Transform

Apply exponential decay to older tokens:

```python
from langcalc.algebra import RecencyWeightTransform

transform = RecencyWeightTransform(decay=0.9)
weighted = model << transform
```

#### Focus Transform

Filter to specific word types:

```python
from langcalc.algebra import FocusTransform

transform = FocusTransform(word_types=['NOUN', 'VERB'])
focused = model << transform
```

### Composing Transformations

Transformations can be chained:

```python
# Sequential: apply one after another
pipeline = transform1 | transform2 | transform3

# Parallel: try multiple paths
multi_path = transform1 & transform2

# Apply to model
transformed_model = model << pipeline
```

---

## Suffix Arrays vs N-grams

LangCalc uses suffix arrays for efficient pattern matching.

### Why Suffix Arrays?

**N-gram Hash Tables:**

- Store counts for every n-gram seen
- Memory: $O(|V|^n)$ where $V$ is vocabulary
- For large n, becomes impractical
- Example: 5-grams on 50K vocabulary = ~3 petabytes!

**Suffix Arrays:**

- Store positions of all suffixes
- Memory: $O(n)$ where $n$ is corpus size
- Query time: $O(m \log n)$ where $m$ is pattern length
- **34x more memory efficient** in practice

### Example Comparison

```python
# N-gram model (fixed length)
ngram = NGramModel(corpus, n=3)  # Only 3-grams

# Infinigram (variable length using suffix arrays)
infini = Infinigram(corpus, max_length=20)  # Up to 20-grams!
```

For 1B token corpus:

| Approach | Memory | Longest Pattern |
|----------|--------|-----------------|
| N-gram (n=5) | ~34 GB | 5 tokens |
| Suffix Array | ~1 GB | Variable (up to corpus size) |

---

## Pattern Matching

How LangCalc finds patterns in the corpus.

### Longest Matching Suffix

Given context `x` and corpus `C`, find:

$$\text{LMS}(x, C) = \arg\max_{s \in \text{Suffixes}(C)} \{|s| : s \text{ is suffix of } x\}$$

**Example:**

```
Context: [the, cat, sat, on]
Corpus:  [the, cat, sat, on, the, mat, ...]

Longest suffix: [the, cat, sat, on] (full match, length 4)
```

### Variable-Length Matching

Unlike fixed n-grams, infinigrams adapt pattern length:

```python
model = Infinigram(corpus, max_length=20)

# Context 1: "the cat"
# Finds: "the cat sat" (3 tokens)

# Context 2: "the quick brown fox jumps over the"
# Finds: full match (7+ tokens)

# Context 3: "xyz"
# Finds: no match (falls back to unigram)
```

---

## Memory vs Speed Tradeoffs

Understanding when to use projections vs augmentations.

### Space-Time Matrix

| Transformation | Projection Cost | Augmentation Space | Recommendation |
|----------------|-----------------|--------------------|-----------------|
| Lowercase | O(n) time | 2× memory | **Augmentation** |
| Full Case | O(n) time | 4× memory | **Augmentation** |
| Whitespace | O(n) time | 2× memory | **Augmentation** |
| Unicode NFC | O(n) time | 2× memory | **Augmentation** |
| Edit Distance | O(n² m) time | Infinite | **Projection** |
| Synonyms | O(k) time | Exponential | **Projection** |
| Recency | O(1) time | N/A | **Projection** |

**General Rule:**

- **Simple, precomputable → Augmentation** (case, whitespace, Unicode)
- **Complex, context-dependent → Projection** (edit distance, recency, semantic)

---

## Best Practices

### 1. Start Simple

```python
# Begin with basic infinigram
model = Infinigram(corpus)

# Add complexity incrementally
model = 0.9 * llm + 0.1 * Infinigram(corpus)
```

### 2. Use Augmentation for Common Cases

```python
# Standard augmentation: case + whitespace + Unicode
from langcalc.augmentations import StandardAugmentation

augmented = StandardAugmentation().augment(corpus)
model = Infinigram(augmented)
```

### 3. Profile Before Optimizing

```python
import time

# Measure query time
start = time.time()
for _ in range(100):
    probs = model.predict(context)
print(f"Avg query time: {(time.time() - start) / 100 * 1000:.2f}ms")
```

### 4. Test Both Approaches

```python
# Try projection
proj_model = ProjectedModel(base, projection, corpus)

# Try augmentation
aug_model = Infinigram(augmentation.augment(corpus))

# Compare performance
```

### 5. Chain Projections Correctly

Follow the canonical ordering (see [Ordering Principles](../projection-system/ordering.md)):

```python
# CORRECT: Error correction → Normalization → Expansion → Matching
correct = (
    EditDistanceProjection(1) >>
    LowercaseProjection() >>
    SynonymProjection() >>
    LongestSuffixProjection()
)

# WRONG: Expansion before normalization
wrong = (
    SynonymProjection() >>
    LowercaseProjection()  # Too late!
)
```

---

## Next Steps

Now that you understand the core concepts:

1. **[User Guide](../user-guide/index.md)** - Practical patterns and examples
2. **[Projection System](../projection-system/index.md)** - Mathematical formalism
3. **[API Reference](../api/index.md)** - Detailed API documentation
4. **[Advanced Topics](../advanced/index.md)** - Performance optimization and extending LangCalc

## Further Reading

- [Mathematical Formalism](../projection-system/formalism.md) - Rigorous definitions
- [Canonical Augmentations](../projection-system/augmentations.md) - Standard transformations catalog
- [Ordering Principles](../projection-system/ordering.md) - Non-commutativity and composition
- [Reference Implementation](../projection-system/implementation.md) - Complete code examples
