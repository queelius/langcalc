# Projection Ordering and Non-Commutativity

## Abstract

While projections form a **monoid** under sequential composition (associative with identity), they are generally **not commutative**. The order in which projections are applied significantly affects the result. This document establishes principles for ordering projections and identifies cases where order matters.

---

## 1. Non-Commutativity of Projections

### 1.1 Mathematical Statement

**Theorem 1.1 (Non-Commutativity):** For projections $\pi_1, \pi_2$:
$$\pi_1 \circ \pi_2 \neq \pi_2 \circ \pi_1 \quad \text{(in general)}$$

**Proof by Example:**

Let:
- $\pi_1 = $ SynonymProjection (replaces words with synonyms)
- $\pi_2 = $ EditDistanceProjection (finds typo corrections)

**Case 1:** $\pi_1 \circ \pi_2$ (typo correction, then synonym)
```
Input:    "the quik cat"
→ π₂:     "the quick cat"      (fix typo: quik → quick)
→ π₁:     "the fast feline"    (synonyms: quick → fast, cat → feline)
```

**Case 2:** $\pi_2 \circ \pi_1$ (synonym, then typo correction)
```
Input:    "the quik cat"
→ π₁:     "the quik feline"    (synonym: cat → feline, but "quik" has no synonym)
→ π₂:     "the quick feline"   (fix typo: quik → quick)
```

**Result:** Different outputs! Order matters. ∎

### 1.2 Algebraic Structure

Projections form a **non-commutative monoid**:

1. **Closure:** $\pi_1 \circ \pi_2$ is a projection
2. **Associativity:** $(\pi_1 \circ \pi_2) \circ \pi_3 = \pi_1 \circ (\pi_2 \circ \pi_3)$
3. **Identity:** $\pi_{\text{id}} \circ \pi = \pi \circ \pi_{\text{id}} = \pi$
4. **Non-commutativity:** $\pi_1 \circ \pi_2 \neq \pi_2 \circ \pi_1$ (generally)

This is similar to function composition, matrix multiplication, or string concatenation.

---

## 2. When Order Matters

### 2.1 Dependency Analysis

**Definition 2.1 (Projection Dependency):** Projection $\pi_2$ **depends on** $\pi_1$ if:
$$\exists x, C: \pi_2(\pi_1(x, C), C) \neq \pi_2(x, C)$$

**Interpretation:** $\pi_2$'s behavior changes based on $\pi_1$'s transformation.

### 2.2 Classes of Dependencies

#### Class A: Lossy → Lossless (Order Critical)

**Pattern:** Apply lossy projections before lossless ones.

**Example:** Truncation before normalization
```python
# WRONG ORDER: Normalize, then truncate
projection = RecencyProjection(10) >> LowercaseProjection()
# Input:  "The Quick Brown Fox Jumps Over"
# → Truncate: "rown Fox J"  (loses "The Quick B")
# → Lowercase: "rown fox j"  (normalized after truncation)

# RIGHT ORDER: Normalize, then truncate
projection = LowercaseProjection() >> RecencyProjection(10)
# Input:  "The Quick Brown Fox Jumps Over"
# → Lowercase: "the quick brown fox jumps over"
# → Truncate: "umps over"  (preserves more normalized context)
```

**Rule:** **Lossy last** - Apply destructive transformations (truncation, sampling) after information-preserving ones (normalization).

#### Class B: Correction → Transformation (Order Critical)

**Pattern:** Fix errors before semantic transformations.

**Example:** Typo correction before synonym expansion
```python
# RIGHT ORDER: Fix typos, then find synonyms
projection = EditDistanceProjection(max_distance=2) >> SynonymProjection()
# Input:  "the quik cat"
# → Fix typo: "the quick cat"
# → Synonyms: "the fast feline"

# WRONG ORDER: Find synonyms, then fix typos
projection = SynonymProjection() >> EditDistanceProjection(max_distance=2)
# Input:  "the quik cat"
# → Synonyms: "the quik feline"  (no synonym for misspelled "quik")
# → Fix typo: "the quick feline"  (partial correction)
```

**Rule:** **Correct first** - Error correction before semantic transformations.

#### Class C: Semantic → Structural (Order Critical)

**Pattern:** Semantic transformations before structural ones.

**Example:** Synonym expansion before longest suffix
```python
# RIGHT ORDER: Expand synonyms, then find longest match
projection = SynonymProjection() >> LongestSuffixProjection()
# Input:  "the cat sat"
# → Synonyms: "the feline sat"
# → Longest suffix in corpus: "the feline sat on the mat"  (longer match)

# WRONG ORDER: Find longest match, then expand synonyms
projection = LongestSuffixProjection() >> SynonymProjection()
# Input:  "the cat sat"
# → Longest suffix: "cat sat"  (shorter match, "cat" not in corpus)
# → Synonyms: "feline sat"  (too late, already truncated)
```

**Rule:** **Semantic first** - Expand semantic space before structural matching.

#### Class D: Independent Transformations (Order Irrelevant)

**Pattern:** Transformations that don't interact.

**Example:** Case + Whitespace normalization
```python
# These commute:
projection1 = LowercaseProjection() >> WhitespaceProjection()
projection2 = WhitespaceProjection() >> LowercaseProjection()

# Input:  "Hello  World"
# Both produce: "hello world"
```

**Rule:** **Independent commute** - If transformations don't interact, order doesn't matter.

---

## 3. Canonical Ordering Principles

### 3.1 General Pipeline

**Recommended Order:**
```
1. Error Correction (typos, encoding issues)
   ↓
2. Normalization (case, whitespace, Unicode)
   ↓
3. Semantic Expansion (synonyms, stemming, lemmatization)
   ↓
4. Structural Matching (longest suffix, edit distance on structure)
   ↓
5. Lossy Operations (truncation, sampling)
```

**Rationale:**
1. Fix errors early (clean data)
2. Normalize to canonical form (consistent representation)
3. Expand semantic space (increase match potential)
4. Find structural patterns (leverage expanded space)
5. Reduce context if needed (preserve only relevant info)

### 3.2 Formal Ordering Rules

**Rule 1 (Error Before Transformation):**
$$\pi_{\text{error}} \gg \pi_{\text{transform}}$$

**Rule 2 (Normalization Before Expansion):**
$$\pi_{\text{normalize}} \gg \pi_{\text{expand}}$$

**Rule 3 (Expansion Before Matching):**
$$\pi_{\text{expand}} \gg \pi_{\text{match}}$$

**Rule 4 (Lossy Last):**
$$\pi_{\text{preserving}} \gg \pi_{\text{lossy}}$$

### 3.3 Example: Complete Pipeline

```python
projection = (
    # 1. Error correction
    EditDistanceProjection(max_distance=1) >>

    # 2. Normalization
    WhitespaceProjection() >>
    LowercaseProjection() >>
    UnicodeNormalizationProjection('NFC') >>

    # 3. Semantic expansion
    SynonymProjection(wordnet_synsets=3) >>

    # 4. Structural matching
    LongestSuffixProjection(min_length=5) >>

    # 5. Lossy operations
    RecencyProjection(max_length=100)
)
```

---

## 4. Commutativity Classes

### 4.1 Identifying Commutative Pairs

**Definition 4.1 (Commutative Projections):** Projections $\pi_1, \pi_2$ commute if:
$$\forall x, C: \pi_1(\pi_2(x, C), C) = \pi_2(\pi_1(x, C), C)$$

**Theorem 4.1 (Independent Transformations):** If $\pi_1$ and $\pi_2$ transform disjoint aspects of the input, they commute.

### 4.2 Commutative Examples

#### Lowercase + Whitespace
```python
# These are equivalent:
LowercaseProjection() >> WhitespaceProjection()
WhitespaceProjection() >> LowercaseProjection()
```
**Reason:** Case and whitespace are independent.

#### Unicode NFC + Lowercase
```python
# These are equivalent:
UnicodeNormalizationProjection('NFC') >> LowercaseProjection()
LowercaseProjection() >> UnicodeNormalizationProjection('NFC')
```
**Reason:** Both preserve semantic content, operate on different aspects.

### 4.3 Non-Commutative Examples

#### Synonym + Edit Distance
**NOT commutative** - see Section 1.1

#### Truncation + Anything
Truncation is non-commutative with almost everything:
```python
# Different results:
RecencyProjection(5) >> LowercaseProjection()
LowercaseProjection() >> RecencyProjection(5)
```

---

## 5. Practical Guidelines

### 5.1 Decision Tree for Ordering

```
Question 1: Does one projection lose information?
  YES → Apply information-preserving first
  NO → Continue

Question 2: Does one correct errors?
  YES → Apply error correction first
  NO → Continue

Question 3: Does one expand semantic space?
  YES → Apply semantic expansion before structural matching
  NO → Continue

Question 4: Are they independent?
  YES → Order doesn't matter
  NO → Test both orders, choose better result
```

### 5.2 Testing for Commutativity

```python
def test_commutativity(proj1, proj2, test_cases):
    """Test if two projections commute on given test cases."""
    for context, corpus in test_cases:
        result1 = (proj1 >> proj2).project(context, corpus)
        result2 = (proj2 >> proj1).project(context, corpus)

        if result1 != result2:
            print(f"Non-commutative: {result1} ≠ {result2}")
            return False

    return True

# Example usage
test_cases = [
    (list("Hello  World".encode('utf-8')), []),
    (list("the quik cat".encode('utf-8')), []),
]

is_commutative = test_commutativity(
    LowercaseProjection(),
    WhitespaceProjection(),
    test_cases
)
```

### 5.3 Documenting Ordering Constraints

Each projection should document its ordering preferences:

```python
class SynonymProjection(Projection):
    """
    Synonym projection using WordNet.

    Ordering constraints:
    - AFTER: EditDistanceProjection (fix typos first)
    - BEFORE: LongestSuffixProjection (expand before matching)
    - COMMUTES WITH: LowercaseProjection, WhitespaceProjection
    """
```

---

## 6. Advanced: Multi-Path Projections

### 6.1 Exploring Multiple Orders

Instead of choosing one order, try multiple:

```python
class MultiOrderProjection(Projection):
    """Try multiple projection orders and return best match."""

    def __init__(self, projections: List[Projection]):
        self.projections = projections

    def project_multi(self, context: List[int], corpus: List[int]) -> Set[Tuple[int, ...]]:
        results = set()

        # Try all permutations of projection orderings
        from itertools import permutations
        for order in permutations(self.projections):
            # Apply projections in this order
            result = context
            for proj in order:
                result = proj.project(result, corpus)
            results.add(tuple(result))

        return results
```

**Use case:** When optimal order is unclear, let the model try all.

**Warning:** Exponential complexity - only feasible for small numbers of projections.

### 6.2 Learned Ordering

Train a model to select optimal projection order:

```python
class LearnedOrderProjection(Projection):
    """Learn optimal projection order based on context."""

    def __init__(self, projections: List[Projection], order_model):
        self.projections = projections
        self.order_model = order_model  # Neural network or decision tree

    def project(self, context: List[int], corpus: List[int]) -> List[int]:
        # Use learned model to predict best ordering
        order = self.order_model.predict_order(context, corpus)

        result = context
        for idx in order:
            result = self.projections[idx].project(result, corpus)

        return result
```

---

## 7. Implications for API Design

### 7.1 Left-to-Right Composition

Use `>>` operator for clarity (matches text reading order):

```python
# Clear intent: apply left-to-right
pipeline = (
    EditDistanceProjection() >>  # First
    LowercaseProjection() >>      # Second
    SynonymProjection() >>        # Third
    LongestSuffixProjection()     # Last
)
```

**Rationale:** Mathematical $\circ$ is right-to-left $(f \circ g)(x) = f(g(x))$, but programming pipelines read left-to-right.

### 7.2 Named Pipelines

Provide pre-configured pipelines with documented ordering:

```python
class StandardPipeline(Projection):
    """
    Standard projection pipeline.

    Order:
    1. Error correction (edit distance ≤ 1)
    2. Normalization (lowercase, whitespace, Unicode NFC)
    3. Truncation (last 100 tokens)
    """

    def __init__(self):
        self.pipeline = (
            EditDistanceProjection(max_distance=1) >>
            WhitespaceProjection() >>
            LowercaseProjection() >>
            UnicodeNormalizationProjection('NFC') >>
            RecencyProjection(max_length=100)
        )

    def project(self, context: List[int], corpus: List[int]) -> List[int]:
        return self.pipeline.project(context, corpus)
```

### 7.3 Validation

Check for common ordering mistakes:

```python
def validate_pipeline(pipeline: Projection):
    """Warn about potential ordering issues."""

    # Example: Warn if lossy operation comes before normalization
    if isinstance(pipeline, SequentialProjection):
        first, second = pipeline.first, pipeline.second

        if is_lossy(first) and is_normalizing(second):
            warnings.warn(
                f"Lossy projection {first} before normalizing {second}. "
                f"Consider reversing order."
            )
```

---

## 8. Mathematical Properties

### 8.1 Associativity (Preserved)

Despite non-commutativity, projections are **associative**:

$$(\pi_1 \circ \pi_2) \circ \pi_3 = \pi_1 \circ (\pi_2 \circ \pi_3)$$

**Proof:**
Both sides equal: $\pi_1(\pi_2(\pi_3(x, C), C), C)$

This allows us to write unambiguous chains:
$$\pi_1 \circ \pi_2 \circ \pi_3 \circ \pi_4$$

### 8.2 Identity Preservation

$$\pi \circ \pi_{\text{id}} = \pi_{\text{id}} \circ \pi = \pi$$

### 8.3 Partial Ordering

We can define a partial order on projections based on "should come before":

$$\pi_1 \prec \pi_2 \quad \text{if $\pi_1$ should be applied before $\pi_2$}$$

**Properties:**
- **Transitive:** $\pi_1 \prec \pi_2 \land \pi_2 \prec \pi_3 \implies \pi_1 \prec \pi_3$
- **Antisymmetric:** $\pi_1 \prec \pi_2 \land \pi_2 \prec \pi_1 \implies \pi_1 = \pi_2$
- **Partial:** Not all pairs are comparable

**Example Partial Order:**
```
EditDistance
    ↓
Lowercase ← → Whitespace  (incomparable/commutative)
    ↓            ↓
    ↓←←←←←←←←←←←←
    ↓
Synonym
    ↓
LongestSuffix
    ↓
Recency
```

---

## 9. Case Studies

### 9.1 Case Study: Query Expansion for Search

**Goal:** Find documents matching user query, accounting for typos and synonyms.

**Naive Pipeline (Wrong):**
```python
# WRONG: Synonym first might miss typos
SynonymProjection() >> EditDistanceProjection()
```

**Correct Pipeline:**
```python
# RIGHT: Fix typos, then expand synonyms
EditDistanceProjection(max_distance=2) >>
LowercaseProjection() >>
SynonymProjection(max_synsets=3) >>
LongestSuffixProjection()
```

**Rationale:**
1. Fix typos first (can't find synonyms of misspellings)
2. Normalize case (consistent matching)
3. Expand to synonyms (increase recall)
4. Find longest match in corpus (best result)

### 9.2 Case Study: Code Completion

**Goal:** Suggest code completions from codebase.

**Pipeline:**
```python
# 1. Normalize whitespace (code formatting varies)
WhitespaceProjection() >>

# 2. Keep recent tokens only (local context matters in code)
RecencyProjection(max_length=50) >>

# 3. Find longest matching suffix (exact match preferred)
LongestSuffixProjection(min_length=3)

# Note: No case normalization (code is case-sensitive)
# Note: No synonym expansion (code tokens are literal)
```

**Rationale:** Code is more literal than natural language, so fewer semantic transformations needed.

### 9.3 Case Study: Chat Message Matching

**Goal:** Find similar previous chat messages.

**Pipeline:**
```python
# 1. Aggressive normalization (chat has inconsistent formatting)
WhitespaceProjection() >>
LowercaseProjection() >>
PunctuationRemovalProjection() >>

# 2. Expand with common chat abbreviations
ChatAbbreviationExpansion() >>  # lol → laughing out loud, etc.

# 3. Emoji normalization
EmojiNormalizationProjection() >>

# 4. Find similar message
EditDistanceProjection(max_distance=3)
```

**Rationale:** Chat messages are informal and vary widely in style, need aggressive normalization.

---

## 10. Conclusion

### 10.1 Key Takeaways

1. **Projections are non-commutative** - order matters
2. **Follow ordering principles:**
   - Error correction first
   - Normalization before expansion
   - Semantic before structural
   - Lossy operations last
3. **Test commutativity** when unsure
4. **Document ordering constraints** in projection classes
5. **Use left-to-right (`>>`)** for readability

### 10.2 Default Pipeline Recommendation

For general text matching:

```python
StandardTextProjection = (
    EditDistanceProjection(max_distance=1) >>  # Fix typos
    WhitespaceProjection() >>                  # Normalize whitespace
    LowercaseProjection() >>                   # Case-insensitive
    UnicodeNormalizationProjection('NFC') >>   # Unicode consistency
    RecencyProjection(max_length=100)          # Keep recent context
)
```

### 10.3 Future Work

- **Automatic ordering:** Learn optimal projection order from data
- **Conditional projections:** Apply different projections based on context type
- **Adaptive ordering:** Reorder based on corpus statistics
- **Parallel exploration:** Try multiple orders and ensemble results

This framework provides principled guidance for ordering projections while acknowledging the inherent non-commutativity of the composition operation.
