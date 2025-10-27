# Mathematical Formalism for Projections

## Abstract

We develop a rigorous mathematical treatment of **projections** as transformations that map query contexts onto a corpus, enabling flexible pattern matching and generalization. We distinguish between **context projections** (query-time transformations) and **corpus augmentations** (training-time normal forms).

---

## 1. Basic Definitions

### 1.1 Fundamental Objects

**Definition 1.1 (Corpus):** A corpus $C$ is a finite sequence over an alphabet $\Sigma$:
$$C = (c_1, c_2, \ldots, c_n) \in \Sigma^*$$

For byte-level models, $\Sigma = \{0, 1, \ldots, 255\}$.

**Definition 1.2 (Context):** A context $x$ is a finite sequence over the same alphabet:
$$x = (x_1, x_2, \ldots, x_m) \in \Sigma^*$$

**Definition 1.3 (Language Model):** A language model $M$ is a function:
$$M: \Sigma^* \times \Sigma \to [0, 1]$$
such that for any context $x$, $\sum_{a \in \Sigma} M(x, a) = 1$.

### 1.2 Pattern Matching

**Definition 1.4 (Suffix):** A sequence $s$ is a suffix of $C$ at position $i$ if:
$$s = (c_{i-|s|+1}, \ldots, c_i)$$

**Definition 1.5 (Longest Matching Suffix):** Given context $x$ and corpus $C$:
$$\text{LMS}(x, C) = \arg\max_{s \in \text{Suffixes}(C)} \{|s| : s \text{ is a suffix of } x\}$$

---

## 2. Projection Theory

### 2.1 Context Projections

**Definition 2.1 (Projection):** A projection $\pi$ is a function:
$$\pi: \Sigma^* \times 2^{\Sigma^*} \to \Sigma^*$$

mapping a context $x$ and corpus $C$ to a transformed context $\pi(x, C)$.

**Interpretation:** $\pi$ "projects" the query context $x$ onto the corpus $C$, finding a representation that facilitates pattern matching.

**Key Properties:**
1. **Corpus-aware:** $\pi$ may depend on $C$ (e.g., finding similar contexts)
2. **Query-time:** Applied when querying the model
3. **Composable:** Projections can be combined

### 2.2 Canonical Projections

#### Identity Projection
$$\pi_{\text{id}}(x, C) = x$$

**Interpretation:** No transformation.

#### Recency Projection
For decay parameter $\lambda \in (0, 1)$:
$$\pi_{\text{rec}}(x, C) = \text{truncate}_k(x)$$
where $k = \arg\max_j \left\{ \sum_{i=1}^j \lambda^{j-i} > \theta \right\}$

**Interpretation:** Focus on recent tokens by truncating old context.

#### Edit Distance Projection
For distance threshold $d$:
$$\pi_{\text{edit}}(x, C) = \arg\min_{s \in \text{Suffixes}(C)} \{\text{edit}(x, s) : \text{edit}(x, s) \leq d\}$$

**Interpretation:** Find most similar context in corpus within edit distance $d$.

#### Case Normalization Projection
$$\pi_{\text{lower}}(x, C) = \text{lowercase}(x)$$
$$\pi_{\text{upper}}(x, C) = \text{uppercase}(x)$$

**Interpretation:** Normalize case to match corpus conventions.

---

## 3. Corpus Augmentation

### 3.1 Normal Forms

**Definition 3.1 (Corpus Augmentation):** An augmentation $\alpha$ is a function:
$$\alpha: 2^{\Sigma^*} \to 2^{\Sigma^*}$$

that expands the corpus by adding transformed variants.

**Definition 3.2 (Normal Form):** A normal form is a canonical representation of sequences. Common normal forms include:

1. **Lowercase Normal Form:**
   $$\alpha_{\text{lower}}(C) = C \cup \{\text{lowercase}(s) : s \in C\}$$

2. **Unicode Normal Form:**
   $$\alpha_{\text{nfc}}(C) = C \cup \{\text{NFC}(s) : s \in C\}$$
   where NFC is Unicode Normalization Form C.

3. **Whitespace Normal Form:**
   $$\alpha_{\text{ws}}(C) = C \cup \{\text{normalize\_ws}(s) : s \in C\}$$
   where consecutive whitespace is collapsed.

### 3.2 Projection-Augmentation Duality

**Theorem 3.1 (Duality):** For certain projections $\pi$ and augmentations $\alpha$, the following equivalence holds:
$$\text{LMS}(\pi(x, C), C) = \text{LMS}(x, \alpha(C))$$

**Proof sketch:**
- If $\pi$ transforms $x$ to match a normal form
- And $\alpha$ places corpus in that normal form
- Then finding patterns in transformed $x$ on original $C$ equals finding patterns in original $x$ on augmented $C$

**Example:** Case-insensitive matching:
$$\text{LMS}(\pi_{\text{lower}}(x, C), C) = \text{LMS}(x, \alpha_{\text{lower}}(C))$$

**Practical implication:** We can implement certain projections efficiently by augmenting the corpus once at training time rather than transforming every query.

---

## 4. Projection Algebra

### 4.1 Composition Operations

**Definition 4.1 (Sequential Composition):**
$$(\pi_1 \circ \pi_2)(x, C) = \pi_1(\pi_2(x, C), C)$$

**Interpretation:** Apply $\pi_2$ first, then $\pi_1$.

**Notation:** We may write $\pi_1 \circ \pi_2$ or $\pi_2 \gg \pi_1$ (left-to-right).

**Definition 4.2 (Parallel Composition - Union):**
$$(\pi_1 \sqcup \pi_2)(x, C) = \pi_1(x, C) \cup \pi_2(x, C)$$

Returns a *set* of projected contexts (multi-valued projection).

**Definition 4.3 (Parallel Composition - Weighted):**
For weights $w_1, w_2$ with $w_1 + w_2 = 1$:
$$(\pi_1 \oplus_{w_1} \pi_2)(x, C) = \begin{cases}
\pi_1(x, C) & \text{with probability } w_1 \\
\pi_2(x, C) & \text{with probability } w_2
\end{cases}$$

**Interpretation:** Stochastically choose between projections.

### 4.2 Algebraic Properties

**Proposition 4.1 (Associativity):** Sequential composition is associative:
$$(\pi_1 \circ \pi_2) \circ \pi_3 = \pi_1 \circ (\pi_2 \circ \pi_3)$$

**Proposition 4.2 (Identity):** $\pi_{\text{id}}$ is the identity element:
$$\pi \circ \pi_{\text{id}} = \pi_{\text{id}} \circ \pi = \pi$$

**Proposition 4.3 (Commutativity - Special Cases):**
Projections commute if they transform independent aspects:
$$\pi_{\text{lower}} \circ \pi_{\text{ws}} = \pi_{\text{ws}} \circ \pi_{\text{lower}}$$

(Case normalization and whitespace normalization are independent)

---

## 5. Projected Language Models

### 5.1 Model with Projection

**Definition 5.1 (Projected Model):** Given a model $M$, corpus $C$, and projection $\pi$, the projected model is:
$$M^\pi(x, a) = M(\pi(x, C), a)$$

**Interpretation:** Transform the context via $\pi$ before querying the model.

### 5.2 Multi-Projection Models

For a set of projections $\{\pi_1, \ldots, \pi_k\}$ with weights $\{w_1, \ldots, w_k\}$:

**Definition 5.2 (Mixture of Projections):**
$$M^{\{\pi_i, w_i\}}(x, a) = \sum_{i=1}^k w_i M(\pi_i(x, C), a)$$

**Definition 5.3 (Recursive Projection):**
$$M^{\text{rec}}(x, a) = \max_{\pi \in \Pi} M(\pi(x, C), a)$$

where $\Pi$ is a set of candidate projections.

**Interpretation:** Try multiple projections and take the maximum (most confident) prediction.

---

## 6. Canonical Augmentations (Normal Forms)

Based on the duality theorem, we identify augmentations that efficiently implement common projections:

### 6.1 Case Normalization
**Augmentation:**
$$\alpha_{\text{case}} = \alpha_{\text{lower}} \cup \alpha_{\text{upper}} \cup \alpha_{\text{title}}$$

**Effect:** Enables case-insensitive matching without query-time transformation.

**Implementation:**
```python
def augment_case(corpus: List[int]) -> List[int]:
    text = bytes(corpus).decode('utf-8')
    variants = [text, text.lower(), text.upper(), text.title()]
    return [byte for variant in variants for byte in variant.encode('utf-8')]
```

### 6.2 Whitespace Normalization
**Augmentation:**
$$\alpha_{\text{ws}}(C) = C \cup \{\text{normalize}(s) : s \in C\}$$
where normalize collapses consecutive whitespace to single space.

**Effect:** Robust to formatting differences.

### 6.3 Unicode Normalization
**Augmentation:**
$$\alpha_{\text{unicode}}(C) = C \cup \{\text{NFC}(s), \text{NFD}(s), \text{NFKC}(s), \text{NFKD}(s) : s \in C\}$$

**Effect:** Handles different Unicode representations of same character.

### 6.4 Stemming/Lemmatization (Language-Specific)
**Augmentation:**
$$\alpha_{\text{stem}}(C) = C \cup \{\text{stem}(w) : w \in \text{words}(C)\}$$

**Effect:** Match different word forms (running → run).

**Note:** Requires linguistic processing, breaks byte-level abstraction.

---

## 7. Complexity Analysis

### 7.1 Space Complexity

**Corpus Augmentation:**
- Original corpus: $|C| = n$
- With $k$ augmentations: $|C'| \leq k \cdot n$
- **Space cost:** $O(kn)$

**Suffix Array:**
- Original: $O(n)$ space
- Augmented: $O(kn)$ space

**Tradeoff:** Pay $k\times$ space to avoid per-query transformation cost.

### 7.2 Time Complexity

**Query with Projection:**
- Projection cost: $O(|x|)$ for simple projections (case, whitespace)
- Edit distance projection: $O(|x| \cdot n)$ (expensive!)
- Suffix array lookup: $O(|x| \log n)$
- **Total:** $O(|x| + |x| \log n) = O(|x| \log n)$

**Query with Augmentation:**
- No projection cost
- Suffix array lookup on augmented corpus: $O(|x| \log(kn)) = O(|x| \log n)$ (logarithm absorbs constant)
- **Total:** $O(|x| \log n)$

**Conclusion:** For simple projections, augmentation is strictly better (no per-query cost, same asymptotic lookup).

---

## 8. Examples

### 8.1 Case-Insensitive Model

**Approach 1: Query-time projection**
```python
class CaseProjection(Projection):
    def project(self, context: List[int], corpus: List[int]) -> List[int]:
        text = bytes(context).decode('utf-8')
        return list(text.lower().encode('utf-8'))

model = InfinigramModel(corpus, projection=CaseProjection())
```

**Approach 2: Training-time augmentation**
```python
def augment_case(corpus: List[int]) -> List[int]:
    text = bytes(corpus).decode('utf-8')
    lower = text.lower().encode('utf-8')
    upper = text.upper().encode('utf-8')
    return list(corpus) + list(lower) + list(upper)

augmented_corpus = augment_case(corpus)
model = InfinigramModel(augmented_corpus)
```

**Tradeoff:** Approach 2 uses 3× space but avoids per-query transformation.

### 8.2 Edit Distance (Must Use Projection)

```python
class EditDistanceProjection(Projection):
    def __init__(self, max_distance: int = 2):
        self.max_distance = max_distance

    def project(self, context: List[int], corpus: List[int]) -> List[int]:
        # Find most similar suffix in corpus within max_distance
        best_suffix = find_closest_suffix(context, corpus, self.max_distance)
        return best_suffix if best_suffix else context
```

**Note:** Cannot precompute all edit distance variants (exponential), must use query-time projection.

### 8.3 Recency Weighting

```python
class RecencyProjection(Projection):
    def __init__(self, max_length: int = 10):
        self.max_length = max_length

    def project(self, context: List[int], corpus: List[int]) -> List[int]:
        # Keep only most recent max_length tokens
        return context[-self.max_length:] if len(context) > self.max_length else context
```

### 8.4 Composed Projection

```python
# Sequential: normalize whitespace, then lowercase, then truncate
projection = WhitespaceProjection() >> CaseProjection() >> RecencyProjection(10)

# Parallel: try both original and lowercased
projection = IdentityProjection() | CaseProjection()
```

---

## 9. Implementation Strategy

### 9.1 Projection Interface

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Set

class Projection(ABC):
    @abstractmethod
    def project(self, context: List[int], corpus: List[int]) -> List[int]:
        """Project context onto corpus."""
        pass

    def project_multi(self, context: List[int], corpus: List[int]) -> Set[List[int]]:
        """Multi-valued projection (returns set of contexts)."""
        return {tuple(self.project(context, corpus))}

    def __rshift__(self, other: 'Projection') -> 'Projection':
        """Sequential composition: self >> other"""
        return SequentialProjection(self, other)

    def __or__(self, other: 'Projection') -> 'Projection':
        """Parallel composition: self | other"""
        return ParallelProjection(self, other)
```

### 9.2 Augmentation Interface

```python
class Augmentation(ABC):
    @abstractmethod
    def augment(self, corpus: List[int]) -> List[int]:
        """Augment corpus with transformed variants."""
        pass

    def __add__(self, other: 'Augmentation') -> 'Augmentation':
        """Combine augmentations."""
        return CombinedAugmentation(self, other)
```

### 9.3 Model Integration

```python
class InfinigramModel(LanguageModel):
    def __init__(self,
                 corpus: List[int],
                 projection: Optional[Projection] = None,
                 augmentation: Optional[Augmentation] = None):
        # Apply augmentation to corpus at initialization
        if augmentation:
            corpus = augmentation.augment(corpus)

        self.corpus = corpus
        self.projection = projection or IdentityProjection()
        self.infinigram = Infinigram(corpus=corpus)

    def logprobs(self, tokens: List[int], context: Optional[List[int]] = None):
        # Apply projection at query time
        if context:
            context = self.projection.project(context, self.corpus)

        probs_dict = self.infinigram.predict(context, top_k=256)
        # ... convert to log probabilities
```

---

## 10. Future Directions

### 10.1 Learnable Projections
Instead of hand-crafted projections, learn transformation parameters:
$$\pi_\theta(x, C) = f_\theta(x, C)$$
where $\theta$ are learned parameters.

### 10.2 Semantic Projections
Use embeddings to find semantically similar contexts:
$$\pi_{\text{sem}}(x, C) = \arg\min_{s \in \text{Suffixes}(C)} \|\text{embed}(x) - \text{embed}(s)\|$$

### 10.3 Multi-Scale Projections
Apply different projections at different context lengths:
$$M^{\text{multi}}(x, a) = \sum_{i=1}^k w_i M(\pi_i(x_{-k_i:}), a)$$

### 10.4 Adaptive Projections
Choose projection based on context:
$$\pi_{\text{adapt}}(x, C) = \pi_{h(x)}(x, C)$$
where $h(x)$ selects which projection to use.

---

## 11. Conclusion

We have developed a rigorous mathematical framework for projections in language models:

1. **Projections** as corpus-aware context transformations
2. **Augmentations** as training-time normal forms
3. **Duality theorem** relating projections and augmentations
4. **Projection algebra** supporting composition
5. **Canonical augmentations** for common use cases

**Key Insights:**
- Simple projections (case, whitespace) → use augmentation (better performance)
- Complex projections (edit distance, semantic) → use query-time projection (infeasible to precompute)
- Projections compose naturally, forming an algebra

**Implementation priority:**
1. Core projection interface
2. Canonical augmentations (case, whitespace, Unicode)
3. Simple projections (identity, recency, truncation)
4. Composition operators (>>, |)
5. Complex projections (edit distance, semantic)

This formalism provides a solid foundation for the projection system in LangCalc.
