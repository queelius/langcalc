# Projection System - Reference Implementation

## Overview

This document provides a complete reference implementation of the projection formalism developed in `PROJECTION_FORMALISM.md`. This serves as both documentation and a specification for the actual implementation in LangCalc.

---

## 1. Core Abstractions

### 1.1 Projection Base Class

```python
from abc import ABC, abstractmethod
from typing import List, Set, Tuple, Optional
import numpy as np

class Projection(ABC):
    """
    Abstract base class for context projections.

    A projection transforms a query context before matching against the corpus.
    Mathematically: π: Σ* × 2^(Σ*) → Σ*
    """

    @abstractmethod
    def project(self, context: List[int], corpus: List[int]) -> List[int]:
        """
        Project context onto corpus.

        Args:
            context: Query context (sequence of token IDs)
            corpus: Corpus (sequence of token IDs)

        Returns:
            Transformed context
        """
        pass

    def project_multi(self, context: List[int], corpus: List[int]) -> Set[Tuple[int, ...]]:
        """
        Multi-valued projection (returns multiple candidate contexts).

        Default implementation returns singleton set. Override for projections
        that generate multiple candidates (e.g., synonym expansion).

        Args:
            context: Query context
            corpus: Corpus

        Returns:
            Set of transformed contexts (as tuples for hashability)
        """
        return {tuple(self.project(context, corpus))}

    # Composition operators

    def __rshift__(self, other: 'Projection') -> 'Projection':
        """
        Sequential composition: self >> other

        Applies self first, then other.
        Mathematically: (π₁ >> π₂)(x, C) = π₂(π₁(x, C), C)
        """
        return SequentialProjection(self, other)

    def __or__(self, other: 'Projection') -> 'Projection':
        """
        Parallel composition (union): self | other

        Returns multiple projected contexts.
        Mathematically: (π₁ | π₂)(x, C) = {π₁(x, C), π₂(x, C)}
        """
        return ParallelProjection(self, other)

    def __matmul__(self, weight: float) -> 'Projection':
        """
        Weighted projection: projection @ weight

        For use in stochastic composition.
        """
        return WeightedProjection(self, weight)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
```

### 1.2 Augmentation Base Class

```python
class Augmentation(ABC):
    """
    Abstract base class for corpus augmentations.

    An augmentation expands the corpus by adding transformed variants.
    Mathematically: α: 2^(Σ*) → 2^(Σ*)
    """

    @abstractmethod
    def augment(self, corpus: List[int]) -> List[int]:
        """
        Augment corpus with transformed variants.

        Args:
            corpus: Original corpus

        Returns:
            Augmented corpus (original + variants)
        """
        pass

    def __add__(self, other: 'Augmentation') -> 'Augmentation':
        """
        Compose augmentations: self + other

        Applies both augmentations to the corpus.
        """
        return ComposedAugmentation(self, other)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
```

---

## 2. Composition Implementations

### 2.1 Sequential Projection

```python
class SequentialProjection(Projection):
    """
    Sequential composition of projections.

    (π₁ >> π₂)(x, C) = π₂(π₁(x, C), C)
    """

    def __init__(self, first: Projection, second: Projection):
        self.first = first
        self.second = second

    def project(self, context: List[int], corpus: List[int]) -> List[int]:
        intermediate = self.first.project(context, corpus)
        return self.second.project(intermediate, corpus)

    def project_multi(self, context: List[int], corpus: List[int]) -> Set[Tuple[int, ...]]:
        # Apply first projection (may be multi-valued)
        intermediate_set = self.first.project_multi(context, corpus)

        # Apply second projection to each result
        result = set()
        for intermediate in intermediate_set:
            result.update(self.second.project_multi(list(intermediate), corpus))

        return result

    def __repr__(self) -> str:
        return f"({self.first} >> {self.second})"
```

### 2.2 Parallel Projection

```python
class ParallelProjection(Projection):
    """
    Parallel composition (union) of projections.

    (π₁ | π₂)(x, C) = {π₁(x, C), π₂(x, C)}
    """

    def __init__(self, *projections: Projection):
        self.projections = projections

    def project(self, context: List[int], corpus: List[int]) -> List[int]:
        # For single-valued interface, return first projection
        # (This is somewhat arbitrary for parallel composition)
        return self.projections[0].project(context, corpus)

    def project_multi(self, context: List[int], corpus: List[int]) -> Set[Tuple[int, ...]]:
        # Union of all projections
        result = set()
        for proj in self.projections:
            result.update(proj.project_multi(context, corpus))
        return result

    def __repr__(self) -> str:
        return " | ".join(str(p) for p in self.projections)
```

### 2.3 Weighted Projection

```python
class WeightedProjection:
    """
    Weighted projection for stochastic composition.

    Not a Projection itself, but used in mixture models.
    """

    def __init__(self, projection: Projection, weight: float):
        self.projection = projection
        self.weight = weight

    def __repr__(self) -> str:
        return f"{self.weight} * {self.projection}"
```

### 2.4 Composed Augmentation

```python
class ComposedAugmentation(Augmentation):
    """
    Composition of multiple augmentations.

    (α₁ + α₂)(C) applies both augmentations.
    """

    def __init__(self, *augmentations: Augmentation):
        self.augmentations = augmentations

    def augment(self, corpus: List[int]) -> List[int]:
        result = corpus
        for aug in self.augmentations:
            result = aug.augment(result)
        return result

    def __repr__(self) -> str:
        return " + ".join(str(a) for a in self.augmentations)
```

---

## 3. Basic Projections

### 3.1 Identity Projection

```python
class IdentityProjection(Projection):
    """
    Identity projection: π(x, C) = x

    No transformation.
    """

    def project(self, context: List[int], corpus: List[int]) -> List[int]:
        return context
```

### 3.2 Recency Projection

```python
class RecencyProjection(Projection):
    """
    Recency projection: truncate to most recent k tokens.

    π_rec(x, C) = x[-k:] if |x| > k else x
    """

    def __init__(self, max_length: int):
        """
        Args:
            max_length: Maximum context length to keep
        """
        self.max_length = max_length

    def project(self, context: List[int], corpus: List[int]) -> List[int]:
        if len(context) <= self.max_length:
            return context
        return context[-self.max_length:]

    def __repr__(self) -> str:
        return f"RecencyProjection(max_length={self.max_length})"
```

### 3.3 Truncation Projection

```python
class TruncationProjection(Projection):
    """
    Truncation projection: keep first k tokens.

    Useful for testing or limiting context scope.
    """

    def __init__(self, max_length: int):
        self.max_length = max_length

    def project(self, context: List[int], corpus: List[int]) -> List[int]:
        return context[:self.max_length]

    def __repr__(self) -> str:
        return f"TruncationProjection(max_length={self.max_length})"
```

---

## 4. Normalization Projections

### 4.1 Lowercase Projection

```python
class LowercaseProjection(Projection):
    """
    Lowercase projection: convert context to lowercase.

    π_lower(x, C) = lowercase(x)

    Note: If corpus is augmented with lowercase variant,
    this projection can be skipped (projection-augmentation duality).
    """

    def project(self, context: List[int], corpus: List[int]) -> List[int]:
        try:
            text = bytes(context).decode('utf-8')
            lower_text = text.lower()
            return list(lower_text.encode('utf-8'))
        except (UnicodeDecodeError, UnicodeEncodeError):
            # If not valid UTF-8, return unchanged
            return context

    def __repr__(self) -> str:
        return "LowercaseProjection()"
```

### 4.2 Uppercase Projection

```python
class UppercaseProjection(Projection):
    """Uppercase projection: convert context to uppercase."""

    def project(self, context: List[int], corpus: List[int]) -> List[int]:
        try:
            text = bytes(context).decode('utf-8')
            return list(text.upper().encode('utf-8'))
        except (UnicodeDecodeError, UnicodeEncodeError):
            return context
```

### 4.3 Whitespace Normalization Projection

```python
import re

class WhitespaceProjection(Projection):
    """
    Whitespace normalization: collapse consecutive whitespace to single space.

    π_ws(x, C) = normalize_whitespace(x)
    """

    def project(self, context: List[int], corpus: List[int]) -> List[int]:
        try:
            text = bytes(context).decode('utf-8')
            normalized = re.sub(r'\s+', ' ', text)
            return list(normalized.encode('utf-8'))
        except (UnicodeDecodeError, UnicodeEncodeError):
            return context

    def __repr__(self) -> str:
        return "WhitespaceProjection()"
```

### 4.4 Unicode Normalization Projection

```python
import unicodedata

class UnicodeNormalizationProjection(Projection):
    """
    Unicode normalization projection.

    π_unicode(x, C) = normalize(x, form)
    """

    def __init__(self, form: str = 'NFC'):
        """
        Args:
            form: Unicode normalization form ('NFC', 'NFD', 'NFKC', 'NFKD')
        """
        if form not in ('NFC', 'NFD', 'NFKC', 'NFKD'):
            raise ValueError(f"Invalid normalization form: {form}")
        self.form = form

    def project(self, context: List[int], corpus: List[int]) -> List[int]:
        try:
            text = bytes(context).decode('utf-8')
            normalized = unicodedata.normalize(self.form, text)
            return list(normalized.encode('utf-8'))
        except (UnicodeDecodeError, UnicodeEncodeError):
            return context

    def __repr__(self) -> str:
        return f"UnicodeNormalizationProjection(form='{self.form}')"
```

---

## 5. Advanced Projections

### 5.1 Edit Distance Projection

```python
class EditDistanceProjection(Projection):
    """
    Edit distance projection: find most similar suffix in corpus.

    π_edit(x, C) = argmin_{s ∈ Suffixes(C)} {edit(x, s) : edit(x, s) ≤ d}

    WARNING: This is expensive (O(|x| * |C|)). Use sparingly.
    """

    def __init__(self, max_distance: int = 2, suffix_length: Optional[int] = None):
        """
        Args:
            max_distance: Maximum edit distance to consider
            suffix_length: Only check suffixes of this length (for efficiency)
        """
        self.max_distance = max_distance
        self.suffix_length = suffix_length

    def project(self, context: List[int], corpus: List[int]) -> List[int]:
        if not context:
            return context

        # Limit search to suffixes of specific length if specified
        search_length = self.suffix_length or len(context)

        # Find best matching suffix (simplified implementation)
        best_suffix = None
        best_distance = float('inf')

        # Search through corpus for matching suffixes
        # (In practice, would use suffix array for efficiency)
        for i in range(len(corpus)):
            suffix = corpus[max(0, i - search_length):i]
            if not suffix:
                continue

            distance = self._edit_distance(context, suffix)
            if distance <= self.max_distance and distance < best_distance:
                best_distance = distance
                best_suffix = suffix

        return best_suffix if best_suffix is not None else context

    def _edit_distance(self, s1: List[int], s2: List[int]) -> int:
        """Compute Levenshtein distance between two sequences."""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)

        if not s2:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def __repr__(self) -> str:
        return f"EditDistanceProjection(max_distance={self.max_distance})"
```

### 5.2 Longest Suffix Projection

```python
class LongestSuffixProjection(Projection):
    """
    Longest suffix projection: find longest matching suffix.

    π_lms(x, C) = LMS(x, C)

    Uses suffix array for efficient lookup.
    """

    def __init__(self, min_length: int = 1):
        """
        Args:
            min_length: Minimum suffix length to consider
        """
        self.min_length = min_length
        self._suffix_array = None

    def project(self, context: List[int], corpus: List[int]) -> List[int]:
        # Build suffix array if not cached
        # (In practice, would build once and reuse)
        if self._suffix_array is None:
            from infinigram import Infinigram
            self._infinigram = Infinigram(corpus=corpus)

        # Find longest matching suffix
        # This is a simplified version - actual implementation would
        # use suffix array binary search
        for length in range(len(context), self.min_length - 1, -1):
            suffix = context[-length:]
            # Check if this suffix exists in corpus
            # (Would use suffix array lookup in practice)
            if self._exists_in_corpus(suffix, corpus):
                return suffix

        return context[-self.min_length:] if len(context) >= self.min_length else context

    def _exists_in_corpus(self, pattern: List[int], corpus: List[int]) -> bool:
        """Check if pattern exists in corpus (naive implementation)."""
        if not pattern:
            return False
        pattern_tuple = tuple(pattern)
        for i in range(len(corpus) - len(pattern) + 1):
            if tuple(corpus[i:i + len(pattern)]) == pattern_tuple:
                return True
        return False

    def __repr__(self) -> str:
        return f"LongestSuffixProjection(min_length={self.min_length})"
```

---

## 6. Basic Augmentations

### 6.1 Lowercase Augmentation

```python
class LowercaseAugmentation(Augmentation):
    """
    Lowercase augmentation: α_lower(C) = C ∪ {lowercase(C)}

    Doubles corpus size.
    """

    def augment(self, corpus: List[int]) -> List[int]:
        try:
            text = bytes(corpus).decode('utf-8')
            lower_text = text.lower()
            lower_bytes = list(lower_text.encode('utf-8'))
            # Return original + lowercase
            return corpus + lower_bytes
        except UnicodeDecodeError:
            return corpus

    def __repr__(self) -> str:
        return "LowercaseAugmentation()"
```

### 6.2 Case Augmentation

```python
class CaseAugmentation(Augmentation):
    """
    Full case augmentation: α_case(C) = C ∪ {lower, upper, title}

    Quadruples corpus size.
    """

    def augment(self, corpus: List[int]) -> List[int]:
        try:
            text = bytes(corpus).decode('utf-8')
            variants = [
                text,
                text.lower(),
                text.upper(),
                text.title(),
            ]
            return [byte for variant in variants
                    for byte in variant.encode('utf-8')]
        except UnicodeDecodeError:
            return corpus

    def __repr__(self) -> str:
        return "CaseAugmentation()"
```

### 6.3 Whitespace Augmentation

```python
class WhitespaceAugmentation(Augmentation):
    """
    Whitespace augmentation: α_ws(C) = C ∪ {normalize_ws(C)}

    Doubles corpus size.
    """

    def augment(self, corpus: List[int]) -> List[int]:
        try:
            text = bytes(corpus).decode('utf-8')
            normalized = re.sub(r'\s+', ' ', text)
            return corpus + list(normalized.encode('utf-8'))
        except UnicodeDecodeError:
            return corpus

    def __repr__(self) -> str:
        return "WhitespaceAugmentation()"
```

---

## 7. Model Integration

### 7.1 Projected Language Model

```python
class ProjectedModel(LanguageModel):
    """
    Language model with projection applied to context.

    M^π(x, a) = M(π(x, C), a)
    """

    def __init__(self, base_model: LanguageModel, projection: Projection, corpus: List[int]):
        """
        Args:
            base_model: Underlying language model
            projection: Projection to apply to context
            corpus: Corpus (needed for projection)
        """
        self.base_model = base_model
        self.projection = projection
        self.corpus = corpus

    def logprobs(self, tokens: List[int], context: Optional[List[int]] = None) -> np.ndarray:
        if context is None:
            context = []

        # Apply projection to context
        projected_context = self.projection.project(context, self.corpus)

        # Query base model with projected context
        return self.base_model.logprobs(tokens, projected_context)

    def sample(self, context: Optional[List[int]] = None,
               temperature: float = 1.0, max_tokens: int = 100) -> List[int]:
        if context is None:
            context = []

        projected_context = self.projection.project(context, self.corpus)
        return self.base_model.sample(projected_context, temperature, max_tokens)

    def score(self, sequence: List[int]) -> float:
        # For scoring, apply projection to increasingly long prefixes
        # This is one possible interpretation
        return self.base_model.score(sequence)

    def __repr__(self) -> str:
        return f"ProjectedModel({self.base_model} @ {self.projection})"
```

### 7.2 Multi-Projection Model

```python
class MultiProjectionModel(LanguageModel):
    """
    Model that tries multiple projections and combines results.

    M^{π_i, w_i}(x, a) = Σ_i w_i M(π_i(x, C), a)
    """

    def __init__(self, base_model: LanguageModel,
                 weighted_projections: List[Tuple[Projection, float]],
                 corpus: List[int]):
        """
        Args:
            base_model: Underlying language model
            weighted_projections: List of (projection, weight) pairs
            corpus: Corpus
        """
        self.base_model = base_model
        self.weighted_projections = weighted_projections
        self.corpus = corpus

        # Normalize weights
        total_weight = sum(w for _, w in weighted_projections)
        self.weighted_projections = [
            (proj, w / total_weight)
            for proj, w in weighted_projections
        ]

    def logprobs(self, tokens: List[int], context: Optional[List[int]] = None) -> np.ndarray:
        if context is None:
            context = []

        # Weighted mixture of projections
        result = np.zeros(len(tokens))
        for projection, weight in self.weighted_projections:
            projected_context = projection.project(context, self.corpus)
            logprobs = self.base_model.logprobs(tokens, projected_context)
            result += weight * np.exp(logprobs)  # Convert to probs, mix, convert back

        return np.log(result + 1e-10)  # Back to log space

    def sample(self, context: Optional[List[int]] = None,
               temperature: float = 1.0, max_tokens: int = 100) -> List[int]:
        if context is None:
            context = []

        # Randomly choose projection based on weights
        import random
        rand = random.random()
        cumsum = 0
        for projection, weight in self.weighted_projections:
            cumsum += weight
            if rand < cumsum:
                projected_context = projection.project(context, self.corpus)
                return self.base_model.sample(projected_context, temperature, max_tokens)

        # Fallback to last projection
        projected_context = self.weighted_projections[-1][0].project(context, self.corpus)
        return self.base_model.sample(projected_context, temperature, max_tokens)

    def score(self, sequence: List[int]) -> float:
        return self.base_model.score(sequence)

    def __repr__(self) -> str:
        proj_str = ", ".join(f"{w}*{p}" for p, w in self.weighted_projections)
        return f"MultiProjectionModel({self.base_model} @ [{proj_str}])"
```

---

## 8. Usage Examples

### 8.1 Simple Case-Insensitive Model

```python
from langcalc.models import InfinigramModel
from langcalc.projections import LowercaseProjection

# Approach 1: Query-time projection
corpus = list("Hello World".encode('utf-8'))
projection = LowercaseProjection()
model = ProjectedModel(
    InfinigramModel(corpus),
    projection=projection,
    corpus=corpus
)

# Approach 2: Training-time augmentation (more efficient)
from langcalc.augmentations import LowercaseAugmentation

augmented_corpus = LowercaseAugmentation().augment(corpus)
model = InfinigramModel(augmented_corpus)
```

### 8.2 Composed Projections

```python
# Normalize whitespace, then lowercase, then truncate to 10 tokens
projection = (
    WhitespaceProjection() >>
    LowercaseProjection() >>
    RecencyProjection(max_length=10)
)

model = ProjectedModel(InfinigramModel(corpus), projection, corpus)
```

### 8.3 Multi-Projection Model

```python
# Try multiple projections with different weights
projections = [
    (IdentityProjection(), 0.5),          # Original context
    (LowercaseProjection(), 0.3),         # Lowercase
    (RecencyProjection(5), 0.2),          # Recent tokens only
]

model = MultiProjectionModel(InfinigramModel(corpus), projections, corpus)
```

### 8.4 Standard Normalization

```python
# Common preprocessing pipeline
projection = (
    WhitespaceProjection() >>
    LowercaseProjection() >>
    UnicodeNormalizationProjection('NFC')
)

model = ProjectedModel(InfinigramModel(corpus), projection, corpus)
```

---

## 9. Testing

### 9.1 Projection Tests

```python
def test_identity_projection():
    proj = IdentityProjection()
    context = [1, 2, 3]
    corpus = [4, 5, 6]
    assert proj.project(context, corpus) == context

def test_recency_projection():
    proj = RecencyProjection(max_length=3)
    context = [1, 2, 3, 4, 5]
    corpus = []
    assert proj.project(context, corpus) == [3, 4, 5]

def test_lowercase_projection():
    proj = LowercaseProjection()
    context = list("Hello".encode('utf-8'))
    corpus = []
    result = bytes(proj.project(context, corpus)).decode('utf-8')
    assert result == "hello"

def test_sequential_composition():
    proj = WhitespaceProjection() >> LowercaseProjection()
    context = list("Hello  World".encode('utf-8'))
    corpus = []
    result = bytes(proj.project(context, corpus)).decode('utf-8')
    assert result == "hello world"
```

### 9.2 Augmentation Tests

```python
def test_lowercase_augmentation():
    aug = LowercaseAugmentation()
    corpus = list("Hello".encode('utf-8'))
    result = aug.augment(corpus)

    text = bytes(result).decode('utf-8')
    assert "Hello" in text
    assert "hello" in text
    assert len(result) == 2 * len(corpus)

def test_augmentation_composition():
    aug = LowercaseAugmentation() + WhitespaceAugmentation()
    corpus = list("Hello  World".encode('utf-8'))
    result = aug.augment(corpus)

    # Should contain: original, lowercase, normalized whitespace, and combinations
    text = bytes(result).decode('utf-8')
    assert "Hello  World" in text
    assert "hello  world" in text
```

---

## 10. Conclusion

This reference implementation provides:

1. **Core abstractions** - `Projection` and `Augmentation` base classes
2. **Composition operators** - Sequential (>>), parallel (|), weighted (@)
3. **Basic projections** - Identity, recency, truncation, normalization
4. **Advanced projections** - Edit distance, longest suffix
5. **Augmentations** - Case, whitespace, Unicode normalization
6. **Model integration** - `ProjectedModel` and `MultiProjectionModel`
7. **Usage examples** - Common patterns and workflows
8. **Testing strategy** - Unit tests for each component

This serves as the specification for implementing the projection system in LangCalc.
