# Canonical Corpus Augmentations

## Overview

This document catalogs the standard corpus augmentations (normal forms) that should be supported in LangCalc. Based on the **projection-augmentation duality theorem**, these augmentations implement common projections efficiently by transforming the corpus once at training time rather than transforming every query.

---

## 1. Case Normalization

### 1.1 Lowercase Normalization
**Mathematical Definition:**
$$\alpha_{\text{lower}}(C) = C \cup \{\text{lowercase}(s) : s \in C\}$$

**Purpose:** Enable case-insensitive matching.

**Effect:** Doubles corpus size (original + lowercase variant).

**Implementation:**
```python
class LowercaseAugmentation(Augmentation):
    """Augment corpus with lowercase variant."""

    def augment(self, corpus: List[int]) -> List[int]:
        try:
            text = bytes(corpus).decode('utf-8')
            lower_text = text.lower()
            lower_bytes = list(lower_text.encode('utf-8'))
            return corpus + lower_bytes
        except UnicodeDecodeError:
            # If corpus is not valid UTF-8, return unchanged
            return corpus
```

**Example:**
```python
# Input corpus: "Hello World"
# Output: "Hello WorldHello world"  # (original + lowercase)
```

### 1.2 Full Case Augmentation
**Mathematical Definition:**
$$\alpha_{\text{case}}(C) = C \cup \{\text{lower}(C), \text{upper}(C), \text{title}(C)\}$$

**Purpose:** Maximize case-insensitive matching coverage.

**Effect:** 4× corpus size (original + 3 variants).

**Implementation:**
```python
class CaseAugmentation(Augmentation):
    """Augment corpus with all case variants."""

    def augment(self, corpus: List[int]) -> List[int]:
        try:
            text = bytes(corpus).decode('utf-8')
            variants = [
                text,              # original
                text.lower(),      # lowercase
                text.upper(),      # uppercase
                text.title(),      # titlecase
            ]
            return [byte for variant in variants
                    for byte in variant.encode('utf-8')]
        except UnicodeDecodeError:
            return corpus
```

**Tradeoff:** Uses 4× space but completely eliminates case sensitivity.

---

## 2. Whitespace Normalization

### 2.1 Whitespace Collapsing
**Mathematical Definition:**
$$\alpha_{\text{ws}}(C) = C \cup \{\text{collapse\_ws}(C)\}$$

where `collapse_ws` replaces sequences of whitespace characters with single space.

**Purpose:** Handle formatting variations (tabs, multiple spaces, etc.).

**Implementation:**
```python
import re

class WhitespaceAugmentation(Augmentation):
    """Augment corpus with normalized whitespace."""

    def augment(self, corpus: List[int]) -> List[int]:
        try:
            text = bytes(corpus).decode('utf-8')
            # Collapse consecutive whitespace to single space
            normalized = re.sub(r'\s+', ' ', text)
            normalized_bytes = list(normalized.encode('utf-8'))
            return corpus + normalized_bytes
        except UnicodeDecodeError:
            return corpus
```

**Example:**
```python
# Input: "hello  world\t\tfoo"
# Output: "hello  world\t\tfoohello world foo"
```

### 2.2 Whitespace Stripping
**Mathematical Definition:**
$$\alpha_{\text{strip}}(C) = C \cup \{\text{strip}(C)\}$$

**Purpose:** Remove leading/trailing whitespace.

**Implementation:**
```python
class StripAugmentation(Augmentation):
    """Augment corpus with stripped variant."""

    def augment(self, corpus: List[int]) -> List[int]:
        try:
            text = bytes(corpus).decode('utf-8')
            stripped = text.strip()
            return corpus + list(stripped.encode('utf-8'))
        except UnicodeDecodeError:
            return corpus
```

---

## 3. Unicode Normalization

### 3.1 NFC Normalization
**Mathematical Definition:**
$$\alpha_{\text{NFC}}(C) = C \cup \{\text{NFC}(C)\}$$

where NFC is Unicode Normalization Form C (Canonical Composition).

**Purpose:** Handle different Unicode representations of same character (e.g., é as single character vs e + combining accent).

**Implementation:**
```python
import unicodedata

class NFCAugmentation(Augmentation):
    """Augment corpus with NFC normalized variant."""

    def augment(self, corpus: List[int]) -> List[int]:
        try:
            text = bytes(corpus).decode('utf-8')
            nfc_text = unicodedata.normalize('NFC', text)
            return corpus + list(nfc_text.encode('utf-8'))
        except UnicodeDecodeError:
            return corpus
```

### 3.2 Full Unicode Normalization
**Mathematical Definition:**
$$\alpha_{\text{unicode}}(C) = C \cup \{\text{NFC}(C), \text{NFD}(C), \text{NFKC}(C), \text{NFKD}(C)\}$$

**Purpose:** Maximum Unicode compatibility.

**Effect:** 5× corpus size.

**Implementation:**
```python
class UnicodeAugmentation(Augmentation):
    """Augment corpus with all Unicode normal forms."""

    FORMS = ['NFC', 'NFD', 'NFKC', 'NFKD']

    def augment(self, corpus: List[int]) -> List[int]:
        try:
            text = bytes(corpus).decode('utf-8')
            variants = [text]  # original
            for form in self.FORMS:
                normalized = unicodedata.normalize(form, text)
                variants.append(normalized)

            return [byte for variant in variants
                    for byte in variant.encode('utf-8')]
        except UnicodeDecodeError:
            return corpus
```

---

## 4. Punctuation Handling

### 4.1 Punctuation Removal
**Mathematical Definition:**
$$\alpha_{\text{nopunct}}(C) = C \cup \{\text{remove\_punct}(C)\}$$

**Purpose:** Match content regardless of punctuation.

**Implementation:**
```python
import string

class NoPunctuationAugmentation(Augmentation):
    """Augment corpus with punctuation removed."""

    def augment(self, corpus: List[int]) -> List[int]:
        try:
            text = bytes(corpus).decode('utf-8')
            # Remove all punctuation
            no_punct = text.translate(str.maketrans('', '', string.punctuation))
            return corpus + list(no_punct.encode('utf-8'))
        except UnicodeDecodeError:
            return corpus
```

**Example:**
```python
# Input: "Hello, world!"
# Output: "Hello, world!Hello world"
```

### 4.2 Punctuation Normalization
**Mathematical Definition:**
$$\alpha_{\text{punct}}(C) = C \cup \{\text{normalize\_punct}(C)\}$$

where normalization converts fancy quotes/dashes to ASCII equivalents.

**Implementation:**
```python
class PunctuationAugmentation(Augmentation):
    """Augment corpus with normalized punctuation."""

    PUNCT_MAP = {
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201C': '"',  # Left double quote
        '\u201D': '"',  # Right double quote
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2026': '...', # Ellipsis
    }

    def augment(self, corpus: List[int]) -> List[int]:
        try:
            text = bytes(corpus).decode('utf-8')
            for fancy, simple in self.PUNCT_MAP.items():
                text = text.replace(fancy, simple)
            return corpus + list(text.encode('utf-8'))
        except UnicodeDecodeError:
            return corpus
```

---

## 5. Composite Augmentations

### 5.1 Standard Normalization
**Mathematical Definition:**
$$\alpha_{\text{std}} = \alpha_{\text{case}} + \alpha_{\text{ws}} + \alpha_{\text{NFC}}$$

**Purpose:** Common baseline normalization (case + whitespace + Unicode).

**Implementation:**
```python
class StandardAugmentation(Augmentation):
    """Standard normalization: case + whitespace + Unicode NFC."""

    def __init__(self):
        self.augmentations = [
            CaseAugmentation(),
            WhitespaceAugmentation(),
            NFCAugmentation(),
        ]

    def augment(self, corpus: List[int]) -> List[int]:
        result = corpus
        for aug in self.augmentations:
            result = aug.augment(result)
        return result
```

**Effect:** Significantly larger corpus but handles most common variations.

### 5.2 Aggressive Normalization
**Mathematical Definition:**
$$\alpha_{\text{aggressive}} = \alpha_{\text{case}} + \alpha_{\text{ws}} + \alpha_{\text{unicode}} + \alpha_{\text{nopunct}}$$

**Purpose:** Maximum robustness to formatting differences.

**Warning:** Very large corpus expansion.

---

## 6. Language-Specific Augmentations

### 6.1 ASCII Folding
**Mathematical Definition:**
$$\alpha_{\text{ascii}}(C) = C \cup \{\text{to\_ascii}(C)\}$$

where accented characters are converted to ASCII equivalents (é → e).

**Purpose:** Match across accented/unaccented variants.

**Implementation:**
```python
class ASCIIFoldingAugmentation(Augmentation):
    """Augment corpus with ASCII-folded variant."""

    def augment(self, corpus: List[int]) -> List[int]:
        try:
            text = bytes(corpus).decode('utf-8')
            # Decompose to NFD and remove combining marks
            nfd = unicodedata.normalize('NFD', text)
            ascii_text = ''.join(
                char for char in nfd
                if unicodedata.category(char) != 'Mn'  # Mn = Mark, Nonspacing
            )
            return corpus + list(ascii_text.encode('utf-8'))
        except UnicodeDecodeError:
            return corpus
```

**Example:**
```python
# Input: "café"
# Output: "cafécafe"  # (é → e)
```

---

## 7. Augmentation Composition

### 7.1 Sequential Composition
Apply augmentations in sequence:

```python
class SequentialAugmentation(Augmentation):
    """Compose augmentations sequentially."""

    def __init__(self, *augmentations: Augmentation):
        self.augmentations = augmentations

    def augment(self, corpus: List[int]) -> List[int]:
        result = corpus
        for aug in self.augmentations:
            result = aug.augment(result)
        return result

# Usage
aug = SequentialAugmentation(
    CaseAugmentation(),
    WhitespaceAugmentation(),
    NFCAugmentation()
)
```

### 7.2 Parallel Composition
Apply augmentations independently and concatenate:

```python
class ParallelAugmentation(Augmentation):
    """Compose augmentations in parallel."""

    def __init__(self, *augmentations: Augmentation):
        self.augmentations = augmentations

    def augment(self, corpus: List[int]) -> List[int]:
        # Start with original
        result = corpus
        # Add each augmentation's output
        for aug in self.augmentations:
            augmented = aug.augment(corpus)  # Apply to original
            # Add only the new variants (skip original)
            result.extend(augmented[len(corpus):])
        return result
```

---

## 8. Recommended Augmentation Sets

### 8.1 Minimal (2× corpus)
```python
# Just lowercase
aug = LowercaseAugmentation()
```
**Use case:** Small corpora, case-insensitive matching.

### 8.2 Standard (≈8× corpus)
```python
# Case + whitespace + Unicode NFC
aug = SequentialAugmentation(
    CaseAugmentation(),       # 4×
    WhitespaceAugmentation(), # 2×
    NFCAugmentation()         # 2×
)
```
**Use case:** General-purpose text matching.

### 8.3 Aggressive (≈20× corpus)
```python
# Everything
aug = SequentialAugmentation(
    CaseAugmentation(),           # 4×
    WhitespaceAugmentation(),     # 2×
    UnicodeAugmentation(),        # 5×
    NoPunctuationAugmentation(),  # 2×
)
```
**Use case:** Maximum robustness, large corpora, plenty of memory.

### 8.4 Web Text (≈10× corpus)
```python
# Case + whitespace + punctuation + Unicode
aug = SequentialAugmentation(
    CaseAugmentation(),
    WhitespaceAugmentation(),
    PunctuationAugmentation(),
    NFCAugmentation()
)
```
**Use case:** Web scraping, mixed formatting sources.

---

## 9. Space-Time Tradeoffs

| Augmentation | Space Multiplier | Query Time Saved | When to Use |
|--------------|------------------|------------------|-------------|
| Lowercase | 2× | Significant | Almost always |
| Full Case | 4× | Significant | Case-insensitive search |
| Whitespace | 2× | Moderate | Mixed formatting |
| Unicode NFC | 2× | Significant | International text |
| Full Unicode | 5× | Significant | Maximum compatibility |
| No Punctuation | 2× | Moderate | Content-focused matching |
| Standard | ≈8× | High | General purpose |
| Aggressive | ≈20× | Very High | Large corpora only |

**Rule of thumb:** If you have memory for $k\times$ corpus expansion, use augmentation. Otherwise, use query-time projection.

---

## 10. Implementation Checklist

### Priority 1 (Must Have)
- [ ] `LowercaseAugmentation` - Case insensitive matching
- [ ] `WhitespaceAugmentation` - Format robustness
- [ ] `NFCAugmentation` - Unicode handling

### Priority 2 (Should Have)
- [ ] `CaseAugmentation` - Full case coverage
- [ ] `StripAugmentation` - Trim whitespace
- [ ] `ASCIIFoldingAugmentation` - ASCII compatibility

### Priority 3 (Nice to Have)
- [ ] `UnicodeAugmentation` - Full Unicode coverage
- [ ] `PunctuationAugmentation` - Punctuation normalization
- [ ] `NoPunctuationAugmentation` - Content matching

### Composition
- [ ] `SequentialAugmentation` - Chain augmentations
- [ ] `ParallelAugmentation` - Independent augmentations

### Presets
- [ ] `StandardAugmentation` - Recommended default
- [ ] `MinimalAugmentation` - Space-efficient
- [ ] `AggressiveAugmentation` - Maximum robustness

---

## 11. Testing Strategy

Each augmentation should be tested for:

1. **Correctness:** Augmented corpus contains expected variants
2. **UTF-8 safety:** Handles invalid UTF-8 gracefully
3. **Idempotency:** `aug.augment(aug.augment(corpus))` is predictable
4. **Composition:** Sequential/parallel composition works correctly
5. **Edge cases:** Empty corpus, non-text data, special characters

Example test:
```python
def test_lowercase_augmentation():
    corpus = list("Hello World".encode('utf-8'))
    aug = LowercaseAugmentation()
    result = aug.augment(corpus)

    # Should contain original + lowercase
    text = bytes(result).decode('utf-8')
    assert "Hello World" in text
    assert "hello world" in text

    # Should be exactly 2× original length
    assert len(result) == 2 * len(corpus)
```

---

## Conclusion

These canonical augmentations implement common normalization needs efficiently. The key insight from the **projection-augmentation duality** is:

> **For simple transformations, pay space (augmentation) to save time (per-query projection).**

This catalog provides a reference implementation for the most common use cases.
