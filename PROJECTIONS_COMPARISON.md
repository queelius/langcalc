# Projections: LangCalc vs Infinigram

## Summary

**LangCalc** and **Infinigram** use the term "projection" for **completely different concepts**. They are NOT the same thing!

## LangCalc Projections

**What they are:** Context transformation before model lookup

**Purpose:** Transform the input context before querying the n-gram model

**Location:** `langcalc/projections/`

**Interface:**
```python
class Projection(ABC):
    def project(self, context: List[int]) -> List[int]:
        """Transform context before lookup"""

    def similarity(self, seq1: List[int], seq2: List[int]) -> float:
        """Compute similarity between sequences"""
```

**Examples:**
- `RecencyProjection` - Weight recent tokens higher
- `EditDistanceProjection` - Find similar contexts
- `SemanticProjection` - Semantic similarity
- `IdentityProjection` - No transformation

**Usage:**
```python
from langcalc.models.ngram import NGramModel
from langcalc.projections import RecencyProjection

# Transform context before n-gram lookup
model = NGramModel(corpus, n=3, projection=RecencyProjection(decay=0.9))
```

**Key Point:** Transforms INPUT context → then looks up in fixed corpus

---

## Infinigram "Projections" (Actually Augmentations)

**What they are:** Data augmentation - creates additional training data

**Purpose:** Add augmented variants of documents to the corpus during training

**Location:** `infinigram/corpus_utils.py`, `infinigram/repl.py`

**Interface:**
```python
def augmentation_function(text: str) -> str:
    """Transform text to create variant"""
    return transformed_text
```

**Examples:**
- `lowercase` - Convert to lowercase
- `uppercase` - Convert to uppercase
- `title` - Convert to title case
- `strip` - Remove whitespace

**Usage:**
```python
# In infinigram REPL:
/load the cat sat on the mat
/augment lowercase uppercase  # Creates 3 versions in corpus:
# 1. "the cat sat on the mat" (original)
# 2. "the cat sat on the mat" (lowercase - same as original)
# 3. "THE CAT SAT ON THE MAT" (uppercase)
```

**Key Point:** Transforms CORPUS data → creates augmented training data

---

## Infinigram Recursive Transformations

**What they are:** Context transformation for OOD generalization (DIFFERENT from augmentations!)

**Purpose:** Transform query context to find similar patterns in corpus

**Location:** `infinigram/recursive.py`

**Interface:**
```python
class Transformer(ABC):
    def generate_transformations(
        self, context: bytes, corpus_matches: List[bytes]
    ) -> List[Tuple[bytes, str]]:
        """Generate transformed contexts"""
```

**Examples:**
- `SynonymTransformer` - Replace words with synonyms
- `EditDistanceTransformer` - Fix typos
- `CaseNormalizer` - Normalize case

**Usage:**
```python
from infinigram.recursive import RecursiveInfinigram, SynonymTransformer

model = RecursiveInfinigram(
    corpus,
    transformers=[SynonymTransformer(), EditDistanceTransformer()]
)

# Tries multiple transformed contexts to find matches
probs = model.predict_recursive(context)
```

**Key Point:** Like LangCalc projections BUT **recursive** - generates multiple variants and tries them all

---

## Comparison Table

| Aspect | LangCalc Projections | Infinigram Augmentations | Infinigram Recursive |
|--------|---------------------|------------------------|---------------------|
| **What** | Context transform | Corpus augmentation | Context transform |
| **When** | Query time | Training time | Query time |
| **Purpose** | Find similar contexts | Increase data diversity | OOD generalization |
| **Input** | Query context | Training documents | Query context |
| **Output** | Transformed context | Augmented corpus | Multiple contexts |
| **Similar to** | - | Data augmentation | LangCalc projections |

## Conceptual Mapping

```
LangCalc Projection ≈ Infinigram RecursiveTransformer
                    ≠ Infinigram Augmentation
```

### LangCalc RecencyProjection
```python
# Transform context before lookup
projection.project([1, 2, 3, 4])
→ [0, 0, 3, 4]  # Weight recent tokens
```

### Infinigram Augmentation
```python
# Create variants in corpus
augment(["hello world"])
→ ["hello world", "HELLO WORLD", "Hello World"]
```

### Infinigram SynonymTransformer
```python
# Generate alternative contexts to try
transformer.generate_transformations("the cat sat")
→ [("the feline sat", "synonym:cat→feline"),
   ("the kitty sat", "synonym:cat→kitty")]
```

## Answer to Your Question

> "are the projections in the original Infinigram model package?"

**No** - LangCalc's projections are NOT in the infinigram package.

**BUT** - Infinigram has something similar called "Recursive Transformers" that serve a similar purpose (context transformation at query time).

## Recommendation

Since NGramModel is the ONLY model in LangCalc that uses projections, and:
1. Infinigram has its own recursive transformation system
2. InfinigramModel is a simple wrapper (no projections)
3. You mentioned having new ideas on projections anyway

**Options:**

### Option 1: Remove NGramModel, lose projections temporarily
- Remove NGramModel now
- Wait for your new projection formalization
- Implement new system when ready

### Option 2: Keep NGramModel for now
- Don't remove NGramModel yet
- Use it when you need projections
- Replace when new projection system is ready

### Option 3: Port projections to InfinigramModel
- Add `projection` parameter to InfinigramModel
- Apply projection before calling `infinigram.predict()`
- This would give InfinigramModel the same capability as NGramModel

**My recommendation:** **Option 1** - Remove NGramModel, focus on the new infinigram package, revisit projections when you have your new formalization ready.

## Code Impact if We Remove NGramModel

**Tests that would break:** ~30+ tests

**Easy fix:** Replace with InfinigramModel:
```python
# Old
from langcalc.models.ngram import NGramModel
model = NGramModel(corpus, n=3)

# New
from langcalc.models import InfinigramModel
model = InfinigramModel(corpus, max_length=3)
```

**Projections:** Would lose this capability temporarily until new system is designed.
