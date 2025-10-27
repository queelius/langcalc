# Ollama Integration & NGramModel Removal Plan

## Summary

Created OllamaModel provider and identified NGramModel for removal. NGramModel has extensive usage and should be carefully deprecated.

## Completed: OllamaModel Provider ✅

### Created File
**`langcalc/models/ollama.py`** - Full LanguageModel implementation

### Features
- ✅ Implements `LanguageModel` interface (`logprobs`, `sample`, `score`)
- ✅ HTTP client for Ollama server (default: `localhost:11434`)
- ✅ Connection testing on initialization
- ✅ Byte-level token handling (UTF-8 encoding/decoding)
- ✅ Algebraic composition support (inherited from base class)
- ✅ Extra methods: `chat()`, `get_embeddings()`

### Usage

```python
from langcalc.models import OllamaModel, InfinigramModel

# Create Ollama model
llm = OllamaModel("mistral", host="localhost")

# Create Infinigram model
corpus = list("the cat sat on the mat".encode('utf-8'))
infinigram = InfinigramModel(corpus)

# Compose them algebraically
model = 0.95 * llm + 0.05 * infinigram

# Generate
context = list("the".encode('utf-8'))
samples = model.sample(context, max_tokens=20)
```

### Limitations

**Note:** Ollama API doesn't provide per-token log probabilities, so:
- `logprobs()` returns uniform distribution (approximation)
- `score()` returns approximate score based on sequence length
- For better probability estimates, would need to sample multiple times

### API Methods

| Method | Status | Notes |
|--------|--------|-------|
| `logprobs()` | ✅ | Approximated (uniform dist) |
| `sample()` | ✅ | Full Ollama API support |
| `score()` | ✅ | Approximated |
| `chat()` | ✅ | Bonus feature (chat API) |
| `get_embeddings()` | ✅ | Bonus feature |

### Testing Needed

To test OllamaModel (requires Ollama running):

```bash
# Start Ollama server
ollama serve

# Pull a model
ollama pull mistral

# Test in Python
python3 -c "
from langcalc.models import OllamaModel
model = OllamaModel('mistral')
samples = model.sample(max_tokens=10)
print(samples)
"
```

## NGramModel Removal Plan

### Why Remove NGramModel?

1. **Redundant with InfinigramModel**
   - NGramModel: Fixed-order n-grams
   - InfinigramModel: Variable-length (more flexible)

2. **InfinigramModel is Superior**
   - Memory efficient (O(n) vs O(V^n))
   - Adaptive context length
   - Backed by external, well-tested package

3. **Projection Features**
   - NGramModel supports projections
   - Need to document/preserve this capability
   - May want to add projection support to other models later

### NGramModel Projection Features

**What projections do:**
Transform context before model lookup:

```python
from langcalc.models.ngram import NGramModel
from langcalc.projections import RecencyProjection

# Create model with projection
model = NGramModel(corpus, n=3, projection=RecencyProjection(decay=0.9))

# Projection transforms context before n-gram lookup
logprobs = model.logprobs(tokens, context)
# ^^ context is projected first
```

**Available projections:**
- `IdentityProjection` - No transformation
- `RecencyProjection` - Weight recent tokens higher
- `EditDistanceProjection` - Find similar contexts
- `SemanticProjection` - Semantic similarity

**Projection operators:**
```python
proj1 >> proj2  # Sequential composition
proj1 | proj2   # Union
proj1 & proj2   # Intersection
weight * proj   # Weighted
```

### Impact Analysis

**Files with NGramModel references:** 15+ files

#### Core Files
- `langcalc/models/ngram.py` - **DELETE**
- `langcalc/__init__.py` - Remove imports
- `langcalc/grounding.py` - Uses `LightweightNGramModel` (different class)

#### Test Files (61 tests total)
- `tests/test_unit/test_ngram_model.py` - **DELETE** (61 tests)
- `tests/test_unit/test_infinigram.py` - 1 test uses NGramModel
- `tests/test_unit/test_projections.py` - 3 tests use NGramModel
- `tests/test_unit/test_algebraic_operations.py` - 3 tests use NGramModel
- `tests/test_integration/test_model_composition.py` - 19 tests use NGramModel
- `tests/conftest.py` - `ngram_model` fixture

### Replacement Strategy

| Use Case | Replace With |
|----------|--------------|
| Fixed n-gram | `InfinigramModel(corpus, max_length=n)` |
| Variable n-gram | `InfinigramModel(corpus)` |
| With projections | **TBD** - Need projection support |

### Projection Support - Future Work

**Option 1:** Add projection parameter to InfinigramModel
```python
InfinigramModel(corpus, projection=RecencyProjection())
```

**Option 2:** Use `@` operator (already in base class)
```python
model @ RecencyProjection()  # ProjectedModel wrapper
```

**Option 3:** Wait for your new paper's formalization
- You mentioned having newer ideas on projections
- Document current projection system for reference
- Implement new approach when ready

### Recommended Next Steps

**Immediate (don't do yet):**
1. ❌ Don't remove NGramModel yet
2. ❌ Don't update tests yet
3. ✅ Document projection features (done above)
4. ✅ Test OllamaModel works

**When ready to remove NGramModel:**
1. Decide on projection strategy
2. Replace NGramModel with InfinigramModel in tests
3. Update documentation
4. Remove ngram.py file
5. Run full test suite

### Files to Modify

```
DELETE:
  langcalc/models/ngram.py
  tests/test_unit/test_ngram_model.py

MODIFY:
  langcalc/models/__init__.py (remove import)
  langcalc/__init__.py (remove from __all__)
  tests/conftest.py (remove fixture)
  tests/test_unit/test_infinigram.py (1 test)
  tests/test_unit/test_projections.py (3 tests)
  tests/test_unit/test_algebraic_operations.py (3 tests)
  tests/test_integration/test_model_composition.py (19 tests)
```

## Current Status

✅ **OllamaModel**: Created and ready to use
⏸️ **NGramModel removal**: Documented, waiting for direction on projections

## Testing OllamaModel

```python
# Test 1: Basic instantiation
from langcalc.models import OllamaModel
model = OllamaModel("mistral")

# Test 2: LanguageModel interface
context = list("hello".encode('utf-8'))
logprobs = model.logprobs(list(range(256)), context)
samples = model.sample(context, max_tokens=10)
score = model.score(samples)

# Test 3: Algebraic composition
from langcalc.models import MockLLM
mixed = 0.5 * model + 0.5 * MockLLM(vocab_size=256)
```

## Questions for You

1. **Projections**: Do you want to:
   - Add projection support to InfinigramModel now?
   - Use the `@` operator with ProjectedModel wrapper?
   - Wait for your new projection formalization?

2. **NGramModel**: Should I:
   - Remove it now and deal with test breakage?
   - Keep it for now until projection strategy is clear?
   - Create a deprecated wrapper that uses InfinigramModel?

3. **OllamaModel**: Should I:
   - Add tests for it?
   - Improve logprobs approximation with sampling?
   - Leave it as-is for now?
