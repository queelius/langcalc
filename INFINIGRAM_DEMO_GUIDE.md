# Infinigram Demo Guide

## Quick Start

The Infinigram model is now ready to use! We've created two demo scripts to help you explore it:

### 1. Simple Demo (Recommended for First Try)

```bash
python infinigram_simple_demo.py
```

This runs 4 focused examples showing:
- **Numeric sequences**: Basic pattern matching
- **Text prediction**: Word-level predictions
- **Variable-length matching**: How the suffix array finds patterns
- **Dynamic updates**: Adding new data to the corpus

### 2. Comprehensive Demo

```bash
python test_infinigram_demo.py
```

This runs 7 in-depth demonstrations:
- Basic usage
- Text corpus handling
- Longest suffix matching
- Confidence scoring
- Dynamic updates
- Performance benchmarking
- Wikipedia-style example

## Interactive Python Usage

```python
from langcalc import Infinigram

# Create a model
corpus = [1, 2, 3, 4, 2, 3, 5, 6, 2, 3, 4]
model = Infinigram(corpus, max_length=10)

# Predict next token
context = [2, 3]
probs = model.predict(context)
print(probs)
# Output: {4: 0.6569, 5: 0.3301, 1: 0.0033, ...}

# Find longest matching suffix
position, length = model.longest_suffix(context)
print(f"Found {length} tokens at position {position}")
# Output: Found 2 tokens at position 8

# Get confidence score
confidence = model.confidence(context)
print(f"Confidence: {confidence:.4f}")
# Output: Confidence: 0.1490

# Update with new data
model.update([2, 3, 7, 8])
print(f"New corpus size: {len(model.corpus)}")
# Output: New corpus size: 15
```

## Demo Results Summary

From running `infinigram_simple_demo.py`:

### Example 1: Numeric Sequences
```
Corpus: [1, 2, 3, 4, 2, 3, 5, 6, 2, 3, 4, 7, 8]
Context: [2, 3]

Most likely next tokens:
  4: 0.653 ████████████████████████████████
  5: 0.328 ████████████████
```

**Analysis**: The pattern `[2, 3, 4]` appears twice in the corpus, so token 4 gets ~65% probability.

### Example 2: Text Prediction
```
Training on 3 sentences:
  "the cat sat on the mat"
  "the dog sat on the rug"
  "the cat ran on the mat"

Context: 'the cat'
Predictions:
  'sat': 0.486
  'ran': 0.486
```

**Analysis**: After "the cat", we've seen both "sat" and "ran" once each, so they get equal probability.

### Example 3: Variable-Length Matching
```
Corpus: [1, 2, 3, 4, 5, 1, 2, 3, 6, 1, 2, 3, 4, 7]

Context: [1, 2, 3, 4]
  ✓ Matched 4 tokens at position 0
  Confidence: 0.286
```

**Analysis**: Shows how longer matches give higher confidence scores.

### Example 4: Dynamic Updates
```
Initial corpus: [1, 2, 3, 4, 5]
Context: [3, 4]
Before: Token 5 has 96.2% probability

After adding: [3, 4, 6, 7, 3, 4, 6, 8]
After: Token 6 has 65.3% probability
```

**Analysis**: The model adapts immediately to new patterns in the data.

## Key Features Demonstrated

### 1. Variable-Length N-grams
Unlike traditional fixed-order n-grams, Infinigram automatically finds the longest matching pattern:
```python
# Will match as much context as possible (up to max_length)
model = Infinigram(corpus, max_length=20)
```

### 2. Suffix Array Efficiency
- **O(m log n)** query time (m = pattern length, n = corpus size)
- **O(n)** space complexity
- Much more memory efficient than hash-based n-grams

### 3. Confidence Scoring
Longer matches = higher confidence:
```python
confidence = model.confidence(context)
# Returns 0.0 to 1.0 based on match length
```

### 4. Dynamic Updates
Add new data without rebuilding from scratch:
```python
model.update([new, tokens, here])
# Suffix array is reconstructed, predictions updated
```

## Performance Characteristics

From the comprehensive demo's benchmark:

| Corpus Size | Construction | Avg Prediction | Avg Suffix Search |
|-------------|--------------|----------------|-------------------|
| 100 tokens  | 0.07 ms      | 0.043 ms       | 0.014 ms          |
| 1K tokens   | 6.09 ms      | 0.390 ms       | 0.184 ms          |
| 10K tokens  | 718 ms       | 4.370 ms       | 2.373 ms          |

**Notes**:
- Construction is one-time cost (or on updates)
- Prediction time includes suffix search + probability computation
- Suffix search is the main bottleneck for large corpora

## Advanced Usage

### Text Corpus Example

```python
from langcalc import Infinigram

# Prepare text data
sentences = [
    "the cat sat on the mat",
    "the dog sat on the rug"
]

# Build vocabulary
vocab = {}
corpus = []
for sent in sentences:
    for word in sent.split():
        if word not in vocab:
            vocab[word] = len(vocab)
        corpus.append(vocab[word])

# Create model
model = Infinigram(corpus, max_length=5, smoothing=0.001)

# Predict
context_words = ["the", "cat"]
context = [vocab[w] for w in context_words]
probs = model.predict(context)

# Convert back to words
id_to_word = {v: k for k, v in vocab.items()}
for token_id, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]:
    print(f"{id_to_word[token_id]}: {prob:.3f}")
```

### Adjusting Parameters

```python
model = Infinigram(
    corpus,
    max_length=20,      # Maximum pattern length to search (default: None)
    min_count=2,        # Minimum occurrences to include (default: 1)
    smoothing=0.01      # Smoothing factor for unseen tokens (default: 0.01)
)
```

**Parameter Effects**:
- `max_length`: Higher = more context considered, slower queries
- `min_count`: Higher = only frequent patterns, more smoothing
- `smoothing`: Higher = more uniform distribution, less peaked

## Comparison with Traditional N-grams

| Feature | Traditional N-gram | Infinigram |
|---------|-------------------|------------|
| Pattern Length | Fixed (n) | Variable (up to max_length) |
| Memory | O(V^n) exponential | O(corpus_size) linear |
| Query Time | O(1) hash lookup | O(m log n) suffix search |
| Updates | Fast (add to hash) | Slow (rebuild suffix array) |
| Best For | Small n, frequent updates | Large patterns, static corpus |

## Integration with LangCalc

The Infinigram can be composed with other models:

```python
from langcalc import Infinigram, NGramModel
from langcalc.models import MockLLM

# Create models
wiki = Infinigram(wikipedia_corpus, max_length=20)
news = NGramModel(news_corpus, n=3)
llm = MockLLM(vocab_size=10000)

# Compose using algebraic operators
model = 0.90 * llm + 0.07 * wiki + 0.03 * news

# Or use algebra module for transformations
from langcalc.algebra import LongestSuffixTransform
grounded = llm + (wiki << LongestSuffixTransform(suffix_array))
```

## Test Suite

The Infinigram has 36 comprehensive tests covering:
- ✅ Suffix array construction and queries
- ✅ Longest suffix matching
- ✅ Continuation probability computation
- ✅ Prediction with smoothing
- ✅ Confidence scoring
- ✅ Dynamic corpus updates
- ✅ Edge cases (empty corpus, long contexts)
- ✅ Integration scenarios

Run the tests:
```bash
pytest tests/test_unit/test_infinigram.py -v
# 36 tests, all passing
```

## Files

- **`infinigram_simple_demo.py`** - Quick 4-example demo (recommended)
- **`test_infinigram_demo.py`** - Comprehensive 7-demo showcase
- **`langcalc/infinigram.py`** - Source code (381 lines)
- **`tests/test_unit/test_infinigram.py`** - Test suite (36 tests)
- **`INFINIGRAM_DESIGN.md`** - Complete API specification

## Next Steps

1. **Try the demos**: Run the simple demo first
2. **Experiment interactively**: Use Python REPL with examples above
3. **Read the tests**: See `tests/test_unit/test_infinigram.py` for usage patterns
4. **Check the design doc**: Read `INFINIGRAM_DESIGN.md` for full details
5. **Compose models**: Try mixing Infinigram with other LangCalc models

## Troubleshooting

### Slow Construction Time
If corpus is very large (>100K tokens), construction may be slow:
```python
# Solution: Use smaller max_length or consider batching
model = Infinigram(corpus, max_length=10)  # Faster than max_length=50
```

### Memory Issues
For very large corpora:
```python
# The suffix array stores the full corpus in memory
# Consider: sampling, compression, or using incremental updates
```

### Low Confidence Scores
If all confidence scores are low:
```python
# May need more training data or longer patterns
model = Infinigram(larger_corpus, max_length=30)
```

## Support

- **Documentation**: `INFINIGRAM_DESIGN.md`
- **Tests**: `tests/test_unit/test_infinigram.py`
- **Examples**: `infinigram_simple_demo.py`, `test_infinigram_demo.py`
- **Source**: `langcalc/infinigram.py`

---

**Happy experimenting with Infinigrams!** 🚀
