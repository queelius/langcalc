# Infinigram Implementation Design

## What is an Infinigram?

An **Infinigram** is a language model that supports **variable-length n-grams** (up to infinite length) using suffix arrays for efficient storage and retrieval. Unlike traditional n-gram models with fixed order n, Infinigrams can query the longest matching suffix from a corpus.

## Key Features

1. **Variable-length matching**: Find longest suffix match, not just fixed n
2. **Efficient storage**: Suffix array = O(N) space vs. O(N^k) for k-grams
3. **Fast queries**: O(m log N) lookup time using binary search
4. **Dynamic updates**: Support incremental corpus updates
5. **Continuation probabilities**: Get P(next_token | longest_matching_suffix)

## API Design

### Core Class: `Infinigram`

```python
class Infinigram(LanguageModel):
    """
    Variable-length n-gram model using suffix arrays.

    Supports queries like:
    - What's the longest suffix of my context in the corpus?
    - What are the continuation probabilities?
    - How confident is this prediction?
    """

    def __init__(self, corpus: List[int],
                 max_length: Optional[int] = None,
                 min_count: int = 1):
        """
        Initialize Infinigram from corpus.

        Args:
            corpus: Token sequence
            max_length: Maximum suffix length to consider (None = unlimited)
            min_count: Minimum frequency threshold
        """

    def predict(self, context: List[int], top_k: int = 50) -> Dict[int, float]:
        """
        Predict next token probabilities.

        Returns:
            Distribution over next tokens based on longest matching suffix
        """

    def longest_suffix(self, context: List[int]) -> Tuple[int, int]:
        """
        Find longest matching suffix in corpus.

        Returns:
            (position, length) of longest match
        """

    def continuations(self, context: List[int]) -> Dict[int, int]:
        """
        Get continuation counts for longest matching suffix.

        Returns:
            Dict mapping next_token -> count
        """

    def update(self, new_tokens: List[int]):
        """
        Dynamically add new tokens to corpus.
        Incremental suffix array update.
        """

    def confidence(self, context: List[int]) -> float:
        """
        How confident is this prediction?
        Based on suffix match length and continuation frequency.
        """
```

### Integration with Algebraic Framework

```python
# Infinigram is a LanguageModel, so it supports algebra:

wiki_gram = Infinigram(wikipedia_corpus)
news_gram = Infinigram(news_corpus)

# Algebraic composition
model = 0.95 * llm + 0.03 * wiki_gram + 0.02 * news_gram

# With transformations
from langcalc import LongestSuffixTransform
grounded = llm + (wiki_gram << LongestSuffixTransform())
```

## Implementation Strategy

### Phase 1: Core Infinigram
```python
langcalc/infinigram.py:
  - Infinigram class
  - Variable-length suffix matching
  - Continuation probability computation
  - Integration with LanguageModel interface
```

### Phase 2: Optimizations
```python
langcalc/data/incremental.py:
  - IncrementalSuffixArray
  - Efficient dynamic updates
  - Caching for repeated queries
```

### Phase 3: Advanced Features
```python
langcalc/infinigram.py:
  - Backoff strategies (if no match, try shorter suffix)
  - Interpolation with lower-order n-grams
  - Confidence scoring
  - Batch queries
```

## Differences from Current NGramModel

| Feature | Current NGramModel | Infinigram |
|---------|-------------------|------------|
| N-gram order | Fixed (e.g., n=3) | Variable (longest match) |
| Storage | Hash table per order | Single suffix array |
| Query | Lookup context[-n:] | Binary search longest suffix |
| Memory | O(N * n) | O(N) |
| Update | Rebuild | Incremental |

## Mathematical Foundation

Given context `c = [w_1, w_2, ..., w_m]`:

1. **Find longest suffix**:
   ```
   s* = argmax_{s ⊂ corpus} |s| where s is suffix of c
   ```

2. **Continuation probability**:
   ```
   P(w | c) = count(s* · w) / count(s*)
   ```

3. **Backoff**:
   ```
   P(w | c) = λ * P_infinigram(w | c) + (1-λ) * P_backoff(w | c[1:])
   ```

## Usage Examples

```python
from langcalc import Infinigram, Ollama

# Create Infinigram from Wikipedia
corpus = load_wikipedia_tokens()
wiki_gram = Infinigram(corpus, max_length=100)

# Query
context = tokenize("The capital of France is")
continuations = wiki_gram.continuations(context)
# => {token("Paris"): 1523, token("the"): 42, ...}

# Compose with LLM
llm = Ollama("llama2")
grounded_llm = 0.97 * llm + 0.03 * wiki_gram

# Generate
text = grounded_llm.sample(context, max_tokens=50)

# Dynamic update
wiki_gram.update(new_articles)  # Incremental, no rebuild!
```

## Performance Targets

- **Construction**: O(N log N) time, O(N) space
- **Query**: O(m log N) time where m = context length
- **Update**: O(k log N) time for k new tokens
- **Memory**: ~10 bytes per token (vs ~100 bytes for hash table)

## Testing Strategy

```python
tests/test_unit/test_infinigram.py:
  - Variable-length matching
  - Continuation probabilities
  - Backoff strategies
  - Edge cases (empty corpus, no match)
  - Comparison with fixed n-gram

tests/test_integration/test_infinigram_composition.py:
  - Integration with LLM
  - Algebraic composition
  - Dynamic updates
  - Real corpus (Wikipedia sample)
```

## File Organization

```
langcalc/
├── infinigram.py           # Main Infinigram class
├── data/
│   ├── suffix_array.py     # Core suffix array (existing)
│   └── incremental.py      # Incremental updates
└── models/
    ├── base.py             # LanguageModel interface
    └── ngram.py            # Keep for fixed-order n-grams
```

## Migration Path

1. **Keep NGramModel** for fixed-order use cases (faster for small n)
2. **Introduce Infinigram** for variable-length needs
3. **Unified API** - both implement LanguageModel
4. **Easy switching**:
   ```python
   # Fixed order (faster for n=3)
   model = NGramModel(corpus, n=3)

   # Variable length (more flexible)
   model = Infinigram(corpus, max_length=10)
   ```

## References

- InfiniGram paper: "Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens"
- Suffix arrays: Manber & Myers (1990)
- Our implementation: Lightweight grounding with algebraic composition
