# Quick Start Guide

Get started with LangCalc in 5 minutes! This guide walks you through creating your first language model using LangCalc's algebraic framework.

## Your First Infinigram Model

Let's create a simple infinigram model and make predictions:

```python
from langcalc import Infinigram

# Create a simple corpus (byte-level tokens)
corpus = [1, 2, 3, 4, 2, 3, 5, 6, 2, 3, 4]

# Create an infinigram model (variable-length n-grams)
model = Infinigram(corpus, max_length=10)

# Make a prediction
context = [2, 3]
probs = model.predict(context, top_k=10)

print(f"Top predictions after [2, 3]: {probs}")
```

**What's happening?**

- `Infinigram` creates a model using suffix arrays for efficient pattern matching
- `max_length=10` means it considers patterns up to 10 tokens long
- `predict()` returns probability distribution over next tokens
- The model finds that after `[2, 3]`, tokens `4` and `5` are likely (they appear in the corpus)

## Composing Models with Algebra

LangCalc's power comes from algebraic composition:

```python
from langcalc import Infinigram, NGramModel

# Create two models
corpus = [1, 2, 3, 4, 2, 3, 5, 6, 2, 3, 4]
infini = Infinigram(corpus, max_length=10)
ngram = NGramModel(corpus, n=3)

# Compose them with weights
model = 0.7 * infini + 0.3 * ngram

# Make predictions
context = [2, 3]
probs = model.predict(context)
```

**What's happening?**

- We create two different models (infinigram and 3-gram)
- Use `*` for weighted mixture: `0.7 * infini` means 70% weight
- Use `+` for ensemble: `0.7 * infini + 0.3 * ngram` combines them
- The result is a new model that leverages both approaches

## Working with Text

Convert text to byte-level tokens:

```python
from langcalc import Infinigram

# Convert text to bytes
text = "the cat sat on the mat"
corpus = list(text.encode('utf-8'))

# Create model
model = Infinigram(corpus, max_length=20)

# Query with context
context_text = "the cat"
context = list(context_text.encode('utf-8'))

# Predict next tokens
probs = model.predict(context, top_k=256)  # All possible bytes

# Convert predictions back to characters
for token_id, prob in probs.items():
    if prob > 0.1:  # Only show high-probability predictions
        char = chr(token_id) if 32 <= token_id < 127 else f"\\x{token_id:02x}"
        print(f"  '{char}': {prob:.3f}")
```

## Using Projections

Transform context before matching:

```python
from langcalc.projections import LowercaseProjection, WhitespaceProjection
from langcalc.models.projected import ProjectedModel

# Create corpus
corpus = list("Hello World! HELLO WORLD!".encode('utf-8'))
base_model = Infinigram(corpus, max_length=20)

# Create projection pipeline (normalize before matching)
projection = WhitespaceProjection() >> LowercaseProjection()

# Apply projection to model
model = ProjectedModel(base_model, projection, corpus)

# Now queries are case-insensitive and whitespace-normalized
context = list("hello  world".encode('utf-8'))
probs = model.logprobs(list(range(256)), context)
```

**What's happening?**

- `WhitespaceProjection()` normalizes whitespace
- `LowercaseProjection()` converts to lowercase
- `>>` chains them left-to-right
- `ProjectedModel` applies the pipeline before each query

## Using Augmentations

Alternatively, augment the corpus once at training time:

```python
from langcalc.augmentations import LowercaseAugmentation
from langcalc import Infinigram

# Create corpus
corpus = list("Hello World".encode('utf-8'))

# Augment corpus (add lowercase variant)
augmentation = LowercaseAugmentation()
augmented_corpus = augmentation.augment(corpus)

# Create model with augmented corpus
model = Infinigram(augmented_corpus, max_length=20)

# Now case-insensitive matching works automatically
context = list("HELLO".encode('utf-8'))
probs = model.predict(context, top_k=256)
```

**Projection vs Augmentation:**

- **Projection**: Transform query at prediction time (flexible, uses less memory)
- **Augmentation**: Transform corpus at training time (faster queries, uses more memory)

See [Core Concepts](concepts.md) for detailed comparison.

## Advanced Example: Lightweight Grounding

Combine LLM with infinigram for factual grounding:

```python
from langcalc import Infinigram
from langcalc.models import OllamaModel

# Create knowledge base (e.g., Wikipedia)
with open('wikipedia.txt', 'rb') as f:
    wiki_corpus = list(f.read())

wiki = Infinigram(wiki_corpus, max_length=20)

# Create LLM
llm = OllamaModel(model_name='llama2')

# Optimal mixture: 95% LLM + 5% infinigram
# (Based on research showing 70% perplexity reduction)
grounded_model = 0.95 * llm + 0.05 * wiki

# Make predictions
context = list("The capital of France is".encode('utf-8'))
probs = grounded_model.predict(context, top_k=50)
```

**Why this works:**

- LLM provides fluent generation
- Infinigram provides factual grounding
- Only 5% weight needed for 70% perplexity reduction
- Infinigram adds only 0.03ms latency

## More Algebraic Operations

LangCalc supports many operators:

```python
# Set operations
best_model = llm | wiki  # max(p_llm, p_wiki)
conservative = llm & wiki  # min(p_llm, p_wiki)
diff = llm ^ wiki  # symmetric difference

# Temperature scaling
creative = model ** 1.5  # Higher temperature
focused = model ** 0.5   # Lower temperature

# Negation (complement)
anti_model = ~model  # 1 - p(x)

# Subtraction (experimental)
residual = llm - ngram  # What LLM learned beyond n-grams
```

## Complete Example

Putting it all together:

```python
from langcalc import Infinigram, NGramModel
from langcalc.models import OllamaModel
from langcalc.projections import (
    EditDistanceProjection,
    LowercaseProjection,
    WhitespaceProjection,
    RecencyProjection
)
from langcalc.models.projected import ProjectedModel

# 1. Load corpus
corpus = list(open('corpus.txt', 'rb').read())

# 2. Create models
wiki = Infinigram(corpus, max_length=20)
ngram = NGramModel(corpus, n=5)
llm = OllamaModel(model_name='llama2')

# 3. Create projection pipeline
projection = (
    EditDistanceProjection(max_distance=1) >>  # Fix typos
    WhitespaceProjection() >>                   # Normalize whitespace
    LowercaseProjection() >>                    # Case-insensitive
    RecencyProjection(max_length=100)           # Recent tokens
)

# 4. Apply projection to wiki
projected_wiki = ProjectedModel(wiki, projection, corpus)

# 5. Compose final model
model = (
    0.85 * llm +                    # 85% LLM
    0.10 * projected_wiki +         # 10% projected wiki
    0.05 * ngram                    # 5% n-gram smoothing
) ** 0.9  # Lower temperature slightly

# 6. Make predictions
context = list("The quick brown fox".encode('utf-8'))
predictions = model.predict(context, top_k=20)

# 7. Sample text
samples = model.sample(context, temperature=1.0, max_tokens=50)
generated_text = bytes(samples).decode('utf-8', errors='ignore')
print(f"Generated: {generated_text}")
```

## Interactive Exploration

Try the Jupyter notebooks for interactive experimentation:

```bash
# Start Jupyter
jupyter notebook

# Open notebooks
# 1. notebooks/explore_algebra.ipynb (45 min - foundations)
# 2. notebooks/lightweight_grounding_demo.ipynb (60 min - practical)
# 3. notebooks/unified_algebra.ipynb (60 min - advanced)
```

## Next Steps

Now that you've created your first models, learn more about:

1. **[Core Concepts](concepts.md)** - Understand projections, augmentations, and algebra
2. **[User Guide](../user-guide/index.md)** - Explore practical patterns and best practices
3. **[Projection System](../projection-system/index.md)** - Deep dive into mathematical formalism
4. **[Examples](../user-guide/examples.md)** - More complete examples and use cases

## Common Patterns

### Pattern 1: Case-Insensitive Matching

```python
from langcalc.augmentations import LowercaseAugmentation

augmented_corpus = LowercaseAugmentation().augment(corpus)
model = Infinigram(augmented_corpus)
```

### Pattern 2: Robust Text Matching

```python
from langcalc.augmentations import StandardAugmentation

# Case + whitespace + Unicode normalization
augmented = StandardAugmentation().augment(corpus)
model = Infinigram(augmented)
```

### Pattern 3: Error Correction

```python
from langcalc.projections import EditDistanceProjection

projection = EditDistanceProjection(max_distance=2)
model = ProjectedModel(base_model, projection, corpus)
```

### Pattern 4: Recent Context Focus

```python
from langcalc.projections import RecencyProjection

projection = RecencyProjection(max_length=50)
model = ProjectedModel(base_model, projection, corpus)
```

## Troubleshooting

### Model predictions are all zero

**Solution**: Ensure context exists in corpus or use projections to normalize:

```python
# Add lowercase augmentation for case-insensitive matching
from langcalc.augmentations import LowercaseAugmentation
augmented = LowercaseAugmentation().augment(corpus)
model = Infinigram(augmented)
```

### Out of memory with large corpus

**Solution**: Use smaller `max_length` or consider streaming approaches:

```python
# Reduce max pattern length
model = Infinigram(corpus, max_length=10)  # Instead of 20
```

### Predictions are too conservative

**Solution**: Increase temperature or use different mixing weights:

```python
# Higher temperature = more diversity
creative_model = model ** 1.5

# Or reduce n-gram weight
model = 0.8 * llm + 0.2 * ngram  # Instead of 0.5/0.5
```

## Getting Help

- **Examples**: Check `/home/spinoza/github/beta/langcalc/examples/` for more examples
- **Tests**: See `/home/spinoza/github/beta/langcalc/tests/` for reference implementations
- **Discussions**: [GitHub Discussions](https://github.com/queelius/langcalc/discussions)
- **Issues**: [GitHub Issues](https://github.com/queelius/langcalc/issues)
