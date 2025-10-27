# TDD Cheat Sheet for LangCalc

Quick reference guide for writing excellent tests in the LangCalc project.

---

## The Golden Rule

**Test behavior, not implementation.**

```python
# ❌ BAD: Tests implementation
assert model._internal_state == expected_state
assert len(model.cache._dict) == 0

# ✅ GOOD: Tests behavior
assert model.predict(context) == expected_output
assert model.is_empty()
```

---

## Test-Driven Development Cycle

```
1. RED    → Write a failing test that defines desired behavior
2. GREEN  → Write minimal code to make test pass
3. REFACTOR → Improve code while keeping tests green
4. REPEAT → Small increments, frequent commits
```

---

## Test Structure: Given-When-Then

```python
def test_user_authentication_fails_after_three_attempts(self):
    """Test that user account locks after 3 failed login attempts."""
    # GIVEN: A user with valid credentials
    user = create_user("alice", "password123")

    # WHEN: User fails to login 3 times
    for _ in range(3):
        user.attempt_login("wrong_password")

    # THEN: Account is locked
    assert user.is_locked()
    assert not user.can_login("password123")
```

---

## What to Test vs What NOT to Test

### ✅ DO Test

**Public API Contracts**
```python
def test_logprobs_returns_valid_distribution(self):
    logprobs = model.logprobs(tokens, context)
    assert isinstance(logprobs, np.ndarray)
    assert len(logprobs) == len(tokens)
    assert all(lp <= 0 for lp in logprobs)  # Log probabilities ≤ 0
```

**Behavioral Properties**
```python
def test_mixture_weights_sum_to_one(self):
    """Verify mathematical constraint."""
    mixture = 0.3 * model1 + 0.7 * model2
    assert abs(sum(mixture.weights) - 1.0) < 1e-10
```

**Edge Cases and Boundaries**
```python
def test_empty_context(self):
    result = model.predict(context=[])
    assert result is not None

def test_very_long_context(self):
    huge_context = list(range(10000))
    result = model.predict(context=huge_context)
    assert result is not None
```

**Error Handling**
```python
def test_invalid_input_raises_clear_error(self):
    with pytest.raises(ValueError, match="smoothing must be positive"):
        model = InfinigramModel(corpus, smoothing=-1)
```

**Integration Points**
```python
def test_composes_with_other_models(self):
    """Test algebraic composition works."""
    combined = 0.5 * model1 + 0.5 * model2
    result = combined.predict(context)
    assert result is not None
```

### ❌ DON'T Test

**Private Methods**
```python
# ❌ BAD
def test_internal_cache_structure(self):
    model._update_cache()
    assert model._cache._internal_dict == {...}
```

**Framework/Library Code**
```python
# ❌ BAD: Testing NumPy
def test_numpy_array_addition(self):
    assert (np.array([1, 2]) + np.array([3, 4])).tolist() == [4, 6]
```

**Implementation Details**
```python
# ❌ BAD: Tests how it works, not what it does
def test_uses_binary_search(self):
    assert model._search_algorithm == "binary_search"
```

**Specific Values (unless they're requirements)**
```python
# ❌ BAD: Brittle, breaks on valid changes
def test_confidence_score(self):
    assert model.confidence(context) == 0.8537

# ✅ GOOD: Tests valid range
def test_confidence_in_valid_range(self):
    confidence = model.confidence(context)
    assert 0.0 <= confidence <= 1.0
```

---

## Test Naming Conventions

```python
# ✅ GOOD: Descriptive, tells you what's tested
def test_empty_corpus_raises_value_error(self)
def test_higher_smoothing_increases_unseen_token_probability(self)
def test_score_equals_sum_of_conditional_logprobs(self)

# ❌ BAD: Vague, doesn't tell you what's tested
def test_corpus(self)
def test_smoothing(self)
def test_score(self)
```

**Template**: `test_<what>_<when>_<expected_behavior>`

---

## Fixtures: Reduce Duplication

```python
# ❌ BAD: Duplication in every test
def test_something(self):
    corpus = [1, 2, 3, 4, 5]
    model = InfinigramModel(corpus, smoothing=0.1)
    # test code...

def test_something_else(self):
    corpus = [1, 2, 3, 4, 5]  # Same setup!
    model = InfinigramModel(corpus, smoothing=0.1)
    # test code...

# ✅ GOOD: Shared fixture
@pytest.fixture
def standard_model():
    corpus = [1, 2, 3, 4, 5]
    return InfinigramModel(corpus, smoothing=0.1)

def test_something(standard_model):
    # Use fixture
    result = standard_model.predict([1, 2])

def test_something_else(standard_model):
    # Reuse same fixture
    result = standard_model.score([3, 4])
```

---

## Parametrized Tests: Test Multiple Cases

```python
# ❌ BAD: Repetitive tests
def test_smoothing_01(self):
    model = InfinigramModel(corpus, smoothing=0.1)
    assert model.smoothing == 0.1

def test_smoothing_05(self):
    model = InfinigramModel(corpus, smoothing=0.5)
    assert model.smoothing == 0.5

# ✅ GOOD: Parametrized
@pytest.mark.parametrize("smoothing", [0.01, 0.1, 0.5, 1.0])
def test_smoothing_parameter(smoothing):
    model = InfinigramModel(corpus, smoothing=smoothing)
    assert model.smoothing == smoothing
```

---

## Assertions: Make Them Clear

```python
# ❌ BAD: Unhelpful failure message
assert result == expected

# ✅ GOOD: Clear failure message
assert result == expected, \
    f"Expected {expected} but got {result} for context {context}"

# ✅ EVEN BETTER: Use pytest helpers
from pytest import approx
assert value == approx(0.5, abs=1e-6)

# ✅ BEST: Use numpy testing for arrays
import numpy.testing as npt
npt.assert_array_almost_equal(result, expected, decimal=6)
```

---

## Testing Adapters (Wrapper Pattern)

When wrapping external libraries:

```python
class InfinigramModel(LanguageModel):
    """Adapter for external infinigram package."""

    def __init__(self, corpus, ...):
        self.infinigram = _Infinigram(corpus, ...)  # Wrapped object
```

**Focus tests on:**

1. **Interface Compliance**
   ```python
   def test_implements_language_model_interface(self):
       assert isinstance(model, LanguageModel)
       assert hasattr(model, 'logprobs')
       assert hasattr(model, 'sample')
       assert hasattr(model, 'score')
   ```

2. **Data Transformation**
   ```python
   def test_converts_text_to_bytes_correctly(self):
       text = "Hello 世界"
       corpus = list(text.encode('utf-8'))
       model = InfinigramModel(corpus)
       # Verify byte-level operations work
   ```

3. **Parameter Pass-Through**
   ```python
   def test_max_length_parameter_affects_behavior(self):
       model_short = InfinigramModel(corpus, max_length=3)
       model_long = InfinigramModel(corpus, max_length=10)
       # Verify different behavior with different params
   ```

4. **Error Handling**
   ```python
   def test_adapter_provides_clear_error_for_invalid_input(self):
       with pytest.raises(ValueError, match="corpus cannot be empty"):
           InfinigramModel([])
   ```

**Don't test the wrapped library's internals:**
```python
# ❌ BAD: Testing infinigram's implementation
def test_suffix_array_is_sorted(self):
    assert is_sorted(model.infinigram._sa)

# ❌ BAD: Accessing internal state
def test_corpus_stored_correctly(self):
    assert model.infinigram.corpus == [1, 2, 3]
```

---

## Common Anti-Patterns to Avoid

### 1. Testing Implementation Details

```python
# ❌ BAD
def test_uses_cache(self):
    model.predict(context)
    assert model._cache_hits == 1

# ✅ GOOD
def test_repeated_queries_are_fast(self):
    import time
    start = time.time()
    for _ in range(100):
        model.predict(context)
    duration = time.time() - start
    assert duration < 0.1  # Should be fast if cached
```

### 2. Non-Deterministic Tests

```python
# ❌ BAD: Flaky due to randomness
def test_temperature_affects_sampling(self):
    samples1 = model.sample(temperature=0.1)
    samples2 = model.sample(temperature=1.0)
    assert samples1 != samples2  # May fail randomly!

# ✅ GOOD: Test statistical property
def test_temperature_affects_diversity(self):
    np.random.seed(42)
    samples_low = [model.sample(temperature=0.1) for _ in range(100)]
    unique_low = len(set(tuple(s) for s in samples_low))

    np.random.seed(42)
    samples_high = [model.sample(temperature=10.0) for _ in range(100)]
    unique_high = len(set(tuple(s) for s in samples_high))

    assert unique_high > unique_low  # Statistical property
```

### 3. Testing Multiple Things

```python
# ❌ BAD: Tests multiple behaviors
def test_model(self):
    assert model.logprobs([1], []) is not None
    assert model.sample() is not None
    assert model.score([1, 2]) < 0
    # If this fails, which part broke?

# ✅ GOOD: One behavior per test
def test_logprobs_returns_array(self):
    result = model.logprobs([1], [])
    assert isinstance(result, np.ndarray)

def test_sample_returns_list(self):
    result = model.sample()
    assert isinstance(result, list)

def test_score_returns_negative_value(self):
    result = model.score([1, 2])
    assert result <= 0
```

### 4. Test Interdependence

```python
# ❌ BAD: Tests depend on execution order
class TestModel:
    def test_a_initialize(self):
        self.model = Model()  # Sets instance variable

    def test_b_predict(self):
        result = self.model.predict()  # Depends on test_a!

# ✅ GOOD: Independent tests
class TestModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = Model()  # Each test gets fresh model

    def test_predict(self):
        result = self.model.predict()
```

---

## Test Organization

```python
tests/
├── test_unit/                 # Fast, isolated tests
│   ├── test_model_algebra.py
│   ├── test_ngram_model.py
│   └── test_infinigram.py
│       ├── TestInfinigramInterface      # Interface compliance
│       ├── TestInfinigramParameters     # Parameter handling
│       ├── TestInfinigramComposition    # Algebraic operations
│       ├── TestInfinigramEdgeCases      # Boundary conditions
│       └── TestInfinigramBehavior       # Mathematical properties
│
├── test_integration/          # Tests combining components
│   ├── test_model_composition.py
│   └── test_lightweight_grounding.py
│
└── test_e2e/                  # End-to-end workflows (slow)
    └── test_complete_pipeline.py
```

---

## Testing Checklist

Before committing tests, verify:

- [ ] Tests focus on behavior, not implementation
- [ ] Each test verifies ONE thing
- [ ] Test names clearly describe what's tested
- [ ] Tests are independent (can run in any order)
- [ ] Tests are deterministic (same result every time)
- [ ] Edge cases are covered (empty, null, extreme values)
- [ ] Error cases are tested
- [ ] Assertions have clear failure messages
- [ ] Tests run fast (< 2s for unit tests)
- [ ] Coverage is adequate (>80% for critical code)

---

## Quick Reference: pytest Commands

```bash
# Run all tests
pytest tests/

# Run specific file
pytest tests/test_unit/test_infinigram.py

# Run specific test
pytest tests/test_unit/test_infinigram.py::test_logprobs_interface

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only failed tests from last run
pytest --lf

# Run tests matching pattern
pytest -k "test_edge"

# Stop on first failure
pytest -x

# Verbose output
pytest -v

# Show print statements
pytest -s

# Run in parallel (requires pytest-xdist)
pytest -n auto
```

---

## Common pytest Patterns

### Expecting Exceptions

```python
# Test that exception is raised
with pytest.raises(ValueError):
    model.predict(invalid_input)

# Test exception message
with pytest.raises(ValueError, match="must be positive"):
    Model(smoothing=-1)
```

### Temporary Files

```python
def test_with_temp_file(tmp_path):
    # tmp_path is a pytest fixture providing temporary directory
    file_path = tmp_path / "data.txt"
    file_path.write_text("test data")
    # File is automatically cleaned up
```

### Marks (Test Categories)

```python
@pytest.mark.slow
def test_large_corpus(self):
    # Slow test, skip in quick runs
    pass

@pytest.mark.integration
def test_with_ollama(self):
    # Integration test
    pass

# Run only fast tests
# pytest -m "not slow"
```

### Skipping Tests

```python
@pytest.mark.skip(reason="Feature not implemented yet")
def test_future_feature(self):
    pass

@pytest.mark.skipif(sys.version_info < (3, 8),
                    reason="Requires Python 3.8+")
def test_modern_feature(self):
    pass
```

---

## Testing Language Models: Specific Tips

### 1. Test Probability Distributions

```python
def test_probabilities_sum_to_one(self):
    logprobs = model.logprobs(list(range(256)), context=[1, 2])
    probs = np.exp(logprobs)
    assert abs(np.sum(probs) - 1.0) < 0.01
```

### 2. Test Consistency

```python
def test_deterministic_predictions(self):
    """Non-sampling methods should be deterministic."""
    logprobs1 = model.logprobs([1, 2, 3], context=[])
    logprobs2 = model.logprobs([1, 2, 3], context=[])
    np.testing.assert_array_equal(logprobs1, logprobs2)
```

### 3. Test Algebraic Properties

```python
def test_addition_is_commutative(self):
    """a + b = b + a"""
    m1 = model1 + model2
    m2 = model2 + model1
    # Both should give similar results
    logprobs1 = m1.logprobs(tokens, context)
    logprobs2 = m2.logprobs(tokens, context)
    np.testing.assert_array_almost_equal(logprobs1, logprobs2)
```

### 4. Test Numerical Stability

```python
def test_no_overflow_on_long_sequences(self):
    """Very negative logprobs shouldn't overflow."""
    long_sequence = list(range(1000))
    score = model.score(long_sequence)
    assert np.isfinite(score)
    assert score <= 0  # Log probability
```

---

## When Tests Fail

### Good Failure Messages

```python
# ❌ BAD
AssertionError

# ✅ GOOD
AssertionError: Expected confidence in [0, 1], got -0.5 for context [1, 2, 3]

# Use clear assertions:
assert 0 <= confidence <= 1, \
    f"Expected confidence in [0, 1], got {confidence} for context {context}"
```

### Debugging Failed Tests

```bash
# Run with maximum verbosity
pytest tests/test_infinigram.py::test_fails -vv

# Show local variables on failure
pytest tests/test_infinigram.py::test_fails -l

# Drop into debugger on failure
pytest tests/test_infinigram.py::test_fails --pdb

# Show print statements
pytest tests/test_infinigram.py::test_fails -s
```

---

## Remember

1. **Tests are specifications** - They define what the code should do
2. **Tests enable refactoring** - Good tests let you change internals safely
3. **Tests are documentation** - New developers learn from tests
4. **Fast tests = fast development** - Slow tests won't be run
5. **Green tests aren't enough** - They must test the right things

---

**"Any fool can write code that a computer can understand. Good programmers write code that humans can understand."** - Martin Fowler

**"The act of writing a unit test is more important than the test itself."** - Kent Beck

---

Generated: 2025-10-22
For: LangCalc Project
Author: Claude (TDD Expert)
