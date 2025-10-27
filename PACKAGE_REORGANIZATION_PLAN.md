# Package Reorganization Plan

**Date**: October 7, 2025
**Status**: Phase 1 Complete (Infinigram implemented)
**Remaining**: Package restructuring

## Summary of Progress

### ✅ Phase 1: Infinigram Implementation (COMPLETE)

1. **Design Document** (`INFINIGRAM_DESIGN.md`)
   - Complete API specification
   - Mathematical foundation
   - Performance targets
   - Testing strategy

2. **Implementation** (`langcalc/infinigram.py`)
   - Variable-length n-gram model
   - Suffix array-based pattern matching
   - Continuation probability computation
   - Confidence scoring
   - Dynamic updates
   - 381 lines of production code

3. **Tests** (`tests/test_unit/test_infinigram.py`)
   - 36 comprehensive tests (all passing) ✅
   - 100% test success rate
   - Coverage of:
     - Suffix array operations
     - Longest suffix matching
     - Continuation computation
     - Probability prediction
     - Confidence scoring
     - Dynamic updates
     - Edge cases
     - Integration scenarios

### 🚧 Phase 2: Package Reorganization (IN PROGRESS)

Current file structure issues:
```
src/
├── suffix_array_demo.py          # Should be in examples/
├── suffix_tree_demo.py           # Should be in examples/
├── algebra_integration.py        # Should be in examples/
├── wikipedia_*.py                # Should be in scripts/
├── lightweight_grounding.py      # Should be in langcalc/
├── lightweight_grounding_sa.py   # Duplicate, should merge
├── model_algebra.py              # Should be in langcalc/
├── incremental_suffix_array.py   # Should be in langcalc/data/
└── ngram_projections/            # Already properly organized ✓
```

## Proposed Clean Structure

```
langcalc/                    # Main package (NEW)
├── __init__.py             # Public API exports
├── infinigram.py           # ✅ DONE - Infinigram model
├── algebra.py              # From model_algebra.py
├── grounding.py            # Merge lightweight_grounding*.py
├── models/
│   ├── __init__.py
│   ├── base.py            # From ngram_projections/models/base.py
│   ├── ngram.py           # From ngram_projections/models/ngram.py
│   ├── llm.py             # From ngram_projections/models/llm.py
│   └── mixture.py         # From ngram_projections/models/mixture.py
├── projections/
│   ├── __init__.py
│   ├── base.py            # From ngram_projections/projections/base.py
│   ├── recency.py
│   ├── edit_distance.py
│   └── semantic.py
├── data/
│   ├── __init__.py
│   ├── suffix_array.py    # From ngram_projections/data/suffix_array.py
│   └── incremental.py     # From incremental_suffix_array.py
└── utils/
    └── __init__.py

examples/                   # Demo and example code
├── suffix_array_demo.py
├── suffix_tree_demo.py
├── algebra_demo.py        # From algebra_integration.py
├── algebra_examples.py    # Already there
├── comprehensive_experiments.py
├── lightweight_experiments.py
└── ollama_integration_example.py

scripts/                    # Utility scripts
├── wikipedia_sample.py
├── wikipedia_suffix_array.py
├── download_wikipedia.py
└── run_experiments.py

src/                        # TO BE DEPRECATED
└── (old files remain temporarily for compatibility)
```

## Migration Strategy

### Step 1: Create langcalc Package Structure
- [x] Create `langcalc/` directory
- [x] Create subdirectories: models/, projections/, data/, utils/
- [x] Implement `langcalc/infinigram.py`
- [ ] Create all `__init__.py` files with proper exports

### Step 2: Copy and Refactor Core Modules
- [ ] Copy `src/model_algebra.py` → `langcalc/algebra.py`
- [ ] Merge `src/lightweight_grounding*.py` → `langcalc/grounding.py`
- [ ] Copy `src/ngram_projections/models/*` → `langcalc/models/`
- [ ] Copy `src/ngram_projections/projections/*` → `langcalc/projections/`
- [ ] Copy `src/ngram_projections/data/*` → `langcalc/data/`
- [ ] Move `src/incremental_suffix_array.py` → `langcalc/data/incremental.py`

### Step 3: Move Demo and Utility Files
- [ ] Move demo files: `src/*_demo.py` → `examples/`
- [ ] Move `src/algebra_integration.py` → `examples/algebra_demo.py`
- [ ] Move `src/wikipedia_*.py` → `scripts/`
- [ ] Ensure all moved files update their imports

### Step 4: Update Configuration
- [ ] Update `setup.py`:
  - Change `package_dir={"": "src"}` to `packages=find_packages()`
  - Update package list to include `langcalc`
- [ ] Update `MANIFEST.in` if exists
- [ ] Update `.gitignore` if needed

### Step 5: Update All Imports
Files that need import updates:
- [ ] All test files (`tests/**/*.py`)
- [ ] All example files (`examples/*.py`)
- [ ] All notebook files (`notebooks/*.ipynb`)
- [ ] Documentation files with code examples

### Step 6: Testing and Validation
- [ ] Run full test suite: `pytest tests/`
- [ ] Fix any import errors
- [ ] Verify all examples still run
- [ ] Check notebooks still work
- [ ] Run coverage report

### Step 7: Documentation Updates
- [ ] Update `README.md` with new import paths
- [ ] Update `docs/README_PACKAGE.md`
- [ ] Update `CLAUDE.md`
- [ ] Update API documentation
- [ ] Update Jupyter notebooks

### Step 8: Cleanup (Optional)
- [ ] Mark `src/` as deprecated
- [ ] Add warning imports in `src/` redirecting to `langcalc/`
- [ ] Eventually remove `src/` completely

## Public API Design

Main `langcalc/__init__.py`:
```python
"""
LangCalc: A Calculus for Language Models

Variable-length n-gram models, algebraic composition, and lightweight grounding.
"""

# Core models
from langcalc.infinigram import Infinigram, create_infinigram
from langcalc.algebra import (
    AlgebraicModel,
    SumModel,
    ScaledModel,
    TransformedModel,
    # ... other algebraic classes
)

# Model implementations
from langcalc.models.ngram import NGramModel
from langcalc.models.llm import HuggingFaceModel, MockLLM
from langcalc.models.mixture import MixtureModel

# Projections
from langcalc.projections.recency import RecencyProjection
from langcalc.projections.semantic import SemanticProjection
from langcalc.projections.edit_distance import EditDistanceProjection

# Data structures
from langcalc.data.suffix_array import SuffixArray

# Grounding
from langcalc.grounding import (
    MixtureModel as GroundingMixture,
    # ... grounding utilities
)

__version__ = "0.4.0"  # Bump for major refactor

__all__ = [
    # Core
    "Infinigram",
    "create_infinigram",
    "AlgebraicModel",
    # Models
    "NGramModel",
    "HuggingFaceModel",
    "MockLLM",
    "MixtureModel",
    # Projections
    "RecencyProjection",
    "SemanticProjection",
    "EditDistanceProjection",
    # Data
    "SuffixArray",
]
```

## Breaking Changes

Users will need to update imports:
```python
# OLD (src-based)
from ngram_projections.models.ngram import NGramModel
from ngram_projections.projections.recency import RecencyProjection
import sys; sys.path.append('src')
from model_algebra import AlgebraicModel

# NEW (langcalc package)
from langcalc.models.ngram import NGramModel
from langcalc.projections.recency import RecencyProjection
from langcalc.algebra import AlgebraicModel

# Or simpler:
from langcalc import NGramModel, RecencyProjection, AlgebraicModel, Infinigram
```

## Testing the Migration

After each step, run:
```bash
# Test imports work
python -c "from langcalc import Infinigram, NGramModel"

# Run test suite
pytest tests/ -v

# Test examples
python examples/algebra_examples.py

# Test notebooks
jupyter nbconvert --execute notebooks/explore_algebra.ipynb
```

## Estimated Effort

- Step 1 (Structure): ✅ Done
- Step 2 (Copy modules): 1-2 hours
- Step 3 (Move demos): 30 minutes
- Step 4 (Config): 30 minutes
- Step 5 (Update imports): 2-3 hours (many files)
- Step 6 (Testing): 1-2 hours
- Step 7 (Docs): 1-2 hours

**Total**: 6-10 hours of focused work

## Benefits of New Structure

1. **Clean package**: `pip install langcalc` works properly
2. **Better imports**: `from langcalc import Infinigram`
3. **Separation of concerns**: code vs examples vs scripts
4. **Professional**: Follows Python packaging best practices
5. **Discoverable**: Clear public API in `__init__.py`
6. **Maintainable**: Organized file structure

## Next Immediate Steps

1. Create `langcalc/__init__.py` with public API
2. Copy core modules to new structure
3. Update setup.py
4. Fix imports in tests
5. Verify everything works

Would you like me to continue with the reorganization, or would you prefer to handle it manually with this plan as a guide?
