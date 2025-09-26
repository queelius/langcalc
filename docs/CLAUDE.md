# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository explores autoregressive models using inductive biases and projections to enhance out-of-distribution (OOD) generalization. It implements infini-gram models that leverage suffix arrays for efficient arbitrary input length management.

## Key Dependencies

- **infini_gram**: Core engine for infini-gram model operations
- **transformers**: HuggingFace library for tokenizers and language models (Llama-2, GPT-2)
- **torch**: PyTorch for neural network operations

## Common Development Commands

### Running Experiments
```bash
# Test infini-gram functionality
python code/test_infini.py

# Run main projection search algorithm
python code/main.py

# Test GPT-2 integration
python code/test_gpt2.py

# Test Llama integration
python code/test_llama.py
```

### Working with the Infini-gram Model
The project uses a pre-built infini-gram index located at `index/v4_pileval_llama`. Ensure this directory exists or update the path in `code/infini.py`.

## Architecture & Key Concepts

### Core Algorithm: Projection Functions
The main algorithmic contribution is treating OOD generalization as a projection problem. Key implementations:

- **BFS Edit Distance Search** (`code/main.py`): Implements breadth-first search to find minimal edit transformations between strings, supporting the shortest edit distance projection bias
- **Neighbor Generation**: Creates variations through character insertions and deletions to explore the projection space

### Inductive Biases Explored
1. **Recency Bias**: Longest suffix matching (default infini-gram behavior)
2. **Similarity Bias**: Shortest edit distance transformations
3. **Semantic Similarity**: Planned extension using embeddings (see TODO.md)

### Research Focus Areas
- Suffix extension strategies for better context matching
- Integration of classical IR techniques (BM25, query expansion)
- Meta-learning approaches for optimizing projection functions
- Bootstrapping methods for uncertainty estimation

## Important Implementation Notes

- The project is in experimental phase - many concepts from the paper are not yet implemented
- Current code focuses on demonstrating edit distance search and basic infini-gram operations
- GPU support is tested but not required (see `cuda-test.py`, `cuda.py`)