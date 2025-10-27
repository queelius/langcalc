# Academic Paper

Research paper on LangCalc.

## Title

**LangCalc: A Calculus for Compositional Language Modeling with Infinigram Grounding**

## Abstract

This paper introduces LangCalc, an algebraic framework for compositional language modeling.
The key innovation is lightweight grounding: combining LLMs with suffix array-based pattern
matching using just 5% weight to achieve 70% perplexity reduction.

## PDF

The full paper is available at: `/home/spinoza/github/beta/langcalc/papers/paper.pdf`

## Key Results

- 70% perplexity reduction with 5% grounding weight
- 34× memory efficiency over n-gram hash tables
- O(m log n) query time with suffix arrays
- Minimal overhead (6.5%, 2.66ms) with real LLMs

## Citation

See [How to Cite](citation.md) for BibTeX.
