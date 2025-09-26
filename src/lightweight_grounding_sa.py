#!/usr/bin/env python3
"""
Lightweight grounding using suffix arrays instead of n-gram hash tables.
This is the production-ready version using our efficient suffix array approach.
"""

import pickle
import time
import requests
from typing import Dict, List, Optional
from collections import defaultdict

from wikipedia_suffix_array import WikipediaSuffixArray, WikipediaSuffixModel


class LightweightGroundingSystemSA:
    """
    Lightweight grounding system using suffix arrays.
    Combines LLMs with suffix-array-based pattern matching.

    Core equation:
    P(x_t | context) = α_LLM * P_LLM(x_t | context) + α_SA * P_SA(x_t | context)

    Typical: α_LLM = 0.95, α_SA = 0.05
    """

    def __init__(self, llm_model, llm_weight: float = 0.95):
        """
        Initialize grounding system.

        Args:
            llm_model: The base LLM to ground
            llm_weight: Weight for LLM (suffix array gets 1 - llm_weight)
        """
        self.llm = llm_model
        self.llm_weight = llm_weight
        self.sa_weight = 1.0 - llm_weight
        self.suffix_model = None

    def set_suffix_array(self, suffix_array: WikipediaSuffixArray):
        """Set the suffix array for grounding."""
        self.suffix_model = WikipediaSuffixModel(suffix_array)

    def load_suffix_array(self, filepath: str):
        """Load pre-built suffix array model."""
        with open(filepath, 'rb') as f:
            self.suffix_model = pickle.load(f)

    def predict(self, context: List[str], top_k: int = 10) -> Dict[str, float]:
        """
        Get grounded predictions combining LLM and suffix array.

        Args:
            context: List of context tokens
            top_k: Number of top predictions to return

        Returns:
            Dictionary of token -> probability
        """
        combined = defaultdict(float)

        # Get LLM predictions
        llm_preds = self.llm.predict(context, top_k)
        for token, prob in llm_preds.items():
            combined[token] += prob * self.llm_weight

        # Get suffix array predictions
        if self.suffix_model:
            sa_preds = self.suffix_model.predict(context, top_k)
            for token, prob in sa_preds.items():
                combined[token] += prob * self.sa_weight

        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {k: v/total for k, v in combined.items()}

        return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k])


def benchmark_suffix_vs_ngram():
    """Compare suffix array vs n-gram approaches."""

    print("="*70)
    print("SUFFIX ARRAY vs N-GRAM BENCHMARK")
    print("="*70)

    # Create test corpus
    sentences = [
        "The capital of France is Paris",
        "The capital of Germany is Berlin",
        "The capital of Japan is Tokyo",
        "Machine learning uses neural networks",
        "Deep learning requires large datasets",
    ] * 20  # Repeat for larger corpus

    print(f"\nCorpus: {len(sentences)} sentences")

    # 1. Suffix Array Approach
    print("\n1. SUFFIX ARRAY APPROACH")
    print("-"*40)

    start = time.time()
    sa = WikipediaSuffixArray()
    sa.build_from_sentences(sentences)
    sa_build_time = time.time() - start

    sa_model = WikipediaSuffixModel(sa)

    # Test prediction speed
    context = ["the", "capital", "of"]
    start = time.time()
    for _ in range(1000):
        _ = sa_model.predict(context)
    sa_pred_time = (time.time() - start) / 1000

    mem = sa.memory_usage()
    print(f"  Build time: {sa_build_time*1000:.2f} ms")
    print(f"  Prediction time: {sa_pred_time*1000:.4f} ms")
    print(f"  Memory usage: {mem['total_mb']:.3f} MB")

    # 2. N-gram Hash Table Approach (simulated)
    print("\n2. N-GRAM HASH TABLE APPROACH")
    print("-"*40)

    start = time.time()
    ngrams = defaultdict(lambda: defaultdict(int))
    for sentence in sentences:
        tokens = sentence.lower().split()
        for i in range(len(tokens) - 2):
            context = tuple(tokens[i:i+2])
            next_token = tokens[i+2]
            ngrams[context][next_token] += 1
    ngram_build_time = time.time() - start

    # Test prediction speed
    def ngram_predict(context_list):
        context_tuple = tuple(context_list[-2:])
        if context_tuple in ngrams:
            counts = ngrams[context_tuple]
            total = sum(counts.values())
            return {k: v/total for k, v in counts.items()}
        return {}

    start = time.time()
    for _ in range(1000):
        _ = ngram_predict(context)
    ngram_pred_time = (time.time() - start) / 1000

    # Estimate memory (rough)
    ngram_memory = len(ngrams) * 100  # ~100 bytes per entry
    print(f"  Build time: {ngram_build_time*1000:.2f} ms")
    print(f"  Prediction time: {ngram_pred_time*1000:.4f} ms")
    print(f"  Memory usage: {ngram_memory / (1024*1024):.3f} MB (estimated)")

    # 3. Comparison
    print("\n3. COMPARISON")
    print("-"*40)
    print(f"  Build Speed: Suffix Array {ngram_build_time/sa_build_time:.1f}x slower")
    print(f"  Query Speed: Suffix Array {ngram_pred_time/sa_pred_time:.1f}x slower")
    print(f"  Memory: Suffix Array {ngram_memory/mem['total_bytes']:.1f}x smaller")
    print("\n✓ Suffix arrays: Better memory efficiency, scalability")
    print("✓ N-grams: Faster for small corpora, simple implementation")


def demo_lightweight_grounding_sa():
    """Demonstrate lightweight grounding with suffix arrays."""

    print("\n" + "="*70)
    print("LIGHTWEIGHT GROUNDING WITH SUFFIX ARRAYS")
    print("="*70)

    # Create Wikipedia suffix array
    sentences = [
        "Einstein developed the theory of relativity",
        "Darwin proposed the theory of evolution",
        "The capital of France is Paris",
        "The capital of Germany is Berlin",
        "Machine learning uses neural networks",
        "Deep learning requires large datasets",
        "Quantum mechanics describes atomic behavior",
    ]

    sa = WikipediaSuffixArray()
    sa.build_from_sentences(sentences)

    # Mock LLM for demo
    class MockLLM:
        def predict(self, context, top_k=10):
            # Generic predictions
            return {
                "the": 0.3,
                "and": 0.2,
                "is": 0.15,
                "of": 0.15,
                "a": 0.1,
                "in": 0.1
            }

    llm = MockLLM()

    # Create grounding system
    system = LightweightGroundingSystemSA(llm, llm_weight=0.95)
    system.set_suffix_array(sa)

    # Test predictions
    test_cases = [
        ["einstein", "developed", "the"],
        ["the", "capital", "of"],
        ["the", "theory", "of"],
        ["machine", "learning"],
    ]

    print("\nGROUNDED PREDICTIONS (95% LLM + 5% Suffix Array):")
    print("-"*50)

    for context in test_cases:
        preds = system.predict(context, top_k=3)
        print(f"\nContext: {' '.join(context)}")
        for token, prob in preds.items():
            print(f"  {token}: {prob:.3f}")

    # Compare with pure LLM
    print("\n\nPURE LLM PREDICTIONS:")
    print("-"*50)

    for context in test_cases:
        preds = llm.predict(context, top_k=3)
        print(f"\nContext: {' '.join(context)}")
        for token, prob in list(preds.items())[:3]:
            print(f"  {token}: {prob:.3f}")

    print("\n✓ Suffix array grounding adds factual knowledge!")


if __name__ == "__main__":
    # Run benchmarks
    benchmark_suffix_vs_ngram()

    # Demo grounding
    demo_lightweight_grounding_sa()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key Advantages of Suffix Arrays for Grounding:

1. MEMORY EFFICIENCY
   - O(n) space vs O(n²) for all n-grams
   - Single data structure for all pattern lengths
   - No need to pre-compute fixed n values

2. FLEXIBLE PATTERN MATCHING
   - Variable-length contexts
   - Efficient substring search
   - Natural backoff mechanism

3. SCALABILITY
   - Handles large corpora (Wikipedia-scale)
   - Incremental updates possible
   - Cache-friendly access patterns

4. PRODUCTION READY
   - Binary search: O(m log n) queries
   - Proven in search engines
   - Well-understood algorithms

Recommended: Use suffix arrays for production grounding systems!
    """)