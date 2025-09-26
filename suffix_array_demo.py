#!/usr/bin/env python3
"""
Suffix Array Implementation and Demonstration
"""

from typing import List, Tuple, Optional
import time


class SuffixArray:
    """
    A suffix array implementation for efficient pattern matching.
    """

    def __init__(self, text: str):
        """
        Build suffix array for the given text.

        Args:
            text: Input string to build suffix array from
        """
        self.text = text
        self.n = len(text)
        self.sa = self._build_suffix_array()
        self.lcp = self._build_lcp_array()

    def _build_suffix_array(self) -> List[int]:
        """
        Build suffix array using simple sorting.
        For production, use DC3 or SA-IS algorithm for O(n) time.
        """
        # Create list of (suffix, original_index) pairs
        suffixes = []
        for i in range(self.n):
            suffixes.append((self.text[i:], i))

        # Sort suffixes lexicographically
        suffixes.sort(key=lambda x: x[0])

        # Extract the indices
        suffix_array = [idx for _, idx in suffixes]
        return suffix_array

    def _build_lcp_array(self) -> List[int]:
        """
        Build Longest Common Prefix array.
        LCP[i] = length of longest common prefix between sa[i] and sa[i-1]
        """
        lcp = [0] * self.n

        for i in range(1, self.n):
            # Compare suffix at sa[i] with suffix at sa[i-1]
            j = 0
            while (self.sa[i] + j < self.n and
                   self.sa[i-1] + j < self.n and
                   self.text[self.sa[i] + j] == self.text[self.sa[i-1] + j]):
                j += 1
            lcp[i] = j

        return lcp

    def find_pattern(self, pattern: str) -> List[int]:
        """
        Find all occurrences of pattern using binary search.

        Args:
            pattern: Pattern to search for

        Returns:
            List of starting positions where pattern occurs
        """
        m = len(pattern)

        # Binary search for leftmost position
        left = self._binary_search_left(pattern)
        if left == -1:
            return []

        # Binary search for rightmost position
        right = self._binary_search_right(pattern)

        # Extract all positions
        positions = []
        for i in range(left, right + 1):
            positions.append(self.sa[i])

        return sorted(positions)

    def _binary_search_left(self, pattern: str) -> int:
        """Find leftmost suffix that starts with pattern."""
        left, right = 0, self.n - 1
        result = -1

        while left <= right:
            mid = (left + right) // 2
            suffix_start = self.sa[mid]
            suffix = self.text[suffix_start:suffix_start + len(pattern)]

            if suffix >= pattern:
                if suffix.startswith(pattern):
                    result = mid
                right = mid - 1
            else:
                left = mid + 1

        return result

    def _binary_search_right(self, pattern: str) -> int:
        """Find rightmost suffix that starts with pattern."""
        left, right = 0, self.n - 1
        result = -1

        while left <= right:
            mid = (left + right) // 2
            suffix_start = self.sa[mid]
            suffix = self.text[suffix_start:suffix_start + len(pattern)]

            if suffix > pattern:
                right = mid - 1
            else:
                if suffix.startswith(pattern):
                    result = mid
                left = mid + 1

        return result

    def count_occurrences(self, pattern: str) -> int:
        """Count how many times pattern appears."""
        positions = self.find_pattern(pattern)
        return len(positions)

    def get_all_ngrams(self, n: int) -> dict:
        """
        Extract all n-grams and their counts.

        Args:
            n: Length of n-grams to extract

        Returns:
            Dictionary mapping n-grams to their counts
        """
        ngrams = {}

        for i in range(self.n - n + 1):
            ngram = self.text[i:i + n]
            ngrams[ngram] = ngrams.get(ngram, 0) + 1

        return ngrams

    def print_suffix_array(self):
        """Print the suffix array for visualization."""
        print(f"Text: '{self.text}'")
        print(f"Length: {self.n}")
        print("\nSuffix Array:")
        print("Index | SA | LCP | Suffix")
        print("------|----|----|-------")

        for i in range(self.n):
            suffix = self.text[self.sa[i]:]
            # Truncate long suffixes for display
            if len(suffix) > 20:
                suffix = suffix[:20] + "..."
            print(f"{i:5} | {self.sa[i]:2} | {self.lcp[i]:3} | {suffix}")


class NGramSuffixModel:
    """
    N-gram model using suffix arrays for efficient storage and retrieval.
    """

    def __init__(self, n: int = 3):
        self.n = n
        self.corpus = ""
        self.suffix_array = None
        self.vocab = set()
        self.separator = " "  # Token separator

    def train(self, tokens: List[str]):
        """
        Train the model on a list of tokens.

        Args:
            tokens: List of tokens to train on
        """
        # Add tokens to vocabulary
        self.vocab.update(tokens)

        # Append to corpus with separator
        if self.corpus:
            self.corpus += self.separator
        self.corpus += self.separator.join(tokens)

        # Rebuild suffix array
        self._rebuild_suffix_array()

    def _rebuild_suffix_array(self):
        """Rebuild the suffix array after corpus update."""
        if self.corpus:
            self.suffix_array = SuffixArray(self.corpus)

    def predict(self, context: List[str]) -> dict:
        """
        Predict next token probabilities given context.

        Args:
            context: List of context tokens

        Returns:
            Dictionary mapping tokens to probabilities
        """
        if not self.suffix_array:
            return {}

        # Use last n-1 tokens as context
        context_str = self.separator.join(context[-(self.n-1):])

        # Find all occurrences of this context
        positions = self.suffix_array.find_pattern(context_str)

        if not positions:
            # Backoff to shorter context
            if len(context) > 1:
                return self.predict(context[-1:])
            else:
                # Uniform distribution over vocabulary
                uniform_prob = 1.0 / len(self.vocab)
                return {token: uniform_prob for token in self.vocab}

        # Count what follows this context
        next_tokens = {}
        for pos in positions:
            # Check what comes after the context
            next_pos = pos + len(context_str)
            if next_pos < len(self.corpus) - 1:
                # Skip separator
                if self.corpus[next_pos] == self.separator:
                    next_pos += 1

                # Find next token
                end_pos = self.corpus.find(self.separator, next_pos)
                if end_pos == -1:
                    end_pos = len(self.corpus)

                next_token = self.corpus[next_pos:end_pos]
                if next_token and next_token != self.separator:
                    next_tokens[next_token] = next_tokens.get(next_token, 0) + 1

        # Convert counts to probabilities
        total = sum(next_tokens.values())
        if total > 0:
            return {token: count/total for token, count in next_tokens.items()}
        else:
            # Uniform distribution if no following tokens found
            uniform_prob = 1.0 / len(self.vocab)
            return {token: uniform_prob for token in self.vocab}


def demonstrate_suffix_array():
    """Demonstrate suffix array functionality."""

    print("=" * 60)
    print("SUFFIX ARRAY DEMONSTRATION")
    print("=" * 60)

    # Example 1: Simple string
    print("\n1. Simple Example: 'banana'")
    print("-" * 40)
    sa = SuffixArray("banana")
    sa.print_suffix_array()

    # Pattern search
    print("\n2. Pattern Search in 'banana'")
    print("-" * 40)
    patterns = ["ana", "na", "ban", "a"]
    for pattern in patterns:
        positions = sa.find_pattern(pattern)
        count = len(positions)
        print(f"Pattern '{pattern}': Found {count} times at positions {positions}")

    # Example 2: Sentence for n-grams
    print("\n3. N-gram Example")
    print("-" * 40)
    text = "the dog ran the dog jumped the cat ran"
    sa2 = SuffixArray(text)

    print(f"Text: '{text}'")
    print("\nSearching for repeated n-grams:")

    ngrams = ["the dog", "dog", "ran", "the"]
    for ngram in ngrams:
        positions = sa2.find_pattern(ngram)
        count = len(positions)
        print(f"'{ngram}': Found {count} times at positions {positions}")

    # Example 3: N-gram model
    print("\n4. N-gram Model with Suffix Arrays")
    print("-" * 40)

    model = NGramSuffixModel(n=3)

    # Train on sample sentences
    sentences = [
        "the dog ran fast",
        "the dog jumped high",
        "the cat ran away",
        "the cat jumped down"
    ]

    print("Training on:")
    for sent in sentences:
        print(f"  - {sent}")
        model.train(sent.split())

    print(f"\nCorpus: '{model.corpus}'")
    print(f"Vocabulary size: {len(model.vocab)}")

    # Make predictions
    print("\nPredictions:")
    contexts = [
        ["the", "dog"],
        ["the", "cat"],
        ["ran"],
        ["jumped"]
    ]

    for context in contexts:
        predictions = model.predict(context)
        print(f"\nContext: {context}")
        if predictions:
            # Show top 3 predictions
            top_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
            for token, prob in top_preds:
                print(f"  {token}: {prob:.3f}")

    # Performance comparison
    print("\n5. Performance Analysis")
    print("-" * 40)

    # Build a larger corpus
    large_text = " ".join(["the quick brown fox"] * 1000)

    # Time suffix array construction
    start = time.time()
    sa_large = SuffixArray(large_text)
    build_time = time.time() - start

    # Time pattern search
    start = time.time()
    for _ in range(1000):
        sa_large.find_pattern("quick brown")
    search_time = (time.time() - start) / 1000

    print(f"Corpus size: {len(large_text)} characters")
    print(f"Build time: {build_time*1000:.2f} ms")
    print(f"Search time: {search_time*1000:.4f} ms per query")
    print(f"Memory usage: ~{len(large_text) * 4 / 1024:.1f} KB (approx)")


if __name__ == "__main__":
    demonstrate_suffix_array()