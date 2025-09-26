#!/usr/bin/env python3
"""
Exploring incremental updates for suffix arrays and alternatives.
"""

import time
from typing import List, Dict, Optional
from collections import defaultdict
import random
import string


class NaiveSuffixArray:
    """
    Standard suffix array - requires full rebuild for updates.
    """

    def __init__(self, text: str = ""):
        self.text = text
        self.sa = []
        if text:
            self._rebuild()

    def _rebuild(self):
        """Full rebuild - O(n log n) time."""
        suffixes = [(self.text[i:], i) for i in range(len(self.text))]
        suffixes.sort()
        self.sa = [idx for _, idx in suffixes]

    def append(self, new_text: str):
        """Append text and rebuild entire suffix array."""
        self.text += new_text
        self._rebuild()  # Expensive! O(n log n) on entire text

    def search(self, pattern: str) -> List[int]:
        """Binary search for pattern."""
        if not self.sa:
            return []

        results = []
        left, right = 0, len(self.sa) - 1

        # Find range of suffixes starting with pattern
        while left <= right:
            mid = (left + right) // 2
            suffix = self.text[self.sa[mid]:]

            if suffix.startswith(pattern):
                # Found match, expand to find all
                i = mid
                while i >= 0 and self.text[self.sa[i]:].startswith(pattern):
                    results.append(self.sa[i])
                    i -= 1
                i = mid + 1
                while i < len(self.sa) and self.text[self.sa[i]:].startswith(pattern):
                    results.append(self.sa[i])
                    i += 1
                return sorted(results)
            elif suffix < pattern:
                left = mid + 1
            else:
                right = mid - 1

        return []


class HybridNGramIndex:
    """
    Hybrid approach: Suffix array for old data + hash table for recent updates.
    Periodically rebuilds suffix array to incorporate new data.
    """

    def __init__(self, rebuild_threshold: int = 10000):
        self.stable_text = ""  # Text in suffix array
        self.stable_sa = None  # Suffix array for stable text
        self.recent_text = ""  # Recent additions (not in SA yet)
        self.recent_index = defaultdict(list)  # Hash index for recent text
        self.rebuild_threshold = rebuild_threshold
        self.n = 3  # n-gram size

    def add_text(self, text: str):
        """Add text to the index."""
        self.recent_text += text

        # Index recent n-grams in hash table
        for i in range(len(text) - self.n + 1):
            ngram = text[i:i + self.n]
            position = len(self.stable_text) + len(self.recent_text) - len(text) + i
            self.recent_index[ngram].append(position)

        # Check if we should rebuild
        if len(self.recent_text) >= self.rebuild_threshold:
            self._merge_and_rebuild()

    def _merge_and_rebuild(self):
        """Merge recent text into stable text and rebuild suffix array."""
        print(f"Rebuilding suffix array (merging {len(self.recent_text)} chars)...")

        self.stable_text += self.recent_text
        self.stable_sa = self._build_suffix_array(self.stable_text)
        self.recent_text = ""
        self.recent_index.clear()

    def _build_suffix_array(self, text: str) -> List[int]:
        """Build suffix array for text."""
        suffixes = [(text[i:], i) for i in range(len(text))]
        suffixes.sort()
        return [idx for _, idx in suffixes]

    def search(self, pattern: str) -> List[int]:
        """Search in both stable and recent data."""
        results = []

        # Search in stable suffix array
        if self.stable_sa:
            results.extend(self._search_sa(pattern))

        # Search in recent hash index
        if pattern in self.recent_index:
            results.extend(self.recent_index[pattern])

        return sorted(results)

    def _search_sa(self, pattern: str) -> List[int]:
        """Binary search in suffix array."""
        if not self.stable_sa:
            return []

        results = []
        for i, idx in enumerate(self.stable_sa):
            if self.stable_text[idx:].startswith(pattern):
                results.append(idx)
        return results


class DynamicNGramIndex:
    """
    Alternative: Pure hash-based index with rolling window.
    Supports true incremental updates but uses more memory.
    """

    def __init__(self, n: int = 3, max_size: int = 1000000):
        self.n = n
        self.ngrams = defaultdict(list)
        self.text = ""
        self.max_size = max_size
        self.window_start = 0

    def add_text(self, text: str):
        """Add text incrementally."""
        old_len = len(self.text)
        self.text += text

        # Add new n-grams
        for i in range(len(text)):
            for n in range(1, self.n + 1):  # Store multiple n-gram sizes
                if i + n <= len(text):
                    ngram = text[i:i + n]
                    position = old_len + i
                    self.ngrams[ngram].append(position)

        # Implement sliding window if text gets too large
        if len(self.text) > self.max_size:
            self._slide_window()

    def _slide_window(self):
        """Remove old data when size limit exceeded."""
        remove_size = len(self.text) - self.max_size

        # Update positions and remove old entries
        new_ngrams = defaultdict(list)
        for ngram, positions in self.ngrams.items():
            new_positions = [p - remove_size for p in positions if p >= remove_size]
            if new_positions:
                new_ngrams[ngram] = new_positions

        self.ngrams = new_ngrams
        self.text = self.text[remove_size:]
        self.window_start += remove_size

    def search(self, pattern: str) -> List[int]:
        """Search for pattern."""
        if pattern in self.ngrams:
            return [p + self.window_start for p in self.ngrams[pattern]]
        return []

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            'text_size': len(self.text),
            'unique_ngrams': len(self.ngrams),
            'total_positions': sum(len(positions) for positions in self.ngrams.values()),
            'window_start': self.window_start
        }


def benchmark_incremental_updates():
    """
    Compare different approaches for incremental updates.
    """
    print("=" * 70)
    print("INCREMENTAL UPDATE COMPARISON")
    print("=" * 70)

    # Generate test data
    def generate_text(size: int) -> str:
        """Generate random text."""
        words = ["the", "dog", "cat", "ran", "jumped", "quickly", "slowly", "big", "small"]
        return " ".join(random.choices(words, k=size))

    # Test parameters
    initial_size = 1000
    update_size = 100
    num_updates = 10

    initial_text = generate_text(initial_size)
    updates = [generate_text(update_size) for _ in range(num_updates)]

    print(f"\nInitial text: {initial_size} words")
    print(f"Updates: {num_updates} x {update_size} words each")
    print()

    # Test 1: Naive suffix array (full rebuild)
    print("1. Naive Suffix Array (full rebuild each time)")
    print("-" * 50)

    naive_sa = NaiveSuffixArray(initial_text)

    start = time.time()
    for update in updates:
        naive_sa.append(" " + update)
    naive_time = time.time() - start

    print(f"Total update time: {naive_time*1000:.2f} ms")
    print(f"Average per update: {naive_time*1000/num_updates:.2f} ms")
    print(f"Final text size: {len(naive_sa.text)} chars")

    # Search test
    start = time.time()
    for _ in range(100):
        naive_sa.search("dog")
    search_time = (time.time() - start) / 100
    print(f"Search time: {search_time*1000:.4f} ms")

    # Test 2: Hybrid approach
    print("\n2. Hybrid Index (SA + Hash Table)")
    print("-" * 50)

    hybrid = HybridNGramIndex(rebuild_threshold=500)
    hybrid.add_text(initial_text)

    start = time.time()
    for update in updates:
        hybrid.add_text(" " + update)
    hybrid_time = time.time() - start

    print(f"Total update time: {hybrid_time*1000:.2f} ms")
    print(f"Average per update: {hybrid_time*1000/num_updates:.2f} ms")

    # Search test
    start = time.time()
    for _ in range(100):
        hybrid.search("dog")
    search_time = (time.time() - start) / 100
    print(f"Search time: {search_time*1000:.4f} ms")

    # Test 3: Dynamic hash-based
    print("\n3. Dynamic Hash Index")
    print("-" * 50)

    dynamic = DynamicNGramIndex(n=3)
    dynamic.add_text(initial_text)

    start = time.time()
    for update in updates:
        dynamic.add_text(" " + update)
    dynamic_time = time.time() - start

    print(f"Total update time: {dynamic_time*1000:.2f} ms")
    print(f"Average per update: {dynamic_time*1000/num_updates:.2f} ms")

    stats = dynamic.get_stats()
    print(f"Unique n-grams: {stats['unique_ngrams']}")
    print(f"Memory positions: {stats['total_positions']}")

    # Search test
    start = time.time()
    for _ in range(100):
        dynamic.search("dog")
    search_time = (time.time() - start) / 100
    print(f"Search time: {search_time*1000:.4f} ms")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nUpdate Performance (relative to naive):")
    print(f"  Naive SA:  1.00x (baseline)")
    print(f"  Hybrid:    {naive_time/hybrid_time:.2f}x faster")
    print(f"  Dynamic:   {naive_time/dynamic_time:.2f}x faster")

    print("\nTrade-offs:")
    print("  Naive SA:  ❌ Slow updates, ✅ Optimal search, ✅ Low memory")
    print("  Hybrid:    ✅ Fast updates, ✅ Good search, ✅ Moderate memory")
    print("  Dynamic:   ✅ Fast updates, ✅ Fast search, ❌ High memory")


def demonstrate_solutions():
    """
    Demonstrate practical solutions for incremental updates.
    """
    print("\n" + "=" * 70)
    print("PRACTICAL SOLUTIONS FOR INCREMENTAL UPDATES")
    print("=" * 70)

    print("\n1. BATCHED UPDATES")
    print("-" * 50)
    print("Collect updates and rebuild periodically:")
    print("  - Keep recent data in memory")
    print("  - Rebuild suffix array nightly/weekly")
    print("  - Good for: Search engines, document indices")

    print("\n2. HYBRID APPROACH")
    print("-" * 50)
    print("Combine suffix array with hash table:")
    print("  - Stable data in suffix array")
    print("  - Recent updates in hash table")
    print("  - Periodically merge and rebuild")
    print("  - Good for: Logs, streaming data")

    print("\n3. SEGMENTED INDICES")
    print("-" * 50)
    print("Multiple suffix arrays for different time periods:")
    print("  - Daily/weekly/monthly segments")
    print("  - Search across all segments")
    print("  - Can delete old segments")
    print("  - Good for: Time-series data")

    print("\n4. ALTERNATIVE STRUCTURES")
    print("-" * 50)
    print("Consider other data structures:")
    print("  - Suffix trees (support incremental updates)")
    print("  - FM-index (compressed, slower updates)")
    print("  - Hash tables (fast updates, more memory)")
    print("  - Good for: Different trade-offs")


if __name__ == "__main__":
    benchmark_incremental_updates()
    demonstrate_solutions()