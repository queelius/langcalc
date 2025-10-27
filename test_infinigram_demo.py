#!/usr/bin/env python3
"""
Interactive demo of Infinigram model in isolation.

This script demonstrates the Infinigram language model's capabilities
without any mixing or composition with other models.
"""

from langcalc import Infinigram, create_infinigram
import time


def demo_basic_usage():
    """Demonstrate basic Infinigram usage."""
    print("=" * 70)
    print("INFINIGRAM DEMO: Basic Usage")
    print("=" * 70)

    # Create a simple corpus
    corpus = [1, 2, 3, 4, 2, 3, 5, 6, 2, 3, 4, 7, 8, 2, 3, 4, 5]

    print(f"\nCorpus: {corpus}")
    print(f"Corpus size: {len(corpus)} tokens")

    # Create Infinigram model
    model = Infinigram(corpus, max_length=10)

    print(f"\nModel created successfully!")
    print(f"  Max length: {model.max_length}")
    print(f"  Min count: {model.min_count}")
    print(f"  Smoothing: {model.smoothing}")

    # Test prediction
    context = [2, 3]
    print(f"\n📊 Predicting next token after context: {context}")

    probs = model.predict(context)
    print(f"\nPredictions:")
    for token, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  Token {token}: {prob:.4f}")


def demo_text_corpus():
    """Demonstrate with text-based corpus."""
    print("\n\n" + "=" * 70)
    print("INFINIGRAM DEMO: Text Corpus")
    print("=" * 70)

    # Create text corpus (using simple character encoding)
    sentences = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "the cat jumped on the mat",
        "a dog ran in the park",
        "the quick brown fox jumps over the lazy dog"
    ]

    # Convert to token IDs (simple word-to-id mapping)
    vocab = {}
    corpus = []

    for sentence in sentences:
        words = sentence.split()
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)
            corpus.append(vocab[word])

    # Create reverse mapping
    id_to_word = {v: k for k, v in vocab.items()}

    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Corpus size: {len(corpus)} tokens")
    print(f"\nSample vocabulary:")
    for word, idx in list(vocab.items())[:10]:
        print(f"  '{word}' -> {idx}")

    # Create Infinigram
    model = Infinigram(corpus, max_length=5, smoothing=0.001)

    # Test predictions
    test_contexts = [
        ["the", "cat"],
        ["on", "the"],
        ["the", "dog"],
        ["brown", "fox"]
    ]

    print("\n📊 Testing predictions:")
    print("-" * 70)

    for words in test_contexts:
        # Convert to IDs
        context = [vocab[w] for w in words if w in vocab]

        if not context:
            continue

        print(f"\nContext: '{' '.join(words)}'")

        # Get predictions
        probs = model.predict(context, top_k=5)

        print("  Top predictions:")
        for token_id, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]:
            word = id_to_word[token_id]
            print(f"    '{word}': {prob:.4f}")


def demo_longest_suffix():
    """Demonstrate longest suffix matching."""
    print("\n\n" + "=" * 70)
    print("INFINIGRAM DEMO: Longest Suffix Matching")
    print("=" * 70)

    corpus = [1, 2, 3, 4, 5, 2, 3, 4, 6, 7, 2, 3, 4, 5, 8]
    model = Infinigram(corpus, max_length=10)

    print(f"\nCorpus: {corpus}")

    test_contexts = [
        [2, 3],
        [2, 3, 4],
        [2, 3, 4, 5],
        [3, 4],
        [9, 10]  # Not in corpus
    ]

    print("\n📊 Testing longest suffix matching:")
    print("-" * 70)

    for context in test_contexts:
        position, length = model.longest_suffix(context)

        print(f"\nContext: {context}")
        if length > 0:
            print(f"  ✓ Found match at position {position}, length {length}")
            print(f"    Matched suffix: {context[-length:]}")
            print(f"    Corpus slice: {corpus[position:position+length]}")
        else:
            print(f"  ✗ No match found")


def demo_confidence_scoring():
    """Demonstrate confidence scoring."""
    print("\n\n" + "=" * 70)
    print("INFINIGRAM DEMO: Confidence Scoring")
    print("=" * 70)

    corpus = [1, 2, 3, 4, 5, 1, 2, 3, 6, 7, 1, 2, 3, 4, 8]
    model = Infinigram(corpus, max_length=10)

    print(f"\nCorpus: {corpus}")

    test_contexts = [
        [1, 2, 3],        # Very common prefix
        [1, 2, 3, 4],     # Less common
        [1, 2, 3, 4, 5],  # Rare
        [9, 10, 11]       # Not in corpus
    ]

    print("\n📊 Confidence scores:")
    print("-" * 70)

    for context in test_contexts:
        confidence = model.confidence(context)

        # Get match info
        position, length = model.longest_suffix(context)

        print(f"\nContext: {context}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Match length: {length}")

        if confidence > 0.8:
            print(f"  → HIGH confidence (very reliable prediction)")
        elif confidence > 0.5:
            print(f"  → MEDIUM confidence (moderately reliable)")
        elif confidence > 0.2:
            print(f"  → LOW confidence (less reliable)")
        else:
            print(f"  → VERY LOW confidence (unreliable, using smoothing)")


def demo_dynamic_updates():
    """Demonstrate dynamic corpus updates."""
    print("\n\n" + "=" * 70)
    print("INFINIGRAM DEMO: Dynamic Updates")
    print("=" * 70)

    # Start with small corpus
    initial_corpus = [1, 2, 3, 4, 5]
    model = Infinigram(initial_corpus, max_length=10)

    print(f"Initial corpus: {initial_corpus}")
    print(f"Corpus size: {len(model.corpus)} tokens")

    # Test prediction before update
    context = [3, 4]
    probs_before = model.predict(context, top_k=3)

    print(f"\nBefore update - Predictions for {context}:")
    for token, prob in sorted(probs_before.items(), key=lambda x: x[1], reverse=True):
        print(f"  Token {token}: {prob:.4f}")

    # Update corpus
    new_data = [3, 4, 6, 7, 8, 3, 4, 6, 9]
    model.update(new_data)

    print(f"\nAdded new data: {new_data}")
    print(f"New corpus size: {len(model.corpus)} tokens")

    # Test prediction after update
    probs_after = model.predict(context, top_k=3)

    print(f"\nAfter update - Predictions for {context}:")
    for token, prob in sorted(probs_after.items(), key=lambda x: x[1], reverse=True):
        print(f"  Token {token}: {prob:.4f}")

    # Show what changed
    print(f"\n📊 Changes in predictions:")
    all_tokens = set(probs_before.keys()) | set(probs_after.keys())
    for token in sorted(all_tokens):
        prob_before = probs_before.get(token, 0.0)
        prob_after = probs_after.get(token, 0.0)
        change = prob_after - prob_before

        if abs(change) > 0.001:
            direction = "↑" if change > 0 else "↓"
            print(f"  Token {token}: {prob_before:.4f} → {prob_after:.4f} ({direction} {abs(change):.4f})")


def demo_performance():
    """Demonstrate performance characteristics."""
    print("\n\n" + "=" * 70)
    print("INFINIGRAM DEMO: Performance Benchmark")
    print("=" * 70)

    # Create increasingly large corpora
    sizes = [100, 1000, 10000]

    for size in sizes:
        print(f"\n📊 Corpus size: {size:,} tokens")
        print("-" * 50)

        # Generate corpus
        import random
        random.seed(42)
        corpus = [random.randint(0, 100) for _ in range(size)]

        # Time construction
        start = time.time()
        model = Infinigram(corpus, max_length=20)
        construction_time = (time.time() - start) * 1000

        print(f"  Construction time: {construction_time:.2f} ms")

        # Time predictions
        context = corpus[:5]
        num_predictions = 100

        start = time.time()
        for _ in range(num_predictions):
            _ = model.predict(context)
        prediction_time = ((time.time() - start) / num_predictions) * 1000

        print(f"  Avg prediction time: {prediction_time:.4f} ms")

        # Time longest suffix
        start = time.time()
        for _ in range(num_predictions):
            _ = model.longest_suffix(context)
        suffix_time = ((time.time() - start) / num_predictions) * 1000

        print(f"  Avg suffix search time: {suffix_time:.4f} ms")


def demo_wikipedia_example():
    """Demonstrate with a more realistic Wikipedia-like example."""
    print("\n\n" + "=" * 70)
    print("INFINIGRAM DEMO: Wikipedia-Style Example")
    print("=" * 70)

    # Simulate Wikipedia sentences (character-level for simplicity)
    wiki_text = """
    Python is a high-level programming language.
    Python was created by Guido van Rossum.
    Python is known for its simple syntax.
    Python supports multiple programming paradigms.
    Python is widely used in data science.
    """

    # Convert to character IDs
    unique_chars = sorted(set(wiki_text))
    char_to_id = {c: i for i, c in enumerate(unique_chars)}
    id_to_char = {i: c for c, i in char_to_id.items()}

    corpus = [char_to_id[c] for c in wiki_text]

    print(f"Text length: {len(wiki_text)} characters")
    print(f"Unique characters: {len(unique_chars)}")
    print(f"Corpus size: {len(corpus)} tokens")

    # Create Infinigram
    model = Infinigram(corpus, max_length=30, smoothing=0.0001)

    # Test completion
    test_strings = [
        "Python is",
        "Python was",
        "programming",
        "data"
    ]

    print("\n📊 Text completion predictions:")
    print("-" * 70)

    for text in test_strings:
        # Convert to IDs
        context = [char_to_id[c] for c in text if c in char_to_id]

        print(f"\nInput: '{text}'")

        # Get predictions
        probs = model.predict(context, top_k=5)

        print("  Most likely next characters:")
        for token_id, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]:
            char = id_to_char[token_id]
            char_display = repr(char) if char in '\n\t ' else char
            print(f"    {char_display}: {prob:.4f}")

        # Show longest match
        position, length = model.longest_suffix(context)
        if length > 0:
            matched_text = ''.join(id_to_char[corpus[position + i]] for i in range(length))
            print(f"  Matched pattern: '{matched_text}'")


def interactive_mode():
    """Interactive mode for user experimentation."""
    print("\n\n" + "=" * 70)
    print("INFINIGRAM DEMO: Interactive Mode")
    print("=" * 70)

    # Create a simple corpus
    corpus = [1, 2, 3, 4, 5, 2, 3, 6, 7, 2, 3, 4, 8, 9, 10]
    model = Infinigram(corpus, max_length=10)

    print(f"\nCorpus: {corpus}")
    print(f"\nYou can now experiment with the Infinigram model!")
    print(f"The model object is available as 'model'")
    print(f"\nExample commands:")
    print(f"  model.predict([2, 3])")
    print(f"  model.longest_suffix([2, 3, 4])")
    print(f"  model.confidence([2, 3])")
    print(f"  model.update([2, 3, 11, 12])")

    print(f"\nEntering interactive Python shell...")
    print(f"(Use Ctrl+D or exit() to quit)")

    import code
    code.interact(local={'model': model, 'corpus': corpus, 'Infinigram': Infinigram})


if __name__ == "__main__":
    print("\n🚀 LangCalc Infinigram - Standalone Demo\n")

    # Run all demos
    demo_basic_usage()
    demo_text_corpus()
    demo_longest_suffix()
    demo_confidence_scoring()
    demo_dynamic_updates()
    demo_performance()
    demo_wikipedia_example()

    # Offer interactive mode
    print("\n\n" + "=" * 70)
    print("All demos complete!")
    print("=" * 70)

    response = input("\nWould you like to enter interactive mode? (y/n): ")
    if response.lower() in ['y', 'yes']:
        interactive_mode()
    else:
        print("\n✅ Demo complete! Try running:")
        print("   python test_infinigram_demo.py")
        print("\nOr import in Python:")
        print("   from langcalc import Infinigram")
        print("   model = Infinigram(corpus)")
