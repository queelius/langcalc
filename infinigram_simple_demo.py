#!/usr/bin/env python3
"""
Simple Infinigram demo - try the model in isolation.

Run this script to see Infinigram in action!
"""

from langcalc import Infinigram


def main():
    print("🚀 Infinigram Language Model - Simple Demo\n")
    print("=" * 60)

    # Example 1: Simple numeric corpus
    print("\n📊 Example 1: Numeric Sequences")
    print("-" * 60)

    corpus = [1, 2, 3, 4, 2, 3, 5, 6, 2, 3, 4, 7, 8]
    model = Infinigram(corpus, max_length=10)

    print(f"Corpus: {corpus}")
    print(f"Size: {len(corpus)} tokens\n")

    # Predict after [2, 3]
    context = [2, 3]
    probs = model.predict(context)

    print(f"Context: {context}")
    print("Most likely next tokens:")
    for token, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]:
        bar = "█" * int(prob * 50)
        print(f"  {token}: {prob:.3f} {bar}")

    # Example 2: Text (word-based)
    print("\n\n📊 Example 2: Text Prediction")
    print("-" * 60)

    # Simple word-to-ID mapping
    sentences = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "the cat ran on the mat"
    ]

    # Build vocabulary
    vocab = {}
    corpus = []
    for sent in sentences:
        for word in sent.split():
            if word not in vocab:
                vocab[word] = len(vocab)
            corpus.append(vocab[word])

    id_to_word = {v: k for k, v in vocab.items()}

    # Create model
    model = Infinigram(corpus, max_length=5)

    print(f"Training on {len(sentences)} sentences")
    print(f"Vocabulary: {len(vocab)} unique words")
    print(f"Corpus: {len(corpus)} tokens\n")

    # Predict "the cat ..."
    context_words = ["the", "cat"]
    context = [vocab[w] for w in context_words]

    print(f"Context: '{' '.join(context_words)}'")
    print("Predictions:")

    probs = model.predict(context, top_k=5)
    for token_id, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]:
        word = id_to_word[token_id]
        bar = "█" * int(prob * 50)
        print(f"  '{word}': {prob:.3f} {bar}")

    # Example 3: Show variable-length matching
    print("\n\n📊 Example 3: Variable-Length Suffix Matching")
    print("-" * 60)

    corpus = [1, 2, 3, 4, 5, 1, 2, 3, 6, 1, 2, 3, 4, 7]
    model = Infinigram(corpus, max_length=10)

    print(f"Corpus: {corpus}\n")

    test_contexts = [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]]

    for context in test_contexts:
        position, length = model.longest_suffix(context)
        confidence = model.confidence(context)

        print(f"Context: {context}")
        if length > 0:
            print(f"  ✓ Matched {length} tokens at position {position}")
            print(f"  Confidence: {confidence:.3f}")
        else:
            print(f"  ✗ No match")
        print()

    # Example 4: Dynamic updates
    print("\n📊 Example 4: Dynamic Corpus Updates")
    print("-" * 60)

    corpus = [1, 2, 3, 4, 5]
    model = Infinigram(corpus, max_length=10)

    print(f"Initial corpus: {corpus}")

    context = [3, 4]
    probs_before = model.predict(context, top_k=3)

    print(f"\nBefore update - Top predictions for {context}:")
    for token, prob in sorted(probs_before.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  Token {token}: {prob:.3f}")

    # Add new data
    new_data = [3, 4, 6, 7, 3, 4, 6, 8]
    model.update(new_data)

    print(f"\nAdded: {new_data}")
    print(f"New corpus size: {len(model.corpus)} tokens")

    probs_after = model.predict(context, top_k=3)

    print(f"\nAfter update - Top predictions for {context}:")
    for token, prob in sorted(probs_after.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  Token {token}: {prob:.3f}")

    # Summary
    print("\n\n✅ Demo Complete!")
    print("=" * 60)
    print("\nKey Infinigram Features Demonstrated:")
    print("  • Variable-length pattern matching")
    print("  • Suffix array-based efficient search")
    print("  • Confidence scoring")
    print("  • Dynamic corpus updates")
    print("  • Text and numeric data support")
    print("\nTry it yourself:")
    print("  from langcalc import Infinigram")
    print("  model = Infinigram(your_corpus)")
    print("  probs = model.predict(your_context)")


if __name__ == "__main__":
    main()
