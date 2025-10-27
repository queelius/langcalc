#!/usr/bin/env python3
"""
Create a sample Wikipedia-based n-gram model using a smaller dataset.
For demonstration purposes, we'll use a curated set of Wikipedia sentences.
"""

import pickle
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def create_wikipedia_sample_ngrams():
    """Create n-gram model from Wikipedia sample sentences."""

    # Sample Wikipedia sentences covering various topics
    wikipedia_sentences = [
        # Science
        "Albert Einstein developed the theory of relativity which revolutionized physics",
        "The theory of general relativity explains gravity as curved spacetime",
        "Marie Curie was the first woman to win a Nobel Prize in physics",
        "Charles Darwin proposed the theory of evolution by natural selection",
        "DNA contains the genetic instructions for all living organisms",
        "Quantum mechanics describes the behavior of matter at atomic scales",
        "The speed of light in vacuum is approximately 299792458 meters per second",
        "Black holes are regions of spacetime where gravity is extremely strong",
        "The Higgs boson was discovered at CERN in 2012",
        "Photosynthesis converts light energy into chemical energy in plants",

        # Geography
        "The capital of France is Paris",
        "The capital of Germany is Berlin",
        "The capital of Japan is Tokyo",
        "The capital of United Kingdom is London",
        "The capital of Italy is Rome",
        "Mount Everest is the highest mountain on Earth",
        "The Amazon River is the longest river in South America",
        "The Sahara Desert is the largest hot desert in the world",
        "The Pacific Ocean is the largest ocean on Earth",
        "Antarctica is the coldest continent on Earth",

        # History
        "World War II ended in 1945",
        "The Roman Empire fell in 476 AD",
        "The American Revolution began in 1776",
        "The French Revolution started in 1789",
        "The Industrial Revolution transformed manufacturing processes",
        "The Renaissance was a period of cultural rebirth in Europe",
        "The Cold War lasted from 1947 to 1991",
        "The Berlin Wall fell in 1989",
        "The ancient Egyptians built the pyramids of Giza",
        "Christopher Columbus reached the Americas in 1492",

        # Technology
        "The internet was invented by Tim Berners-Lee in 1989",
        "Artificial intelligence simulates human intelligence in machines",
        "Machine learning enables computers to learn from data",
        "Deep learning uses neural networks with multiple layers",
        "The transistor was invented at Bell Labs in 1947",
        "Moore's law predicts the doubling of transistors every two years",
        "Quantum computers use quantum bits or qubits",
        "Blockchain is a distributed ledger technology",
        "The first programmable computer was ENIAC built in 1945",
        "Alan Turing is considered the father of computer science",

        # Mathematics
        "The Pythagorean theorem relates the sides of a right triangle",
        "Prime numbers are divisible only by one and themselves",
        "Euler's identity is considered the most beautiful equation",
        "The Fibonacci sequence appears frequently in nature",
        "Calculus was independently developed by Newton and Leibniz",
        "Pi is the ratio of a circle's circumference to its diameter",
        "The golden ratio is approximately 1.618",
        "Fermat's last theorem was proved by Andrew Wiles",
        "The Riemann hypothesis remains unproven",
        "Zero was invented independently by several civilizations",

        # Biology
        "Evolution is the change in heritable characteristics over generations",
        "Natural selection is the primary mechanism of evolution",
        "Cells are the basic units of life",
        "Mitochondria are the powerhouses of the cell",
        "DNA replication is semiconservative",
        "Proteins are made of amino acids",
        "Photosynthesis produces oxygen as a byproduct",
        "The human genome contains approximately 3 billion base pairs",
        "Neurons transmit information through electrical and chemical signals",
        "Antibodies are proteins that help fight infections",

        # Physics
        "Energy cannot be created or destroyed only transformed",
        "The four fundamental forces are gravity electromagnetic weak and strong",
        "Light exhibits both wave and particle properties",
        "Time dilation occurs at high speeds or in strong gravitational fields",
        "The universe is approximately 13.8 billion years old",
        "Dark matter makes up about 27 percent of the universe",
        "The Big Bang theory explains the origin of the universe",
        "Entropy always increases in isolated systems",
        "Matter and antimatter annihilate when they meet",
        "The Planck length is the smallest measurable distance",
    ]

    # Build n-gram models
    ngram_models = {}

    for n in [2, 3, 4]:
        print(f"\nBuilding {n}-gram model from Wikipedia sample...")

        ngrams = defaultdict(lambda: defaultdict(int))
        vocab = set()
        total_ngrams = 0

        for sentence in wikipedia_sentences:
            tokens = sentence.lower().split()
            vocab.update(tokens)

            for i in range(len(tokens) - n + 1):
                if i + n <= len(tokens):
                    context = tuple(tokens[i:i + n - 1])
                    next_token = tokens[i + n - 1]
                    ngrams[context][next_token] += 1
                    total_ngrams += 1

        # Convert to regular dict for pickling
        ngrams = {k: dict(v) for k, v in ngrams.items()}

        model_data = {
            'n': n,
            'ngrams': ngrams,
            'vocab': list(vocab),
            'metadata': {
                'source': 'Wikipedia sample sentences',
                'num_sentences': len(wikipedia_sentences),
                'unique_contexts': len(ngrams),
                'vocab_size': len(vocab),
                'total_ngrams': total_ngrams
            }
        }

        ngram_models[n] = model_data

        print(f"  Unique contexts: {len(ngrams):,}")
        print(f"  Vocabulary size: {len(vocab):,}")
        print(f"  Total n-grams: {total_ngrams:,}")

        # Show top n-grams
        all_ngrams = []
        for context, next_tokens in ngrams.items():
            for token, count in next_tokens.items():
                all_ngrams.append((' '.join(context) + ' ' + token, count))

        all_ngrams.sort(key=lambda x: x[1], reverse=True)

        print(f"  Top {n}-grams:")
        for ngram_text, count in all_ngrams[:5]:
            print(f"    '{ngram_text}': {count}")

    return ngram_models


def save_wikipedia_ngrams():
    """Save Wikipedia n-gram models to disk."""

    print("="*70)
    print("WIKIPEDIA SAMPLE N-GRAM BUILDER")
    print("="*70)

    # Create data directory
    data_dir = "wikipedia_data"
    os.makedirs(data_dir, exist_ok=True)

    # Build models
    ngram_models = create_wikipedia_sample_ngrams()

    # Save models
    print("\nSaving models...")
    for n, model in ngram_models.items():
        filename = os.path.join(data_dir, f"wikipedia_sample_{n}grams.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

        size_kb = os.path.getsize(filename) / 1024
        print(f"  Saved {n}-gram model: {filename} ({size_kb:.1f} KB)")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("✓ Created Wikipedia sample n-gram models")
    print("✓ Models saved to wikipedia_data/")
    print("\nThese models contain factual knowledge from Wikipedia including:")
    print("  - Scientific facts (relativity, evolution, quantum mechanics)")
    print("  - Geographic facts (capitals, mountains, oceans)")
    print("  - Historical facts (wars, revolutions, discoveries)")
    print("  - Mathematical and technological facts")
    print("\nReady to use in the demonstration notebook!")


if __name__ == "__main__":
    save_wikipedia_ngrams()