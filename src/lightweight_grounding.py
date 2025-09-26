#!/usr/bin/env python3
"""
Lightweight Grounding System: Small weights, big impact.

This implements the core insight from our paper - that n-gram models with
just 1-5% weight can dramatically improve LLM factual accuracy and grounding.
"""

import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict, Counter
from dataclasses import dataclass
import requests
from abc import ABC, abstractmethod


# ============================================================================
# Core Language Model Interfaces
# ============================================================================

class LanguageModel(ABC):
    """Base interface for all language models."""

    @abstractmethod
    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        """Return probability distribution over next tokens."""
        pass

    def __mul__(self, weight: float):
        """Enable weight * model syntax."""
        return WeightedModel(self, weight)

    def __add__(self, other):
        """Enable model1 + model2 syntax."""
        if isinstance(other, WeightedModel):
            return MixtureModel([self, other.model], [1.0 - other.weight, other.weight])
        return MixtureModel([self, other], [0.5, 0.5])


class WeightedModel:
    """Wrapper for weighted models in mixtures."""

    def __init__(self, model: LanguageModel, weight: float):
        self.model = model
        self.weight = weight

    def __add__(self, other):
        """Enable weighted combinations."""
        if isinstance(other, WeightedModel):
            return MixtureModel([self.model, other.model], [self.weight, other.weight])
        elif isinstance(other, LanguageModel):
            return MixtureModel([self.model, other], [self.weight, 1.0 - self.weight])
        return NotImplemented


class MixtureModel(LanguageModel):
    """Mixture of multiple language models with weights."""

    def __init__(self, models: List[LanguageModel], weights: List[float]):
        self.models = models
        self.weights = weights
        # Normalize weights
        total = sum(weights)
        self.weights = [w/total for w in weights]

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        """Combine predictions from all models."""
        combined = defaultdict(float)

        for model, weight in zip(self.models, self.weights):
            predictions = model.predict(context, top_k)
            for token, prob in predictions.items():
                combined[token] += weight * prob

        # Keep only top_k
        sorted_preds = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


# ============================================================================
# Incremental Suffix Extension
# ============================================================================

class IncrementalSuffixExtender:
    """
    Implements the incremental suffix extension algorithm.
    Extends suffix matches backwards through local transformations.
    """

    def __init__(self, corpus_index):
        self.corpus_index = corpus_index
        self.synonym_cache = self._build_synonym_cache()
        self.transformations = []

    def _build_synonym_cache(self) -> Dict[str, List[str]]:
        """Build a simple synonym cache."""
        # In production, use WordNet or word embeddings
        return {
            'dog': ['hound', 'canine', 'pup', 'pooch'],
            'cat': ['feline', 'kitty', 'kitten'],
            'big': ['large', 'huge', 'great', 'grand'],
            'small': ['little', 'tiny', 'petite', 'mini'],
            'fast': ['quick', 'rapid', 'swift', 'speedy'],
            'slow': ['leisurely', 'gradual', 'unhurried'],
            'said': ['stated', 'declared', 'announced', 'mentioned'],
            'good': ['great', 'excellent', 'fine', 'superior'],
            'bad': ['poor', 'terrible', 'awful', 'inferior'],
            # Add more as needed
        }

    def extend_suffix(self, context: List[str], current_suffix_len: int) -> Tuple[List[str], Dict]:
        """
        Try to extend the suffix match backwards by one word.
        Returns (extended_suffix, transformation_record).
        """
        if current_suffix_len >= len(context):
            return None, None

        # Get the word just before current suffix
        boundary_idx = -(current_suffix_len + 1)
        boundary_word = context[boundary_idx].lower()

        # Build suffix to test
        suffix_start = boundary_idx if boundary_idx != -len(context) else 0

        # Strategy 1: Try exact extension
        test_suffix = context[suffix_start:]
        if self.corpus_index.contains(" ".join(test_suffix)):
            return test_suffix, {}

        # Strategy 2: Try synonyms
        if boundary_word in self.synonym_cache:
            for synonym in self.synonym_cache[boundary_word]:
                test = context[:suffix_start] + [synonym] + context[suffix_start+1:]
                test_suffix = test[suffix_start:]
                if self.corpus_index.contains(" ".join(test_suffix)):
                    return test_suffix, {boundary_word: synonym, 'position': boundary_idx}

        # Strategy 3: Handle function words
        if boundary_word in ['the', 'a', 'an']:
            # Try removing it
            test_suffix = context[suffix_start+1:]
            if self.corpus_index.contains(" ".join(test_suffix)):
                return test_suffix, {boundary_word: 'REMOVED', 'position': boundary_idx}

            # Try replacing with other determiners
            for replacement in ['the', 'a', 'an']:
                if replacement != boundary_word:
                    test = context[:suffix_start] + [replacement] + context[suffix_start+1:]
                    test_suffix = test[suffix_start:]
                    if self.corpus_index.contains(" ".join(test_suffix)):
                        return test_suffix, {boundary_word: replacement, 'position': boundary_idx}

        # Strategy 4: Try stemming (simplified)
        stem_map = {
            'dogs': 'dog', 'cats': 'cat',
            'running': 'run', 'walked': 'walk',
            'bigger': 'big', 'smallest': 'small'
        }
        if boundary_word in stem_map:
            stemmed = stem_map[boundary_word]
            test = context[:suffix_start] + [stemmed] + context[suffix_start+1:]
            test_suffix = test[suffix_start:]
            if self.corpus_index.contains(" ".join(test_suffix)):
                return test_suffix, {boundary_word: stemmed, 'position': boundary_idx, 'type': 'stem'}

        return None, None

    def find_best_match(self, context: List[str]) -> Tuple[List[str], List[Dict]]:
        """
        Find the longest pseudo-suffix match using incremental extension.
        Returns (matched_suffix, list_of_transformations).
        """
        # Start with longest exact suffix
        best_suffix = []
        best_suffix_len = 0
        transformations = []

        # First, find longest exact suffix
        for length in range(len(context), 0, -1):
            suffix = context[-length:]
            if self.corpus_index.contains(" ".join(suffix)):
                best_suffix = suffix
                best_suffix_len = length
                break

        # Now try to extend backwards
        while best_suffix_len < len(context):
            extended, transform = self.extend_suffix(context, best_suffix_len)
            if extended:
                best_suffix = extended
                best_suffix_len += 1
                if transform:
                    transformations.append(transform)
            else:
                break  # Can't extend further

        return best_suffix, transformations


# ============================================================================
# Lightweight N-gram Models
# ============================================================================

class LightweightNGramModel(LanguageModel):
    """
    Simple, fast n-gram model designed for lightweight grounding.
    Optimized for small weight contributions (1-5%) in mixtures.
    """

    def __init__(self, n: int = 3, smoothing: float = 0.001):
        self.n = n
        self.smoothing = smoothing
        self.counts = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)
        self.vocabulary = set()
        self.total_tokens = 0

    def train(self, tokens: List[str]):
        """Train on a sequence of tokens."""
        self.vocabulary.update(tokens)

        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i+self.n-1])
            next_token = tokens[i+self.n-1]
            self.counts[context][next_token] += 1
            self.context_counts[context] += 1
            self.total_tokens += 1

    def train_on_text(self, text: str):
        """Convenience method to train on raw text."""
        tokens = text.lower().split()
        self.train(tokens)

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        """Get probability distribution over next tokens."""
        # Use last n-1 tokens as context
        context_tuple = tuple(context[-(self.n-1):]) if len(context) >= self.n-1 else tuple(context)

        predictions = {}

        if context_tuple in self.counts:
            # Calculate probabilities with smoothing
            context_total = self.context_counts[context_tuple]
            vocab_size = len(self.vocabulary)

            # Get all possible next tokens
            for token in self.vocabulary:
                count = self.counts[context_tuple].get(token, 0)
                prob = (count + self.smoothing) / (context_total + self.smoothing * vocab_size)
                predictions[token] = prob
        else:
            # Fallback: uniform distribution over vocabulary
            uniform_prob = 1.0 / len(self.vocabulary) if self.vocabulary else 0.0
            predictions = {token: uniform_prob for token in self.vocabulary}

        # Return top k
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


# ============================================================================
# Specialized N-gram Models
# ============================================================================

class WikipediaNGram(LightweightNGramModel):
    """N-gram model trained on Wikipedia for factual grounding."""

    def __init__(self, n: int = 3):
        super().__init__(n=n)
        self.source = "wikipedia"
        self._load_wikipedia_sample()

    def _load_wikipedia_sample(self):
        """Load sample Wikipedia-like content."""
        sample_text = """
        Albert Einstein was a theoretical physicist who developed the theory of relativity.
        Einstein revolutionized our understanding of space time and gravity.
        Marie Curie was the first woman to win a Nobel Prize.
        Alan Turing is considered the father of computer science.
        The theory of relativity consists of special and general relativity.
        """
        self.train_on_text(sample_text)


class NewsNGram(LightweightNGramModel):
    """N-gram model for recent news, updated frequently."""

    def __init__(self, n: int = 3):
        super().__init__(n=n)
        self.source = "news"
        self.last_update = time.time()

    def update_with_recent_news(self, news_text: str):
        """Update with recent news articles."""
        self.train_on_text(news_text)
        self.last_update = time.time()


class UserContextNGram(LightweightNGramModel):
    """N-gram model personalized to user's conversation history."""

    def __init__(self, n: int = 3, max_history: int = 10000):
        super().__init__(n=n)
        self.source = "user"
        self.max_history = max_history
        self.conversation_history = []

    def add_conversation(self, text: str):
        """Add a conversation to the history."""
        self.conversation_history.append(text)
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
        self.train_on_text(text)


# ============================================================================
# Corpus Index for Suffix Matching
# ============================================================================

class SimpleCorpusIndex:
    """Simple corpus index for suffix matching."""

    def __init__(self):
        self.phrases = set()
        self.suffix_map = defaultdict(set)

    def add_text(self, text: str):
        """Add text to the index."""
        tokens = text.lower().split()

        # Index all n-grams up to length 10
        for n in range(1, min(11, len(tokens) + 1)):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i:i+n])
                self.phrases.add(phrase)

                # Index by first word for faster lookup
                self.suffix_map[tokens[i]].add(phrase)

    def contains(self, phrase: str) -> bool:
        """Check if phrase exists in corpus."""
        return phrase.lower() in self.phrases


# ============================================================================
# LLM Wrappers
# ============================================================================

class OllamaLLM(LanguageModel):
    """Wrapper for Ollama API."""

    def __init__(self, model_name: str = "mistral:latest", host: str = "localhost", port: int = 11434):
        self.model_name = model_name
        self.api_url = f"http://{host}:{port}/api/generate"

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        """Get predictions from Ollama."""
        prompt = " ".join(context)

        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_k": top_k,
                        "num_predict": 1
                    }
                },
                timeout=10
            )

            if response.status_code == 200:
                # Parse response and extract token probabilities
                # Note: Ollama's exact API may vary
                result = response.json()
                generated = result.get('response', '').strip()

                # For demo, return simple distribution
                if generated:
                    first_word = generated.split()[0] if generated else ""
                    return {first_word: 0.9, "the": 0.05, "a": 0.05}

        except Exception as e:
            print(f"Ollama API error: {e}")

        # Fallback
        return {"the": 0.3, "a": 0.2, "and": 0.2, "of": 0.15, "in": 0.15}


class MockLLM(LanguageModel):
    """Mock LLM for testing."""

    def __init__(self, name: str = "mock"):
        self.name = name

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        """Generate mock predictions."""
        # Base distribution
        predictions = {
            'the': 0.15, 'of': 0.10, 'and': 0.08, 'to': 0.07,
            'a': 0.06, 'in': 0.05, 'that': 0.04, 'is': 0.04,
            'was': 0.03, 'for': 0.03, 'with': 0.02, 'as': 0.02,
        }

        # Context-aware adjustments
        if context:
            last_word = context[-1].lower()
            if last_word in ['the', 'a', 'an']:
                predictions.update({'dog': 0.1, 'cat': 0.08, 'theory': 0.05})
            elif 'einstein' in " ".join(context).lower():
                predictions.update({'relativity': 0.2, 'physics': 0.15, 'theory': 0.1})

        # Normalize
        total = sum(predictions.values())
        predictions = {k: v/total for k, v in predictions.items()}

        # Return top k
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_preds[:top_k])


# ============================================================================
# Main Grounding System
# ============================================================================

class LightweightGroundingSystem:
    """
    The main system that combines LLMs with lightweight n-gram grounding.
    """

    def __init__(self, llm: LanguageModel, llm_weight: float = 0.95):
        self.llm = llm
        self.llm_weight = llm_weight
        self.ngram_models = {}
        self.ngram_weights = {}

    def add_ngram_model(self, name: str, model: LightweightNGramModel, weight: float):
        """Add a specialized n-gram model."""
        self.ngram_models[name] = model
        self.ngram_weights[name] = weight

    def build_mixture(self) -> LanguageModel:
        """Build the weighted mixture model."""
        models = [self.llm]
        weights = [self.llm_weight]

        for name, model in self.ngram_models.items():
            models.append(model)
            weights.append(self.ngram_weights[name])

        return MixtureModel(models, weights)

    def predict(self, context: List[str], top_k: int = 50) -> Dict[str, float]:
        """Make predictions with the grounded system."""
        mixture = self.build_mixture()
        return mixture.predict(context, top_k)

    def predict_with_explanation(self, context: List[str], top_k: int = 10) -> Dict[str, Any]:
        """
        Make predictions with detailed explanation of contributions.
        """
        # Get individual predictions
        llm_pred = self.llm.predict(context, top_k)
        ngram_preds = {name: model.predict(context, top_k)
                      for name, model in self.ngram_models.items()}

        # Get mixture prediction
        mixture_pred = self.predict(context, top_k)

        # Analyze contributions
        contributions = {}
        for token in mixture_pred:
            contrib = {'llm': llm_pred.get(token, 0) * self.llm_weight}
            for name, model in self.ngram_models.items():
                contrib[name] = ngram_preds[name].get(token, 0) * self.ngram_weights[name]
            contributions[token] = contrib

        return {
            'predictions': mixture_pred,
            'llm_predictions': llm_pred,
            'ngram_predictions': ngram_preds,
            'contributions': contributions,
            'weights': {
                'llm': self.llm_weight,
                **self.ngram_weights
            }
        }


# ============================================================================
# Demonstration
# ============================================================================

def demo_lightweight_grounding():
    """Demonstrate the lightweight grounding system."""

    print("=" * 70)
    print("LIGHTWEIGHT GROUNDING SYSTEM DEMONSTRATION")
    print("=" * 70)
    print()

    # Create specialized n-gram models
    print("1. Creating specialized n-gram models...")

    wiki_ngram = WikipediaNGram(n=3)
    news_ngram = NewsNGram(n=3)
    news_ngram.train_on_text("Anthropic announced Claude 3.5 yesterday with improved performance")

    user_ngram = UserContextNGram(n=3)
    user_ngram.add_conversation("I love programming in Python")

    # Create LLM (mock for demo)
    llm = MockLLM()

    # Create grounding system
    print("2. Building grounding system with algebraic composition...")
    system = LightweightGroundingSystem(llm, llm_weight=0.95)
    system.add_ngram_model('wikipedia', wiki_ngram, 0.03)
    system.add_ngram_model('news', news_ngram, 0.01)
    system.add_ngram_model('user', user_ngram, 0.01)

    print(f"   Composition: 0.95*LLM + 0.03*Wikipedia + 0.01*News + 0.01*User")
    print()

    # Test predictions
    test_contexts = [
        ["Albert", "Einstein", "was"],
        ["The", "theory", "of"],
        ["Anthropic", "announced"],
        ["I", "love", "programming"]
    ]

    print("3. Testing predictions...")
    print("-" * 50)

    for context in test_contexts:
        result = system.predict_with_explanation(context, top_k=5)

        print(f"\nContext: {' '.join(context)}")
        print(f"Top predictions:")
        for token, prob in list(result['predictions'].items())[:3]:
            print(f"  {token:15} {prob:.3f}")

            # Show contribution breakdown
            contrib = result['contributions'][token]
            print(f"    LLM: {contrib['llm']:.4f}, Wiki: {contrib.get('wikipedia', 0):.4f}, "
                  f"News: {contrib.get('news', 0):.4f}, User: {contrib.get('user', 0):.4f}")

    # Demonstrate incremental suffix extension
    print("\n" + "=" * 70)
    print("INCREMENTAL SUFFIX EXTENSION DEMONSTRATION")
    print("=" * 70)

    # Create corpus index
    corpus = SimpleCorpusIndex()
    corpus.add_text("the hound barked at the mailman")
    corpus.add_text("a large hound barked loudly")
    corpus.add_text("barked at the stranger")

    extender = IncrementalSuffixExtender(corpus)

    test_input = ["the", "big", "dog", "barked", "at"]
    suffix, transforms = extender.find_best_match(test_input)

    print(f"\nInput: {' '.join(test_input)}")
    print(f"Longest suffix match: {' '.join(suffix)}")
    print(f"Transformations applied: {transforms}")

    print("\n✓ System demonstration complete!")


if __name__ == "__main__":
    demo_lightweight_grounding()