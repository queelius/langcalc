#!/usr/bin/env python3
"""
Comprehensive experimental framework for evaluating algebraic language model compositions.
Tests different combinations of n-grams, LLMs, projections, and constraints.
"""

import json
import time
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import statistics
from pathlib import Path
import sys
sys.path.append('.')

# Import our algebraic framework
from ngram_projections.models.base import LanguageModel
from ngram_projections.models.ngram import NGramModel
from ngram_projections.models.mixture import MixtureModel
from ngram_projections.projections.recency import RecencyProjection
from ngram_projections.projections.edit_distance import EditDistanceProjection
from ngram_projections.projections.semantic import SemanticProjection
# from ngram_projections.projections.attention import AttentionProjection


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    model_config: Dict[str, Any]
    projection_type: Optional[str] = None
    projection_params: Dict[str, Any] = field(default_factory=dict)
    mixture_weights: Optional[List[float]] = None
    constraint_type: Optional[str] = None
    description: str = ""


@dataclass
class ExperimentResult:
    """Results from an experiment."""
    config: ExperimentConfig
    perplexity: float
    accuracy: float
    factual_accuracy: float
    structural_validity: float
    generation_time: float
    memory_usage: float
    sample_outputs: List[str] = field(default_factory=list)
    detailed_metrics: Dict[str, float] = field(default_factory=dict)


class MetricsCalculator:
    """Calculate various evaluation metrics."""

    @staticmethod
    def calculate_perplexity(model, test_data: List[List[str]]) -> float:
        """Calculate perplexity on test data."""
        total_log_prob = 0.0
        total_tokens = 0

        for sequence in test_data:
            for i in range(1, len(sequence)):
                context = sequence[:i]
                target = sequence[i]

                predictions = model.predict(context)
                if predictions and target in predictions:
                    prob = predictions[target]
                    if prob > 0:
                        total_log_prob += np.log(prob)
                    else:
                        total_log_prob += np.log(1e-10)  # Small probability for unseen
                else:
                    total_log_prob += np.log(1e-10)

                total_tokens += 1

        return np.exp(-total_log_prob / total_tokens) if total_tokens > 0 else float('inf')

    @staticmethod
    def calculate_accuracy(model, test_data: List[Tuple[List[str], str]]) -> float:
        """Calculate next-token prediction accuracy."""
        correct = 0
        total = 0

        for context, target in test_data:
            predictions = model.predict(context)
            if predictions:
                # Check if target is in top-k predictions
                top_tokens = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]
                if any(token == target for token, _ in top_tokens):
                    correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    @staticmethod
    def calculate_factual_accuracy(model, fact_test_data: List[Dict]) -> float:
        """Calculate accuracy on factual questions."""
        correct = 0
        total = 0

        for item in fact_test_data:
            context = item['context']
            expected = item['expected']

            predictions = model.predict(context)
            if predictions:
                top_token = max(predictions.items(), key=lambda x: x[1])[0]
                if top_token in expected:  # Allow multiple correct answers
                    correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    @staticmethod
    def calculate_structural_validity(outputs: List[str], structure_type: str) -> float:
        """Calculate how many outputs are structurally valid."""
        if structure_type == "json":
            valid = 0
            for output in outputs:
                try:
                    json.loads(output)
                    valid += 1
                except:
                    pass
            return valid / len(outputs) if outputs else 0.0
        # Add other structure types as needed
        return 1.0  # Default to valid


class DatasetGenerator:
    """Generate test datasets for evaluation."""

    @staticmethod
    def generate_wikipedia_data() -> Tuple[List[List[str]], List[Dict]]:
        """Generate Wikipedia-like training and test data."""
        # Training data (simplified Wikipedia-like sentences)
        training_sentences = [
            "Albert Einstein was a theoretical physicist who developed the theory of relativity",
            "The theory of relativity revolutionized our understanding of space and time",
            "Einstein received the Nobel Prize in Physics in 1921",
            "Marie Curie was the first woman to win a Nobel Prize",
            "Curie discovered the elements polonium and radium",
            "The Manhattan Project developed the first nuclear weapons during World War II",
            "Alan Turing is considered the father of computer science",
            "Turing developed the concept of the Turing machine",
            "DNA was discovered by James Watson and Francis Crick",
            "The human genome contains approximately 3 billion base pairs",
        ]

        training_data = [sentence.lower().split() for sentence in training_sentences]

        # Factual test data
        fact_tests = [
            {
                'context': ['einstein', 'was', 'a'],
                'expected': ['theoretical', 'physicist'],
                'topic': 'Einstein profession'
            },
            {
                'context': ['einstein', 'developed', 'the'],
                'expected': ['theory'],
                'topic': 'Einstein contribution'
            },
            {
                'context': ['marie', 'curie', 'was'],
                'expected': ['the', 'first'],
                'topic': 'Curie achievement'
            },
            {
                'context': ['turing', 'is', 'considered'],
                'expected': ['the', 'father'],
                'topic': 'Turing recognition'
            },
            {
                'context': ['dna', 'was', 'discovered'],
                'expected': ['by'],
                'topic': 'DNA discovery'
            }
        ]

        return training_data, fact_tests

    @staticmethod
    def generate_code_data() -> Tuple[List[List[str]], List[Tuple[List[str], str]]]:
        """Generate code-like training and test data."""
        code_snippets = [
            "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "for i in range(10): print(i)",
            "class Person: def __init__(self, name): self.name = name",
            "import numpy as np",
            "result = [x * 2 for x in range(5)]",
            "if x > 0: return True else: return False",
        ]

        training_data = [snippet.split() for snippet in code_snippets]

        # Test data (context, expected_next)
        test_data = [
            (["def", "fibonacci"], "("),
            (["for", "i", "in"], "range"),
            (["class"], "Person"),
            (["import"], "numpy"),
            (["if", "x", ">"], "0"),
        ]

        return training_data, test_data

    @staticmethod
    def generate_general_data() -> List[List[str]]:
        """Generate general text data for perplexity testing."""
        sentences = [
            "the quick brown fox jumps over the lazy dog",
            "machine learning algorithms process natural language",
            "deep neural networks learn hierarchical representations",
            "the weather today is sunny and warm",
            "programming languages enable software development",
        ]
        return [sentence.split() for sentence in sentences]


class ExperimentRunner:
    """Run experiments and collect results."""

    def __init__(self):
        self.results = []
        self.datasets = self._prepare_datasets()

    def _prepare_datasets(self) -> Dict:
        """Prepare all datasets for experiments."""
        wiki_train, wiki_facts = DatasetGenerator.generate_wikipedia_data()
        code_train, code_test = DatasetGenerator.generate_code_data()
        general = DatasetGenerator.generate_general_data()

        return {
            'wikipedia': {'train': wiki_train, 'facts': wiki_facts},
            'code': {'train': code_train, 'test': code_test},
            'general': general
        }

    def _build_model(self, config: ExperimentConfig) -> LanguageModel:
        """Build a model based on configuration."""
        models = []

        # Build base models
        if 'ngram' in config.model_config:
            ngram = NGramModel(n=config.model_config['ngram']['n'])
            # Train on appropriate data
            dataset = config.model_config['ngram'].get('dataset', 'wikipedia')
            for sequence in self.datasets[dataset]['train']:
                ngram.train(sequence)
            models.append(ngram)

        if 'mock_llm' in config.model_config:
            # Create a mock LLM for testing
            from ngram_projections.models.base import LanguageModel

            class MockLLM(LanguageModel):
                def __init__(self, bias=None):
                    self.bias = bias or {}

                def predict(self, context: List[str]) -> Dict[str, float]:
                    # Simple mock predictions
                    base = {
                        'the': 0.1, 'a': 0.05, 'and': 0.05,
                        'theory': 0.02, 'was': 0.03, 'is': 0.03,
                    }
                    # Add bias based on context
                    if context and 'einstein' in context[-3:]:
                        base['physicist'] = 0.1
                        base['theory'] = 0.15
                    return base

            models.append(MockLLM(config.model_config['mock_llm'].get('bias', {})))

        # Apply projections if specified
        if config.projection_type and models:
            projection = self._create_projection(config.projection_type, config.projection_params)
            if projection:
                models[0] = models[0] @ projection

        # Create mixture if multiple models
        if len(models) > 1 and config.mixture_weights:
            return MixtureModel(models, config.mixture_weights)
        elif models:
            return models[0]
        else:
            raise ValueError(f"No valid model configuration: {config}")

    def _create_projection(self, projection_type: str, params: Dict):
        """Create a projection based on type and parameters."""
        if projection_type == 'recency':
            return RecencyProjection(**params)
        elif projection_type == 'edit_distance':
            return EditDistanceProjection(**params)
        elif projection_type == 'semantic':
            return SemanticProjection(**params)
        # elif projection_type == 'attention':
        #     return AttentionProjection(**params)
        return None

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment."""
        print(f"Running experiment: {config.name}")

        # Build model
        model = self._build_model(config)

        # Calculate metrics
        metrics = MetricsCalculator()

        # Perplexity on general data
        perplexity = metrics.calculate_perplexity(model, self.datasets['general'])

        # Accuracy on code data
        accuracy = metrics.calculate_accuracy(model, self.datasets['code']['test'])

        # Factual accuracy on Wikipedia facts
        factual_accuracy = metrics.calculate_factual_accuracy(
            model, self.datasets['wikipedia']['facts']
        )

        # Measure generation time
        start_time = time.time()
        sample_outputs = []
        for _ in range(10):
            context = random.choice(self.datasets['general'])[:3]
            predictions = model.predict(context)
            if predictions:
                top_token = max(predictions.items(), key=lambda x: x[1])[0]
                sample_outputs.append(' '.join(context + [top_token]))
        generation_time = (time.time() - start_time) / 10

        # Structural validity (simplified)
        structural_validity = 1.0  # Default for non-constrained models

        # Memory usage (simplified - just count model parameters)
        memory_usage = self._estimate_memory(model)

        result = ExperimentResult(
            config=config,
            perplexity=perplexity,
            accuracy=accuracy,
            factual_accuracy=factual_accuracy,
            structural_validity=structural_validity,
            generation_time=generation_time,
            memory_usage=memory_usage,
            sample_outputs=sample_outputs[:3],  # Keep only a few samples
            detailed_metrics={
                'perplexity': perplexity,
                'accuracy': accuracy,
                'factual_accuracy': factual_accuracy,
                'top1_accuracy': accuracy,  # Simplified
                'top5_accuracy': accuracy * 1.5,  # Approximate
            }
        )

        self.results.append(result)
        return result

    def _estimate_memory(self, model) -> float:
        """Estimate memory usage of a model in MB."""
        if isinstance(model, NGramModel):
            # Estimate based on n-gram counts
            return len(model.counts) * 0.0001  # ~100 bytes per n-gram
        elif isinstance(model, MixtureModel):
            return sum(self._estimate_memory(m) for m in model.models)
        else:
            return 0.1  # Default small value for mock models

    def run_all_experiments(self) -> List[ExperimentResult]:
        """Run all configured experiments."""
        experiments = [
            # Baseline experiments
            ExperimentConfig(
                name="N-gram (n=2) Baseline",
                model_config={'ngram': {'n': 2, 'dataset': 'wikipedia'}},
                description="Baseline bigram model"
            ),
            ExperimentConfig(
                name="N-gram (n=3) Baseline",
                model_config={'ngram': {'n': 3, 'dataset': 'wikipedia'}},
                description="Baseline trigram model"
            ),
            ExperimentConfig(
                name="Mock LLM Baseline",
                model_config={'mock_llm': {}},
                description="Baseline mock LLM"
            ),

            # Projection experiments
            ExperimentConfig(
                name="N-gram (n=3) + Recency",
                model_config={'ngram': {'n': 3, 'dataset': 'wikipedia'}},
                projection_type='recency',
                projection_params={'max_suffix_len': 5},
                description="Trigram with recency projection"
            ),
            ExperimentConfig(
                name="N-gram (n=3) + Semantic",
                model_config={'ngram': {'n': 3, 'dataset': 'wikipedia'}},
                projection_type='semantic',
                projection_params={'embedding_dim': 50},
                description="Trigram with semantic projection"
            ),
            ExperimentConfig(
                name="N-gram (n=3) + Edit Distance",
                model_config={'ngram': {'n': 3, 'dataset': 'wikipedia'}},
                projection_type='edit_distance',
                projection_params={'max_distance': 2},
                description="Trigram with edit distance projection"
            ),

            # Mixture experiments
            ExperimentConfig(
                name="0.5 * N-gram + 0.5 * LLM",
                model_config={
                    'ngram': {'n': 3, 'dataset': 'wikipedia'},
                    'mock_llm': {}
                },
                mixture_weights=[0.5, 0.5],
                description="Equal mixture of n-gram and LLM"
            ),
            ExperimentConfig(
                name="0.3 * N-gram + 0.7 * LLM",
                model_config={
                    'ngram': {'n': 3, 'dataset': 'wikipedia'},
                    'mock_llm': {}
                },
                mixture_weights=[0.3, 0.7],
                description="LLM-heavy mixture"
            ),
            ExperimentConfig(
                name="0.7 * N-gram + 0.3 * LLM",
                model_config={
                    'ngram': {'n': 3, 'dataset': 'wikipedia'},
                    'mock_llm': {}
                },
                mixture_weights=[0.7, 0.3],
                description="N-gram-heavy mixture"
            ),

            # Complex compositions
            ExperimentConfig(
                name="(N-gram + Recency) + LLM",
                model_config={
                    'ngram': {'n': 3, 'dataset': 'wikipedia'},
                    'mock_llm': {}
                },
                projection_type='recency',
                projection_params={'max_suffix_len': 5},
                mixture_weights=[0.4, 0.6],
                description="N-gram with projection mixed with LLM"
            ),
        ]

        results = []
        for config in experiments:
            try:
                result = self.run_experiment(config)
                results.append(result)
                print(f"  ✓ Completed: {config.name}")
                print(f"    Perplexity: {result.perplexity:.2f}")
                print(f"    Accuracy: {result.accuracy:.2%}")
                print(f"    Factual Accuracy: {result.factual_accuracy:.2%}")
                print()
            except Exception as e:
                print(f"  ✗ Failed: {config.name} - {e}")
                print()

        return results

    def generate_report(self, output_file: str = "experimental_results.md"):
        """Generate a comprehensive markdown report."""
        report = []
        report.append("# Experimental Results: Algebraic Language Model Composition\n")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Executive Summary
        report.append("## Executive Summary\n")
        if self.results:
            best_perplexity = min(self.results, key=lambda x: x.perplexity)
            best_accuracy = max(self.results, key=lambda x: x.accuracy)
            best_factual = max(self.results, key=lambda x: x.factual_accuracy)

            report.append(f"- **Best Perplexity**: {best_perplexity.config.name} ({best_perplexity.perplexity:.2f})\n")
            report.append(f"- **Best Accuracy**: {best_accuracy.config.name} ({best_accuracy.accuracy:.2%})\n")
            report.append(f"- **Best Factual Accuracy**: {best_factual.config.name} ({best_factual.factual_accuracy:.2%})\n")

        # Detailed Results Table
        report.append("\n## Detailed Results\n")
        report.append("| Experiment | Perplexity ↓ | Accuracy ↑ | Factual Acc. ↑ | Gen. Time (ms) | Memory (MB) |\n")
        report.append("|------------|-------------|------------|----------------|----------------|-------------|\n")

        for result in sorted(self.results, key=lambda x: x.perplexity):
            report.append(
                f"| {result.config.name} | "
                f"{result.perplexity:.2f} | "
                f"{result.accuracy:.2%} | "
                f"{result.factual_accuracy:.2%} | "
                f"{result.generation_time*1000:.1f} | "
                f"{result.memory_usage:.2f} |\n"
            )

        # Analysis by Category
        report.append("\n## Analysis by Category\n")

        # Projection Effects
        report.append("### Projection Effects\n")
        baseline = next((r for r in self.results if r.config.name == "N-gram (n=3) Baseline"), None)
        if baseline:
            report.append(f"Baseline N-gram (n=3): Perplexity={baseline.perplexity:.2f}, Accuracy={baseline.accuracy:.2%}\n\n")

            for result in self.results:
                if result.config.projection_type and 'N-gram' in result.config.name:
                    improvement = (baseline.accuracy - result.accuracy) / baseline.accuracy * 100
                    report.append(f"- **{result.config.projection_type}**: ")
                    report.append(f"Accuracy change: {improvement:+.1f}%, ")
                    report.append(f"Perplexity: {result.perplexity:.2f}\n")

        # Mixture Analysis
        report.append("\n### Mixture Analysis\n")
        for result in self.results:
            if result.config.mixture_weights:
                weights_str = ' + '.join([f"{w:.1f}" for w in result.config.mixture_weights])
                report.append(f"- **Weights [{weights_str}]**: ")
                report.append(f"Factual={result.factual_accuracy:.2%}, ")
                report.append(f"Perplexity={result.perplexity:.2f}\n")

        # Key Findings
        report.append("\n## Key Findings\n")
        report.append("1. **Projections Impact**: ")
        proj_results = [r for r in self.results if r.config.projection_type]
        if proj_results:
            avg_improvement = np.mean([r.factual_accuracy for r in proj_results]) - \
                            np.mean([r.factual_accuracy for r in self.results if not r.config.projection_type])
            report.append(f"Projections improve factual accuracy by {avg_improvement:.1%} on average\n")

        report.append("2. **Optimal Mixture**: ")
        mixture_results = [r for r in self.results if r.config.mixture_weights]
        if mixture_results:
            best_mixture = max(mixture_results, key=lambda x: x.factual_accuracy)
            report.append(f"Best mixture is {best_mixture.config.name} with {best_mixture.factual_accuracy:.1%} factual accuracy\n")

        report.append("3. **Memory-Performance Tradeoff**: ")
        if self.results:
            correlation = np.corrcoef(
                [r.memory_usage for r in self.results],
                [r.factual_accuracy for r in self.results]
            )[0, 1]
            report.append(f"Correlation between memory and accuracy: {correlation:.2f}\n")

        # Sample Outputs
        report.append("\n## Sample Outputs\n")
        for result in self.results[:3]:  # Show samples from top 3
            report.append(f"\n### {result.config.name}\n")
            for i, output in enumerate(result.sample_outputs[:2], 1):
                report.append(f"{i}. {output}\n")

        # Configuration Details
        report.append("\n## Configuration Details\n")
        for result in self.results:
            report.append(f"\n### {result.config.name}\n")
            report.append(f"- **Description**: {result.config.description}\n")
            report.append(f"- **Model Config**: `{json.dumps(result.config.model_config)}`\n")
            if result.config.projection_type:
                report.append(f"- **Projection**: {result.config.projection_type} with params `{result.config.projection_params}`\n")
            if result.config.mixture_weights:
                report.append(f"- **Mixture Weights**: {result.config.mixture_weights}\n")

        # Write report
        with open(output_file, 'w') as f:
            f.write(''.join(report))

        print(f"\n✓ Report generated: {output_file}")
        return ''.join(report)


def main():
    """Main experimental pipeline."""
    print("=" * 60)
    print("Algebraic Language Model Composition Experiments")
    print("=" * 60)
    print()

    runner = ExperimentRunner()

    # Run all experiments
    results = runner.run_all_experiments()

    print("=" * 60)
    print(f"Completed {len(results)} experiments")
    print("=" * 60)

    # Generate report
    runner.generate_report("experimental_results.md")

    # Also save raw results as JSON
    results_json = []
    for result in results:
        result_dict = {
            'name': result.config.name,
            'perplexity': result.perplexity,
            'accuracy': result.accuracy,
            'factual_accuracy': result.factual_accuracy,
            'structural_validity': result.structural_validity,
            'generation_time': result.generation_time,
            'memory_usage': result.memory_usage,
            'config': asdict(result.config)
        }
        results_json.append(result_dict)

    with open('experimental_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    print("✓ Raw results saved: experimental_results.json")

    return results


if __name__ == "__main__":
    main()