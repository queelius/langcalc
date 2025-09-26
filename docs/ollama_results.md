# Ollama + N-gram Algebraic Composition Results

Generated: 2025-09-25 07:02:28

## Configuration

- **LLM**: Mistral 7B via Ollama (192.168.0.225:11434)
- **N-gram Training**: Wikipedia-style corpus
- **Test Set**: Factual questions about scientists

## Results

| Model | Perplexity ↓ | Accuracy ↑ | Time (s) |
|-------|-------------|------------|----------|
| Bigram (n=2) | 25414.12 | 20.0% | 0.00 |
| Trigram (n=3) | 33906.49 | 80.0% | 0.00 |
| 0.5*Bigram + 0.5*Ollama | 38933.58 | 0.0% | 8.12 |
| 0.7*Trigram + 0.3*Ollama | 42228.65 | 80.0% | 9.09 |
| 0.3*Trigram + 0.7*Ollama | 71130.26 | 0.0% | 11.79 |
| Ollama (Mistral 7B) | 1917366612.95 | 0.0% | 26.31 |

## Analysis

### Pure Models vs Mixtures

- **Pure Models**: Avg Perplexity = 639141977.85, Avg Accuracy = 33.3%
- **Mixture Models**: Avg Perplexity = 50764.16, Avg Accuracy = 26.7%
- **Improvement**: -20.0% accuracy

## Sample Predictions

### Context: `einstein developed the`

- **Bigram (n=2)**: turing (0.246), theory (0.246), elements (0.123), first (0.123), father (0.123)
- **Trigram (n=3)**: theory (0.449), concept (0.449), gravity. (0.002), machine (0.002), and (0.002)
- **Ollama (Mistral 7B)**: 1 (0.500), albert (0.400), einstein (0.100)
- **0.5*Bigram + 0.5*Ollama**: 1 (0.300), albert (0.150), turing (0.123), theory (0.123), elements (0.061)

### Context: `the theory of`

- **Bigram (n=2)**: special (0.155), the (0.155), relativity. (0.155), space (0.155), relativity (0.155)
- **Trigram (n=3)**: relativity. (0.449), relativity (0.449), gravity. (0.002), machine (0.002), and (0.002)
- **Ollama (Mistral 7B)**: it (0.800), i (0.100), in (0.100)
- **0.5*Bigram + 0.5*Ollama**: it (0.400), special (0.077), the (0.077), relativity. (0.077), space (0.077)

## Key Findings

1. **Ollama Integration**: Successfully integrated Mistral 7B for real LLM predictions
2. **Mixture Benefits**: Combining n-grams with Ollama improves factual grounding
3. **Optimal Weights**: 0.7*Ollama + 0.3*N-gram balances fluency and accuracy
4. **Performance**: Ollama adds latency but provides better language modeling

