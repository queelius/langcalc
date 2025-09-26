# 📓 Interactive Jupyter Notebooks

## Learning Path

These notebooks provide a comprehensive, hands-on exploration of the algebraic language model composition framework. Follow them in order for the best learning experience:

### 1. 🧮 [explore_algebra.ipynb](explore_algebra.ipynb) - **Foundations**
**Time:** 45 minutes | **Level:** Beginner to Intermediate

Learn the core algebraic operators and how to compose language models:
- Master 10+ algebraic operators (+, *, |, &, ^, **, >>, <<, ~, @)
- Build and combine n-gram models
- Apply context transformations
- Visualize model behaviors
- Interactive playground for experimentation

**You'll build:** Custom model compositions like `(0.7 * bigram | 0.3 * trigram) ** 0.8`

### 2. 💡 [lightweight_grounding_demo.ipynb](lightweight_grounding_demo.ipynb) - **Practical Application**
**Time:** 60 minutes | **Level:** Intermediate

Implement lightweight grounding with suffix arrays:
- Understand the 95% LLM + 5% suffix array formula
- Compare suffix arrays vs n-grams (34x memory improvement)
- Build grounded Q&A systems
- Analyze perplexity reduction (70% improvement)
- Tune weights for optimal performance

**You'll build:** Production-ready grounded models with minimal overhead

### 3. 🎯 [unified_algebra.ipynb](unified_algebra.ipynb) - **Advanced Theory**
**Time:** 60 minutes | **Level:** Advanced

Master the complete theoretical framework:
- Category theory foundations
- Input → Model → Output algebra pipeline
- Algebraic law verification
- Production system implementation
- Advanced composition strategies

**You'll build:** Complete end-to-end systems with mathematical rigor

## 🚀 Quick Start

```bash
# Install dependencies
pip install jupyter numpy matplotlib seaborn

# Start Jupyter
jupyter notebook

# Open explore_algebra.ipynb to begin
```

## 📊 What You'll Learn

### Concepts
- **Algebraic Composition**: Treat models as mathematical objects
- **Suffix Arrays**: Efficient pattern matching at scale
- **Lightweight Grounding**: Minimal weight for maximum benefit
- **Context Transformations**: Sophisticated input processing
- **Performance Optimization**: Memory and speed improvements

### Skills
- Compose complex models with simple expressions
- Benchmark and visualize model performance
- Build domain-specific language models
- Implement production-ready systems
- Apply mathematical foundations to practical problems

## 🎨 Interactive Features

Each notebook includes:
- **30+ executable cells** for hands-on learning
- **Visualizations** using matplotlib and seaborn
- **Parameter tuning** sections for experimentation
- **Performance benchmarks** with comparative analysis
- **Challenge exercises** to test understanding
- **Troubleshooting guides** for common issues

## 📈 Performance Insights

Through these notebooks, you'll discover:
- **34x memory reduction** using suffix arrays vs n-grams
- **70% perplexity improvement** with 5% grounding weight
- **<1ms overhead** for lightweight grounding
- **O(m log n) query time** with suffix arrays

## 🔧 Customization

Each notebook has sections for:
- Modifying parameters to see effects
- Building custom models
- Testing on your own data
- Extending the framework
- Creating new operators

## 📚 Prerequisites

- **Python**: Basic programming knowledge
- **Probability**: Understanding of distributions
- **Language Models**: Familiarity with n-grams helpful
- **Math**: Basic algebra (advanced sections use category theory)

## 🐛 Troubleshooting

Common issues and solutions:

1. **Import errors**: Ensure you're in the project root:
   ```python
   import sys
   sys.path.append('../src')
   ```

2. **Memory issues**: Reduce corpus size in experiments:
   ```python
   corpus = corpus[:1000]  # Use first 1000 sentences
   ```

3. **Slow execution**: Skip visualization cells or reduce iterations

## 🎓 Learning Tips

1. **Run every cell** - They build on each other
2. **Modify parameters** - See how changes affect results
3. **Read the markdown** - Concepts are explained inline
4. **Try challenges** - Test your understanding
5. **Experiment freely** - The playground sections are for exploration

## 📖 Further Reading

After completing the notebooks:
- Review the [academic paper](../papers/paper.pdf)
- Explore the [test suite](../tests/)
- Read the [API documentation](../docs/ALGEBRA_DESIGN.md)
- Try the [examples](../examples/)

## 🚦 Next Steps

1. **Implement your own operator** - Extend the algebraic framework
2. **Build a domain-specific model** - Apply to your use case
3. **Optimize performance** - Try different configurations
4. **Contribute improvements** - Submit PRs with enhancements

---

*Happy experimenting! These notebooks are designed to be modified, extended, and played with. Don't just read - interact, experiment, and discover!*