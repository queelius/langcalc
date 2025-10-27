# Installation

This guide covers everything you need to install LangCalc and its dependencies.

## Requirements

### System Requirements

- **Python**: 3.8 or higher (tested with 3.12.3)
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large corpora)

### Python Dependencies

#### Core Dependencies (Production)

- `numpy>=1.19.0` - Numerical operations
- `scipy>=1.5.0` - Statistical functions
- `infinigram>=0.2.0` - Suffix array pattern matching

#### Development Dependencies

- `pytest>=6.0` - Testing framework
- `pytest-cov` - Coverage reporting
- `black` - Code formatting
- `flake8` - Linting
- `mypy` - Type checking

#### Experiment Dependencies

- `matplotlib>=3.3.0` - Visualization
- `jupyter>=1.0.0` - Interactive notebooks
- `requests>=2.25.0` - HTTP requests (for Ollama integration)

---

## Installation Methods

### Method 1: From Source (Recommended for Development)

This method is recommended if you want to contribute or stay up-to-date with the latest changes.

```bash
# Clone the repository
git clone https://github.com/queelius/langcalc.git
cd langcalc

# Install in development mode
pip install -e .
```

**Development mode** (`-e` flag) allows you to edit the code and see changes immediately without reinstalling.

### Method 2: With Development Dependencies

If you plan to run tests or contribute code:

```bash
# Clone and install with dev dependencies
git clone https://github.com/queelius/langcalc.git
cd langcalc
pip install -e .[dev]
```

This installs all testing, linting, and formatting tools.

### Method 3: With Experiment Dependencies

If you want to run experiments and create visualizations:

```bash
# Clone and install with experiment dependencies
git clone https://github.com/queelius/langcalc.git
cd langcalc
pip install -e .[experiments]
```

This includes matplotlib, jupyter, and other tools for experimentation.

### Method 4: Complete Installation

To install everything (development + experiments):

```bash
git clone https://github.com/queelius/langcalc.git
cd langcalc
pip install -e .[dev,experiments]
```

---

## Verifying Installation

### Quick Verification

Run this Python snippet to verify the installation:

```python
import langcalc
print(f"LangCalc version: {langcalc.__version__}")

# Test basic functionality
from langcalc import Infinigram
corpus = [1, 2, 3, 4, 2, 3, 5]
model = Infinigram(corpus, max_length=5)
context = [2, 3]
probs = model.predict(context)
print("Installation successful!")
```

### Running Tests

Verify everything works by running the test suite:

```bash
# Navigate to project directory
cd langcalc

# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=langcalc --cov-report=html
```

You should see:

```
=========================== test session starts ============================
collected 299 items

tests/test_unit/test_infinigram.py ................... [ XX%]
tests/test_unit/test_model_algebra_core.py ........... [ XX%]
...
========================== 299 passed in XXs ===========================
```

---

## Optional: Ollama Integration

If you want to use LangCalc with real LLMs via Ollama:

### Install Ollama

1. **Download Ollama**: Visit [ollama.ai](https://ollama.ai/) and follow installation instructions

2. **Pull a model**:
   ```bash
   ollama pull llama2
   ```

3. **Start Ollama server**:
   ```bash
   ollama serve
   ```

### Test Ollama Integration

```python
from langcalc.models import OllamaModel

# Create Ollama model (assumes server at localhost:11434)
llm = OllamaModel(model_name='llama2')

# Test prediction
context = list("The capital of France is".encode('utf-8'))
probs = llm.predict(context, top_k=10)
print(f"Predictions: {probs}")
```

---

## Troubleshooting

### Common Issues

#### Import Error: No module named 'langcalc'

**Solution**: Make sure you're in the correct directory and ran `pip install -e .`

```bash
cd /path/to/langcalc
pip install -e .
```

#### Import Error: No module named 'infinigram'

**Solution**: Install the infinigram dependency:

```bash
pip install infinigram>=0.2.0
```

#### numpy or scipy not found

**Solution**: Install core dependencies:

```bash
pip install numpy>=1.19.0 scipy>=1.5.0
```

#### Tests failing

**Solution**: Install dev dependencies and ensure you're using Python 3.8+:

```bash
python --version  # Should be 3.8+
pip install -e .[dev]
pytest tests/
```

#### Ollama connection refused

**Solution**: Ensure Ollama server is running:

```bash
# Start Ollama server
ollama serve

# In another terminal, test connection
curl http://localhost:11434/api/generate -d '{"model":"llama2","prompt":"test"}'
```

---

## Upgrading

To upgrade to the latest version:

```bash
cd langcalc
git pull origin master
pip install -e .[dev,experiments]
```

To upgrade dependencies:

```bash
pip install --upgrade -r requirements.txt
```

---

## Uninstalling

To uninstall LangCalc:

```bash
pip uninstall langcalc
```

To completely remove including the source:

```bash
pip uninstall langcalc
rm -rf /path/to/langcalc  # Be careful with this command!
```

---

## Development Environment Setup

For contributors, we recommend this setup:

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install in Development Mode

```bash
pip install -e .[dev,experiments]
```

### 3. Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### 4. Configure IDE

For VS Code, add to `.vscode/settings.json`:

```json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true
}
```

---

## Docker (Optional)

For a containerized environment:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY . /app
RUN pip install -e .[dev,experiments]

CMD ["python", "-m", "pytest", "tests/"]
```

Build and run:

```bash
docker build -t langcalc .
docker run langcalc
```

---

## Next Steps

Now that LangCalc is installed, continue to:

- [Quick Start](quickstart.md) - Create your first model
- [Core Concepts](concepts.md) - Understand the fundamentals
- [User Guide](../user-guide/index.md) - Explore practical applications

## Getting Help

If you encounter installation issues:

1. Check [GitHub Issues](https://github.com/queelius/langcalc/issues) for similar problems
2. Ask in [GitHub Discussions](https://github.com/queelius/langcalc/discussions)
3. Review the [troubleshooting section](#troubleshooting) above
