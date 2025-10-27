# Building the Documentation

This guide explains how to build and serve the LangCalc documentation locally.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

Install MkDocs and required plugins:

```bash
# From the project root directory
pip install -r docs/requirements.txt
```

Or install individually:

```bash
pip install mkdocs-material mkdocs-minify-plugin
```

## Building the Documentation

### Build HTML Documentation

Generate static HTML files:

```bash
# From the project root directory
mkdocs build
```

The built documentation will be in the `site/` directory.

### Serve Locally

Start a local development server with auto-reload:

```bash
mkdocs serve
```

Then open your browser to: http://127.0.0.1:8000

The documentation will automatically rebuild when you make changes to any markdown files.

### Serve on Custom Port

```bash
mkdocs serve --dev-addr 0.0.0.0:8080
```

## Documentation Structure

```
docs/
├── index.md                      # Homepage
├── getting-started/              # Getting started guides
│   ├── installation.md
│   ├── quickstart.md
│   └── concepts.md
├── projection-system/            # Projection system documentation
│   ├── overview.md
│   ├── formalism.md
│   ├── augmentations.md
│   ├── ordering.md
│   └── implementation.md
├── user-guide/                   # User guides
│   ├── models.md
│   ├── algebra.md
│   ├── transformations.md
│   ├── examples.md
│   └── best-practices.md
├── api/                          # API reference
│   ├── core.md
│   ├── models.md
│   ├── projections.md
│   ├── augmentations.md
│   └── algebra.md
├── advanced/                     # Advanced topics
│   ├── suffix-arrays.md
│   ├── grounding.md
│   ├── performance.md
│   └── extending.md
├── development/                  # Development guides
│   ├── contributing.md
│   ├── testing.md
│   ├── style.md
│   └── releases.md
├── about/                        # About pages
│   ├── license.md
│   ├── changelog.md
│   ├── paper.md
│   └── citation.md
├── javascripts/                  # JavaScript files
│   └── mathjax.js               # MathJax configuration
└── stylesheets/                  # CSS files
    └── extra.css                # Custom styles
```

## Configuration

The documentation is configured in `mkdocs.yml` at the project root.

Key configuration sections:

- **Site metadata**: Site name, description, URL
- **Theme**: Material theme with dark/light mode
- **Navigation**: Page organization and structure
- **Plugins**: Search, minify
- **Markdown extensions**: Math rendering, code highlighting, admonitions
- **JavaScript**: MathJax for LaTeX equations

## Math Rendering

The documentation supports LaTeX math equations:

### Inline Math

Use `\(...\)` for inline math:

```markdown
The projection is defined as \(\pi: \Sigma^* \to \Sigma^*\).
```

### Display Math

Use `$$...$$` for display math:

```markdown
$$
\text{LMS}(\pi(x, C), C) = \text{LMS}(x, \alpha(C))
$$
```

## Code Blocks

Use fenced code blocks with language specifiers:

````markdown
```python
from langcalc import Infinigram

model = Infinigram(corpus, max_length=10)
probs = model.predict(context)
```
````

## Admonitions

Use admonitions for notes, warnings, tips:

```markdown
!!! note "Optional Title"
    This is a note.

!!! warning
    This is a warning.

!!! tip
    This is a tip.
```

## Checking for Broken Links

Build the documentation and check for warnings:

```bash
mkdocs build 2>&1 | grep -E "(WARNING|ERROR)"
```

No output means all links are valid.

## Deploying Documentation

### To GitHub Pages

```bash
mkdocs gh-deploy
```

This builds the docs and pushes to the `gh-pages` branch.

### To Read the Docs

The documentation is configured for Read the Docs deployment. Simply:

1. Connect your GitHub repository to Read the Docs
2. RTD will automatically build using `mkdocs.yml`
3. Documentation will be available at `https://langcalc.readthedocs.io/`

## Updating Documentation

### Adding a New Page

1. Create a new markdown file in the appropriate directory
2. Add it to the `nav` section in `mkdocs.yml`
3. Build and verify: `mkdocs serve`

Example:

```yaml
# In mkdocs.yml
nav:
  - Getting Started:
      - installation.md
      - quickstart.md
      - concepts.md
      - new-page.md  # Add here
```

### Updating Existing Content

1. Edit the markdown file
2. The local server will auto-reload
3. Verify changes in browser
4. Commit when satisfied

## Troubleshooting

### MkDocs not found

```bash
pip install --upgrade mkdocs mkdocs-material
```

### Math not rendering

Check that `docs/javascripts/mathjax.js` exists and is configured correctly.

### Theme issues

Ensure Material theme is installed:

```bash
pip install mkdocs-material
```

### Build warnings

Review warnings carefully:

```bash
mkdocs build --strict  # Fail on warnings
```

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [MathJax Documentation](https://www.mathjax.org/)
- [PyMdown Extensions](https://facelessuser.github.io/pymdown-extensions/)

## Questions?

If you encounter issues building the documentation:

1. Check [GitHub Issues](https://github.com/queelius/langcalc/issues)
2. Ask in [GitHub Discussions](https://github.com/queelius/langcalc/discussions)
3. Review the MkDocs documentation
