"""
Setup script for ngram_projections package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ngram-projections",
    version="0.1.0",
    author="NGram Projections Contributors",
    description="An elegant algebraic API for composing language models with n-gram projections",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ngram-projections",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "mypy>=0.900",
        ],
        "llm": [
            "transformers>=4.20.0",
            "torch>=1.9.0",
        ],
        "infini": [
            "infini-gram>=0.1.0",
        ],
    },
)