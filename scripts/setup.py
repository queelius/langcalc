"""
Setup script for langcalc package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="langcalc",
    version="0.3.0",
    author="LangCalc Contributors",
    description="LangCalc: A Calculus for Compositional Language Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/queelius/langcalc",
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