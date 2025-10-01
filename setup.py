"""Setup configuration for langcalc package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="langcalc",
    version="0.3.0",
    author="",
    description="LangCalc: A Calculus for Compositional Language Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/queelius/langcalc",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
        "experiments": [
            "matplotlib>=3.3.0",
            "jupyter>=1.0.0",
            "requests>=2.25.0",
        ],
    },
)
