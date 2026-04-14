"""Setup script for RNA 3D Folding Pipeline."""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rna-3d-folding",
    version="0.1.0",
    author="RNA Folding Developer",
    description="RNA 3D structure prediction pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rnafold/rna-3d-folding",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "mypy>=0.910",
            "flake8>=3.9",
            "pre-commit>=2.15",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.17",
        ],
        "benchmark": [
            "matplotlib>=3.5",
            "seaborn>=0.11",
            "pandas>=1.3",
            "jupyter>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rna-predict=rna_model.cli:predict",
            "rna-train=rna_model.cli:train",
            "rna-evaluate=rna_model.cli:evaluate",
            "rna-prepare=rna_model.cli:prepare_data",
        ],
    },
    include_package_data=True,
    package_data={
        "rna_model": [
            "data/motif_library/*.json",
            "data/parameters/*.npz",
            "configs/*.yaml",
        ],
    },
    zip_safe=False,
)