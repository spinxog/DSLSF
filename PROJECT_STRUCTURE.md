# RNA 3D Folding Pipeline - Project Structure

This document outlines the complete folder organization for the RNA 3D folding pipeline project.

## Root Directory Structure

```
DSLSF/
# Core Project Files
README.md                    # Project overview and quick start
LICENSE                     # License file
setup.py                    # Package setup and installation
requirements.txt            # Dependencies
Makefile                    # Build and development commands
.gitignore                  # Git ignore patterns
pyproject.toml              # Modern Python project configuration

# Configuration
config/                     # Global configuration files
docs/                       # Documentation
examples/                   # Usage examples and tutorials
tests/                      # Test suite
scripts/                    # Utility and automation scripts

# Data and Results
data/                       # Raw and processed data
cache/                      # Cached computations and embeddings
checkpoints/               # Model checkpoints and weights
logs/                       # Log files
results/                    # Experiment results and outputs
benchmarks/                 # Performance benchmarks
experiments/               # Research experiments

# Development
tools/                      # Development and utility tools
assets/                     # Static assets and resources
```

## Core Package Structure (`rna_model/`)

```
rna_model/
# Core Package Files
__init__.py                 # Package initialization and public API
config.py                   # Configuration management
logging_config.py            # Structured logging setup

# Core Modules
language_model.py           # BERT-style RNA language model
secondary_structure.py      # Secondary structure prediction
structure_encoder.py        # Sparse attention encoder
geometry_module.py          # SE(3)-equivariant geometry module
sampler.py                  # Decoy generation and sampling
refinement.py               # Structure refinement
pipeline.py                 # Main pipeline orchestration
training.py                 # Training utilities
evaluation.py               # Evaluation metrics
data.py                     # Data loading and preprocessing
utils.py                    # Utility functions

# Subpackages
rna_model/
    data/                   # Data-related utilities
        __init__.py
        loaders.py
        preprocessors.py
        datasets.py
        msa.py
    
    parameters/             # Model parameters and weights
        __init__.py
        pretrained.py
        checkpoints.py
    
    configs/                # Model configurations
        __init__.py
        lm_config.py
        geometry_config.py
        training_config.py
    
    motifs/                 # RNA motif libraries
        __init__.py
        tetraloops.py
        junctions.py
        hairpins.py
    
    utils/                  # Additional utilities
        __init__.py
        geometry.py
        sequence.py
        visualization.py
    
    cli/                    # Command-line interface
        __init__.py
        predict.py
        train.py
        evaluate.py
    
    evaluation/             # Evaluation utilities
        __init__.py
        metrics.py
        benchmarks.py
        reports.py
    
    training/               # Training utilities
        __init__.py
        optimizers.py
        schedulers.py
        callbacks.py
```

## Scripts Organization (`scripts/`)

```
scripts/
# Core Scripts
core/
    __init__.py
    test_pipeline.py          # Pipeline testing
    input_processing.py       # Data input preprocessing
    submission_formatting.py  # Competition submission formatting

# Advanced ML Techniques
advanced/
    __init__.py
    advanced_optimizations.py
    automated_benchmarking.py
    clustering_ranking_calibration.py
    competition_deployment.py
    contact_graph_preprocessing.py
    data_collection.py
    ensemble_prediction.py
    finetuning.py
    fragment_library.py
    model_interpretation.py
    multimodal_learning.py
    offline_training.py
    parallel_tempering_mcmc.py
    pretraining.py
    quality_calibration.py
    relaxer_rescoring.py
    rescoring_ensemble.py
    retrieval_optimization.py
    robustness_features.py
    sampling_refinement.py
    self_distillation.py
    stitched_domain_assembly.py
    student_model_inference.py
    template_integration.py
    topology_aware_sampler.py
    validation_experiments.py

# Performance Optimization
optimization/
    __init__.py
    advanced_optimizations.py
    automated_benchmarking.py
    retrieval_optimization.py
    robustness_features.py

# Evaluation and Benchmarking
evaluation/
    __init__.py
    automated_benchmarking.py
    clustering_ranking_calibration.py
    competition_deployment.py
    quality_calibration.py
    validation_experiments.py
    monitoring_diagnostics.py
```

## Data Organization

```
data/
# Raw Data
raw/
    pdb/                    # PDB structure files
    fasta/                  # Sequence files
    msa/                    # Multiple sequence alignments
    metadata/               # Metadata and annotations

# Processed Data
processed/
    embeddings/             # Precomputed embeddings
    structures/             # Processed structures
    datasets/               # Training/validation datasets
    contacts/               # Contact maps

# External Databases
external/
    rnacentral/             # RNAcentral data
    rfam/                   # Rfam families
    pdb/                    # PDB structures
```

## Configuration Files

```
config/
default_config.json         # Default configuration
development_config.json      # Development settings
production_config.json       # Production settings
competition_config.json      # Competition-specific settings

rna_model/configs/
lm_config.yaml                # Language model configuration
geometry_config.yaml          # Geometry module configuration
training_config.yaml           # Training configuration
```

## Documentation

```
docs/
# User Documentation
user_guide.md                # User guide
api_reference.md              # API reference
tutorials/                    # Step-by-step tutorials
    quick_start.md
    advanced_usage.md
    competition_guide.md

# Developer Documentation
developer_guide.md           # Developer guide
architecture.md               # System architecture
contributing.md               # Contribution guidelines
code_style.md                # Code style guidelines

# Research Documentation
papers/                       # Research papers
experiments/                  # Experiment documentation
benchmarks/                  # Benchmark results

# Deployment
deployment/                   # Deployment guides
    hpc_setup.md
    cloud_setup.md
    competition_setup.md
```

## Examples

```
examples/
# Basic Usage
basic/
    predict_structure.py       # Basic structure prediction
    train_model.py             # Basic training example
    evaluate_model.py          # Basic evaluation

# Advanced Usage
advanced/
    custom_pipeline.py          # Custom pipeline configuration
    distributed_training.py    # Distributed training setup
    ensemble_prediction.py      # Ensemble prediction

# Competition
competition/
    submission_pipeline.py     # Competition submission
    benchmarking.py             # Performance benchmarking
    optimization.py             # Competition optimization
```

## Testing

```
tests/
# Unit Tests
unit/
    test_language_model.py
    test_geometry_module.py
    test_pipeline.py
    test_data.py

# Integration Tests
integration/
    test_end_to_end.py
    test_compatibility.py
    test_performance.py

# Test Data
test_data/
    sequences/
    structures/
    configs/
```

## Development Tools

```
tools/
# Data Processing
data_processing/
    pdb_parser.py
    msa_generator.py
    contact_calculator.py

# Model Analysis
model_analysis/
    parameter_count.py
    memory_profiler.py
    performance_profiler.py

# Utilities
utilities/
    format_check.py
    lint_check.py
    dependency_check.py
```

## Git Ignore Patterns

```
.gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environments
venv/
env/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project Specific
logs/
results/
checkpoints/
cache/
*.pth
*.pkl
*.npy
data/raw/
data/external/
wandb/
.mlflow/
```

## Installation and Setup

```bash
# Clone repository
git clone <repository-url>
cd DSLSF

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Setup development environment
pip install -e .[dev]

# Run tests
pytest tests/

# Run linting
black rna_model/
isort rna_model/
mypy rna_model/
```

## Usage Examples

```python
# Basic usage
from rna_model import RNAFoldingPipeline, PipelineConfig

# Initialize pipeline
config = PipelineConfig(device="cuda")
pipeline = RNAFoldingPipeline(config)

# Predict structure
sequence = "GGGAAAUCCGCCUUGGCAAC"
result = pipeline.predict_single_sequence(sequence)
```

This structure provides:
- **Clear separation** of concerns
- **Scalable organization** for large projects
- **Standard Python package** structure
- **Professional development** workflow
- **Easy navigation** and maintenance
