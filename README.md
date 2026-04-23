## Overview

DSLSF - Deep Learning Structure Folding is a deep learning structure folding model for RNA molecules. It combines physics-based modeling to accurately predict RNA tertiary structures from sequence data. It implements topology-aware sampling methods and SE(3)-equivariant neural networks for robust structural prediction.

## Key Features

### Advanced Sampling Methods
- **Graph-edit operators** for structural exploration (stem rewiring, junction splitting)
- **Parallel tempering MCMC** with adaptive temperature control
- **Motif recombination** from curated structural libraries
- **Adaptive budgeting** based on sequence complexity

### Deep Learning Architecture
- **SE(3)-equivariant transformers** for 3D geometric learning
- **Invariant point attention** for structural features
- **Multi-scale representations** from local to global structure
- **MSA integration** for evolutionary sequence information
- **End-to-end differentiable** pipeline

### Analysis Tools
- **TM-score and RMSD** calculations for structural validation
- **Contact map prediction** with vectorized optimization
- **Dihedral and bond angle** analysis
- **Kabsch alignment** for structural superposition

## Installation

```bash
# Clone the repository
git clone https://github.com/spinxog/DSLSF
cd rna_model

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Basic Usage

```python
from rna_model import RNAPredictor

# Initialize the predictor
predictor = RNAPredictor()

# Predict structure from sequence
sequence = "GGGAAAUCC"
structure = predictor.predict(sequence)

# Get coordinates and confidence scores
coordinates = structure.coordinates
confidence = structure.confidence_scores
```

### Command Line Interface

```bash
# Predict structure
rna-predict --sequence "GGGAAAUCC" --output structure.pdb

# Train model
rna-train --data-dir data/ --config config.yaml

# Evaluate predictions
rna-evaluate --pred-dir predictions/ --true-dir true_structures/
```

## Architecture

### Geometric Deep Learning
Our approach builds on recent advances in SE(3)-equivariant neural networks for molecular modeling:

```python
# Core geometric operations
from rna_model.geometry_module import RigidTransform
from rna_model.utils import compute_tm_score, superimpose_coordinates

# SE(3)-equivariant attention
transformer = InvariantPointAttention(d_model=256, n_heads=8)
```

### Sampling Framework
The sampling methodology incorporates techniques from multiple RNA structure prediction works:

- **Graph-based sampling** inspired by RNA secondary structure graph algorithms
- **Monte Carlo methods** adapted from protein folding literature
- **Motif-based approaches** using known RNA structural motifs
- **Diverse temperature sampling** for ensemble predictions

## Mathematical Foundation

### Quaternion-Based Rotations
All 3D transformations use quaternion mathematics for numerical stability:

```python
# Convert between quaternions and rotation matrices
quaternions = RigidTransform.matrix_to_quaternion(rotation_matrix)
rotation_matrix = RigidTransform.quaternion_to_matrix(quaternions)
```

### Structural Alignment
We implement the Kabsch algorithm for optimal structural superposition:

```python
# Align two structures
aligned_coords1, aligned_coords2 = superimpose_coordinates(coords1, coords2)
tm_score = compute_tm_score(aligned_coords1, aligned_coords2)
```

## Model Configuration

### Architecture
- **Language Model**: 512d, 12 layers, 8 heads
- **Geometry Module**: 512d, 8 layers (scaled up for better refinement)
- **Total Parameters**: ~51M

### Competition
- 8-hour inference limit compliant
- Best-of-5 TM-score predictions with diverse sampling

## Computational Efficiency
- **Vectorized operations** for distance calculations
- **GPU acceleration** for neural network inference
- **Parallel processing** for batch predictions

## Dependencies

### Core Libraries
- `torch` - Deep learning framework
- `numpy` - Numerical computations
- `biopython` - Bioinformatics utilities
- `scipy` - Scientific computing

### Optional Dependencies
- `matplotlib` - Visualization
- `jupyter` - Interactive notebooks
- `pytest` - Testing framework

## Data Formats

### Input Formats
- **FASTA sequences** for RNA input
- **PDB files** for reference structures
- **MSA files** for evolutionary information
- **JSON configs** for model parameters

### Output Formats
- **PDB files** for 3D coordinates
- **NPZ files** for compressed data
- **JSON reports** for analysis results

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black rna_model/
isort rna_model/
```

## Citation

If you use this software in your research, please cite:

```bibtex
@software{DSLSF,
  title={Deep Learning Structure Folding},
  author={Pradip Debnath},
  year={2025},
  url={https://github.com/spinxog/DSLSF}
}
```

### Related Work
This work builds upon several important contributions to RNA structure prediction and geometric deep learning:

- **SE(3)-Transformers** for molecular modeling
- **Invariant Point Attention** for 3D structures
- **RNA-specific sampling** methodologies
- **Graph-based RNA structure** algorithms

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the RNA structure prediction community for valuable datasets and benchmarks that made this work possible. Special appreciation to the developers of foundational geometric deep learning methods that inspired our approach.

## Contact

For questions, issues, or collaborations, please open an issue on GitHub or visit our [discussions page](https://github.com/spinxog/DSLSF/discussions).

---

*Note: This software is provided for research purposes. Please validate predictions experimentally before drawing biological conclusions.*
