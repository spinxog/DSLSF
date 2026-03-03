# RNA 3D Folding Pipeline

A state-of-the-art end-to-end neural pipeline for RNA 3D structure prediction, inspired by recent advances in protein folding (AlphaFold2, RhoFold+, DRfold) and tailored specifically for RNA molecules.

## Overview

This implementation combines:

- **Massive sequence pretraining** on ~23M RNA sequences with contact-aware objectives
- **Evolutionary features** (MSA, secondary structure) with graceful degradation
- **Transformer-based geometry modules** with SE(3)-equivariant operations
- **Multi-task supervision** (distances, angles, torsions, sugar pucker)
- **Energy-guided refinement** with learned potentials
- **Ensemble prediction** for best-of-5 TM-score optimization

## Architecture

### Core Components

1. **Language Model** (`language_model.py`)
   - BERT-style transformer with masked span objectives
   - Contact prediction auxiliary head for long-range interactions
   - 512-d embeddings, 12 layers, 8 attention heads

2. **Secondary Structure Predictor** (`secondary_structure.py`)
   - Top-k hypothesis generation (k=3)
   - Pseudoknot-aware prediction
   - Pairwise attention with sequence features

3. **Structure Encoder** (`structure_encoder.py`)
   - Sparse/axial attention for long sequences
   - Compact student model (≤150M parameters)
   - Window-based attention for memory efficiency

4. **Geometry Module** (`geometry_module.py`)
   - SE(3)-equivariant operations (IPA adaptation)
   - Multi-part rigid body representation
   - Sugar pucker and torsion prediction

5. **Sampler** (`sampler.py`)
   - Fast decoy generation via multiple strategies
   - MC-dropout, SS hypothesis switching, MSA subsampling
   - Clustering and diverse selection

6. **Refinement** (`refinement.py`)
   - Internal-coordinate optimization
   - Bond length/angle constraints
   - Steric clash resolution

## Installation

```bash
# Clone repository
git clone <repository-url>
cd DSLSF

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

```python
from rna_model import RNAFoldingPipeline, PipelineConfig

# Initialize pipeline
config = PipelineConfig()
pipeline = RNAFoldingPipeline(config)

# Enable competition mode
pipeline.enable_competition_mode()

# Predict structure for a single sequence
sequence = "GGGAAAUCCGCCUUGGCAAC"
result = pipeline.predict_single_sequence(sequence)

# Access coordinates (5 decoys × n_residues × 3)
coordinates = result["coordinates"]
print(f"Predicted {coordinates.shape[0]} decoys for {len(sequence)} residues")
```

## Competition Deployment

The pipeline is optimized for the 8-hour notebook constraint:

- **Precomputed embeddings**: LM embeddings cached to avoid runtime computation
- **Sparse attention**: Efficient handling of sequences >200nt
- **Adaptive budgeting**: Per-sequence compute allocation based on complexity
- **Fast sampling**: 20 decoys → cluster → select 5 diverse representatives
- **Lightweight refinement**: 1-3 internal-coordinate optimization steps

### Performance Targets

- **Inference time**: <144s per sequence (200 sequences in 8h)
- **Memory usage**: <8GB GPU peak
- **Bundle size**: ≤15GB compressed artifacts
- **Accuracy**: Best-of-5 TM-score competitive with state-of-the-art

## Advanced Features

### Compression & Caching

- **PCA + 8-bit quantization** for LM embeddings
- **Reconstruction QC** with residual storage for critical sequences
- **Family-adaptive thresholds** for compression quality
- **Memory-mapped libraries** for efficient I/O

### Robust Domain Splitting

- **Ensemble proposals** (spectral clustering, community detection)
- **Multi-hub tokens** per domain (3-6 hubs)
- **Verification & backtracking** for assembly failures
- **Progressive assembly** with overlap constraints

### Topology-Aware Sampling

- **Graph-edit operators** (stem rewiring, junction splitting)
- **Parallel tempering MCMC** with adaptive temperature control
- **Motif recombination** from curated library
- **Adaptive budgeting** based on sequence complexity

### Consensus Rescoring

- **Knowledge-based potentials** from RNA structural statistics
- **Learned rescoring network** (1-2M parameters)
- **Torsion-strain detection** with motif-class statistics
- **Two-stage acceptance** with consensus requirements

## Training

### Data Preparation

```python
from rna_model.data import RNADatasetLoader, create_sample_dataset

# Create sample dataset
data = create_sample_dataset("sample_data")

# Load PDB structures
loader = RNADatasetLoader()
structures = [loader.load_pdb_structure(f) for f in pdb_files]

# Preprocess for training
processed = loader.preprocess_for_training(structures)
```

### Model Training

```python
from rna_model.training import train_model, IntegratedModel, PipelineConfig

# Create model
config = PipelineConfig()
model = IntegratedModel(config)

# Train with custom configuration
trainer = train_model(
    model=model,
    train_dataset=train_data,
    val_dataset=val_data,
    config_overrides={
        "batch_size": 4,
        "learning_rate": 1e-4,
        "max_steps": 50000
    }
)
```

### Loss Functions

- **Masked span LM**: Language modeling with long-span masking
- **Contact prediction**: Binary cross-entropy for pairwise contacts
- **FAPE**: Frame-aligned point error for 3D coordinates
- **Multi-task geometry**: Distance, angle, torsion, pucker losses
- **Confidence ranking**: TM-score prediction calibration

## Evaluation

### Structure Metrics

```python
from rna_model.evaluation import StructureEvaluator

evaluator = StructureEvaluator()

# Evaluate single prediction
metrics = evaluator.evaluate_single_prediction(pred_coords, true_coords)

# Evaluate ensemble (best-of-5)
ensemble_metrics = evaluator.evaluate_ensemble(pred_decoys, true_coords)

# Full dataset evaluation
report = evaluator.create_evaluation_report(
    predictions=all_predictions,
    true_structures=true_structures,
    output_file="evaluation_report.json"
)
```

### Competition Format

```python
from rna_model.evaluation import CompetitionEvaluator

comp_eval = CompetitionEvaluator()

# Evaluate submission format
metrics = comp_eval.evaluate_competition_submission(
    submission_coords=submission_array,
    true_coords=true_array,
    sequence_lengths=lengths
)

# Create leaderboard entry
entry = comp_eval.create_leaderboard_entry(
    team_name="RNA_Fold_Team",
    submission_coords=submission_array,
    true_coords=true_array,
    sequence_lengths=lengths
)
```

## Configuration

### Pipeline Configuration

```python
from rna_model import PipelineConfig, LMConfig, GeometryConfig

config = PipelineConfig(
    device="cuda",
    mixed_precision=True,
    max_sequence_length=512,
    lm_config=LMConfig(
        d_model=512,
        n_layers=12,
        n_heads=8
    ),
    geometry_config=GeometryConfig(
        d_model=256,
        n_layers=4,
        distance_bins=64
    )
)
```

### Training Configuration

```python
from rna_model.training import TrainingConfig

train_config = TrainingConfig(
    batch_size=8,
    learning_rate=1e-4,
    max_steps=100000,
    mixed_precision=True,
    save_every=1000,
    eval_every=500
)
```

## Advanced Usage

### Custom Motif Library

```python
from rna_model.data import RNADatasetLoader

loader = RNADatasetLoader()

# Load custom motifs
custom_motifs = loader.load_motif_library("custom_motifs/")

# Use in pipeline
pipeline = RNAFoldingPipeline(config)
pipeline.set_motif_library(custom_motifs)
```

### Retrieval-Augmented LM

```python
from rna_model.language_model import RNALanguageModel

# Load with retrieval augmentation
lm = RNALanguageModel(config)
lm.enable_retrieval(
    index_path="embedding_index/",
    top_k=64,
    diversity_weight=0.4
)
```

### Adaptive Sampling

```python
from rna_model.sampler import RNASampler, SamplerConfig

sampler_config = SamplerConfig(
    n_decoys=60,  # Increased for complex sequences
    adaptive_budget=True,
    complexity_threshold=0.7
)

sampler = RNASampler(sampler_config)
```

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce batch size or enable gradient checkpointing
2. **Slow inference**: Enable mixed precision and model compilation
3. **Poor accuracy**: Check MSA quality and secondary structure predictions
4. **Bundle size**: Use aggressive compression for non-critical families

### Performance Tuning

```python
# Enable all optimizations
pipeline.enable_competition_mode()

# Profile performance
import time
start = time.time()
result = pipeline.predict_single_sequence(sequence)
print(f"Inference time: {time.time() - start:.2f}s")

# Monitor memory
from rna_model.utils import memory_usage
print(memory_usage())
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black rna_model/
isort rna_model/

# Type checking
mypy rna_model/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
RNA 3D Folding Pipeline: End-to-End Neural Prediction of RNA Structure
[Authors], [Year]
```

## Acknowledgments

- Inspired by AlphaFold2's SE(3)-equivariant architecture
- Incorporates ideas from RhoFold+ and DRfold
- Built on the RNAcentral database for sequence pretraining
- Uses structural insights from the PDB RNA database
