# User Guide

This comprehensive guide covers all aspects of using the RNA 3D folding pipeline, from basic usage to advanced features and competition deployment.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Configuration](#configuration)
6. [Competition Deployment](#competition-deployment)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#frequently-asked-questions)

## Installation

### System Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM for training, 4GB+ for inference
- 20GB+ disk space for models and data

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/rnafold/rna-3d-folding.git
cd rna-3d-folding

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Verification

```python
# Test installation
from rna_model import RNAFoldingPipeline, PipelineConfig

# Basic test
config = PipelineConfig(device="cpu")
pipeline = RNAFoldingPipeline(config)
print("Installation successful!")
```

## Quick Start

### Basic Prediction

```python
from rna_model import RNAFoldingPipeline, PipelineConfig

# Initialize pipeline
config = PipelineConfig(device="cuda")  # Use GPU if available
pipeline = RNAFoldingPipeline(config)

# Predict structure for a single sequence
sequence = "GGGAAAUCCGCCUUGGCAAC"
result = pipeline.predict_single_sequence(sequence)

# Access results
print(f"Sequence: {result['sequence']}")
print(f"Residues: {result['n_residues']}")
print(f"Decoys: {result['n_decoys']}")
print(f"Coordinates shape: {result['coordinates'].shape}")
```

### Batch Prediction

```python
# Multiple sequences
sequences = [
    "GGGAAAUCC",
    "GCCUUGGCAAC", 
    "AUGCUAAUCGAU",
    "CGGAUCUCCGAGUCC"
]

results = pipeline.predict_batch(sequences)

# Process results
for i, result in enumerate(results):
    if result.get("success", True):
        print(f"Sequence {i+1}: Success")
    else:
        print(f"Sequence {i+1}: {result.get('error', 'Unknown error')}")
```

## Basic Usage

### Pipeline Configuration

```python
from rna_model import PipelineConfig, LMConfig, GeometryConfig

# Custom configuration
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

pipeline = RNAFoldingPipeline(config)
```

### Competition Mode

```python
# Enable competition optimizations
pipeline.enable_competition_mode()

# Now the pipeline is optimized for:
# - Fast inference
# - Memory efficiency
# - 8-hour time constraint
# - Batch processing
```

### Saving and Loading Models

```python
# Save the complete pipeline
pipeline.save_model("my_pipeline.pth")

# Load a saved pipeline
pipeline = RNAFoldingPipeline(config)
pipeline.load_model("my_pipeline.pth")
```

## Advanced Features

### Custom Motif Libraries

```python
from rna_model.data import RNADatasetLoader

# Load custom motifs
loader = RNADatasetLoader()
custom_motifs = loader.load_motif_library("path/to/motifs/")

# Use in pipeline
pipeline.set_motif_library(custom_motifs)
```

### Retrieval-Augmented Language Model

```python
from rna_model.language_model import RNALanguageModel

# Enable retrieval augmentation
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

# Configure adaptive sampling
sampler_config = SamplerConfig(
    n_decoys=60,  # Increased for complex sequences
    adaptive_budget=True,
    complexity_threshold=0.7
)

sampler = RNASampler(sampler_config)
```

### Custom Secondary Structure

```python
from rna_model.secondary_structure import SecondaryStructurePredictor, SSConfig

# Custom secondary structure predictor
ss_config = SSConfig(
    n_hypotheses=5,  # More hypotheses
    contact_bins=128,  # Higher resolution
    pseudoknot_dim=256  # Better pseudoknot modeling
)

ss_predictor = SecondaryStructurePredictor(ss_config)
```

## Configuration

### Configuration Files

Create JSON configuration files:

```json
{
  "DEFAULT_D_MODEL": 512,
  "DEFAULT_N_LAYERS": 12,
  "DEFAULT_N_HEADS": 8,
  "DEFAULT_BATCH_SIZE": 8,
  "DEFAULT_LEARNING_RATE": 0.0001,
  "DEFAULT_MAX_SEQUENCE_LENGTH": 512,
  "GPU_MEMORY_THRESHOLD": 40.0,
  "SPARSE_ATTENTION_THRESHOLD": 128,
  "WINDOW_SIZE": 64
}
```

### Using Configuration Files

```python
from rna_model import get_config

# Load configuration from file
config = get_config(Path("config/my_config.json"))

# Use configuration values
batch_size = config.DEFAULT_BATCH_SIZE
max_seq_len = config.DEFAULT_MAX_SEQUENCE_LENGTH
```

### Environment-Specific Configurations

```python
# Development configuration
dev_config = get_config(Path("config/development.json"))

# Production configuration  
prod_config = get_config(Path("config/production.json"))

# Competition configuration
comp_config = get_config(Path("config/competition.json"))
```

## Competition Deployment

### Competition Setup

```python
from rna_model import RNAFoldingPipeline, PipelineConfig
from rna_model.logging_config import setup_logger
from pathlib import Path

# Setup logging
logger = setup_logger("competition", Path("logs"))

# Competition-optimized configuration
config = PipelineConfig(
    device="cuda",
    mixed_precision=True,
    max_sequence_length=512,
    compile_model=True
)

pipeline = RNAFoldingPipeline(config)
pipeline.enable_competition_mode()
```

### Adaptive Budgeting

```python
from competition_submission import CompetitionSubmission

# Initialize competition submission
competition = CompetitionSubmission(
    model_path="checkpoints/competition_model.pth",
    cache_dir="cache/embeddings",
    output_dir="results",
    time_limit_hours=8.0
)

# Setup and run
competition.setup_pipeline()
competition.run_competition("test_sequences.txt", "submission.npy")
```

### Performance Monitoring

```python
from rna_model.logging_config import PerformanceLogger

# Setup performance monitoring
perf_logger = PerformanceLogger(logger)

# Track performance
perf_logger.start_timer("inference")
result = pipeline.predict_single_sequence(sequence)
perf_logger.end_timer("inference", sequence_length=len(sequence))

# Monitor memory usage
perf_logger.log_memory_usage(stage="inference")
```

## Data Handling

### Input Validation

```python
# Validate sequences
from rna_model.utils import validate_sequence

sequences = ["GGGAAAUCC", "GCCUUGGCAAC"]

for seq in sequences:
    if validate_sequence(seq):
        print(f"Valid sequence: {seq}")
    else:
        print(f"Invalid sequence: {seq}")
```

### Data Loading

```python
from rna_model.data import RNADatasetLoader

# Load PDB structures
loader = RNADatasetLoader(cache_dir="data/cache")

# Load single structure
structure = loader.load_pdb_structure("data/pdb/1RNA.pdb")

# Load dataset
dataset = loader.load_dataset("data/processed/dataset.json")
```

### Custom Datasets

```python
from rna_model.training import RNADataset, RNACollator

# Create custom dataset
sequences = ["seq1", "seq2", "seq3"]
structures = [coords1, coords2, coords3]

dataset = RNADataset(sequences, structures)
collator = RNACollator(max_seq_len=512)

# Create data loader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collator
)
```

## Evaluation

### Structure Evaluation

```python
from rna_model.evaluation import StructureEvaluator
from rna_model.utils import compute_tm_score, compute_rmsd

# Initialize evaluator
evaluator = StructureEvaluator()

# Evaluate single prediction
metrics = evaluator.evaluate_single_prediction(
    pred_coords=predicted_coordinates,
    true_coords=true_coordinates
)

# Compute specific metrics
tm_score = compute_tm_score(predicted_coordinates, true_coordinates)
rmsd = compute_rmsd(predicted_coordinates, true_coordinates)

print(f"TM-score: {tm_score:.4f}")
print(f"RMSD: {rmsd:.4f}")
```

### Batch Evaluation

```python
# Evaluate multiple predictions
predictions = [pred1, pred2, pred3]
true_structures = [true1, true2, true3]

report = evaluator.create_evaluation_report(
    predictions=predictions,
    true_structures=true_structures,
    output_file="evaluation_report.json"
)
```

### Competition Evaluation

```python
from rna_model.evaluation import CompetitionEvaluator

# Competition evaluation
comp_eval = CompetitionEvaluator()

# Evaluate submission format
metrics = comp_eval.evaluate_competition_submission(
    submission_coords=submission_array,
    true_coords=true_array,
    sequence_lengths=lengths
)

print(f"Competition metrics: {metrics}")
```

## Troubleshooting

### Common Issues

#### Memory Errors

```python
# Reduce batch size
config = PipelineConfig(batch_size=4)  # Reduce from 8

# Enable gradient checkpointing
config.enable_gradient_checkpointing = True

# Use CPU for very long sequences
config.device = "cpu"
```

#### Slow Inference

```python
# Enable mixed precision
config.mixed_precision = True

# Model compilation
config.compile_model = True

# Use sparse attention
config.sparse_attention = True
```

#### Poor Accuracy

```python
# Check sequence length
if len(sequence) > 512:
    print("Sequence too long for current model")

# Validate sequence
if not validate_sequence(sequence):
    print("Invalid sequence characters")

# Check model loading
try:
    pipeline.load_model("checkpoints/latest.pth")
except Exception as e:
    print(f"Model loading failed: {e}")
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use CPU for debugging
config = PipelineConfig(device="cpu")

# Disable mixed precision
config.mixed_precision = False
```

### Performance Profiling

```python
from rna_model.utils import memory_usage
import time

# Profile memory
start_memory = memory_usage()
start_time = time.time()

# Run prediction
result = pipeline.predict_single_sequence(sequence)

end_time = time.time()
end_memory = memory_usage()

print(f"Inference time: {end_time - start_time:.2f}s")
print(f"Memory usage: {end_memory}")
```

## Frequently Asked Questions

### Q: What sequence lengths are supported?
A: The default model supports sequences up to 512 nucleotides. Longer sequences require sparse attention and may have reduced accuracy.

### Q: Can I use the pipeline without a GPU?
A: Yes, but inference will be significantly slower. Use `device="cpu"` in the configuration.

### Q: How do I improve prediction accuracy?
A: 
- Use MSA data if available
- Enable retrieval augmentation
- Increase the number of sampling decoys
- Use ensemble predictions

### Q: What's the difference between decoys?
A: The pipeline generates multiple structural predictions (decoys) to capture structural uncertainty. The top 5 are returned by default.

### Q: How do I handle very long sequences?
A: 
- Enable sparse attention
- Use window-based processing
- Consider sequence splitting for very long sequences (>1000nt)

### Q: Can I train my own model?
A: Yes, see the training documentation for detailed instructions.

### Q: What format should input sequences be in?
A: Use standard RNA nucleotides (A, U, G, C) in 5' to 3' orientation. Use 'N' for unknown nucleotides.

### Q: How do I interpret the confidence scores?
A: Confidence scores range from 0 to 1, with higher values indicating more reliable predictions.

### Q: Can I use the pipeline for protein sequences?
A: No, the model is specifically trained for RNA sequences.

### Q: What's the best way to cite this work?
A: See the citation guidelines in the main README file.

## Getting Help

### Resources

- **API Documentation**: See `docs/api_reference.md`
- **Mathematical Foundation**: See `docs/mathematical_foundation.md`
- **Architecture Guide**: See `docs/architecture.md`

### Community

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences
- **Documentation**: Contribute to improving documentation

### Support Channels

For technical support:
1. Check this user guide first
2. Search existing GitHub issues
3. Check the FAQ section above
4. Open a new issue with detailed information

## Next Steps

After mastering the basics:

1. Read the [API Reference](api_reference.md) for detailed method documentation
2. Review the [Mathematical Foundation](mathematical_foundation.md) for theoretical understanding
3. Study the [Architecture Guide](architecture.md) for system design
4. Explore the [Competition Guide](deployment/competition_guide.md) for competition preparation

This user guide provides everything you need to effectively use the RNA 3D folding pipeline for research, competition, and production applications.