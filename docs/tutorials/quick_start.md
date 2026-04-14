# Quick Start Tutorial

Get started with the RNA 3D folding pipeline in just 5 minutes. This tutorial covers the basics of predicting RNA 3D structures.

## Introduction

In this tutorial, you will learn:
- How to initialize the RNA folding pipeline
- How to predict structures for single sequences
- How to process multiple sequences
- How to interpret the results

## Prerequisites

- Python 3.8+
- Installed RNA 3D folding pipeline
- Basic knowledge of RNA sequences
- GPU (recommended, but CPU works too)

## Step 1: Basic Setup

First, let's import the necessary components:

```python
from rna_model import RNAFoldingPipeline, PipelineConfig
from rna_model.logging_config import setup_logger
from pathlib import Path
import numpy as np
```

## Step 2: Initialize the Pipeline

Create a pipeline with default configuration:

```python
# Setup logging for monitoring
logger = setup_logger("quick_start", Path("logs"))

# Initialize pipeline with default configuration
config = PipelineConfig(
    device="cuda",  # Use GPU if available, otherwise CPU
    mixed_precision=True,  # Enable mixed precision for speed
    max_sequence_length=512  # Maximum sequence length
)

pipeline = RNAFoldingPipeline(config)
logger.info("Pipeline initialized successfully")
```

## Step 3: Predict a Single Sequence

Let's predict the structure of a simple RNA sequence:

```python
# Define your RNA sequence
sequence = "GGGAAAUCC"

# Predict the structure
logger.info(f"Predicting structure for: {sequence}")
result = pipeline.predict_single_sequence(sequence)

# Display results
print(f"Sequence: {result['sequence']}")
print(f"Number of residues: {result['n_residues']}")
print(f"Number of decoys: {result['n_decoys']}")
print(f"Coordinates shape: {result['coordinates'].shape}")
```

## Step 4: Understand the Output

The result dictionary contains:

- `sequence`: Input RNA sequence
- `coordinates`: Predicted 3D coordinates (5 decoys × N residues × 3)
- `n_decoys`: Number of decoys (always 5)
- `n_residues`: Number of residues in the sequence
- `decoys`: Full decoy information (if requested)

```python
# Access the coordinates
coordinates = result['coordinates']
print(f"Coordinates shape: {coordinates.shape}")

# First decoy coordinates
first_decoy = coordinates[:result['n_residues']]
print(f"First decoy shape: {first_decoy.shape}")

# Save coordinates to file
np.save("quick_start_prediction.npy", coordinates)
print("Coordinates saved to quick_start_prediction.npy")
```

## Step 5: Batch Processing

Process multiple sequences efficiently:

```python
# Define multiple sequences
sequences = [
    "GGGAAAUCC",      # 9 nucleotides
    "GCCUUGGCAAC",    # 10 nucleotides
    "AUGCUAAUCGAU",  # 12 nucleotides
    "CGGAUCUCCGAGUCC" # 14 nucleotides
]

# Predict structures for all sequences
logger.info(f"Processing {len(sequences)} sequences")
results = pipeline.predict_batch(sequences)

# Process results
successful = 0
for i, result in enumerate(results):
    if result.get("success", True):
        successful += 1
        print(f"Sequence {i+1}: Success - {result['n_residues']} residues")
    else:
        print(f"Sequence {i+1}: Failed - {result.get('error', 'Unknown error')}")

print(f"\nSummary: {successful}/{len(sequences)} successful predictions")
```

## Step 6: Enable Competition Mode

For optimal performance in competitions:

```python
# Enable competition optimizations
pipeline.enable_competition_mode()
logger.info("Competition mode enabled")

# Now the pipeline is optimized for:
# - Fast inference
# - Memory efficiency  
# - 8-hour time constraint
# - Batch processing

# Test with competition mode
result = pipeline.predict_single_sequence(sequence)
print(f"Competition mode prediction completed")
```

## Complete Code Example

Here's the complete working example:

```python
#!/usr/bin/env python3
"""
Quick start example for RNA 3D folding pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from rna_model import RNAFoldingPipeline, PipelineConfig
from rna_model.logging_config import setup_logger
import numpy as np

def main():
    """Run quick start example."""
    
    # Setup logging
    logger = setup_logger("quick_start", Path("logs"))
    logger.info("Starting RNA 3D folding quick start")
    
    # Initialize pipeline
    config = PipelineConfig(device="cuda", mixed_precision=True)
    pipeline = RNAFoldingPipeline(config)
    
    # Test sequences
    sequences = [
        "GGGAAAUCC",
        "GCCUUGGCAAC", 
        "AUGCUAAUCGAU",
        "CGGAUCUCCGAGUCC"
    ]
    
    # Process sequences
    results = []
    for sequence in sequences:
        try:
            result = pipeline.predict_single_sequence(sequence)
            results.append(result)
            logger.info(f"Successfully predicted {sequence}")
        except Exception as e:
            logger.error(f"Failed to predict {sequence}: {e}")
            results.append({"sequence": sequence, "error": str(e)})
    
    # Summary
    successful = sum(1 for r in results if "error" not in r)
    logger.info(f"Completed: {successful}/{len(sequences)} successful")
    
    # Save results
    all_coords = []
    for result in results:
        if "coordinates" in result:
            all_coords.append(result["coordinates"])
    
    if all_coords:
        combined_coords = np.concatenate(all_coords, axis=0)
        np.save("quick_start_results.npy", combined_coords)
        logger.info(f"Saved results to quick_start_results.npy")
    
    return results

if __name__ == "__main__":
    main()
```

## Exercises

1. **Try Different Sequences**: Test with your own RNA sequences
2. **Sequence Length**: Try sequences of different lengths (5-500 nucleotides)
3. **Batch Size**: Experiment with different batch sizes
4. **Error Handling**: Add error handling for invalid sequences
5. **Performance**: Compare CPU vs GPU performance

## Next Steps

After completing this quick start:

1. Read the [Basic Usage](basic_usage.md) tutorial
2. Review the [User Guide](../user_guide.md) for detailed information
3. Check the [API Reference](../api_reference.md) for method details
4. Try the [Data Preparation](data_preparation.md) tutorial

## Troubleshooting

### Common Issues

**Import Error**: Make sure the package is installed:
```bash
pip install -e .
```

**CUDA Error**: If CUDA is not available, use CPU:
```python
config = PipelineConfig(device="cpu")
```

**Memory Error**: Reduce batch size or use CPU:
```python
config = PipelineConfig(batch_size=1, device="cpu")
```

**Sequence Too Long**: Reduce sequence length:
```python
config = PipelineConfig(max_sequence_length=256)
```

### Getting Help

If you encounter issues:

1. Check the [User Guide](../user_guide.md)
2. Review the [FAQ](../user_guide.md#frequently-asked-questions)
3. Open an issue on GitHub

Congratulations! You've completed the RNA 3D folding pipeline quick start tutorial.