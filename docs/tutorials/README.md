# Tutorials

This directory contains step-by-step tutorials for learning and using the RNA 3D folding pipeline.

## Available Tutorials

### Beginner Tutorials

- **[Quick Start](quick_start.md)** - Get started with basic predictions in 5 minutes
- **[Basic Usage](basic_usage.md)** - Learn fundamental concepts and operations
- **[Data Preparation](data_preparation.md)** - Prepare your data for the pipeline

### Intermediate Tutorials

- **[Advanced Usage](advanced_usage.md)** - Advanced features and customizations
- **[Training Guide](training_guide.md)** - Train your own models
- **[Evaluation Guide](evaluation_guide.md)** - Evaluate model performance

### Advanced Tutorials

- **[Competition Preparation](competition_prep.md)** - Prepare for RNA structure prediction competitions
- **[Custom Components](custom_components.md)** - Develop custom pipeline components
- **[Performance Optimization](performance_optimization.md)** - Optimize for speed and memory

## Tutorial Structure

Each tutorial follows this structure:

1. **Introduction** - What you'll learn
2. **Prerequisites** - What you need to know
3. **Step-by-Step Instructions** - Detailed walkthrough
4. **Code Examples** - Complete, working code
5. **Exercises** - Practice what you've learned
6. **Next Steps** - Where to go next

## Getting Started

If you're new to the RNA 3D folding pipeline:

1. Start with the [Quick Start](quick_start.md) tutorial
2. Follow the [Basic Usage](basic_usage.md) tutorial
3. Try the [Data Preparation](data_preparation.md) tutorial

For more experienced users:

1. Review [Advanced Usage](advanced_usage.md)
2. Explore the [Training Guide](training_guide.md)
3. Check the [Competition Preparation](competition_prep.md)

## Running Tutorials

### Prerequisites

All tutorials require:

- Python 3.8+
- Installed RNA 3D folding pipeline
- Basic knowledge of Python and RNA biology

### Setting Up

```bash
# Clone the repository
git clone https://github.com/rnafold/rna-3d-folding.git
cd rna-3d-folding

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Navigate to tutorials
cd tutorials
```

### Running Tutorial Code

Each tutorial includes complete code examples that you can run:

```python
# Example from quick_start.py
from rna_model import RNAFoldingPipeline, PipelineConfig

# Initialize pipeline
config = PipelineConfig(device="cuda")
pipeline = RNAFoldingPipeline(config)

# Predict structure
sequence = "GGGAAAUCC"
result = pipeline.predict_single_sequence(sequence)

print(f"Predicted structure for {sequence}")
print(f"Coordinates shape: {result['coordinates'].shape}")
```

## Tutorial Data

Some tutorials require sample data. Download it with:

```bash
# Download tutorial data
python download_tutorial_data.py

# This will create:
# tutorials/data/
#   - sample_sequences.fasta
#   - sample_structures/
#   - test_data/
```

## Contributing

Contributions to tutorials are welcome! Please:

1. Follow the existing tutorial format
2. Include complete, tested code examples
3. Add exercises for practice
4. Update this index when adding new tutorials
5. Test tutorials on different platforms

### Tutorial Template

```markdown
# Tutorial Title

## Introduction
Brief description of what users will learn.

## Prerequisites
List of prerequisites and requirements.

## Step 1: Title
Detailed instructions with code examples.

## Step 2: Title
Continue with more detailed instructions.

## Code Example
```python
# Complete, working code example
```

## Exercises
Practice problems and exercises.

## Next Steps
Where to go after this tutorial.
```

## Support

If you have issues with tutorials:

1. Check the prerequisites
2. Verify your installation
3. Review the code examples carefully
4. Check the main documentation
5. Open an issue on GitHub

## Tutorial Index

| Tutorial | Level | Time Required | Topics Covered |
|----------|--------|--------------|----------------|
| Quick Start | Beginner | 5 minutes | Basic prediction |
| Basic Usage | Beginner | 30 minutes | Core concepts |
| Data Preparation | Beginner | 45 minutes | Data handling |
| Advanced Usage | Intermediate | 60 minutes | Advanced features |
| Training Guide | Intermediate | 90 minutes | Model training |
| Evaluation Guide | Intermediate | 30 minutes | Performance metrics |
| Competition Prep | Advanced | 2 hours | Competition setup |
| Custom Components | Advanced | 2 hours | Component development |
| Performance Optimization | Advanced | 90 minutes | Speed and memory |

## Additional Resources

- **[User Guide](../user_guide.md)** - Comprehensive user manual
- **[API Reference](../api_reference.md)** - Complete API documentation
- **[Mathematical Foundation](../mathematical_foundation.md)** - Theoretical background
- **[Architecture Guide](../architecture.md)** - System design

Happy learning and happy folding!