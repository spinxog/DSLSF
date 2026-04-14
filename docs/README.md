# RNA 3D Folding Pipeline - Documentation Index

This directory contains comprehensive documentation for the RNA 3D folding pipeline.

## Available Documentation

### User Documentation

- **[User Guide](user_guide.md)** - Complete user manual and getting started guide
- **[API Reference](api_reference.md)** - Complete API documentation
- **[Configuration Guide](configuration.md)** - Configuration options and examples

### Technical Documentation

- **[Mathematical Foundation](mathematical_foundation.md)** - Mathematical principles and formulations
- **[System Architecture](architecture.md)** - High-level system architecture and design
- **[Development Guide](developer_guide.md)** - Developer setup and contribution guidelines

### Deployment Documentation

- **[Competition Guide](deployment/competition_guide.md)** - Competition deployment instructions
- **[HPC Setup](deployment/hpc_setup.md)** - High-performance computing setup
- **[Cloud Setup](deployment/cloud_setup.md)** - Cloud deployment instructions

### Research Documentation

- **[Papers](papers/)** - Research papers and publications
- **[Experiments](experiments/)** - Experimental results and analysis
- **[Benchmarks](benchmarks/)** - Performance benchmarks and comparisons

## Documentation Structure

```
docs/
README.md                    # This file
user_guide.md               # Complete user manual
api_reference.md             # API documentation
mathematical_foundation.md   # Mathematical principles
architecture.md              # System architecture
developer_guide.md           # Developer guide
configuration.md             # Configuration guide

tutorials/
quick_start.md              # Quick start tutorial
advanced_usage.md           # Advanced usage examples
competition_guide.md         # Competition preparation

deployment/
competition_guide.md        # Competition deployment
hpc_setup.md               # HPC environment setup
cloud_setup.md             # Cloud deployment

papers/                      # Research papers
experiments/                 # Experiment documentation
benchmarks/                 # Performance benchmarks
```

## Getting Started

1. **New Users**: Start with the [Quick Start](tutorials/quick_start.md) tutorial
2. **API Reference**: See [API Reference](api_reference.md) for detailed method documentation
3. **Configuration**: Check [Configuration Guide](configuration.md) for setup options
4. **Competition**: Follow [Competition Guide](deployment/competition_guide.md) for competition prep

## Contributing to Documentation

Documentation contributions are welcome! Please:

1. Follow the existing documentation style
2. Use clear, concise language
3. Include code examples where appropriate
4. Test all examples and commands
5. Update the table of contents when adding new sections

### Documentation Style Guidelines

- Use Markdown format
- Include code blocks with language specification
- Use proper section headers (#, ##, ###)
- Include cross-references to related documentation
- Keep examples simple and focused

## Building Documentation

To build the documentation locally:

```bash
# Install documentation dependencies
pip install -e .[docs]

# Build HTML documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

## Documentation Updates

When updating the codebase, please:

1. Update relevant API documentation
2. Add new configuration options to the configuration guide
3. Update examples if interfaces change
4. Add new tutorials for major features
5. Update the changelog

## Support

For documentation questions or issues:

1. Check existing documentation first
2. Search the [API Reference](api_reference.md)
3. Review the [User Guide](user_guide.md)
4. Check the [FAQ](user_guide.md#frequently-asked-questions)
5. Open an issue for documentation problems

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{rna_3d_folding,
  title={RNA 3D Folding Pipeline},
  author={RNA Folding Developer},
  year={2024},
  url={https://github.com/rnafold/rna-3d-folding}
```

For specific components, please cite the relevant papers mentioned in the [Mathematical Foundation](mathematical_foundation.md) documentation.