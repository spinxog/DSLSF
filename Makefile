# Makefile for RNA 3D Folding Pipeline

.PHONY: help install install-dev test lint format clean train evaluate predict setup-data benchmark docs

# Default target
help:
	@echo "RNA 3D Folding Pipeline - Available Commands:"
	@echo ""
	@echo "  install      Install package and dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run test suite"
	@echo "  lint         Run code linting"
	@echo "  format       Format code with black and isort"
	@echo "  clean        Clean temporary files and artifacts"
	@echo "  train        Train model on sample data"
	@echo "  evaluate     Evaluate model performance"
	@echo "  predict      Run prediction on sample sequences"
	@echo "  setup-data   Create sample dataset"
	@echo "  benchmark    Run performance benchmarks"
	@echo "  docs         Generate documentation"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs,benchmark]"
	pre-commit install

# Code quality
test:
	pytest tests/ -v --cov=rna_model --cov-report=html --cov-report=term

lint:
	flake8 rna_model/ tests/ examples/
	mypy rna_model/

format:
	black rna_model/ tests/ examples/
	isort rna_model/ tests/ examples/

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/

# Data preparation
setup-data:
	python -c "from rna_model.data import create_sample_dataset; create_sample_dataset()"

# Training
train:
	python examples/basic_usage.py

# Evaluation
evaluate:
	python -c "
from rna_model.evaluation import benchmark_model
from rna_model.data import create_sample_dataset
import torch

# Create sample data
data = create_sample_dataset()
sequences = data['sequences'][:5]

# Create dummy structures for testing
structures = []
for seq in sequences:
    coords = torch.randn(len(seq), 3, 3).numpy()
    from rna_model.data import RNAStructure
    structures.append(RNAStructure(seq, coords, [], [], 'A'))

# Run benchmark
benchmark_model(None, sequences, structures)
"

# Prediction
predict:
	python examples/basic_usage.py

# Benchmarking
benchmark:
	python -c "
import time
import torch
from rna_model import RNAFoldingPipeline, PipelineConfig

# Initialize pipeline
config = PipelineConfig(device='cuda' if torch.cuda.is_available() else 'cpu')
pipeline = RNAFoldingPipeline(config)
pipeline.enable_competition_mode()

# Test sequences
sequences = ['GGGAAAUCC' * 10, 'GCCUUGGCAAC' * 10]  # Longer sequences

print('Running performance benchmark...')
start_time = time.time()

results = pipeline.predict_batch(sequences)

end_time = time.time()
total_time = end_time - start_time

print(f'Processed {len(sequences)} sequences in {total_time:.2f}s')
print(f'Average time per sequence: {total_time/len(sequences):.2f}s')
print(f'Sequences per hour: {len(sequences)/total_time*3600:.1f}')

# Memory usage
if torch.cuda.is_available():
    print(f'GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB')
    print(f'GPU memory cached: {torch.cuda.memory_reserved()/1024**3:.2f}GB')
"

# Documentation
docs:
	cd docs && make html

# Development workflow
dev-setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation."

# Quick test
quick-test:
	python -c "
from rna_model import RNAFoldingPipeline, PipelineConfig
config = PipelineConfig(device='cpu')
pipeline = RNAFoldingPipeline(config)
result = pipeline.predict_single_sequence('GGGAAAUCC')
print('Quick test passed! Generated coordinates shape:', result['coordinates'].shape)
"

# Competition simulation
competition-sim:
	python -c "
import time
from rna_model import RNAFoldingPipeline, PipelineConfig

# Simulate competition environment
config = PipelineConfig(device='cuda' if torch.cuda.is_available() else 'cpu')
pipeline = RNAFoldingPipeline(config)
pipeline.enable_competition_mode()

# Simulate test set (200 sequences, avg length 150)
import random
nucleotides = ['A', 'U', 'G', 'C']
test_sequences = [''.join(random.choices(nucleotides, k=150)) for _ in range(200)]

print(f'Simulating competition with {len(test_sequences)} sequences...')
print(f'Total sequence length: {sum(len(s) for s in test_sequences)}')

start_time = time.time()
results = pipeline.predict_batch(test_sequences)
end_time = time.time()

total_time = end_time - start_time
print(f'Total time: {total_time/60:.1f} minutes')
print(f'Time per sequence: {total_time/len(test_sequences):.2f}s')
print(f'Within 8-hour limit: {\"Yes\" if total_time < 8*3600 else \"No\"}')

# Check results
successful = sum(1 for r in results if 'coordinates' in r)
print(f'Successful predictions: {successful}/{len(test_sequences)}')
"

# Model size analysis
analyze-model:
	python -c "
import torch
from rna_model import RNAFoldingPipeline, PipelineConfig

config = PipelineConfig()
pipeline = RNAFoldingPipeline(config)

# Count parameters
total_params = sum(p.numel() for p in pipeline.model.parameters())
trainable_params = sum(p.numel() for p in pipeline.model.parameters() if p.requires_grad)

print(f'Model parameter count:')
print(f'  Total: {total_params:,}')
print(f'  Trainable: {trainable_params:,}')
print(f'  Model size: {total_params * 4 / 1024**2:.1f} MB (float32)')

# Component breakdown
components = {
    'Language Model': pipeline.model.language_model,
    'SS Predictor': pipeline.model.ss_predictor,
    'Structure Encoder': pipeline.model.structure_encoder,
    'Geometry Module': pipeline.model.geometry_module,
}

for name, component in components.items():
    params = sum(p.numel() for p in component.parameters())
    print(f'  {name}: {params:,} ({params/total_params*100:.1f}%)')
"

# Export for competition
export-competition:
	python -c "
import torch
from rna_model import RNAFoldingPipeline, PipelineConfig
import numpy as np

# Initialize pipeline
config = PipelineConfig()
pipeline = RNAFoldingPipeline(config)

# Save model weights
pipeline.save_model('competition_model.pth')
print('Model saved to competition_model.pth')

# Create sample submission
sequences = ['GGGAAAUCC', 'GCCUUGGCAAC']
results = pipeline.predict_batch(sequences)

# Format submission
all_coords = []
for result in results:
    all_coords.append(result['coordinates'])

submission_coords = np.concatenate(all_coords, axis=0)

# Save submission
np.save('submission_coordinates.npy', submission_coords)
print(f'Submission coordinates saved: {submission_coords.shape}')
"

# Continuous integration
ci: lint test

# All checks
check-all: format lint test
	@echo "All checks passed!"
