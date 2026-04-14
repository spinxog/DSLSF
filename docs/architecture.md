# System Architecture

This document describes the high-level architecture and design principles of the RNA 3D folding pipeline.

## Overview

The RNA 3D folding pipeline is a modular, end-to-end system that predicts RNA 3D structures from primary sequences using deep learning. The architecture follows the successful patterns established in protein folding while being specifically adapted for RNA molecules.

## Architecture Diagram

```
Input Sequence
       |
       v
+-----------------+
| Language Model  |  (BERT-style transformer)
+-----------------+
       |
       v
+---------------------+
| Secondary Structure |  (Top-k hypotheses)
|    Predictor       |
+---------------------+
       |
       v
+-------------------+
| Structure Encoder |  (Sparse attention)
+-------------------+
       |
       v
+-------------------+
| Geometry Module   |  (SE(3)-equivariant)
+-------------------+
       |
       v
+-------------+
|   Sampler   |  (Diverse decoys)
+-------------+
       |
       v
+-------------+
|  Refiner    |  (Internal coordinates)
+-------------+
       |
       v
+-----------------+
| 5 Decoys (N×3) |
+-----------------+
```

## Core Components

### 1. Language Model (LM)

**Purpose**: Extract sequence embeddings and predict long-range contacts

**Architecture**:
- BERT-style transformer with 12 layers
- 512-dimensional embeddings
- 8 attention heads
- Masked span language modeling objective
- Auxiliary contact prediction head

**Key Features**:
- Cached embeddings for competition deployment
- Thread-safe cache management
- Input validation and error handling

**Input**: RNA sequence tokens (batch_size × seq_len)
**Output**: Sequence embeddings (batch_size × seq_len × 512)

### 2. Secondary Structure Predictor

**Purpose**: Generate multiple secondary structure hypotheses

**Architecture**:
- Pairwise attention mechanism
- Top-k hypothesis generation (k=3)
- Pseudoknot-aware prediction
- Confidence estimation

**Key Features**:
- Contact probability prediction
- Pseudoknot type classification
- Hypothesis diversity sampling

**Input**: Sequence embeddings
**Output**: k secondary structure hypotheses with confidence scores

### 3. Structure Encoder

**Purpose**: Process sequence representations for geometry prediction

**Architecture**:
- Sparse attention for long sequences
- Window-based attention (window_size=64)
- Efficient memory management
- Periodic cache clearing

**Key Features**:
- O(n×w) complexity instead of O(n²)
- Memory-efficient for sequences >200nt
- Thread-safe operations

**Input**: Sequence embeddings + secondary structure
**Output**: Encoded representations (batch_size × seq_len × d_model)

### 4. Geometry Module

**Purpose**: Predict 3D coordinates using SE(3)-equivariant operations

**Architecture**:
- Invariant Point Attention (IPA)
- Multi-part rigid body representation
- Quaternion-based rotations
- Frame-aligned point error (FAPE)

**Key Features**:
- Numerical stability in quaternion operations
- Distance, angle, and torsion prediction
- Sugar pucker conformation prediction

**Input**: Encoded representations
**Output**: 3D coordinates + geometric features

### 5. Sampler

**Purpose**: Generate diverse decoy structures

**Architecture**:
- Multiple sampling strategies
- MC-dropout for uncertainty
- SS hypothesis switching
- MSA subsampling

**Key Features**:
- 20 decoys generated per sequence
- Clustering for diversity
- Adaptive budgeting based on complexity

**Input**: Predicted coordinates
**Output**: 20 diverse decoy structures

### 6. Refiner

**Purpose**: Improve coordinate quality through optimization

**Architecture**:
- Internal-coordinate optimization
- Bond length/angle constraints
- Steric clash resolution
- Fast refiner for competition

**Key Features**:
- Physics-based constraints
- Energy minimization
- Memory-efficient implementation

**Input**: Decoy coordinates
**Output**: Refined coordinates

## Data Flow

### Training Pipeline

```
Training Data
     |
     v
+--------------+    +-------------------+
| Tokenization | -> | Data Augmentation |
+--------------+    +-------------------+
     |
     v
+--------------+    +-------------------+
| Language Model| -> | Contact Prediction |
+--------------+    +-------------------+
     |
     v
+-------------------+    +-----------------+
| Secondary Structure| -> | Hypothesis Sampling |
+-------------------+    +-----------------+
     |
     v
+----------------+    +-----------------+
| Structure Encoder| -> | Sparse Attention |
+----------------+    +-----------------+
     |
     v
+----------------+    +-----------------+
| Geometry Module | -> | IPA Operations |
+----------------+    +-----------------+
     |
     v
+-------------+    +----------------+
|   Loss      | <- | Multi-task Loss |
+-------------+    +----------------+
```

### Inference Pipeline

```
Input Sequence
     |
     v
+--------------+    +-----------------+
| Tokenization | -> | Cached Embeddings |
+--------------+    +-----------------+
     |
     v
+----------------+    +-----------------+
| Language Model| -> | Embedding Lookup |
+----------------+    +-----------------+
     |
     v
+-------------------+    +-----------------+
| Secondary Structure| -> | Top-k Hypotheses |
+-------------------+    +-----------------+
     |
     v
+----------------+    +-----------------+
| Structure Encoder| -> | Sparse Processing |
+----------------+    +-----------------+
     |
     v
+----------------+    +-----------------+
| Geometry Module | -> | Coordinate Prediction |
+----------------+    +-----------------+
     |
     v
+-------------+    +-----------------+
|   Sampler   | <- | Diverse Sampling |
+-------------+    +-----------------+
     |
     v
+-------------+    +-----------------+
|  Refiner    | <- | Fast Optimization |
+-------------+    +-----------------+
     |
     v
+-----------------+
| 5 Best Decoys |
+-----------------+
```

## Design Principles

### 1. Modularity

Each component is independently designed and can be:
- Replaced with alternative implementations
- Trained separately
- Optimized for specific hardware

### 2. Scalability

The architecture supports:
- Long sequences (>200nt) through sparse attention
- Batch processing with memory management
- Distributed training across multiple GPUs

### 3. Competition Optimization

Special features for competition deployment:
- Cached embeddings to avoid LM computation
- Adaptive budgeting based on sequence complexity
- Fast refiner for time constraints
- Memory-efficient implementations

### 4. Numerical Stability

Robust mathematical operations:
- Quaternion normalization
- Matrix conditioning
- Gradient clipping
- Mixed precision support

## Memory Management

### GPU Memory Optimization

1. **Sparse Attention**: Reduces O(n²) to O(n×w) complexity
2. **Periodic Cache Clearing**: Prevents memory buildup
3. **Tensor Cleanup**: Explicit deletion of intermediate tensors
4. **Context Managers**: Automatic memory cleanup

### CPU Memory Optimization

1. **Efficient Data Loading**: Streaming data processing
2. **Cache Management**: LRU cache with size limits
3. **Batch Processing**: Optimal batch sizes for hardware

## Error Handling

### Fail-Fast Strategy

The pipeline follows a fail-fast approach:
- No fallback coordinate generation
- Clear error messages with context
- Graceful degradation for non-critical failures
- Comprehensive logging for debugging

### Validation Layers

Input validation at each stage:
- Sequence length and content validation
- Tensor shape and type checking
- Numerical stability checks
- Device compatibility verification

## Performance Characteristics

### Inference Time

- **Short sequences** (<100nt): ~30-60 seconds
- **Medium sequences** (100-200nt): ~60-120 seconds  
- **Long sequences** (>200nt): ~120-144 seconds

### Memory Usage

- **Base model**: ~2-4GB GPU memory
- **With sparse attention**: ~4-6GB GPU memory
- **Batch processing**: Scales linearly with batch size

### Accuracy Metrics

- **TM-score**: Competitive with state-of-the-art methods
- **RMSD**: Typically <3Å for well-behaved sequences
- **GDT-TS**: >0.5 for most test cases

## Extensibility

### Plugin Architecture

The system supports:
- Custom geometry modules
- Alternative sampling strategies
- Different refinement methods
- Additional loss functions

### Configuration Management

Flexible configuration system:
- JSON-based configuration files
- Runtime parameter updates
- Environment-specific settings
- Validation of configuration parameters

## Deployment Considerations

### Competition Deployment

- **Time constraint**: 8 hours for 200 sequences
- **Memory constraint**: <8GB GPU memory
- **Bundle size**: <15GB compressed
- **Reproducibility**: Deterministic with fixed seeds

### Production Deployment

- **Scalability**: Multi-GPU and distributed training
- **Monitoring**: Comprehensive logging and metrics
- **Reliability**: Robust error handling and recovery
- **Maintainability**: Clean, documented codebase

## Future Extensions

### Planned Enhancements

1. **Template Integration**: Homology modeling capabilities
2. **MSA Processing**: Evolutionary information incorporation
3. **Energy Functions**: Physics-based scoring
4. **Web Interface**: User-friendly prediction service

### Research Directions

1. **RNA-Protein Complexes**: Multi-chain prediction
2. **RNA Dynamics**: Time-dependent structure prediction
3. **Design Applications**: Inverse folding for RNA design
4. **Clinical Applications**: Disease-associated variant prediction

This architecture provides a solid foundation for RNA 3D structure prediction while maintaining flexibility for future research and development.
