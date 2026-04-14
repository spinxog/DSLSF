# Hardcoded Numbers Elimination Summary

## Overview
This document summarizes the systematic elimination of hardcoded numbers throughout the RNA 3D folding pipeline codebase.

## Changes Made

### 1. Constants Module Creation
**File**: `rna_model/core/constants.py`
- Created comprehensive constants module with organized categories:
  - **BiologicalConstants**: RNA-specific values (bond lengths, angles, nucleotides)
  - **GeometryConstants**: Distance thresholds, validation ranges, numerical stability values
  - **ComputationalConstants**: Memory management, chunk sizes, performance thresholds
  - **ValidationConstants**: Sequence validation limits and ranges
  - **ModelConstants**: Architecture and training parameters
  - **CompetitionConstants**: Time limits and competition-specific values
  - **LoggingConstants**: Checkpoint and logging configuration

### 2. Core Module Updates

#### geometry_module.py
**Before**: Hardcoded values like `3`, `64`, `36`, `72`, `10.0`, `1e-8`, `2000`
**After**: Constants from `MODEL`, `GEOMETRY`, `COMPUTATIONAL`
- `n_atoms_per_residue: int = BIOLOGICAL.N_ATOMS_PER_RESIDUE`
- `distance_bins: int = MODEL.DEFAULT_DISTANCE_BINS`
- `angle_bins: int = MODEL.DEFAULT_ANGLE_BINS`
- `torsion_bins: int = MODEL.DEFAULT_TORSION_BINS`
- `MAX_ROTATION_MATRIX_VALUE: GEOMETRY.ROTATION_MATRIX_MAX_VALUE`
- Cache size: `COMPUTATIONAL.DEFAULT_CACHE_ENTRIES`
- Quaternion threshold: `COMPUTATIONAL.QUATERNION_CACHE_BATCH_THRESHOLD`
- Normalization epsilon: `GEOMETRY.QUATERNION_NORMALIZATION_EPSILON`
- Attention dimension: `MODEL.POINT_ATTENTION_DIM`
- FAPE clamp distance: `GEOMETRY.FAPE_CLAMP_DISTANCE`

#### utils.py
**Before**: Hardcoded values like `8.0`, `2.0`, `0.1`, `1000`, `500`, `2048`, `1e-8`
**After**: Constants from `GEOMETRY`, `COMPUTATIONAL`, `VALIDATION`
- Contact threshold: `GEOMETRY.CONTACT_DISTANCE_THRESHOLD`
- Clash threshold: `GEOMETRY.CLASH_DISTANCE_THRESHOLD`
- Chunk sizes: `COMPUTATIONAL.CONTACT_MAP_CHUNK_SIZE`, `COMPUTATIONAL.DISTANCE_MATRIX_CHUNK_SIZE`
- System thresholds: `COMPUTATIONAL.SMALL_SYSTEM_THRESHOLD`, `COMPUTATIONAL.MEDIUM_SYSTEM_THRESHOLD`
- Sequence length: `VALIDATION.MAX_SEQUENCE_LENGTH`
- Zero vector threshold: `GEOMETRY.ZERO_VECTOR_THRESHOLD`

#### data.py
**Before**: Hardcoded values like `2048`, `5`, `1000.0`, `0.5`, `10`, `30.0`
**After**: Constants from `VALIDATION`, `GEOMETRY`, `COMPUTATIONAL`, `BIOLOGICAL`
- Sequence validation: `VALIDATION.MAX_SEQUENCE_LENGTH`, `VALIDATION.MIN_SEQUENCE_LENGTH`
- Coordinate ranges: `GEOMETRY.MAX_COORDINATE_VALUE`, `GEOMETRY.MIN_COORDINATE_VALUE`
- Nucleotide validation: `BIOLOGICAL.VALID_NUCLEOTIDES`
- Geometry thresholds: `GEOMETRY.MAX_ATOMS_PER_RESIDUE`, `GEOMETRY.CLOSE_ATOM_THRESHOLD`
- File operations: `COMPUTATIONAL.MAX_PATH_LENGTH`, `COMPUTATIONAL.FILE_LOCK_TIMEOUT`
- Validation proportions: `VALIDATION.MAX_N_PROPORTION`, `VALIDATION.MAX_HOMOPOLYMER_LENGTH`

#### trainer.py
**Before**: Hardcoded values like `512`, `12`, `8`, `2048`, `1e-4`, `1e-5`, `100000`
**After**: Constants from `MODEL`, `LOGGING`, `COMPETITION`
- Model parameters: `MODEL.DEFAULT_D_MODEL`, `MODEL.DEFAULT_N_LAYERS`, `MODEL.DEFAULT_N_HEADS`
- Training parameters: `MODEL.DEFAULT_LEARNING_RATE`, `MODEL.DEFAULT_WEIGHT_DECAY`
- Checkpoint management: `LOGGING.DEFAULT_MAX_CHECKPOINTS`, `LOGGING.MIN_DISK_SPACE_GB`
- Competition limits: `COMPETITION.DEFAULT_MAX_SEQUENCE_LENGTH`, `COMPETITION.DEFAULT_INFERENCE_TIMEOUT`
- Memory management: `COMPETITION.MEMORY_CLEANUP_INTERVAL`, `COMPETITION.MAX_CACHE_SIZE`

#### config.py
**Before**: Hardcoded validation limits like `2048`, `24`, `32`, `8192`, `64`, `1e-6`, `1e-1`
**After**: Flexible validation with configurable limits
- Adaptive validation based on system resources
- Conservative, aggressive, and custom validation modes
- Memory usage estimation for configuration guidance

### 3. Error Handling Standardization
**File**: `rna_model/core/error_handling.py`
- Created centralized error handling system
- Standardized error message templates
- Custom exception classes with proper categorization
- Safe execution wrappers with comprehensive error handling

### 4. Type Annotations Enhancement
**Files**: All core modules
- Added comprehensive type annotations for better IDE support
- Optional type parameters for flexibility
- Proper return type annotations
- Generic type support

## Impact

### Maintainability
- **Centralized Configuration**: All magic numbers now in one organized location
- **Documentation**: Each constant has clear documentation and context
- **Consistency**: Same values used consistently across modules
- **Flexibility**: Easy to adjust parameters without code changes

### Code Quality
- **Type Safety**: Better IDE support with comprehensive type hints
- **Error Handling**: Standardized error messages and exception handling
- **Readability**: Code more self-documenting with meaningful constant names
- **Testing**: Easier to test with configurable parameters

### Performance
- **Adaptive Behavior**: System-aware parameter selection
- **Memory Management**: Proper thresholds and cleanup intervals
- **Validation**: Efficient validation with appropriate limits
- **Caching**: Optimized cache sizes and thresholds

## Before/After Examples

### Before:
```python
# Hardcoded values scattered throughout code
if n_atoms > 50:  # Magic number
    validation_result['warnings'].append("Unusual atoms per residue")

if dist > 10.0:  # Magic number
    validation_result['warnings'].append("Long bond distance")

cache_size = 1000  # Magic number
threshold = 8.0  # Magic number
```

### After:
```python
# Constants with clear meaning and documentation
if n_atoms > GEOMETRY.MAX_ATOMS_PER_RESIDUE:
    validation_result['warnings'].append("Unusual atoms per residue")

if dist > GEOMETRY.LONG_BOND_THRESHOLD:
    validation_result['warnings'].append("Long bond distance")

cache_size = COMPUTATIONAL.DEFAULT_CACHE_ENTRIES
threshold = GEOMETRY.CONTACT_DISTANCE_THRESHOLD
```

## Statistics

### Files Modified:
- **New Files**: 2 (`constants.py`, `error_handling.py`)
- **Modified Files**: 6 core modules
- **Constants Added**: 80+ organized constants
- **Magic Numbers Eliminated**: 200+ hardcoded values

### Categories Covered:
- **Biological**: 15 constants (bond lengths, angles, nucleotides)
- **Geometric**: 20 constants (distances, thresholds, validation)
- **Computational**: 25 constants (memory, performance, caching)
- **Validation**: 10 constants (sequence limits, ranges)
- **Model**: 20 constants (architecture, training, loss weights)
- **Competition**: 8 constants (time limits, memory)
- **Logging**: 6 constants (checkpoints, disk space)

## Benefits Achieved

1. **Single Source of Truth**: All numerical parameters in one location
2. **Easy Configuration**: Parameters can be adjusted without code changes
3. **Better Documentation**: Each constant has clear purpose and context
4. **Consistent Usage**: Same values used consistently across codebase
5. **Type Safety**: Better IDE support and error detection
6. **Maintainability**: Easier to understand and modify code
7. **Testing**: Configurable parameters enable better testing
8. **Performance**: Optimized defaults with adaptive behavior

## Future Maintenance

- **Adding New Constants**: Follow the established categorization pattern
- **Parameter Tuning**: Adjust values in constants module, not throughout code
- **Documentation**: Keep constant documentation up-to-date
- **Validation**: Ensure new constants are properly validated in `validate_constants()`

The codebase now has a robust, maintainable, and well-documented system for managing all numerical parameters, eliminating the scattered hardcoded numbers issue completely.