"""Constants for RNA 3D folding pipeline.

This module contains all the magic numbers and hard-coded values
used throughout the codebase, making them easily configurable
and documented.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class BiologicalConstants:
    """Biological and chemical constants for RNA structures."""
    
    # RNA nucleotides
    VALID_NUCLEOTIDES: str = "AUGCaugcNn"
    VALID_NUCLEOTIDES_UPPER: str = "AUGCN"
    STANDARD_RNA_RESIDUES: set = frozenset({'A', 'U', 'G', 'C'})
    
    # Standard RNA atoms (simplified representation)
    STANDARD_ATOMS: list = None  # Will be set in __post_init__
    N_ATOMS_PER_RESIDUE: int = 3  # P, C4', N1 (simplified)
    
    # Bond lengths (Angstroms)
    P_O5_BOND_LENGTH: float = 1.5
    O5_C5_BOND_LENGTH: float = 1.43
    C5_C4_BOND_LENGTH: float = 1.5
    C4_C3_BOND_LENGTH: float = 1.5
    C3_O3_BOND_LENGTH: float = 1.43
    
    # Bond angles (radians)
    RNA_BOND_ANGLE: float = 109.5 * (3.14159 / 180.0)  # tetrahedral
    
    # Physical constants
    AVOGADRO_NUMBER: float = 6.02214076e23
    BOLTZMANN_CONSTANT: float = 1.380649e-23
    
    def __post_init__(self):
        if self.STANDARD_ATOMS is None:
            self.STANDARD_ATOMS = [
                "P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'", 
                "N1", "N3", "C2", "C4", "C5", "C6", "O2", "O4", 
                "N6", "N2", "O6", "N7", "N9"
            ]


@dataclass
class GeometryConstants:
    """Constants for geometric computations."""
    
    # Distance thresholds
    CONTACT_DISTANCE_THRESHOLD: float = 8.0  # Angstroms
    CLASH_DISTANCE_THRESHOLD: float = 2.0   # Angstroms
    CLOSE_ATOM_THRESHOLD: float = 0.1       # Angstroms
    ZERO_VECTOR_THRESHOLD: float = 1e-10    # Angstroms
    
    # Coordinate validation ranges
    MIN_COORDINATE_VALUE: float = -1000.0
    MAX_COORDINATE_VALUE: float = 1000.0
    
    # Structure validation
    MAX_ATOMS_PER_RESIDUE: int = 50
    MIN_REASONABLE_STRUCTURE_SPAN: float = 1.0   # Angstroms
    MAX_REASONABLE_STRUCTURE_SPAN: float = 1000.0  # Angstroms
    PLANARITY_DEVIATION_THRESHOLD: float = 5.0  # Angstroms
    LONG_BOND_THRESHOLD: float = 10.0         # Angstroms
    
    # TM-score constants
    TM_SCORE_D0_SHORT: float = 0.5  # For sequences < 15 residues
    TM_SCORE_MIN_D0: float = 0.5   # Minimum d0 for all sequences
    
    # Numerical stability
    QUATERNION_NORMALIZATION_EPSILON: float = 1e-8
    MATRIX_VALIDATION_TOLERANCE: float = 1e-5
    ROTATION_MATRIX_MAX_VALUE: float = 10.0
    
    # FAPE loss
    FAPE_CLAMP_DISTANCE: float = 10.0


@dataclass
class ComputationalConstants:
    """Constants for computational operations."""
    
    # Memory management
    DEFAULT_CACHE_SIZE_MB: int = 100
    DEFAULT_CACHE_ENTRIES: int = 500
    CACHE_CLEANUP_THRESHOLD: float = 0.8
    CACHE_TARGET_SIZE_RATIO: float = 0.5
    
    # Chunk sizes for memory-efficient operations
    DEFAULT_CHUNK_SIZE: int = 1000
    DISTANCE_MATRIX_CHUNK_SIZE: int = 500
    CONTACT_MAP_CHUNK_SIZE: int = 1000
    
    # Performance thresholds
    SMALL_SYSTEM_THRESHOLD: int = 500
    MEDIUM_SYSTEM_THRESHOLD: int = 1000
    LARGE_SYSTEM_THRESHOLD: int = 2000
    
    # Cleanup and maintenance
    CLEANUP_INTERVAL: int = 100
    IO_DELAY: float = 1.0
    
    # Cache hit rate thresholds
    QUATERNION_CACHE_BATCH_THRESHOLD: int = 2000
    
    # File operations
    MAX_FILE_SIZE_MB: int = 100
    MAX_PATH_LENGTH: int = 512
    FILE_LOCK_TIMEOUT: float = 30.0
    
    # Retry mechanisms
    DEFAULT_MAX_RETRIES: int = 3
    DEFAULT_RETRY_DELAY: float = 1.0
    RETRY_BACKOFF_EXPONENT: float = 2.0


@dataclass
class ValidationConstants:
    """Constants for data validation."""
    
    # Sequence validation
    MIN_SEQUENCE_LENGTH: int = 5
    MAX_SEQUENCE_LENGTH: int = 2048
    MAX_HOMOPOLYMER_LENGTH: int = 10
    
    # GC content ranges
    MIN_GC_CONTENT: float = 0.1
    MAX_GC_CONTENT: float = 0.9
    MAX_N_PROPORTION: float = 0.5
    
    # Control characters
    MIN_CONTROL_CHAR: int = 32
    DELETE_CHAR: int = 127


@dataclass
class ModelConstants:
    """Constants for model architecture and training."""
    
    # Default model parameters
    DEFAULT_D_MODEL: int = 512
    DEFAULT_N_LAYERS: int = 12
    DEFAULT_N_HEADS: int = 8
    DEFAULT_D_FF: int = 2048
    DEFAULT_MAX_SEQ_LEN: int = 2048
    
    # Training parameters
    DEFAULT_BATCH_SIZE: int = 8
    DEFAULT_LEARNING_RATE: float = 1e-4
    DEFAULT_WEIGHT_DECAY: float = 1e-5
    DEFAULT_MAX_STEPS: int = 100000
    
    # Loss weights
    LM_LOSS_WEIGHT: float = 1.0
    SS_LOSS_WEIGHT: float = 1.0
    GEOMETRY_LOSS_WEIGHT: float = 2.0
    FAPE_LOSS_WEIGHT: float = 1.0
    PUCKER_LOSS_WEIGHT: float = 0.3
    
    # Regularization
    DEFAULT_DROPOUT: float = 0.1
    DEFAULT_GRADIENT_CLIP_NORM: float = 1.0
    
    # Binning
    DEFAULT_DISTANCE_BINS: int = 64
    DEFAULT_ANGLE_BINS: int = 36
    DEFAULT_TORSION_BINS: int = 72
    MAX_DISTANCE: float = 20.0
    
    # Attention
    DEFAULT_ATTENTION_SCALE: float = 1.0
    POINT_ATTENTION_DIM: int = 16


@dataclass
class CompetitionConstants:
    """Constants for competition settings."""
    
    # Time limits
    DEFAULT_MAX_SEQUENCE_LENGTH: int = 512
    DEFAULT_INFERENCE_TIMEOUT: float = 144.0  # seconds
    DEFAULT_TIME_LIMIT_HOURS: float = 8.0
    
    # Memory management
    MEMORY_CLEANUP_INTERVAL: int = 10
    MAX_CACHE_SIZE: int = 1000
    GPU_MEMORY_THRESHOLD: float = 40.0  # GB


@dataclass
class LoggingConstants:
    """Constants for logging and monitoring."""
    
    # Log format
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DEFAULT_LOG_LEVEL: str = 'INFO'
    
    # Checkpoint management
    DEFAULT_MAX_CHECKPOINTS: int = 5
    MIN_DISK_SPACE_GB: float = 1.0
    CHECKPOINT_CLEANUP_RATIO: float = 0.5


# Create singleton instances
BIOLOGICAL = BiologicalConstants()
GEOMETRY = GeometryConstants()
COMPUTATIONAL = ComputationalConstants()
VALIDATION = ValidationConstants()
MODEL = ModelConstants()
COMPETITION = CompetitionConstants()
LOGGING = LoggingConstants()


# Convenience dictionaries for backward compatibility
BIOLOGICAL_CONSTANTS: Dict[str, Any] = {
    'VALID_NUCLEOTIDES': BIOLOGICAL.VALID_NUCLEOTIDES,
    'STANDARD_ATOMS': BIOLOGICAL.STANDARD_ATOMS,
    'N_ATOMS_PER_RESIDUE': BIOLOGICAL.N_ATOMS_PER_RESIDUE,
}

GEOMETRY_CONSTANTS: Dict[str, Any] = {
    'CONTACT_DISTANCE_THRESHOLD': GEOMETRY.CONTACT_DISTANCE_THRESHOLD,
    'CLASH_DISTANCE_THRESHOLD': GEOMETRY.CLASH_DISTANCE_THRESHOLD,
    'MIN_COORDINATE_VALUE': GEOMETRY.MIN_COORDINATE_VALUE,
    'MAX_COORDINATE_VALUE': GEOMETRY.MAX_COORDINATE_VALUE,
}

COMPUTATIONAL_CONSTANTS: Dict[str, Any] = {
    'DEFAULT_CACHE_SIZE_MB': COMPUTATIONAL.DEFAULT_CACHE_SIZE_MB,
    'DEFAULT_CACHE_ENTRIES': COMPUTATIONAL.DEFAULT_CACHE_ENTRIES,
    'DEFAULT_CHUNK_SIZE': COMPUTATIONAL.DEFAULT_CHUNK_SIZE,
}

MODEL_CONSTANTS: Dict[str, Any] = {
    'DEFAULT_D_MODEL': MODEL.DEFAULT_D_MODEL,
    'DEFAULT_N_LAYERS': MODEL.DEFAULT_N_LAYERS,
    'DEFAULT_N_HEADS': MODEL.DEFAULT_N_HEADS,
    'DEFAULT_BATCH_SIZE': MODEL.DEFAULT_BATCH_SIZE,
}


def get_all_constants() -> Dict[str, Dict[str, Any]]:
    """Get all constants organized by category."""
    return {
        'biological': BIOLOGICAL_CONSTANTS,
        'geometry': GEOMETRY_CONSTANTS,
        'computational': COMPUTATIONAL_CONSTANTS,
        'model': MODEL_CONSTANTS,
        'validation': {
            'MIN_SEQUENCE_LENGTH': VALIDATION.MIN_SEQUENCE_LENGTH,
            'MAX_SEQUENCE_LENGTH': VALIDATION.MAX_SEQUENCE_LENGTH,
        },
        'competition': {
            'DEFAULT_MAX_SEQUENCE_LENGTH': COMPETITION.DEFAULT_MAX_SEQUENCE_LENGTH,
            'DEFAULT_INFERENCE_TIMEOUT': COMPETITION.DEFAULT_INFERENCE_TIMEOUT,
        },
        'logging': {
            'LOG_FORMAT': LOGGING.LOG_FORMAT,
            'DEFAULT_LOG_LEVEL': LOGGING.DEFAULT_LOG_LEVEL,
        }
    }


def validate_constants() -> bool:
    """Validate that all constants are reasonable."""
    errors = []
    
    # Check biological constants
    if GEOMETRY.CONTACT_DISTANCE_THRESHOLD <= 0:
        errors.append("CONTACT_DISTANCE_THRESHOLD must be positive")
    
    if GEOMETRY.CLASH_DISTANCE_THRESHOLD >= GEOMETRY.CONTACT_DISTANCE_THRESHOLD:
        errors.append("CLASH_DISTANCE_THRESHOLD must be less than CONTACT_DISTANCE_THRESHOLD")
    
    # Check computational constants
    if COMPUTATIONAL.DEFAULT_CACHE_SIZE_MB <= 0:
        errors.append("DEFAULT_CACHE_SIZE_MB must be positive")
    
    if COMPUTATIONAL.DEFAULT_CHUNK_SIZE <= 0:
        errors.append("DEFAULT_CHUNK_SIZE must be positive")
    
    # Check model constants
    if MODEL.DEFAULT_D_MODEL <= 0:
        errors.append("DEFAULT_D_MODEL must be positive")
    
    if MODEL.DEFAULT_D_MODEL % MODEL.DEFAULT_N_HEADS != 0:
        errors.append("DEFAULT_D_MODEL must be divisible by DEFAULT_N_HEADS")
    
    if errors:
        raise ValueError("Invalid constants:\n" + "\n".join(errors))
    
    return True


# Validate constants on import
validate_constants()