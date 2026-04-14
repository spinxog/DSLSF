"""Configuration management for RNA 3D folding pipeline."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
from pathlib import Path


@dataclass
class GlobalConfig:
    """Global configuration constants and settings."""
    
    # Model architecture constants
    DEFAULT_D_MODEL: int = 512
    DEFAULT_N_LAYERS: int = 12
    DEFAULT_N_HEADS: int = 8
    DEFAULT_D_FF: int = 2048
    DEFAULT_MAX_SEQ_LEN: int = 2048
    
    # Training constants
    DEFAULT_BATCH_SIZE: int = 8
    DEFAULT_LEARNING_RATE: float = 1e-4
    DEFAULT_WEIGHT_DECAY: float = 1e-5
    DEFAULT_MAX_STEPS: int = 100000
    
    # Competition constants
    DEFAULT_MAX_SEQUENCE_LENGTH: int = 512
    DEFAULT_INFERENCE_TIMEOUT: float = 144.0  # seconds
    DEFAULT_TIME_LIMIT_HOURS: float = 8.0
    
    # Memory management constants
    MEMORY_CLEANUP_INTERVAL: int = 10
    MAX_CACHE_SIZE: int = 1000
    GPU_MEMORY_THRESHOLD: float = 40.0  # GB
    
    # Validation constants
    MIN_SEQUENCE_LENGTH: int = 1
    MAX_SEQUENCE_LENGTH_HARD: int = 2048
    VALID_NUCLEOTIDES: str = "AUGCaugcNn"
    
    # Performance constants
    SPARSE_ATTENTION_THRESHOLD: int = 128
    WINDOW_SIZE: int = 64
    COMPLEXITY_THRESHOLD: float = 0.7
    
    # Security constants
    MAX_FILE_SIZE_MB: int = 100
    ALLOWED_FILE_EXTENSIONS: list = field(default_factory=lambda: ['.json', '.txt', '.csv', '.fasta', '.fa', '.fna'])
    SUSPICIOUS_PATTERNS: list = field(default_factory=lambda: ['..', '\\\\', '//'])
    
    # Logging constants
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_LEVEL: str = 'INFO'
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'GlobalConfig':
        """Load configuration from JSON file."""
        if not config_path.exists():
            return cls()
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def to_file(self, config_path: Path) -> None:
        """Save configuration to JSON file."""
        config_dict = self.__dict__.copy()
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")


# Default global configuration instance
DEFAULT_CONFIG = GlobalConfig()


def get_config(config_path: Optional[Path] = None) -> GlobalConfig:
    """Get configuration, loading from file if provided."""
    if config_path and config_path.exists():
        return GlobalConfig.from_file(config_path)
    return DEFAULT_CONFIG


def validate_config(config: GlobalConfig) -> bool:
    """Validate configuration values."""
    errors = []
    
    # Validate sequence length constraints
    if config.MIN_SEQUENCE_LENGTH >= config.MAX_SEQUENCE_LENGTH_HARD:
        errors.append("MIN_SEQUENCE_LENGTH must be less than MAX_SEQUENCE_LENGTH_HARD")
    
    if config.DEFAULT_MAX_SEQ_LEN > config.MAX_SEQUENCE_LENGTH_HARD:
        errors.append("DEFAULT_MAX_SEQ_LEN cannot exceed MAX_SEQUENCE_LENGTH_HARD")
    
    # Validate model architecture
    if config.DEFAULT_D_MODEL % config.DEFAULT_N_HEADS != 0:
        errors.append("d_model must be divisible by n_heads")
    
    # Validate training parameters
    if config.DEFAULT_LEARNING_RATE <= 0:
        errors.append("learning_rate must be positive")
    
    if config.DEFAULT_BATCH_SIZE <= 0:
        errors.append("batch_size must be positive")
    
    # Validate memory thresholds
    if config.GPU_MEMORY_THRESHOLD <= 0:
        errors.append("GPU memory threshold must be positive")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True
