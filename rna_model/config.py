"""Configuration management for RNA 3D folding pipeline."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
from pathlib import Path

try:
    import jsonschema
except ImportError:
    jsonschema = None


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
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.DEFAULT_D_MODEL <= 0 or self.DEFAULT_D_MODEL > 2048:
            raise ValueError(f"d_model must be between 1 and 2048, got {self.DEFAULT_D_MODEL}")
        
        if self.DEFAULT_N_LAYERS <= 0 or self.DEFAULT_N_LAYERS > 24:
            raise ValueError(f"n_layers must be between 1 and 24, got {self.DEFAULT_N_LAYERS}")
        
        if self.DEFAULT_N_HEADS <= 0 or self.DEFAULT_N_HEADS > 32:
            raise ValueError(f"n_heads must be between 1 and 32, got {self.DEFAULT_N_HEADS}")
        
        if self.DEFAULT_D_FF <= 0 or self.DEFAULT_D_FF > 8192:
            raise ValueError(f"d_ff must be between 1 and 8192, got {self.DEFAULT_D_FF}")
        
        if self.DEFAULT_BATCH_SIZE <= 0 or self.DEFAULT_BATCH_SIZE > 64:
            raise ValueError(f"batch_size must be between 1 and 64, got {self.DEFAULT_BATCH_SIZE}")
        
        if not (1e-6 <= self.DEFAULT_LEARNING_RATE <= 1e-1):
            raise ValueError(f"learning_rate must be between 1e-6 and 1e-1, got {self.DEFAULT_LEARNING_RATE}")
        
        if not (1e-8 <= self.DEFAULT_WEIGHT_DECAY <= 1e-2):
            raise ValueError(f"weight_decay must be between 1e-8 and 1e-2, got {self.DEFAULT_WEIGHT_DECAY}")
        
        if self.DEFAULT_MAX_STEPS <= 0 or self.DEFAULT_MAX_STEPS > 1000000:
            raise ValueError(f"max_steps must be between 1 and 1000000, got {self.DEFAULT_MAX_STEPS}")
        
        if self.DEFAULT_MAX_SEQ_LEN <= 0 or self.DEFAULT_MAX_SEQ_LEN > 10000:
            raise ValueError(f"max_seq_len must be between 1 and 10000, got {self.DEFAULT_MAX_SEQ_LEN}")
    
    def validate_schema(self, config_dict: Dict[str, Any]) -> None:
        """Validate configuration against JSON schema."""
        if jsonschema is None:
            return  # Skip schema validation if jsonschema not available
        
        schema = {
            "type": "object",
            "properties": {
                "d_model": {"type": "integer", "minimum": 1, "maximum": 2048},
                "n_layers": {"type": "integer", "minimum": 1, "maximum": 24},
                "n_heads": {"type": "integer", "minimum": 1, "maximum": 32},
                "d_ff": {"type": "integer", "minimum": 1, "maximum": 8192},
                "batch_size": {"type": "integer", "minimum": 1, "maximum": 64},
                "learning_rate": {"type": "number", "minimum": 1e-6, "maximum": 1e-1},
                "weight_decay": {"type": "number", "minimum": 1e-8, "maximum": 1e-2},
                "max_steps": {"type": "integer", "minimum": 1, "maximum": 1000000},
                "max_seq_len": {"type": "integer", "minimum": 1, "maximum": 10000}
            },
            "required": ["d_model", "n_layers", "batch_size", "learning_rate"]
        }
        
        try:
            jsonschema.validate(config_dict, schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
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
