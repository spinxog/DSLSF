"""Configuration management for RNA 3D folding pipeline."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
import logging
from pathlib import Path

try:
    import jsonschema
except ImportError:
    jsonschema = None


@dataclass
class ValidationLimits:
    """Configurable validation limits for different use cases."""
    # Model architecture limits
    D_MODEL_MIN: int = 1
    D_MODEL_MAX: int = 4096
    N_LAYERS_MIN: int = 1
    N_LAYERS_MAX: int = 48
    N_HEADS_MIN: int = 1
    N_HEADS_MAX: int = 64
    D_FF_MIN: int = 1
    D_FF_MAX: int = 16384
    MAX_SEQ_LEN_MIN: int = 1
    MAX_SEQ_LEN_MAX: int = 32768
    
    # Training limits
    BATCH_SIZE_MIN: int = 1
    BATCH_SIZE_MAX: int = 512
    LEARNING_RATE_MIN: float = 1e-8
    LEARNING_RATE_MAX: float = 1.0
    WEIGHT_DECAY_MIN: float = 0.0
    WEIGHT_DECAY_MAX: float = 1.0
    MAX_STEPS_MIN: int = 1
    MAX_STEPS_MAX: int = 10000000
    
    # System resource limits
    GPU_MEMORY_THRESHOLD_MIN: float = 1.0  # GB
    GPU_MEMORY_THRESHOLD_MAX: float = 128.0  # GB
    CACHE_SIZE_MIN: int = 10
    CACHE_SIZE_MAX: int = 100000
    
    @classmethod
    def get_conservative_limits(cls) -> 'ValidationLimits':
        """Get conservative limits for production use."""
        limits = cls()
        limits.D_MODEL_MAX = 2048
        limits.N_LAYERS_MAX = 24
        limits.N_HEADS_MAX = 32
        limits.D_FF_MAX = 8192
        limits.BATCH_SIZE_MAX = 64
        limits.MAX_SEQ_LEN_MAX = 10000
        limits.MAX_STEPS_MAX = 1000000
        return limits
    
    @classmethod
    def get_aggressive_limits(cls) -> 'ValidationLimits':
        """Get aggressive limits for research/experimental use."""
        limits = cls()
        limits.D_MODEL_MAX = 8192
        limits.N_LAYERS_MAX = 96
        limits.N_HEADS_MAX = 128
        limits.D_FF_MAX = 32768
        limits.BATCH_SIZE_MAX = 1024
        limits.MAX_SEQ_LEN_MAX = 65536
        limits.MAX_STEPS_MAX = 50000000
        return limits
    
    @classmethod
    def adapt_to_system(cls, gpu_memory_gb: float = None, cpu_memory_gb: float = None) -> 'ValidationLimits':
        """Adapt limits based on available system resources."""
        limits = cls()
        
        # Try to detect GPU memory
        if gpu_memory_gb is None:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except Exception:
                gpu_memory_gb = 0
        
        # Try to detect CPU memory
        if cpu_memory_gb is None:
            try:
                import psutil
                cpu_memory_gb = psutil.virtual_memory().total / (1024**3)
            except Exception:
                cpu_memory_gb = 16  # Conservative default
        
        # Adapt batch size based on GPU memory
        if gpu_memory_gb >= 32:
            limits.BATCH_SIZE_MAX = 512
        elif gpu_memory_gb >= 16:
            limits.BATCH_SIZE_MAX = 256
        elif gpu_memory_gb >= 8:
            limits.BATCH_SIZE_MAX = 128
        else:
            limits.BATCH_SIZE_MAX = 64
        
        # Adapt model size based on available memory
        total_memory_gb = gpu_memory_gb + cpu_memory_gb
        if total_memory_gb >= 64:
            limits.D_MODEL_MAX = 4096
            limits.N_LAYERS_MAX = 48
            limits.MAX_SEQ_LEN_MAX = 16384
        elif total_memory_gb >= 32:
            limits.D_MODEL_MAX = 2048
            limits.N_LAYERS_MAX = 24
            limits.MAX_SEQ_LEN_MAX = 8192
        else:
            limits.D_MODEL_MAX = 1024
            limits.N_LAYERS_MAX = 12
            limits.MAX_SEQ_LEN_MAX = 4096
        
        return limits


@dataclass
class GlobalConfig:
    """Global configuration constants and settings with flexible validation."""
    
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
    
    # Validation configuration
    validation_mode: str = "adaptive"  # "conservative", "adaptive", "aggressive", "custom"
    custom_limits: Optional[ValidationLimits] = None
    
    def _get_validation_limits(self) -> ValidationLimits:
        """Get appropriate validation limits based on mode."""
        if self.validation_mode == "conservative":
            return ValidationLimits.get_conservative_limits()
        elif self.validation_mode == "aggressive":
            return ValidationLimits.get_aggressive_limits()
        elif self.validation_mode == "custom" and self.custom_limits is not None:
            return self.custom_limits
        else:  # adaptive or default
            return ValidationLimits.adapt_to_system()
    
    def validate(self) -> None:
        """Validate configuration parameters with flexible limits."""
        limits = self._get_validation_limits()
        
        # Validate model architecture
        if not (limits.D_MODEL_MIN <= self.DEFAULT_D_MODEL <= limits.D_MODEL_MAX):
            raise ValueError(f"d_model must be between {limits.D_MODEL_MIN} and {limits.D_MODEL_MAX}, got {self.DEFAULT_D_MODEL}")
        
        if not (limits.N_LAYERS_MIN <= self.DEFAULT_N_LAYERS <= limits.N_LAYERS_MAX):
            raise ValueError(f"n_layers must be between {limits.N_LAYERS_MIN} and {limits.N_LAYERS_MAX}, got {self.DEFAULT_N_LAYERS}")
        
        if not (limits.N_HEADS_MIN <= self.DEFAULT_N_HEADS <= limits.N_HEADS_MAX):
            raise ValueError(f"n_heads must be between {limits.N_HEADS_MIN} and {limits.N_HEADS_MAX}, got {self.DEFAULT_N_HEADS}")
        
        if not (limits.D_FF_MIN <= self.DEFAULT_D_FF <= limits.D_FF_MAX):
            raise ValueError(f"d_ff must be between {limits.D_FF_MIN} and {limits.D_FF_MAX}, got {self.DEFAULT_D_FF}")
        
        if not (limits.BATCH_SIZE_MIN <= self.DEFAULT_BATCH_SIZE <= limits.BATCH_SIZE_MAX):
            raise ValueError(f"batch_size must be between {limits.BATCH_SIZE_MIN} and {limits.BATCH_SIZE_MAX}, got {self.DEFAULT_BATCH_SIZE}")
        
        if not (limits.LEARNING_RATE_MIN <= self.DEFAULT_LEARNING_RATE <= limits.LEARNING_RATE_MAX):
            raise ValueError(f"learning_rate must be between {limits.LEARNING_RATE_MIN} and {limits.LEARNING_RATE_MAX}, got {self.DEFAULT_LEARNING_RATE}")
        
        if not (limits.WEIGHT_DECAY_MIN <= self.DEFAULT_WEIGHT_DECAY <= limits.WEIGHT_DECAY_MAX):
            raise ValueError(f"weight_decay must be between {limits.WEIGHT_DECAY_MIN} and {limits.WEIGHT_DECAY_MAX}, got {self.DEFAULT_WEIGHT_DECAY}")
        
        if not (limits.MAX_STEPS_MIN <= self.DEFAULT_MAX_STEPS <= limits.MAX_STEPS_MAX):
            raise ValueError(f"max_steps must be between {limits.MAX_STEPS_MIN} and {limits.MAX_STEPS_MAX}, got {self.DEFAULT_MAX_STEPS}")
        
        if not (limits.MAX_SEQ_LEN_MIN <= self.DEFAULT_MAX_SEQ_LEN <= limits.MAX_SEQ_LEN_MAX):
            raise ValueError(f"max_seq_len must be between {limits.MAX_SEQ_LEN_MIN} and {limits.MAX_SEQ_LEN_MAX}, got {self.DEFAULT_MAX_SEQ_LEN}")
        
        # Additional logical validation
        if self.DEFAULT_D_MODEL % self.DEFAULT_N_HEADS != 0:
            raise ValueError(f"d_model ({self.DEFAULT_D_MODEL}) must be divisible by n_heads ({self.DEFAULT_N_HEADS})")
        
        if self.DEFAULT_D_FF < self.DEFAULT_D_MODEL:
            logging.warning(f"d_ff ({self.DEFAULT_D_FF}) is smaller than d_model ({self.DEFAULT_D_MODEL}), which may limit model capacity")
        
        # Memory usage estimation
        estimated_memory_gb = self._estimate_memory_usage()
        if estimated_memory_gb > 32:  # Warn for high memory usage
            logging.warning(f"Estimated memory usage: {estimated_memory_gb:.1f}GB - ensure sufficient resources")
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in GB for current configuration."""
        # Rough estimation based on model parameters and batch size
        params_per_layer = self.DEFAULT_D_MODEL * self.DEFAULT_D_FF * 3  # Approximate
        total_params = params_per_layer * self.DEFAULT_N_LAYERS
        
        # Memory for parameters (4 bytes per float32)
        param_memory_gb = total_params * 4 / (1024**3)
        
        # Memory for activations (rough estimate)
        seq_len = min(self.DEFAULT_MAX_SEQ_LEN, 1024)  # Cap for estimation
        activation_memory_gb = self.DEFAULT_BATCH_SIZE * seq_len * self.DEFAULT_D_MODEL * 4 / (1024**3)
        
        # Gradient memory (same as parameters)
        gradient_memory_gb = param_memory_gb
        
        # Optimizer state (2x parameters for Adam)
        optimizer_memory_gb = param_memory_gb * 2
        
        total_memory_gb = param_memory_gb + activation_memory_gb + gradient_memory_gb + optimizer_memory_gb
        
        return total_memory_gb
    
    def validate_schema(self, config_dict: Dict[str, Any]) -> None:
        """Validate configuration against JSON schema with flexible limits."""
        if jsonschema is None:
            return  # Skip schema validation if jsonschema not available
        
        limits = self._get_validation_limits()
        
        schema = {
            "type": "object",
            "properties": {
                "d_model": {
                    "type": "integer", 
                    "minimum": limits.D_MODEL_MIN, 
                    "maximum": limits.D_MODEL_MAX
                },
                "n_layers": {
                    "type": "integer", 
                    "minimum": limits.N_LAYERS_MIN, 
                    "maximum": limits.N_LAYERS_MAX
                },
                "n_heads": {
                    "type": "integer", 
                    "minimum": limits.N_HEADS_MIN, 
                    "maximum": limits.N_HEADS_MAX
                },
                "d_ff": {
                    "type": "integer", 
                    "minimum": limits.D_FF_MIN, 
                    "maximum": limits.D_FF_MAX
                },
                "batch_size": {
                    "type": "integer", 
                    "minimum": limits.BATCH_SIZE_MIN, 
                    "maximum": limits.BATCH_SIZE_MAX
                },
                "learning_rate": {
                    "type": "number", 
                    "minimum": limits.LEARNING_RATE_MIN, 
                    "maximum": limits.LEARNING_RATE_MAX
                },
                "weight_decay": {
                    "type": "number", 
                    "minimum": limits.WEIGHT_DECAY_MIN, 
                    "maximum": limits.WEIGHT_DECAY_MAX
                },
                "max_steps": {
                    "type": "integer", 
                    "minimum": limits.MAX_STEPS_MIN, 
                    "maximum": limits.MAX_STEPS_MAX
                },
                "max_seq_len": {
                    "type": "integer", 
                    "minimum": limits.MAX_SEQ_LEN_MIN, 
                    "maximum": limits.MAX_SEQ_LEN_MAX
                }
            },
            "required": ["d_model", "n_layers", "batch_size", "learning_rate"]
        }
        
        try:
            jsonschema.validate(config_dict, schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation limits and current values."""
        limits = self._get_validation_limits()
        
        return {
            "validation_mode": self.validation_mode,
            "current_config": {
                "d_model": self.DEFAULT_D_MODEL,
                "n_layers": self.DEFAULT_N_LAYERS,
                "n_heads": self.DEFAULT_N_HEADS,
                "d_ff": self.DEFAULT_D_FF,
                "batch_size": self.DEFAULT_BATCH_SIZE,
                "learning_rate": self.DEFAULT_LEARNING_RATE,
                "weight_decay": self.DEFAULT_WEIGHT_DECAY,
                "max_steps": self.DEFAULT_MAX_STEPS,
                "max_seq_len": self.DEFAULT_MAX_SEQ_LEN
            },
            "validation_limits": {
                "d_model": [limits.D_MODEL_MIN, limits.D_MODEL_MAX],
                "n_layers": [limits.N_LAYERS_MIN, limits.N_LAYERS_MAX],
                "n_heads": [limits.N_HEADS_MIN, limits.N_HEADS_MAX],
                "d_ff": [limits.D_FF_MIN, limits.D_FF_MAX],
                "batch_size": [limits.BATCH_SIZE_MIN, limits.BATCH_SIZE_MAX],
                "learning_rate": [limits.LEARNING_RATE_MIN, limits.LEARNING_RATE_MAX],
                "weight_decay": [limits.WEIGHT_DECAY_MIN, limits.WEIGHT_DECAY_MAX],
                "max_steps": [limits.MAX_STEPS_MIN, limits.MAX_STEPS_MAX],
                "max_seq_len": [limits.MAX_SEQ_LEN_MIN, limits.MAX_SEQ_LEN_MAX]
            },
            "estimated_memory_gb": self._estimate_memory_usage()
        }
    
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
