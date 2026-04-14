# API Reference

This document provides comprehensive API documentation for the RNA 3D folding pipeline.

## Core Classes

### RNAFoldingPipeline

The main class for RNA 3D structure prediction.

```python
class RNAFoldingPipeline:
    """Main pipeline for RNA 3D structure prediction."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline with configuration."""
        
    def predict_single_sequence(self, 
                              sequence: str,
                              msa_data: Optional[np.ndarray] = None,
                              return_all_decoys: bool = False) -> Dict[str, Any]:
        """Predict structure for a single RNA sequence.
        
        Args:
            sequence: RNA sequence string
            msa_data: Optional MSA data (numpy array)
            return_all_decoys: Whether to return all decoys or just top 5
            
        Returns:
            Dictionary containing:
            - sequence: Input sequence
            - coordinates: Predicted coordinates (5×N×3)
            - n_decoys: Number of decoys (always 5)
            - n_residues: Number of residues
            - decoys: All decoys with metadata (if return_all_decoys=True)
        """
        
    def predict_batch(self,
                     sequences: List[str],
                     msa_data: Optional[List[np.ndarray]] = None) -> List[Dict[str, Any]]:
        """Predict structures for multiple sequences."""
        
    def enable_competition_mode(self):
        """Enable optimizations for competition deployment."""
        
    def save_model(self, filepath: str):
        """Save the complete pipeline state."""
        
    def load_model(self, filepath: str):
        """Load pipeline state from file."""
```

### PipelineConfig

Configuration class for the pipeline.

```python
@dataclass
class PipelineConfig:
    """Configuration for RNA folding pipeline."""
    
    device: str = "cuda"
    mixed_precision: bool = True
    max_sequence_length: int = 512
    compile_model: bool = True
    inference_timeout: float = 144.0
    
    # Model configurations
    lm_config: Optional[LMConfig] = None
    ss_config: Optional[SSConfig] = None
    geometry_config: Optional[GeometryConfig] = None
    sampler_config: Optional[SamplerConfig] = None
    refiner_config: Optional[RefinementConfig] = None
```

### IntegratedModel

Combined model containing all components.

```python
class IntegratedModel(nn.Module):
    """Integrated model containing all pipeline components."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize integrated model."""
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_contacts: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass through all components."""
```

## Model Components

### RNALanguageModel

BERT-style transformer for RNA sequences.

```python
class RNALanguageModel(nn.Module):
    """RNA Language Model with masked span LM and contact prediction."""
    
    def __init__(self, config: LMConfig):
        """Initialize language model."""
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_contacts: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Returns:
            Dictionary containing:
            - embeddings: Sequence embeddings
            - logits: LM logits
            - contacts: Contact predictions (if return_contacts=True)
        """
        
    def create_span_mask(self, 
                        seq_len: int, 
                        batch_size: int,
                        device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create span masks for masked language modeling."""
        
    def get_embeddings(self, 
                      input_ids: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None,
                      use_cache: bool = True) -> torch.Tensor:
        """Get cached embeddings for downstream tasks."""
        
    def clear_cache(self):
        """Clear all cached embeddings."""
        
    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
```

### SecondaryStructurePredictor

Predicts secondary structure with top-k hypotheses.

```python
class SecondaryStructurePredictor(nn.Module):
    """Secondary structure predictor with top-k hypotheses and pseudoknot support."""
    
    def __init__(self, config: SSConfig):
        """Initialize secondary structure predictor."""
        
    def forward(self, 
                embeddings: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Returns:
            Dictionary containing:
            - contact_logits: Contact predictions
            - pseudoknot_logits: Pseudoknot predictions
            - pair_repr: Pairwise representations
        """
        
    def sample_hypotheses(self, 
                         contact_logits: torch.Tensor,
                         pseudoknot_logits: torch.Tensor) -> List[List[Dict]]:
        """Sample top-k secondary structure hypotheses."""
```

### StructureEncoder

Efficient encoder with sparse attention for long sequences.

```python
class StructureEncoder(nn.Module):
    """Compact structure encoder with efficient sparse attention."""
    
    def __init__(self, config: EncoderConfig):
        """Initialize structure encoder."""
        
    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with sparse attention for long sequences."""
```

### GeometryModule

SE(3)-equivariant geometry module for 3D coordinate prediction.

```python
class GeometryModule(nn.Module):
    """SE(3)-equivariant geometry module for 3D coordinate prediction."""
    
    def __init__(self, config: GeometryConfig):
        """Initialize geometry module."""
        
    def forward(self, 
                seq_repr: torch.Tensor,
                pair_repr: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Returns:
            Dictionary containing:
            - coordinates: Predicted 3D coordinates
            - frames: Rigid body frames
            - distance_logits: Distance predictions
            - angle_logits: Angle predictions
            - torsion_logits: Torsion predictions
            - pucker_logits: Sugar pucker predictions
            - confidence: Confidence scores
        """
```

### RNASampler

Generates diverse decoy structures.

```python
class RNASampler(nn.Module):
    """RNA sampler for generating diverse decoy structures."""
    
    def __init__(self, config: SamplerConfig):
        """Initialize RNA sampler."""
        
    def sample(self, 
               coordinates: torch.Tensor,
               secondary_structures: List[Dict],
               msa_data: Optional[torch.Tensor] = None) -> List[Dict]:
        """Sample diverse decoy structures.
        
        Returns:
            List of decoy dictionaries containing coordinates and metadata
        """
```

### GeometryRefiner

Refines predicted structures using physics-based optimization.

```python
class GeometryRefiner(nn.Module):
    """Geometry refiner for improving predicted structures."""
    
    def __init__(self, config: RefinementConfig):
        """Initialize geometry refiner."""
        
    def forward(self,
                coords: torch.Tensor,
                distance_restraints: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Refine coordinates using gradient-based optimization."""
```

## Configuration Classes

### LMConfig

Language model configuration.

```python
@dataclass
class LMConfig:
    """Configuration for RNA language model."""
    
    vocab_size: int = 5
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    d_ff: int = 2048
    max_seq_len: int = 2048
    dropout: float = 0.1
    span_mask_prob: float = 0.15
    contact_head_dim: int = 256
```

### SSConfig

Secondary structure predictor configuration.

```python
@dataclass
class SSConfig:
    """Configuration for secondary structure predictor."""
    
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 2048
    dropout: float = 0.1
    n_hypotheses: int = 3
    contact_bins: int = 64
    pseudoknot_dim: int = 128
```

### GeometryConfig

Geometry module configuration.

```python
@dataclass
class GeometryConfig:
    """Configuration for geometry module."""
    
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    distance_bins: int = 64
    angle_bins: int = 36
    torsion_bins: int = 72
    pucker_bins: int = 4
```

## Data Classes

### RNAStructure

Container for RNA structure data.

```python
@dataclass
class RNAStructure:
    """Container for RNA structure data."""
    
    sequence: str
    coordinates: np.ndarray  # Shape: (n_residues, n_atoms, 3)
    atom_names: List[str]    # Atom names per residue
    residue_names: List[str] # Residue names (A, U, G, C)
    chain_id: str
    pdb_id: Optional[str] = None
    resolution: Optional[float] = None
    secondary_structure: Optional[np.ndarray] = None
    contacts: Optional[np.ndarray] = None
```

## Utility Functions

### Tokenization

```python
def tokenize_rna_sequence(sequence: str) -> torch.Tensor:
    """Convert RNA sequence to token IDs."""
    
def decode_tokens(tokens: torch.Tensor) -> str:
    """Convert token IDs back to RNA sequence."""
```

### Structure Metrics

```python
def compute_tm_score(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute TM-score between two coordinate sets."""
    
def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD between two coordinate sets."""
    
def superimpose_coordinates(coords1: np.ndarray, coords2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Superimpose two coordinate sets using Kabsch algorithm."""
```

### Sequence Processing

```python
def validate_sequence(sequence: str) -> bool:
    """Validate RNA sequence."""
    
def mask_sequence(sequence: str, mask_prob: float = 0.15) -> Tuple[str, List[int]]:
    """Create masked sequence for pretraining."""
    
def compute_contact_map(coords: np.ndarray, threshold: float = 8.0) -> np.ndarray:
    """Compute contact map from coordinates."""
```

## Configuration Management

### GlobalConfig

Global configuration with validation.

```python
class GlobalConfig:
    """Global configuration constants and settings."""
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'GlobalConfig':
        """Load configuration from JSON file."""
        
    def to_file(self, config_path: Path) -> None:
        """Save configuration to JSON file."""
        
    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
```

### Configuration Functions

```python
def get_config(config_path: Optional[Path] = None) -> GlobalConfig:
    """Get configuration, loading from file if provided."""
    
def validate_config(config: GlobalConfig) -> bool:
    """Validate configuration values."""
```

## Logging

### StructuredLogger

Professional structured logging with JSON output.

```python
class StructuredLogger:
    """Structured logger with JSON output and performance tracking."""
    
    def __init__(self, name: str, log_file: Optional[Path] = None, level: str = "INFO"):
        """Initialize structured logger."""
        
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
```

### PerformanceLogger

Performance tracking utilities.

```python
class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def start_timer(self, name: str):
        """Start a named timer."""
        
    def end_timer(self, name: str, **metadata):
        """End a named timer and log the duration."""
        
    def log_memory_usage(self, **metadata):
        """Log current memory usage."""
        
    def log_model_stats(self, model, **metadata):
        """Log model statistics."""
```

### Logging Setup

```python
def setup_logger(name: str, 
                 log_dir: Optional[Path] = None,
                 level: str = "INFO",
                 structured: bool = True) -> StructuredLogger:
    """Setup logger with optional structured logging."""
```

## Command Line Interface

### Predict Command

```python
def predict_command():
    """Command-line interface for structure prediction."""
    
    # Usage:
    # rna-predict --sequence "GGGAAAUCC" --output results/
    # rna-predict --sequence-file sequences.txt --format json
```

### Train Command

```python
def train_command():
    """Command-line interface for training."""
    
    # Usage:
    # rna-train --config config.yaml --data data/train/
    # rna-train --model checkpoints/latest.pth --epochs 100
```

### Evaluate Command

```python
def evaluate_command():
    """Command-line interface for evaluation."""
    
    # Usage:
    # rna-evaluate --model checkpoints/latest.pth --test-data data/test/
    # rna-evaluate --predictions predictions.json --true-structures data/true/
```

## Error Handling

### Custom Exceptions

```python
class RNAFoldingError(Exception):
    """Base exception for RNA folding errors."""
    pass

class SequenceValidationError(RNAFoldingError):
    """Exception for sequence validation errors."""
    pass

class ModelLoadError(RNAFoldingError):
    """Exception for model loading errors."""
    pass

class PredictionError(RNAFoldingError):
    """Exception for prediction errors."""
    pass
```

## Constants

### Token Mapping

```python
TOKEN_MAP = {
    'A': 0,
    'U': 1, 
    'G': 2,
    'C': 3,
    'N': 4  # Unknown nucleotide
}
```

### Standard RNA Atoms

```python
STANDARD_ATOMS = [
    "P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'", 
    "N1", "N3", "C2", "C4", "C5", "C6", "O2", "O4", 
    "N6", "N2", "O6", "N7", "N9"
]
```

### Default Paths

```python
DEFAULT_CACHE_DIR = "cache"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_LOG_DIR = "logs"
DEFAULT_CONFIG_FILE = "config/default_config.json"
```

This API reference provides comprehensive documentation for all public interfaces of the RNA 3D folding pipeline.
