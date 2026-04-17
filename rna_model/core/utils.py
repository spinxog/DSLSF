"""Utility functions for RNA 3D folding pipeline."""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Sequence
import math
import hashlib
import pickle
from functools import lru_cache

from .constants import BIOLOGICAL, GEOMETRY, COMPUTATIONAL, MODEL, VALIDATION
from .error_handling import ValidationError, ComputationError


def tokenize_rna_sequence(sequence: str) -> torch.Tensor:
    """Convert RNA sequence to token IDs with comprehensive validation."""
    # Type validation
    if not isinstance(sequence, str):
        raise ValueError(f"sequence must be a string, got {type(sequence)}")
    
    # Empty sequence check
    if not sequence:
        return torch.tensor([], dtype=torch.long)
    
    # Length validation
    if len(sequence) > VALIDATION.MAX_SEQUENCE_LENGTH:
        raise ValueError(f"sequence too long: {len(sequence)} > {VALIDATION.MAX_SEQUENCE_LENGTH}")
    
    if len(sequence) < VALIDATION.MIN_SEQUENCE_LENGTH:
        raise ValueError(f"sequence too short: {len(sequence)} < {VALIDATION.MIN_SEQUENCE_LENGTH}")
    
    # Content validation
    if sequence.isspace():
        raise ValueError("sequence cannot contain only whitespace")
    
    # Check for control characters
    if any(ord(c) < 32 or ord(c) == 127 for c in sequence):
        raise ValueError("sequence contains control characters")
    
    # Check for suspicious Unicode characters
    try:
        sequence.encode('ascii')
    except UnicodeEncodeError:
        raise ValueError("sequence contains non-ASCII characters")
    
    # Validate nucleotide composition
    upper_sequence = sequence.upper()
    valid_nucleotides = set(BIOLOGICAL.VALID_NUCLEOTIDES_UPPER)
    invalid_chars = set(upper_sequence) - valid_nucleotides
    
    if invalid_chars:
        raise ValueError(f"Invalid nucleotides found: {sorted(invalid_chars)}. Valid nucleotides: {sorted(valid_nucleotides)}")
    
    # Check for excessive N content
    n_count = upper_sequence.count('N')
    if n_count > len(upper_sequence) * VALIDATION.MAX_N_PROPORTION:
        raise ValueError(f"Too many N nucleotides: {n_count}/{len(upper_sequence)} ({n_count/len(upper_sequence)*100:.1f}%)")
    
    # Create token map
    token_map = {nuc: i for i, nuc in enumerate(BIOLOGICAL.VALID_NUCLEOTIDES_UPPER)}
    
    # Tokenize
    tokens = []
    for nucleotide in upper_sequence:
        tokens.append(token_map[nucleotide])
    
    return torch.tensor(tokens, dtype=torch.long)


def decode_tokens(tokens: torch.Tensor) -> str:
    """Convert token IDs back to RNA sequence."""
    token_map = {i: nuc for i, nuc in enumerate(BIOLOGICAL.VALID_NUCLEOTIDES_UPPER)}
    
    sequence = ""
    for token in tokens:
        sequence += token_map.get(token.item(), 'N')
    
    return sequence


def compute_contact_map(
    coords: np.ndarray, 
    threshold: Optional[float] = None, 
    chunk_size: Optional[int] = None, 
    memory_efficient: bool = True
) -> np.ndarray:
    """Compute contact map from coordinates using optimized operations.
    
    Args:
        coords: Coordinate array of shape (N,3)
        threshold: Distance threshold for contact definition
        chunk_size: Size of chunks for memory-efficient computation
        memory_efficient: Whether to use memory-efficient computation for large systems
        
    Returns:
        Boolean contact map of shape (N, N)
        
    Raises:
        ValueError: If coords is not 2D or has wrong shape
    """
    # Set defaults from constants
    if threshold is None:
        threshold = GEOMETRY.CONTACT_DISTANCE_THRESHOLD
    if chunk_size is None:
        chunk_size = COMPUTATIONAL.CONTACT_MAP_CHUNK_SIZE
    
    # Input validation
    if not isinstance(coords, np.ndarray):
        raise ValueError("coords must be a numpy array")
    
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected 2D coordinates with 3 columns, got shape {coords.shape}")
    
    if not isinstance(threshold, (int, float)) or threshold <= 0:
        raise ValueError(f"threshold must be a positive number, got {threshold}")
    
    if len(coords) == 0:
        return np.zeros((0, 0), dtype=bool)
    
    # Check for NaN or Inf values
    if np.isnan(coords).any() or np.isinf(coords).any():
        raise ValueError("coords contains NaN or Inf values")
    
    n_atoms = len(coords)
    threshold_squared = threshold * threshold
    
    # Use memory-efficient computation for large systems
    if memory_efficient and n_atoms > COMPUTATIONAL.MEDIUM_SYSTEM_THRESHOLD:
        return _memory_efficient_contact_map(coords, threshold, chunk_size)
    
    # Optimized computation using squared distances to avoid sqrt
    if n_atoms <= COMPUTATIONAL.SMALL_SYSTEM_THRESHOLD:
        # Small systems: use vectorized approach with early termination
        contact_map = np.zeros((n_atoms, n_atoms), dtype=bool)
        
        # Compute only upper triangle to avoid redundant calculations
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Compute squared distance
                diff = coords[i] - coords[j]
                squared_dist = np.dot(diff, diff)
                
                # Check against threshold without sqrt
                if squared_dist < threshold_squared:
                    contact_map[i, j] = True
                    contact_map[j, i] = True
        
        return contact_map
    else:
        # Medium systems: use chunked computation
        return _memory_efficient_contact_map(coords, threshold, chunk_size)

def _memory_efficient_contact_map(
    coords: np.ndarray, 
    threshold: float, 
    chunk_size: int
) -> np.ndarray:
    """Memory-efficient contact map computation for large systems."""
    n_atoms = len(coords)
    contact_map = np.zeros((n_atoms, n_atoms), dtype=bool)
    threshold_squared = threshold * threshold  # Compare squared distances
    
    # Process in chunks to reduce memory usage
    for i in range(0, n_atoms, chunk_size):
        end_i = min(i + chunk_size, n_atoms)
        
        for j in range(i, n_atoms, chunk_size):  # Start from i for symmetry
            end_j = min(j + chunk_size, n_atoms)
            
            # Skip diagonal chunks
            if i == j:
                continue
            
            # Compute chunk distances efficiently
            chunk_i = coords[i:end_i]
            chunk_j = coords[j:end_j]
            
            # Use broadcasting for chunk computation
            diff = chunk_i[:, np.newaxis, :] - chunk_j[np.newaxis, :, :]
            squared_distances = np.einsum('ijk,ijk->ij', diff, diff)
            
            # Compare with squared threshold to avoid sqrt
            chunk_contacts = squared_distances < threshold_squared
            
            # Store results
            contact_map[i:end_i, j:end_j] = chunk_contacts
            contact_map[j:end_j, i:end_i] = chunk_contacts.T  # Symmetric
    
    return contact_map




def tensor_hash(tensor: torch.Tensor) -> str:
    """Generate hash for tensor caching."""
    # Convert tensor to bytes and hash
    tensor_bytes = tensor.detach().cpu().numpy().tobytes()
    return hashlib.md5(tensor_bytes).hexdigest()


import gc
import threading
from functools import wraps

# Memory-aware cache management
class MemoryAwareCache:
    """Memory-aware caching with automatic cleanup."""
    
    def __init__(self, max_cache_size_mb: int = 100, max_entries: int = 500) -> None:
        self.max_cache_size_mb: int = max_cache_size_mb
        self.max_entries: int = max_entries
        self._cache: Dict[Tuple, np.ndarray] = {}
        self._cache_sizes: Dict[Tuple, float] = {}
        self._access_count: Dict[Tuple, int] = {}
        self._lock = threading.RLock()
        self._total_size_mb: float = 0.0
        self._cleanup_threshold: float = 0.8  # Cleanup when 80% full
    
    def get(self, key: Tuple) -> Optional[np.ndarray]:
        """Get value from cache with access tracking."""
        with self._lock:
            if key in self._cache:
                self._access_count[key] = self._access_count.get(key, 0) + 1
                return self._cache[key].copy()  # Return copy to prevent modification
            return None
    
    def put(self, key: Tuple, value: np.ndarray) -> None:
        """Put value in cache with memory management."""
        with self._lock:
            # Estimate memory usage
            value_size_mb = value.nbytes / (1024 * 1024)
            
            # Check if we need to cleanup
            if (self._total_size_mb + value_size_mb > self.max_cache_size_mb * self._cleanup_threshold or
                len(self._cache) >= self.max_entries * self._cleanup_threshold):
                self._cleanup_lru()
            
            # Add to cache
            if key in self._cache:
                # Update existing entry
                self._total_size_mb -= self._cache_sizes.get(key, 0)
            
            self._cache[key] = value.copy()  # Store copy
            self._cache_sizes[key] = value_size_mb
            self._access_count[key] = 1
            self._total_size_mb += value_size_mb
    
    def _cleanup_lru(self) -> None:
        """Remove least recently used entries with aggressive memory management."""
        if not self._cache:
            return
        
        # Sort by access count (least used first)
        sorted_keys = sorted(self._cache.keys(), key=lambda k: self._access_count.get(k, 0))
        
        # More aggressive cleanup targets
        target_size = self.max_cache_size_mb * 0.3  # Target 30% after cleanup
        target_entries = self.max_entries * 0.3
        
        # Check memory pressure and adjust targets
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 80:  # High memory pressure
                target_size = self.max_cache_size_mb * 0.1  # Aggressive cleanup
                target_entries = self.max_entries * 0.1
        except ImportError:
            pass  # psutil not available, use default targets
        
        removed = 0
        removed_size = 0.0
        
        for key in sorted_keys:
            if (self._total_size_mb <= target_size and 
                len(self._cache) <= target_entries):
                break
            
            entry_size = self._cache_sizes.get(key, 0)
            self._total_size_mb -= entry_size
            removed_size += entry_size
            
            # Clear the cached array to free memory immediately
            if key in self._cache:
                del self._cache[key]
            del self._cache_sizes[key]
            del self._access_count[key]
            removed += 1
        
        # Force garbage collection
        gc.collect()
        
        # Log cleanup activity for monitoring
        if removed > 0:
            logging.debug(f"Cache cleanup: removed {removed} entries, freed {removed_size:.2f} MB")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._cache_sizes.clear()
            self._access_count.clear()
            self._total_size_mb = 0.0
            gc.collect()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'entries': len(self._cache),
                'size_mb': self._total_size_mb,
                'max_entries': self.max_entries,
                'max_size_mb': self.max_cache_size_mb
            }

# Global cache instance
_distance_cache = MemoryAwareCache(
    max_cache_size_mb=COMPUTATIONAL.DEFAULT_CACHE_SIZE_MB, 
    max_entries=COMPUTATIONAL.DEFAULT_CACHE_ENTRIES
)

def cached_distance_matrix(coords_tuple: Tuple) -> np.ndarray:
    """Cached distance matrix computation with memory management."""
    # Try to get from cache
    cached_result = _distance_cache.get(coords_tuple)
    if cached_result is not None:
        return cached_result
    
    # Compute distance matrix
    coords = np.array(coords_tuple).reshape(-1, 3)
    n_atoms = len(coords)
    
    # Use memory-efficient computation for large systems
    if n_atoms > COMPUTATIONAL.MEDIUM_SYSTEM_THRESHOLD:
        distances = _compute_distance_matrix_chunked(coords)
    else:
        # Vectorized computation using broadcasting
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
    
    # Cache the result
    _distance_cache.put(coords_tuple, distances)
    
    return distances

def _compute_distance_matrix_chunked(
    coords: np.ndarray, 
    chunk_size: Optional[int] = None
) -> np.ndarray:
    """Compute distance matrix in chunks to save memory."""
    if chunk_size is None:
        chunk_size = COMPUTATIONAL.DISTANCE_MATRIX_CHUNK_SIZE
    
    n_atoms = len(coords)
    distances = np.zeros((n_atoms, n_atoms), dtype=coords.dtype)
    
    for i in range(0, n_atoms, chunk_size):
        end_i = min(i + chunk_size, n_atoms)
        for j in range(i, n_atoms, chunk_size):  # Start from i for symmetry
            end_j = min(j + chunk_size, n_atoms)
            
            # Compute chunk distances
            chunk_i = coords[i:end_i]
            chunk_j = coords[j:end_j]
            
            diff = chunk_i[:, np.newaxis, :] - chunk_j[np.newaxis, :, :]
            squared_dist = np.einsum('ijk,ijk->ij', diff, diff)
            distances[i:end_i, j:end_j] = np.sqrt(squared_dist, out=squared_dist)
            if i != j:  # Fill symmetric part
                distances[j:end_j, i:end_i] = distances[i:end_i, j:end_j].T
    
    return distances

def compute_dihedrals(coords: np.ndarray) -> np.ndarray:
    """Compute dihedral angles from coordinates."""
    n_atoms = len(coords)
    dihedrals = []
    
    for i in range(n_atoms - 3):
        # Define four consecutive atoms
        p0, p1, p2, p3 = coords[i], coords[i + 1], coords[i + 2], coords[i + 3]
        
        # Compute dihedral angle
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        
        # Normal vectors
        n0 = np.cross(b0, b1)
        n1 = np.cross(b1, b2)
        
        # Check for zero vectors (colinear atoms)
        n0_norm = np.linalg.norm(n0)
        n1_norm = np.linalg.norm(n1)
        b1_norm = np.linalg.norm(b1)
        
        if (n0_norm < GEOMETRY.ZERO_VECTOR_THRESHOLD or 
            n1_norm < GEOMETRY.ZERO_VECTOR_THRESHOLD or 
            b1_norm < GEOMETRY.ZERO_VECTOR_THRESHOLD):
            # Colinear atoms, dihedral is undefined, skip
            continue
        
        # Normalize
        n0 = n0 / n0_norm
        n1 = n1 / n1_norm
        b1 = b1 / b1_norm
        
        # Dihedral angle
        m1 = np.cross(n0, b1)
        x = np.dot(n0, n1)
        y = np.dot(m1, n1)
        
        dihedral = np.arctan2(y, x)
        dihedrals.append(dihedral)
    
    return np.array(dihedrals)


def create_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Create pairwise distance matrix using optimized vectorized operations."""
    if len(coords) == 0:
        return np.zeros((0, 0), dtype=coords.dtype)
    
    n_atoms = len(coords)
    
    # For very large systems, use chunked computation to save memory
    if n_atoms > 5000:
        return _chunked_distance_matrix(coords)
    
    # Use more efficient computation with scipy if available, fallback to numpy
    try:
        from scipy.spatial.distance import pdist, squareform
        return squareform(pdist(coords, metric='euclidean'))
    except ImportError:
        # Fallback to optimized numpy computation
        # Use squared distances to avoid sqrt until the end
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        squared_dist = np.einsum('ijk,ijk->ij', diff, diff)
        return np.sqrt(squared_dist, out=squared_dist)  # In-place sqrt


def _chunked_distance_matrix(
    coords: np.ndarray, 
    chunk_size: int = 1000
) -> np.ndarray:
    """Compute distance matrix in chunks to save memory for large systems."""
    n_atoms = len(coords)
    distances = np.zeros((n_atoms, n_atoms), dtype=coords.dtype)
    
    for i in range(0, n_atoms, chunk_size):
        end_i = min(i + chunk_size, n_atoms)
        for j in range(0, n_atoms, chunk_size):
            end_j = min(j + chunk_size, n_atoms)
            
            # Compute chunk distances
            chunk_i = coords[i:end_i]
            chunk_j = coords[j:end_j]
            
            diff = chunk_i[:, np.newaxis, :] - chunk_j[np.newaxis, :, :]
            squared_dist = np.einsum('ijk,ijk->ij', diff, diff)
            distances[i:end_i, j:end_j] = np.sqrt(squared_dist, out=squared_dist)
    
    return distances


def apply_symmetry_operations(coords: np.ndarray) -> List[np.ndarray]:
    """Apply symmetry operations to generate equivalent structures."""
    sym_coords = []
    
    # Original
    sym_coords.append(coords.copy())
    
    # 180-degree rotation around z-axis
    rotation_z = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    sym_coords.append(np.dot(coords, rotation_z.T))
    
    # 180-degree rotation around x-axis
    rotation_x = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    sym_coords.append(np.dot(coords, rotation_x.T))
    
    # 180-degree rotation around y-axis
    rotation_y = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    sym_coords.append(np.dot(coords, rotation_y.T))
    
    return sym_coords


def check_clashes(coords: np.ndarray, threshold: Optional[float] = None) -> int:
    """Count number of steric clashes."""
    if threshold is None:
        threshold = GEOMETRY.CLASH_DISTANCE_THRESHOLD
    
    n_atoms = len(coords)
    clashes = 0
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = np.linalg.norm(coords[i] - coords[j])
            if distance < threshold:
                clashes += 1
    
    return clashes


def compute_bond_lengths(
    coords: np.ndarray, 
    bonds: List[Tuple[int, int]]
) -> np.ndarray:
    """Compute bond lengths for specified bonds."""
    bond_lengths = []
    
    for i, j in bonds:
        length = np.linalg.norm(coords[i] - coords[j])
        bond_lengths.append(length)
    
    return np.array(bond_lengths)


def check_bond_geometry(
    coords: np.ndarray,
    bonds: List[Tuple[int, int]],
    target_lengths: List[float],
    tolerance: float = 0.1
) -> float:
    """Check bond geometry violations."""
    bond_lengths = compute_bond_lengths(coords, bonds)
    violations = 0
    
    for i, (actual, target) in enumerate(zip(bond_lengths, target_lengths)):
        if abs(actual - target) > tolerance:
            violations += 1
    
    return violations / len(bonds)


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.2f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def memory_usage() -> Dict[str, Union[float, bool]]:
    """Get current memory usage.
    
    Returns:
        Dictionary containing memory usage statistics in GB or cpu_only flag
    """
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "cached": torch.cuda.memory_reserved() / 1024**3,     # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,  # GB
        }
    else:
        return {"cpu_only": True}


def clear_cache() -> None:
    """Clear GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_tensor(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """Validate tensor for NaN/Inf values and reasonable ranges."""
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        raise ValidationError(f"{name} contains {nan_count} NaN values")
    
    if torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        raise ValidationError(f"{name} contains {inf_count} infinite values")
    
    # Check for reasonable value ranges
    if tensor.numel() > 0:
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        if abs(min_val) > GEOMETRY.MAX_COORDINATE_VALUE or abs(max_val) > GEOMETRY.MAX_COORDINATE_VALUE:
            raise ValidationError(f"{name} has unreasonable value range: [{min_val:.2f}, {max_val:.2f}]")
    
    return tensor


def validate_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], name: str = "tensor") -> torch.Tensor:
    """Validate tensor shape matches expected dimensions."""
    validate_tensor(tensor, name)
    
    if len(expected_shape) != tensor.dim():
        raise ValidationError(f"{name} has {tensor.dim()} dimensions, expected {len(expected_shape)}")
    
    for i, (expected, actual) in enumerate(zip(expected_shape, tensor.shape)):
        if expected != -1 and expected != actual:  # -1 means any size allowed
            raise ValidationError(f"{name} dimension {i} is {actual}, expected {expected}")
    
    return tensor


def safe_tensor_operation(operation, *args, **kwargs):
    """Safely execute tensor operations with validation."""
    try:
        result = operation(*args, **kwargs)
        if isinstance(result, torch.Tensor):
            validate_tensor(result, "operation_result")
        return result
    except Exception as e:
        raise ComputationError(f"Tensor operation failed: {e}", cause=e)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD between two coordinate sets using Kabsch algorithm.
    
    Args:
        coords1: First coordinate array (N, 3)
        coords2: Second coordinate array (N, 3)
        
    Returns:
        RMSD value in Angstroms
    """
    if coords1.shape != coords2.shape:
        raise ValueError(f"Coordinate shapes must match: {coords1.shape} vs {coords2.shape}")
    
    # Center the coordinates
    c1 = coords1 - coords1.mean(axis=0)
    c2 = coords2 - coords2.mean(axis=0)
    
    # Compute optimal rotation using SVD (Kabsch algorithm)
    h = c1.T @ c2
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    
    # Handle reflection case
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    
    # Rotate coords2
    c2_rotated = c2 @ r
    
    # Compute RMSD
    rmsd = np.sqrt(np.mean(np.sum((c1 - c2_rotated) ** 2, axis=1)))
    return float(rmsd)


def compute_tm_score(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute TM-score between two coordinate sets.
    
    TM-score is a measure of structural similarity that is length-independent.
    
    Args:
        coords1: First coordinate array (N, 3)
        coords2: Second coordinate array (N, 3)
        
    Returns:
        TM-score (0 to 1, higher is better)
    """
    if coords1.shape != coords2.shape:
        raise ValueError(f"Coordinate shapes must match: {coords1.shape} vs {coords2.shape}")
    
    n_atoms = len(coords1)
    if n_atoms == 0:
        return 0.0
    
    # Center coordinates
    c1 = coords1 - coords1.mean(axis=0)
    c2 = coords2 - coords2.mean(axis=0)
    
    # Compute optimal rotation using SVD (Kabsch algorithm)
    h = c1.T @ c2
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    
    # Handle reflection case
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    
    # Rotate coords2
    c2_rotated = c2 @ r
    
    # Compute distances
    distances = np.sqrt(np.sum((c1 - c2_rotated) ** 2, axis=1))
    
    # TM-score parameters
    d0 = 1.24 * (n_atoms - 15) ** (1.0/3.0) - 1.8 if n_atoms > 15 else 0.5
    d0 = max(d0, 0.5)
    
    # Compute TM-score
    tm_score = np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / n_atoms
    return float(tm_score)


def superimpose_coordinates(coords1: np.ndarray, coords2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Superimpose two coordinate sets using Kabsch algorithm.
    
    Args:
        coords1: First coordinate array (N, 3)
        coords2: Second coordinate array (N, 3)
        
    Returns:
        Tuple of (aligned_coords1, aligned_coords2)
    """
    if coords1.shape != coords2.shape:
        raise ValueError(f"Coordinate shapes must match: {coords1.shape} vs {coords2.shape}")
    
    # Center coordinates
    c1 = coords1 - coords1.mean(axis=0)
    c2 = coords2 - coords2.mean(axis=0)
    
    # Compute optimal rotation using SVD (Kabsch algorithm)
    h = c1.T @ c2
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    
    # Handle reflection case
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    
    # Rotate and center both
    aligned_c1 = c1
    aligned_c2 = c2 @ r
    
    return aligned_c1, aligned_c2


def bin_distances(distances: np.ndarray, bins: int = 64, max_distance: float = 20.0) -> np.ndarray:
    """Bin distances into discrete bins.
    
    Args:
        distances: Distance array
        bins: Number of bins
        max_distance: Maximum distance value
        
    Returns:
        Binned distance indices
    """
    bin_edges = np.linspace(0, max_distance, bins + 1)
    binned = np.digitize(distances, bin_edges) - 1
    binned = np.clip(binned, 0, bins - 1)
    return binned


def mask_sequence(tokens: torch.Tensor, mask_prob: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create mask for sequence tokens.
    
    Args:
        tokens: Token tensor
        mask_prob: Probability of masking each token
        
    Returns:
        Tuple of (masked_tokens, mask)
    """
    mask = torch.rand(tokens.shape, device=tokens.device) < mask_prob
    masked_tokens = tokens.clone()
    masked_tokens[mask] = 4  # Mask token ID (N)
    return masked_tokens, mask
