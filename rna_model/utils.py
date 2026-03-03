"""Utility functions for RNA 3D folding pipeline."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


def tokenize_rna_sequence(sequence: str) -> torch.Tensor:
    """Convert RNA sequence to token IDs."""
    token_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
    
    tokens = []
    for nucleotide in sequence.upper():
        if nucleotide in token_map:
            tokens.append(token_map[nucleotide])
        else:
            tokens.append(token_map['N'])
    
    return torch.tensor(tokens, dtype=torch.long)


def decode_tokens(tokens: torch.Tensor) -> str:
    """Convert token IDs back to RNA sequence."""
    token_map = {0: 'A', 1: 'U', 2: 'G', 3: 'C', 4: 'N'}
    
    sequence = ""
    for token in tokens:
        sequence += token_map.get(token.item(), 'N')
    
    return sequence


def compute_tm_score(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute TM-score between two coordinate sets."""
    # Simplified TM-score computation
    # In practice, this would use the proper TM-score algorithm
    
    # Center coordinates
    coords1_centered = coords1 - np.mean(coords1, axis=0)
    coords2_centered = coords2 - np.mean(coords2, axis=0)
    
    # Compute RMSD
    diff = coords1_centered - coords2_centered
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    
    # Simplified TM-score (not the exact formula)
    d0 = 1.24 * (len(coords1) - 15) ** (1/3) - 1.8
    d0 = max(0.5, d0)  # Minimum d0
    
    tm_score = 1 / (1 + (rmsd / d0) ** 2)
    
    return tm_score


def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD between two coordinate sets."""
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


def superimpose_coordinates(coords1: np.ndarray, coords2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Superimpose two coordinate sets using Kabsch algorithm."""
    # Center coordinates
    center1 = np.mean(coords1, axis=0)
    center2 = np.mean(coords2, axis=0)
    
    coords1_centered = coords1 - center1
    coords2_centered = coords2 - center2
    
    # Compute covariance matrix
    cov_matrix = np.dot(coords2_centered.T, coords1_centered)
    
    # SVD
    U, S, Vt = np.linalg.svd(cov_matrix)
    
    # Rotation matrix
    rotation = np.dot(U, Vt)
    
    # Ensure proper orientation
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = np.dot(U, Vt)
    
    # Apply rotation
    coords2_aligned = np.dot(coords2_centered, rotation) + center1
    
    return coords1, coords2_aligned


def create_submission_format(predictions: List[Dict]) -> np.ndarray:
    """Create submission format from predictions."""
    all_coords = []
    
    for pred in predictions:
        coords = pred["coordinates"]  # Should be (n_decoys * n_residues, 3)
        all_coords.append(coords)
    
    return np.concatenate(all_coords, axis=0)


def validate_sequence(sequence: str) -> bool:
    """Validate RNA sequence."""
    valid_nucleotides = set('AUGCaugcNn')
    
    for nucleotide in sequence:
        if nucleotide not in valid_nucleotides:
            return False
    
    return True


def mask_sequence(sequence: str, mask_prob: float = 0.15) -> Tuple[str, List[int]]:
    """Create masked sequence for pretraining."""
    tokens = list(sequence)
    mask_positions = []
    
    for i, token in enumerate(tokens):
        if np.random.random() < mask_prob:
            mask_positions.append(i)
            tokens[i] = 'N'
    
    masked_sequence = ''.join(tokens)
    return masked_sequence, mask_positions


def compute_contact_map(coords: np.ndarray, threshold: float = 8.0) -> np.ndarray:
    """Compute contact map from coordinates."""
    n_residues = len(coords)
    contact_map = np.zeros((n_residues, n_residues), dtype=int)
    
    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            distance = np.linalg.norm(coords[i] - coords[j])
            if distance < threshold:
                contact_map[i, j] = 1
                contact_map[j, i] = 1
    
    return contact_map


def bin_distances(distances: np.ndarray, n_bins: int = 64, max_dist: float = 20.0) -> np.ndarray:
    """Bin distances for distance prediction."""
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    binned = np.digitize(distances, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned


def unbin_distances(binned_distances: np.ndarray, n_bins: int = 64, max_dist: float = 20.0) -> np.ndarray:
    """Convert binned distances back to continuous values."""
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers[binned_distances]


def compute_angles(coords: np.ndarray) -> np.ndarray:
    """Compute bond angles from coordinates."""
    n_atoms = len(coords)
    angles = []
    
    for i in range(n_atoms - 2):
        v1 = coords[i] - coords[i + 1]
        v2 = coords[i + 2] - coords[i + 1]
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        angles.append(angle)
    
    return np.array(angles)


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
        
        # Normalize
        n0 = n0 / np.linalg.norm(n0)
        n1 = n1 / np.linalg.norm(n1)
        b1 = b1 / np.linalg.norm(b1)
        
        # Dihedral angle
        m1 = np.cross(n0, b1)
        x = np.dot(n0, n1)
        y = np.dot(m1, n1)
        
        dihedral = np.arctan2(y, x)
        dihedrals.append(dihedral)
    
    return np.array(dihedrals)


def create_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Create pairwise distance matrix."""
    n_residues = len(coords)
    dist_matrix = np.zeros((n_residues, n_residues))
    
    for i in range(n_residues):
        for j in range(n_residues):
            dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    
    return dist_matrix


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


def check_clashes(coords: np.ndarray, threshold: float = 2.0) -> int:
    """Count number of steric clashes."""
    n_atoms = len(coords)
    clashes = 0
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = np.linalg.norm(coords[i] - coords[j])
            if distance < threshold:
                clashes += 1
    
    return clashes


def compute_bond_lengths(coords: np.ndarray, bonds: List[Tuple[int, int]]) -> np.ndarray:
    """Compute bond lengths for specified bonds."""
    bond_lengths = []
    
    for i, j in bonds:
        length = np.linalg.norm(coords[i] - coords[j])
        bond_lengths.append(length)
    
    return np.array(bond_lengths)


def check_bond_geometry(coords: np.ndarray,
                       bonds: List[Tuple[int, int]],
                       target_lengths: List[float],
                       tolerance: float = 0.1) -> float:
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


def memory_usage() -> Dict[str, float]:
    """Get current memory usage."""
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "cached": torch.cuda.memory_reserved() / 1024**3,     # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,  # GB
        }
    else:
        return {"cpu_only": True}


def clear_cache():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
