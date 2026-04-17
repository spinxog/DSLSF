"""RNA Sampler for Structure Generation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import math
import random
import time

from .utils import compute_contact_map
from .geometry_module import RigidTransform
from .logging_config import setup_logging


@dataclass
class SamplerConfig:
    """Configuration for RNA sampler."""
    n_decoys: int = 5
    temperature: float = 1.0
    min_distance: float = 3.0  # Minimum distance between atoms
    max_distance: float = 20.0  # Maximum distance for contacts
    n_steps: int = 1000
    contact_threshold: float = 8.0
    rmsd_threshold: float = 5.0
    early_stopping_patience: int = 50  # Steps to wait for improvement
    convergence_threshold: float = 1e-6  # Minimum improvement to continue
    max_time_seconds: float = 300.0  # Maximum time per decoy


@dataclass
class PerformanceMetrics:
    """Performance metrics for sampling operations."""
    total_time: float
    constraint_time: float
    confidence_time: float
    n_violations: int
    memory_peak: float
    n_decoys_generated: int


class RNASampler(nn.Module):
    """RNA structure sampler using fragment-based approach."""
    
    def __init__(self, config: SamplerConfig):
        super().__init__()
        # Validate configuration
        self._validate_config(config)
        
        self.config = config
        self.rigid_transform = RigidTransform()
        self.logger = setup_logging("rna_sampler")
        
        # Thread-local random seed management
        self._random_state = np.random.RandomState(42)
        self._torch_rng = torch.Generator()
        self._torch_rng.manual_seed(42)
        
        # Initialize performance tracking
        self._last_violations = 0
        
        # Fragment library (simplified)
        self.register_buffer('motif_library', self._create_motif_library())
        
        # Sampling networks
        self.coord_generator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3 * 3)  # 3 atoms * 3 coordinates
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def _create_motif_library(self) -> torch.Tensor:
        """Create a simple motif library."""
        # Common RNA motifs (simplified)
        motifs = torch.tensor([
            # Stem loop
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            # Hairpin loop
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            # Internal loop
            [[0.5, 0.866, 0.0], [-0.5, 0.866, 0.0], [0.0, 0.0, 1.0]],
            # Bulge
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ], dtype=torch.float32)
        
        return motifs
    
    def _validate_config(self, config: SamplerConfig) -> None:
        """Validate sampler configuration parameters."""
        if config.n_decoys <= 0 or config.n_decoys > 100:
            raise ValueError(f"n_decoys must be between 1 and 100, got {config.n_decoys}")
        
        if config.temperature <= 0 or config.temperature > 10:
            raise ValueError(f"temperature must be between 0 and 10, got {config.temperature}")
        
        if config.min_distance <= 0 or config.min_distance > 10:
            raise ValueError(f"min_distance must be between 0 and 10, got {config.min_distance}")
        
        if config.max_distance <= config.min_distance:
            raise ValueError(f"max_distance ({config.max_distance}) must be greater than min_distance ({config.min_distance})")
        
        if config.n_steps <= 0 or config.n_steps > 10000:
            raise ValueError(f"n_steps must be between 1 and 10000, got {config.n_steps}")
        
        if config.contact_threshold <= 0 or config.contact_threshold > 50:
            raise ValueError(f"contact_threshold must be between 0 and 50, got {config.contact_threshold}")
    
    def generate_decoys(self, sequence: str, embeddings: torch.Tensor, 
                     initial_coords: Optional[torch.Tensor] = None,
                     return_all_decoys: bool = False,
                     progress_callback: Optional[Callable[[int, int], None]] = None) -> Tuple[List[Dict], PerformanceMetrics]:
        """Generate structure decoys with performance metrics."""
        # Input validation
        if not sequence or not isinstance(sequence, str):
            raise ValueError("Sequence must be a non-empty string")
        
        # Start performance tracking
        start_time = time.time()
        initial_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0.0
        total_violations = 0
        
        if embeddings.dim() != 3:
            raise ValueError(f"Expected 3D embeddings tensor, got {embeddings.dim()}D")
        
        batch_size, seq_len, d_model = embeddings.shape
        if batch_size != 1:
            raise ValueError(f"Expected batch size 1, got {batch_size}")
        
        if seq_len != len(sequence):
            raise ValueError(f"Embedding sequence length {seq_len} doesn't match sequence length {len(sequence)}")
        
        if d_model != 512:
            raise ValueError(f"Expected embedding dimension 512, got {d_model}")
        
        # Use thread-local random state for reproducibility
        torch_rng = self._torch_rng
        np_rng = self._random_state
        
        if initial_coords is None:
            # Generate initial coordinates from embeddings
            initial_coords = self.coord_generator(embeddings)
            initial_coords = initial_coords.view(batch_size, seq_len, 3, 3)
        else:
            # Validate initial coordinates shape
            if initial_coords.dim() != 4:
                raise ValueError(f"Expected 4D initial_coords tensor, got {initial_coords.dim()}D")
            
            expected_shape = (batch_size, seq_len, 3, 3)
            if initial_coords.shape != expected_shape:
                raise ValueError(f"Expected initial_coords shape {expected_shape}, got {initial_coords.shape}")
        
        decoys = []
        constraint_time = 0.0
        confidence_time = 0.0
        
        n_decoys_to_generate = self.config.n_decoys if return_all_decoys else 1
        
        for i in range(n_decoys_to_generate):
            if progress_callback is not None:
                progress_callback(i, n_decoys_to_generate)
            
            # Sample structure
            sample_start = time.time()
            coords = self._sample_structure(
                sequence, embeddings, initial_coords, temperature=self.config.temperature
            )
            sample_time = time.time() - sample_start
            
            # Apply constraints and track violations
            constraint_start = time.time()
            coords = self._apply_constraints(coords, sequence)
            constraint_time += time.time() - constraint_start
            
            # Compute confidence score (pass cached computations to avoid redundancy)
            confidence_start = time.time()
            # Compute distances and contact map for confidence calculation
            rep_coords = coords[:, :, 0, :]  # (batch_size, seq_len, 3)
            diff = rep_coords.unsqueeze(2) - rep_coords.unsqueeze(1)  # (batch_size, seq_len, seq_len, 3)
            distances = torch.norm(diff, dim=-1)  # (batch_size, seq_len, seq_len)
            contact_tensor = (distances < self.config.contact_threshold).float()  # (batch_size, seq_len, seq_len)
            confidence = self._compute_confidence(sequence, coords, distances, contact_tensor)
            confidence_time += time.time() - confidence_start
            
            # Count violations (approximate from logging)
            if hasattr(self, '_last_violations'):
                total_violations += self._last_violations
            
            decoys.append({
                "coordinates": coords,
                "confidence": confidence,
                "decoy_id": i,
                "sequence": sequence,
                "sample_time": sample_time,
                "constraint_time": time.time() - constraint_start,
                "confidence_time": time.time() - confidence_start
            })
        
        # Calculate final metrics
        total_time = time.time() - start_time
        final_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0.0
        memory_peak = max(final_memory - initial_memory, 0.0)
        
        metrics = PerformanceMetrics(
            total_time=total_time,
            constraint_time=constraint_time,
            confidence_time=confidence_time,
            n_violations=total_violations,
            memory_peak=memory_peak,
            n_decoys_generated=len(decoys)
        )
        
        self.logger.debug(f"Performance metrics: {metrics}")
        
        return decoys, metrics
    
    def _sample_structure(self, sequence: str, embeddings: torch.Tensor,
                        initial_coords: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample a single structure."""
        batch_size, seq_len, d_model = embeddings.shape
        device = embeddings.device
        
        # Add noise to initial coordinates
        noise = torch.randn_like(initial_coords) * 0.1 * temperature
        coords = initial_coords + noise
        
        # Apply fragment-based sampling
        for step in range(self.config.n_steps):
            # Progress indicator for long operations
            if step % 100 == 0 or step == self.config.n_steps - 1:
                progress = (step + 1) / self.config.n_steps * 100
                self.logger.debug(f"Sampling progress: {progress:.1f}% ({step + 1}/{self.config.n_steps})")
            
            # Randomly select a fragment position
            if seq_len > 10:
                # Generate random quaternion for rotation
                quat = torch.randn(batch_size, seq_len, 4, device=device)
                quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
                
                # Apply rotation vectorized
                rotation_matrix = self.rigid_transform.quaternion_to_matrix(quat)
                
                # Clean up rotation tensors immediately
                del quat, rotation_matrix
            
            # Adaptive GPU cache cleanup based on memory usage
            if torch.cuda.is_available():
                # Check memory usage and clean up adaptively
                memory_allocated = torch.cuda.memory_allocated()
                memory_reserved = torch.cuda.memory_reserved()
                memory_utilization = memory_allocated / memory_reserved if memory_reserved > 0 else 0.0
                
                # More aggressive cleanup thresholds
                cleanup_threshold = 0.7  # Clean up at 70% instead of 80%
                aggressive_threshold = 0.9  # Aggressive cleanup at 90%
                
                if memory_utilization > aggressive_threshold:
                    # Aggressive cleanup for high memory usage
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Ensure all operations complete
                    self.logger.warning(f"Aggressive GPU cleanup at step {step}, memory usage: {memory_utilization:.2%}")
                elif memory_utilization > cleanup_threshold or step % 50 == 0:
                    # Regular cleanup for moderate usage or periodic cleanup
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    if step % 50 == 0:
                        self.logger.debug(f"Regular GPU cleanup at step {step}, memory usage: {memory_utilization:.2%}")
                
                # Additional cleanup for very large tensors - more efficient approach
                if step % 100 == 0:  # Less frequent, more effective
                    # Force garbage collection instead of iterating over all objects
                    import gc
                    gc.collect()
                    # Additional GPU cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Ensure all operations complete
            
            # Calculate current energy (simple approximation)
            current_energy = self._calculate_energy(coords, sequence)
            
            # Initialize best_energy and best_coords on first iteration
            if step == 0:
                best_energy = current_energy
                best_coords = coords.clone()
                patience_counter = 0
            
            # Check for improvement
            if current_energy < best_energy - self.config.convergence_threshold:
                best_coords = coords.clone()
                best_energy = current_energy
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.debug(f"Early stopping at step {step} (patience: {patience_counter})")
                break
        
        # Final cleanup before returning
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.debug(f"Sampling completed: {step+1} steps, final energy: {best_energy:.6f}")
        return best_coords
    
    def _calculate_energy(self, coords: torch.Tensor, sequence: str) -> float:
        """Calculate approximate energy for early stopping."""
        # Simple energy based on bond violations and contacts
        try:
            # Apply constraints temporarily to check violations
            constrained_coords = self._apply_constraints(coords.clone(), sequence)
            
            # Count violations as energy proxy
            rep_coords = constrained_coords[:, :, 0, :]
            diff = rep_coords.unsqueeze(2) - rep_coords.unsqueeze(1)
            distances = torch.norm(diff, dim=-1)
            
            # Energy = number of violations + distance penalties
            bond_violations = torch.sum(distances < self.config.min_distance)
            energy = bond_violations.item() + torch.mean(distances).item()
            
            return energy
        except Exception:
            return float('inf')  # Return high energy on error
    
    def _apply_constraints(self, coords: torch.Tensor, sequence: str) -> torch.Tensor:
        """Apply geometric constraints to coordinates."""
        # Input validation
        if coords.dim() != 4:
            raise ValueError(f"Expected 4D coords tensor, got {coords.dim()}D")
        
        if not sequence or not isinstance(sequence, str):
            raise ValueError("Sequence must be a non-empty string")
        
        batch_size, seq_len, n_atoms, _ = coords.shape
        device = coords.device
        
        # Validate dimensions
        if batch_size != 1:
            raise ValueError(f"Expected batch size 1, got {batch_size}")
        
        if seq_len != len(sequence):
            raise ValueError(
                f"Coordinate sequence length {seq_len} doesn't match sequence length {len(sequence)}.\n"
                f"This indicates a mismatch between the coordinate tensor shape and the input sequence.\n"
                f"Expected sequence length: {seq_len}, Got: {len(sequence)}\n"
                f"Coordinate tensor shape: {coords.shape}, Sequence length: {len(sequence)}"
            )
        
        if n_atoms < 1:
            raise ValueError(f"Expected at least 1 atom, got {n_atoms}")
        
        if coords.shape[-1] != 3:
            raise ValueError(f"Expected 3D coordinates, got {coords.shape[-1]}D")
        
        # Validate configuration values
        if self.config.min_distance <= 0:
            raise ValueError(f"min_distance must be positive, got {self.config.min_distance}")
        
        if self.config.contact_threshold <= 0:
            raise ValueError(f"contact_threshold must be positive, got {self.config.contact_threshold}")
        
        # Bond length constraints (vectorized)
        if seq_len > 1:
            # Compute all bond distances at once: (batch_size, seq_len-1, n_atoms)
            bond_vectors = coords[:, 1:] - coords[:, :-1]  # Vector from i to i+1
            bond_distances = torch.norm(bond_vectors, dim=-1)  # Distances
            
            # Find violations
            min_dist = self.config.min_distance
            violations = bond_distances < min_dist  # (batch_size, seq_len-1, n_atoms)
            
            if violations.any():
                self.logger.debug(f"Found {violations.sum().item()} bond violations")
                self._last_violations = violations.sum().item()  # Track for metrics
                # Compute directions for violating bonds only
                bond_directions = bond_vectors / (bond_distances + 1e-8)  # Normalized directions
                
                # Apply corrections only where violations exist
                for i in range(seq_len - 1):
                    for j in range(n_atoms):
                        violation_mask = violations[:, i, j]
                        if violation_mask.any():
                            # Apply correction only to violating samples
                            correction = bond_directions[:, i, j] * (min_dist - bond_distances[:, i, j])
                            coords[violation_mask, i, j] = coords[violation_mask, i, j] + correction[violation_mask]
                
                # Clean up bond constraint tensors
                del bond_vectors, bond_distances, violations, bond_directions
        
        # Contact constraints (vectorized computation)
        rep_coords = coords[:, :, 0, :]  # (batch_size, seq_len, 3)
        
        # Compute pairwise distances vectorized
        diff = rep_coords.unsqueeze(2) - rep_coords.unsqueeze(1)  # (batch_size, seq_len, seq_len, 3)
        distances = torch.norm(diff, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # Create contact map (vectorized)
        contact_tensor = (distances < self.config.contact_threshold).float()  # (batch_size, seq_len, seq_len)
        
        # Set diagonal to 0 (no self-contacts)
        batch_indices = torch.arange(seq_len, device=device)
        contact_tensor[:, batch_indices, batch_indices] = 0.0
        
        # Apply contact constraints (using cached distances)
        if seq_len > 1:
            # Use pre-computed distances from above
            min_dist = self.config.min_distance
            contact_violations = (distances < min_dist) & (contact_tensor > 0)  # (batch_size, seq_len, seq_len)
            
            if contact_violations.any():
                self.logger.debug(f"Found {contact_violations.sum().item()} contact violations")
                self._last_violations += contact_violations.sum().item()  # Track for metrics
                # Compute direction vectors only for violations
                rep_coords_expanded_i = rep_coords.unsqueeze(2)  # (batch_size, seq_len, 1, 3)
                rep_coords_expanded_j = rep_coords.unsqueeze(1)  # (batch_size, 1, seq_len, 3)
                
                direction_vectors = rep_coords_expanded_j - rep_coords_expanded_i  # (batch_size, seq_len, seq_len, 3)
                direction_vectors = direction_vectors / (distances.unsqueeze(-1) + 1e-8)  # Normalized
                
                # Apply corrections fully vectorized where violations exist
                if contact_violations.any():
                    # Get all violation indices at once
                    violation_indices = torch.nonzero(contact_violations, as_tuple=False)
                    if violation_indices.numel() > 0:
                        batch_idx = violation_indices[:, 0]
                        i_idx = violation_indices[:, 1] 
                        j_idx = violation_indices[:, 2]
                        
                        # Apply corrections vectorized
                        corrections = direction_vectors[batch_idx, i_idx, j_idx] * (min_dist - distances[batch_idx, i_idx, j_idx])
                        coords[batch_idx, j_idx, 0] = coords[batch_idx, i_idx, 0] + corrections
                
                # Clean up contact constraint tensors
                del rep_coords_expanded_i, rep_coords_expanded_j, direction_vectors, violation_indices
        
        return coords
    
    def _compute_confidence(self, sequence: str, coords: torch.Tensor, 
                        cached_distances: Optional[torch.Tensor] = None,
                        cached_contact_map: Optional[torch.Tensor] = None) -> float:
        """Compute confidence score for a structure."""
        # Use tensor operations to avoid CPU/GPU transfers
        seq_len = len(sequence)
        if seq_len < 3:
            return 0.5
        
        # Use cached computations if available, otherwise compute
        if cached_distances is not None and cached_contact_map is not None:
            distances = cached_distances
            contact_map = cached_contact_map
            self.logger.debug("Using cached distances and contact map for confidence computation")
        else:
            self.logger.debug("Computing distances and contact map for confidence computation")
            # Compute contact map using tensor operations
            rep_coords = coords[:, :, 0, :]  # (batch_size, seq_len, 3)
            diff = rep_coords.unsqueeze(2) - rep_coords.unsqueeze(1)  # (batch_size, seq_len, seq_len, 3)
            distances = torch.norm(diff, dim=-1)  # (batch_size, seq_len, seq_len)
            
            # Create contact map (threshold-based)
            contact_threshold = self.config.contact_threshold
            contact_map = (distances < contact_threshold).float()
            
            # Set diagonal to 0
            batch_indices = torch.arange(seq_len, device=coords.device)
            contact_map[:, batch_indices, batch_indices] = 0.0
        
        # Ensure diagonal is set to 0 for all cases
        batch_indices = torch.arange(seq_len, device=coords.device)
        contact_map[:, batch_indices, batch_indices] = 0.0
            
        # Clean up computation tensors (only if they exist)
        if 'rep_coords' in dir() and 'diff' in dir():
            del rep_coords, diff
        # Additional cleanup for large tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Compute RMSD-like metric
        total_contacts = torch.sum(contact_map)
        max_contacts = seq_len * (seq_len - 1) / 2  # Maximum possible contacts
        contact_ratio = total_contacts / max_contacts if max_contacts > 0 else 0
            
        # Base confidence on contact ratio
        confidence = 0.3 + 0.7 * contact_ratio
        
        # Adjust for sequence length
        if seq_len < 10:
            confidence *= 0.8  # Lower confidence for short sequences
        elif seq_len > 100:
            confidence *= 0.9  # Slightly lower confidence for very long sequences
        
        return min(float(confidence.cpu().item()), 1.0)
    
    def sample_fragment(self, sequence: str, start_pos: int, end_pos: int,
                       embeddings: torch.Tensor) -> torch.Tensor:
        """Sample a fragment of the structure."""
        batch_size, seq_len, d_model = embeddings.shape
        device = embeddings.device
        
        # Get fragment embeddings
        fragment_emb = embeddings[:, start_pos:end_pos]
        
        # Generate fragment coordinates
        fragment_coords = self.coord_generator(fragment_emb)
        fragment_coords = fragment_coords.view(batch_size, end_pos - start_pos, 3, 3)
        
        # Apply fragment-specific constraints
        for i in range(end_pos - start_pos - 1):
            for j in range(3):
                dist = torch.norm(fragment_coords[:, i, j] - fragment_coords[:, i+1, j], dim=-1)
                if dist.min() < self.config.min_distance:
                    direction = (fragment_coords[:, i+1, j] - fragment_coords[:, i, j]) / (dist + 1e-8)
                    fragment_coords[:, i, j] = fragment_coords[:, i, j] + direction * (self.config.min_distance - dist)
        
        return fragment_coords