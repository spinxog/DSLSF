"""RNA Sampler for Structure Generation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import math
import random

from .utils import compute_contact_map, compute_rmsd
from .geometry_module import RigidTransform


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


class RNASampler(nn.Module):
    """RNA structure sampler using fragment-based approach."""
    
    def __init__(self, config: SamplerConfig):
        super().__init__()
        self.config = config
        self.rigid_transform = RigidTransform()
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
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
    
    def generate_decoys(self, sequence: str, embeddings: torch.Tensor, 
                     initial_coords: Optional[torch.Tensor] = None,
                     return_all_decoys: bool = False) -> List[Dict]:
        """Generate structure decoys."""
        # Input validation
        if not sequence or not isinstance(sequence, str):
            raise ValueError("Sequence must be a non-empty string")
        
        if embeddings.dim() != 3:
            raise ValueError(f"Expected 3D embeddings tensor, got {embeddings.dim()}D")
        
        batch_size, seq_len, d_model = embeddings.shape
        if batch_size != 1:
            raise ValueError(f"Expected batch size 1, got {batch_size}")
        
        if seq_len != len(sequence):
            raise ValueError(f"Embedding sequence length {seq_len} doesn't match sequence length {len(sequence)}")
        
        if d_model != 512:
            raise ValueError(f"Expected embedding dimension 512, got {d_model}")
        
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
        
        for i in range(self.config.n_decoys if not return_all_decoys else 1):
            # Sample structure
            coords = self._sample_structure(
                sequence, embeddings, initial_coords, temperature=self.config.temperature
            )
            
            # Compute confidence score
            confidence = self._compute_confidence(sequence, coords)
            
            decoys.append({
                "coordinates": coords,
                "confidence": confidence,
                "decoy_id": i,
                "sequence": sequence
            })
        
        return decoys
    
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
            # Randomly select a fragment position
            if seq_len > 10:
                start_pos = random.randint(0, seq_len - 10)
                end_pos = min(start_pos + 5, seq_len)
            else:
                start_pos, end_pos = 0, seq_len
            
            # Apply motif transformation
            if random.random() < 0.3:  # 30% chance to apply motif
                motif_idx = random.randint(0, len(self.motif_library) - 1)
                motif = self.motif_library[motif_idx].to(device)
                
                # Apply motif transformation
                motif_coords = motif.unsqueeze(0).expand(batch_size, -1, -1, -1)
                coords[:, start_pos:end_pos] = motif_coords[:, :end_pos-start_pos]
                
                # Clean up motif tensors
                del motif_coords
            
            # Apply small random rotations
            if random.random() < 0.5:
                # Generate random quaternion
                quat = torch.randn(batch_size, 4)
                quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
                
                # Apply rotation
                for j in range(seq_len):
                    coords[:, j] = self.rigid_transform.apply_transform(
                        coords[:, j:j+1], quat, torch.zeros(batch_size, 3, device=device)
                    )
                
                # Clean up quaternion
                del quat
            
            # Apply small random translations
            if random.random() < 0.5:
                translation = torch.randn(batch_size, 3) * 0.5 * temperature
                coords = coords + translation.unsqueeze(1)
                
                # Clean up translation
                del translation
            
            # Apply constraints
            coords = self._apply_constraints(coords, sequence)
            
            # Periodic GPU cache cleanup
            if step % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return coords
    
    def _apply_constraints(self, coords: torch.Tensor, sequence: str) -> torch.Tensor:
        """Apply geometric constraints to coordinates."""
        batch_size, seq_len, n_atoms, _ = coords.shape
        device = coords.device
        
        # Bond length constraints
        for i in range(seq_len - 1):
            for j in range(n_atoms):
                # Enforce minimum bond length
                dist = torch.norm(coords[:, i, j] - coords[:, i+1, j], dim=-1)
                min_dist = self.config.min_distance
                if dist.min() < min_dist:
                    # Adjust positions
                    direction = (coords[:, i+1, j] - coords[:, i, j]) / (dist + 1e-8)
                    coords[:, i, j] = coords[:, i, j] + direction * (min_dist - dist)
        
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
        
        # Apply contact constraints (vectorized where possible)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                if contact_tensor[:, i, j].any():  # Contact exists
                    dist = torch.norm(coords[:, i, 0] - coords[:, j, 0], dim=-1)
                    min_dist_mask = dist < self.config.min_distance
                    if min_dist_mask.any():
                        direction = (coords[:, j, 0] - coords[:, i, 0]) / (dist + 1e-8)
                        coords[:, j, 0, min_dist_mask] = coords[:, i, 0, min_dist_mask] + direction[min_dist_mask] * (self.config.min_distance - dist[min_dist_mask])
        
        return coords
    
    def _compute_confidence(self, sequence: str, coords: torch.Tensor) -> float:
        """Compute confidence score for a structure."""
        # Use tensor operations to avoid CPU/GPU transfers
        seq_len = len(sequence)
        if seq_len < 3:
            return 0.5
        
        # Compute contact map using tensor operations
        rep_coords = coords[:, :, 0, :]  # (batch_size, seq_len, 3)
        diff = rep_coords.unsqueeze(2) - rep_coords.unsqueeze(1)  # (batch_size, seq_len, seq_len, 3)
        distances = torch.norm(diff, dim=-1)  # (batch_size, seq_len, seq_len)
        contact_map = (distances < self.config.contact_threshold).float()  # (batch_size, seq_len, seq_len)
        
        # Set diagonal to 0
        batch_indices = torch.arange(seq_len, device=coords.device)
        contact_map[:, batch_indices, batch_indices] = 0.0
        
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