"""Geometry Refinement Module for RNA Structure Optimization"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class RefinementConfig:
    n_iterations: int = 3
    learning_rate: float = 0.01
    bond_length_weight: float = 10.0
    angle_weight: float = 5.0
    clash_weight: float = 20.0
    distance_restraint_weight: float = 2.0
    min_bond_length: float = 1.3  # Å
    max_bond_length: float = 1.7  # Å
    clash_threshold: float = 2.0  # Å


class GeometryRefiner(nn.Module):
    """Fast internal-coordinate optimizer for RNA structure refinement."""
    
    def __init__(self, config: RefinementConfig):
        super().__init__()
        self.config = config
        
        # Standard RNA bond lengths and angles
        self.register_buffer('p_o5_bond_length', torch.tensor(1.6))
        self.register_buffer('o5_c5_bond_length', torch.tensor(1.43))
        self.register_buffer('c5_c4_bond_length', torch.tensor(1.46))
        self.register_buffer('c4_c3_bond_length', torch.tensor(1.48))
        self.register_buffer('c3_o3_bond_length', torch.tensor(1.43))
        
        # Bond angles (in radians)
        self.register_buffer('p_o5_c5_angle', torch.tensor(109.5 * math.pi / 180))
        self.register_buffer('o5_c5_c4_angle', torch.tensor(109.5 * math.pi / 180))
        self.register_buffer('c5_c4_c3_angle', torch.tensor(109.5 * math.pi / 180))
        self.register_buffer('c4_c3_o3_angle', torch.tensor(109.5 * math.pi / 180))
    
    def forward(self,
                coords: torch.Tensor,
                distance_restraints: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Refine RNA coordinates using gradient-based optimization.
        
        Args:
            coords: Input coordinates (batch_size, seq_len, n_atoms, 3)
            distance_restraints: Optional distance restraints (batch_size, seq_len, seq_len)
            mask: Optional sequence mask
            
        Returns:
            Dictionary containing refined coordinates and optimization info
        """
        batch_size, seq_len, n_atoms, _ = coords.shape
        device = coords.device
        
        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=device)
        
        # Clone coordinates for optimization
        refined_coords = coords.clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([refined_coords], lr=self.config.learning_rate)
        
        losses = []
        
        for iteration in range(self.config.n_iterations):
            optimizer.zero_grad()
            
            # Compute all loss terms
            bond_loss = self._bond_length_loss(refined_coords, mask)
            angle_loss = self._bond_angle_loss(refined_coords, mask)
            clash_loss = self._clash_loss(refined_coords, mask)
            
            total_loss = (self.config.bond_length_weight * bond_loss +
                         self.config.angle_weight * angle_loss +
                         self.config.clash_weight * clash_loss)
            
            # Add distance restraints if provided
            if distance_restraints is not None:
                restraint_loss = self._distance_restraint_loss(
                    refined_coords, distance_restraints, mask
                )
                total_loss += self.config.distance_restraint_weight * restraint_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
        
        return {
            "refined_coordinates": refined_coords.detach(),
            "losses": losses,
            "final_loss": losses[-1] if losses else 0.0
        }
    
    def _bond_length_loss(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute bond length violation loss."""
        batch_size, seq_len, n_atoms, _ = coords.shape
        device = coords.device
        
        losses = []
        
        # Backbone bonds (simplified - using consecutive atoms)
        for i in range(seq_len - 1):
            for j in range(n_atoms - 1):
                # Bond between atom j and j+1 in residue i
                if mask[:, i].sum() > 0:  # Only if residue exists
                    bond_vec = coords[:, i, j + 1] - coords[:, i, j]
                    bond_length = torch.norm(bond_vec, dim=-1)
                    
                    # Target bond length (simplified)
                    target_length = 1.5  # Average RNA bond length
                    
                    length_loss = F.mse_loss(bond_length, 
                                           target_length * torch.ones_like(bond_length))
                    losses.append(length_loss)
        
        # Inter-residue bonds (simplified)
        for i in range(seq_len - 1):
            if mask[:, i].sum() > 0 and mask[:, i + 1].sum() > 0:
                # Bond between last atom of residue i and first atom of residue i+1
                bond_vec = coords[:, i + 1, 0] - coords[:, i, -1]
                bond_length = torch.norm(bond_vec, dim=-1)
                
                target_length = 1.5
                length_loss = F.mse_loss(bond_length,
                                       target_length * torch.ones_like(bond_length))
                losses.append(length_loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
    
    def _bond_angle_loss(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute bond angle violation loss."""
        batch_size, seq_len, n_atoms, _ = coords.shape
        device = coords.device
        
        losses = []
        
        # Angles within residues
        for i in range(seq_len):
            if mask[:, i].sum() > 0:
                for j in range(n_atoms - 2):
                    # Angle between atoms j, j+1, j+2
                    v1 = coords[:, i, j] - coords[:, i, j + 1]
                    v2 = coords[:, i, j + 2] - coords[:, i, j + 1]
                    
                    # Compute angle
                    cos_angle = F.cosine_similarity(v1, v2, dim=-1)
                    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                    angle = torch.acos(cos_angle)
                    
                    # Target angle (simplified)
                    target_angle = 109.5 * math.pi / 180  # Tetrahedral angle
                    angle_loss = F.mse_loss(angle, target_angle * torch.ones_like(angle))
                    losses.append(angle_loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
    
    def _clash_loss(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute steric clash loss."""
        batch_size, seq_len, n_atoms, _ = coords.shape
        device = coords.device
        
        losses = []
        
        # Check all pairs of atoms
        for i in range(seq_len):
            for j in range(seq_len):
                if i == j:
                    continue
                
                # Skip if either residue doesn't exist
                if mask[:, i].sum() == 0 or mask[:, j].sum() == 0:
                    continue
                
                # Skip bonded atoms (simplified check)
                if abs(i - j) <= 1:
                    continue
                
                for a1 in range(n_atoms):
                    for a2 in range(n_atoms):
                        # Compute distance
                        dist_vec = coords[:, i, a1] - coords[:, j, a2]
                        distance = torch.norm(dist_vec, dim=-1)
                        
                        # Penalize distances below threshold
                        clash_penalty = torch.clamp(
                            self.config.clash_threshold - distance, min=0.0
                        )
                        
                        losses.append(clash_penalty.mean())
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
    
    def _distance_restraint_loss(self,
                                coords: torch.Tensor,
                                distance_restraints: torch.Tensor,
                                mask: torch.Tensor) -> torch.Tensor:
        """Compute distance restraint loss."""
        batch_size, seq_len, _, _ = coords.shape
        device = coords.device
        
        losses = []
        
        # Use C1' coordinates (or first atom) for distance restraints
        rep_coords = coords[:, :, 0, :]  # (batch_size, seq_len, 3)
        
        for i in range(seq_len):
            for j in range(seq_len):
                if i >= j:  # Only compute upper triangle
                    continue
                
                if mask[:, i].sum() == 0 or mask[:, j].sum() == 0:
                    continue
                
                # Compute actual distance
                dist_vec = rep_coords[:, i] - rep_coords[:, j]
                actual_distance = torch.norm(dist_vec, dim=-1)
                
                # Get target distance from restraints
                target_distance = distance_restraints[:, i, j]
                
                # Compute loss (only if restraint is meaningful)
                restraint_mask = target_distance > 0
                if restraint_mask.sum() > 0:
                    distance_loss = F.mse_loss(
                        actual_distance[restraint_mask],
                        target_distance[restraint_mask]
                    )
                    losses.append(distance_loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)


class FastRefiner(nn.Module):
    """Ultra-fast refiner for competition deployment."""
    
    def __init__(self):
        super().__init__()
        # Very simple coordinate smoothing
        self.smoothing_kernel = nn.Parameter(
            torch.tensor([[0.25, 0.5, 0.25]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )
    
    def forward(self, coords: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply simple smoothing to remove minor clashes."""
        batch_size, seq_len, n_atoms, _ = coords.shape
        
        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=coords.device)
        
        # Apply 1D smoothing along sequence for each atom
        refined = coords.clone()
        
        for atom in range(n_atoms):
            atom_coords = coords[:, :, atom, :]  # (batch_size, seq_len, 3)
            
            # Pad for convolution
            padded = F.pad(atom_coords, (0, 0, 1, 1), mode='replicate')
            
            # Apply smoothing
            smoothed = F.conv1d(
                padded.transpose(1, 2),  # (batch_size, 3, seq_len+2)
                self.smoothing_kernel.expand(3, 1, -1),  # (3, 1, 3)
                groups=3
            ).transpose(1, 2)  # (batch_size, seq_len, 3)
            
            refined[:, :, atom, :] = smoothed
        
        return refined
    
    def refine_structure(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Refine a single structure."""
        if coords.dim() == 3:
            coords = coords.unsqueeze(0)  # Add batch dimension
        
        refined = self.forward(coords)
        
        return {
            "coordinates": refined.squeeze(0),
            "loss": 0.0,
            "refined": True
        }
