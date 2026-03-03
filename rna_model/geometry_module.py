"""RNA-Specific Geometry Module with SE(3)-Equivariant Operations"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class GeometryConfig:
    """Configuration for geometry module."""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    n_atoms_per_residue: int = 3  # P, C4', N1 (simplified representation)
    distance_bins: int = 64
    angle_bins: int = 36
    torsion_bins: int = 72


class RigidTransform:
    """Rigid transformation utilities for 3D coordinates."""
    
    @staticmethod
    def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to rotation matrices."""
        w, x, y, z = quaternions.unbind(-1)
        
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        matrix = torch.stack([
            1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy),
            2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx),
            2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)
        ], dim=-1).view(*quaternions.shape[:-1], 3, 3)
        
        return matrix
    
    @staticmethod
    def matrix_to_quaternion(matrices: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrices to quaternions."""
        trace = matrices[..., 0, 0] + matrices[..., 1, 1] + matrices[..., 2, 2]
        
        # Handle numerical issues
        trace = torch.clamp(trace, min=-1.0, max=3.0)
        
        q_w = 0.5 * torch.sqrt(1 + trace)
        q_x = (matrices[..., 2, 1] - matrices[..., 1, 2]) / (4 * q_w + 1e-8)
        q_y = (matrices[..., 0, 2] - matrices[..., 2, 0]) / (4 * q_w + 1e-8)
        q_z = (matrices[..., 1, 0] - matrices[..., 0, 1]) / (4 * q_w + 1e-8)
        
        return torch.stack([q_w, q_x, q_y, q_z], dim=-1)
    
    @staticmethod
    def apply_transform(coords: torch.Tensor, 
                       rotations: torch.Tensor, 
                       translations: torch.Tensor) -> torch.Tensor:
        """Apply rigid transformation to coordinates."""
        # coords: (..., N, 3)
        # rotations: (..., 3, 3) or (..., 4) quaternions
        # translations: (..., 3)
        
        if rotations.shape[-1] == 4:
            rotations = RigidTransform.quaternion_to_matrix(rotations)
        
        transformed = torch.einsum('...ij,...kj->...ki', rotations, coords) + translations
        return transformed


class InvariantPointAttention(nn.Module):
    """Invariant Point Attention (IPA) adapted for RNA."""
    
    def __init__(self, config: GeometryConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = self.d_model // self.n_heads
        
        # Query, key, value projections
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        
        # Point attention projections
        self.point_q_proj = nn.Linear(3, self.n_heads * 16)
        self.point_k_proj = nn.Linear(3, self.n_heads * 16)
        
        # Output projections
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.point_out_proj = nn.Linear(self.n_heads * 16, 3)
        
        # Attention bias
        self.attention_bias = nn.Parameter(torch.zeros(self.n_heads))
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, 
                seq_repr: torch.Tensor,
                coords: torch.Tensor,
                frames: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            seq_repr: Sequence representations (batch_size, seq_len, d_model)
            coords: 3D coordinates (batch_size, seq_len, n_atoms, 3)
            frames: Local frames (batch_size, seq_len, 4) quaternions
            mask: Optional sequence mask
        """
        batch_size, seq_len, _ = seq_repr.shape
        n_atoms = coords.shape[-2]
        
        # Standard attention
        q = self.q_proj(seq_repr).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(seq_repr).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(seq_repr).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Point attention (use representative atom, e.g., C4')
        rep_coords = coords[..., 1, :]  # Use second atom as representative
        point_q = self.point_q_proj(rep_coords).view(batch_size, seq_len, self.n_heads, 16)
        point_k = self.point_k_proj(rep_coords).view(batch_size, seq_len, self.n_heads, 16)
        
        # Compute attention scores
        seq_scores = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale
        
        # Point attention scores
        point_scores = torch.einsum('bihd,bjhd->bhij', point_q, point_k)
        point_scores = point_scores / math.sqrt(16)
        
        # Combine scores
        combined_scores = seq_scores + point_scores + self.attention_bias.unsqueeze(-1).unsqueeze(-1)
        
        if mask is not None:
            mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
            combined_scores = combined_scores.masked_fill(mask_2d.unsqueeze(1) == 0, -1e9)
        
        # Softmax attention
        attn_weights = F.softmax(combined_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        seq_context = torch.einsum('bhij,bjhd->bihd', attn_weights, v)
        seq_context = seq_context.contiguous().view(batch_size, seq_len, self.d_model)
        
        # Apply attention to point coordinates
        point_context = torch.einsum('bhij,bjhd->bihd', attn_weights, point_q)
        point_context = point_context.contiguous().view(batch_size, seq_len, self.n_heads * 16)
        
        # Output projections
        seq_out = self.out_proj(seq_context)
        point_out = self.point_out_proj(point_context)
        
        return seq_out, point_out


class GeometryBlock(nn.Module):
    """Geometry-aware transformer block."""
    
    def __init__(self, config: GeometryConfig):
        super().__init__()
        self.ipa = InvariantPointAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Frame update network
        self.frame_update = nn.Sequential(
            nn.Linear(config.d_model + 3, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 4)  # Quaternion output
        )
    
    def forward(self, 
                seq_repr: torch.Tensor,
                coords: torch.Tensor,
                frames: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # IPA attention
        ipa_out, point_update = self.ipa(self.norm1(seq_repr), coords, frames, mask)
        seq_repr = seq_repr + ipa_out
        
        # Update coordinates
        coords = coords + point_update.unsqueeze(2)  # Broadcast to all atoms
        
        # Feedforward
        ff_out = self.ff(self.norm2(seq_repr))
        seq_repr = seq_repr + ff_out
        
        # Update frames
        frame_input = torch.cat([seq_repr, point_update], dim=-1)
        frame_delta = self.frame_update(frame_input)
        
        # Apply frame update (quaternion multiplication)
        frames = self._quaternion_multiply(frames, frame_delta)
        
        return seq_repr, coords, frames
    
    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return torch.stack([w, x, y, z], dim=-1)


class GeometryModule(nn.Module):
    """RNA-specific geometry module with SE(3)-equivariant operations."""
    
    def __init__(self, config: GeometryConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.d_model, config.d_model)
        
        # Geometry blocks
        self.blocks = nn.ModuleList([
            GeometryBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output heads
        self.distance_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.distance_bins)
        )
        
        self.angle_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.angle_bins)
        )
        
        self.torsion_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.torsion_bins)
        )
        
        # Sugar pucker prediction
        self.pucker_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 2)  # C3'-endo vs C2'-endo
        )
        
        # Confidence prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, 
                seq_repr: torch.Tensor,
                pair_repr: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            seq_repr: Sequence representations (batch_size, seq_len, d_model)
            pair_repr: Pairwise representations (batch_size, seq_len, seq_len, d_model)
            mask: Optional sequence mask
            
        Returns:
            Dictionary containing geometry predictions and coordinates
        """
        batch_size, seq_len, _ = seq_repr.shape
        device = seq_repr.device
        
        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=device)
        
        # Project input
        x = self.input_proj(seq_repr)
        x = self.dropout(x)
        
        # Initialize coordinates (simple linear chain)
        coords = self._initialize_coordinates(batch_size, seq_len, device)
        
        # Initialize frames (identity quaternions)
        frames = torch.zeros(batch_size, seq_len, 4, device=device)
        frames[..., 0] = 1.0  # w = 1, x = y = z = 0 (identity)
        
        # Apply geometry blocks
        for block in self.blocks:
            x, coords, frames = block(x, coords, frames, mask)
        
        # Predict geometric properties
        outputs = {}
        
        # Pairwise distances
        pairwise_repr = pair_repr.mean(dim=2)  # Aggregate pairwise info
        outputs["distance_logits"] = self.distance_head(pairwise_repr)
        
        # Angles and torsions
        outputs["angle_logits"] = self.angle_head(x)
        outputs["torsion_logits"] = self.torsion_head(x)
        
        # Sugar pucker
        outputs["pucker_logits"] = self.pucker_head(x)
        
        # Confidence
        outputs["confidence"] = self.confidence_head(x)
        
        # Final coordinates
        outputs["coordinates"] = coords
        outputs["frames"] = frames
        
        return outputs
    
    def _initialize_coordinates(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Initialize coordinates as a simple linear chain."""
        coords = torch.zeros(batch_size, seq_len, self.config.n_atoms_per_residue, 3, device=device)
        
        # Place atoms in a simple linear arrangement
        for i in range(seq_len):
            for j in range(self.config.n_atoms_per_residue):
                coords[:, i, j, 0] = i * 3.4  # 3.4 Å spacing along x-axis
                coords[:, i, j, 1] = j * 1.0  # Small offset in y for different atoms
        
        return coords


def fape_loss(pred_coords: torch.Tensor,
              pred_frames: torch.Tensor,
              true_coords: torch.Tensor,
              true_frames: torch.Tensor,
              mask: torch.Tensor,
              clamp_distance: float = 10.0) -> torch.Tensor:
    """Frame-Aligned Point Error (FAPE) loss."""
    batch_size, seq_len, n_atoms, _ = pred_coords.shape
    
    # Transform coordinates to local frames
    pred_local = RigidTransform.apply_transform(
        pred_coords - pred_frames[..., :3, 3].unsqueeze(2),
        RigidTransform.quaternion_to_matrix(pred_frames).inverse(),
        torch.zeros_like(pred_frames[..., :3, 3]).unsqueeze(2)
    )
    
    true_local = RigidTransform.apply_transform(
        true_coords - true_frames[..., :3, 3].unsqueeze(2),
        RigidTransform.quaternion_to_matrix(true_frames).inverse(),
        torch.zeros_like(true_frames[..., :3, 3]).unsqueeze(2)
    )
    
    # Compute squared errors
    squared_errors = ((pred_local - true_local) ** 2).sum(dim=-1)
    
    # Clamp and apply mask
    squared_errors = torch.clamp(squared_errors, max=clamp_distance**2)
    squared_errors = squared_errors * mask.unsqueeze(-1)
    
    # Return mean error
    return squared_errors.sum() / (mask.sum() * n_atoms + 1e-8)


def geometry_loss(distance_logits: torch.Tensor,
                 angle_logits: torch.Tensor,
                 torsion_logits: torch.Tensor,
                 pucker_logits: torch.Tensor,
                 target_distances: torch.Tensor,
                 target_angles: torch.Tensor,
                 target_torsions: torch.Tensor,
                 target_puckers: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
    """Compute multi-task geometry loss."""
    loss = 0.0
    
    # Distance loss
    if target_distances is not None:
        distance_flat = distance_logits.view(-1, distance_logits.size(-1))
        target_flat = target_distances.view(-1)
        mask_flat = mask.view(-1)
        
        distance_loss = F.cross_entropy(
            distance_flat[mask_flat], 
            target_flat[mask_flat], 
            ignore_index=-100
        )
        loss += distance_loss
    
    # Angle loss
    if target_angles is not None:
        angle_flat = angle_logits.view(-1, angle_logits.size(-1))
        target_flat = target_angles.view(-1)
        mask_flat = mask.view(-1)
        
        angle_loss = F.cross_entropy(
            angle_flat[mask_flat],
            target_flat[mask_flat],
            ignore_index=-100
        )
        loss += 0.5 * angle_loss
    
    # Torsion loss
    if target_torsions is not None:
        torsion_flat = torsion_logits.view(-1, torsion_logits.size(-1))
        target_flat = target_torsions.view(-1)
        mask_flat = mask.view(-1)
        
        torsion_loss = F.cross_entropy(
            torsion_flat[mask_flat],
            target_flat[mask_flat],
            ignore_index=-100
        )
        loss += 0.5 * torsion_loss
    
    # Sugar pucker loss
    if target_puckers is not None:
        pucker_flat = pucker_logits.view(-1, pucker_logits.size(-1))
        target_flat = target_puckers.view(-1)
        mask_flat = mask.view(-1)
        
        pucker_loss = F.cross_entropy(
            pucker_flat[mask_flat],
            target_flat[mask_flat],
            ignore_index=-100
        )
        loss += 0.3 * pucker_loss
    
    return loss
