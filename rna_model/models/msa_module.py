"""MSA (Multiple Sequence Alignment) Processing Module for RNA 3D Folding

This module implements MSA-aware processing following the approach used in
RhoFold and AlphaFold2. MSA provides evolutionary covariation signals that
are crucial for accurate structure prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class MSAConfig:
    """Configuration for MSA module."""
    d_model: int = 512  # Match language model dimension
    n_heads: int = 8
    n_layers: int = 4   # MSA transformer layers
    d_ff: int = 2048
    dropout: float = 0.1
    max_msa_depth: int = 256  # Maximum number of sequences in MSA
    max_seq_len: int = 2048
    use_msa: bool = True  # Toggle MSA processing


class MSAAttention(nn.Module):
    """Attention layer for MSA processing with sequence and row/column attention."""
    
    def __init__(self, config: MSAConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        
        # Row-wise attention (within each sequence)
        self.row_q_proj = nn.Linear(config.d_model, config.d_model)
        self.row_k_proj = nn.Linear(config.d_model, config.d_model)
        self.row_v_proj = nn.Linear(config.d_model, config.d_model)
        
        # Column-wise attention (across sequences at same position)
        self.col_q_proj = nn.Linear(config.d_model, config.d_model)
        self.col_k_proj = nn.Linear(config.d_model, config.d_model)
        self.col_v_proj = nn.Linear(config.d_model, config.d_model)
        
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, msa_repr: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process MSA with row and column attention.
        
        Args:
            msa_repr: (batch_size, n_seqs, seq_len, d_model)
            mask: Optional mask (batch_size, n_seqs, seq_len)
        
        Returns:
            Updated MSA representations
        """
        batch_size, n_seqs, seq_len, d_model = msa_repr.shape
        device = msa_repr.device
        
        # Row-wise attention (standard attention within each sequence)
        row_q = self.row_q_proj(msa_repr).view(batch_size, n_seqs, seq_len, self.n_heads, self.head_dim)
        row_k = self.row_k_proj(msa_repr).view(batch_size, n_seqs, seq_len, self.n_heads, self.head_dim)
        row_v = self.row_v_proj(msa_repr).view(batch_size, n_seqs, seq_len, self.n_heads, self.head_dim)
        
        # Reshape for attention: (batch*n_seqs, n_heads, seq_len, head_dim)
        row_q = row_q.transpose(2, 3).reshape(batch_size * n_seqs, self.n_heads, seq_len, self.head_dim)
        row_k = row_k.transpose(2, 3).reshape(batch_size * n_seqs, self.n_heads, seq_len, self.head_dim)
        row_v = row_v.transpose(2, 3).reshape(batch_size * n_seqs, self.n_heads, seq_len, self.head_dim)
        
        # Row attention scores
        row_scores = torch.matmul(row_q, row_k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            # Expand mask: (batch, n_seqs, seq_len) -> (batch*n_seqs, 1, 1, seq_len)
            mask_expanded = mask.view(batch_size * n_seqs, 1, 1, seq_len)
            row_scores = row_scores.masked_fill(~mask_expanded, float('-inf'))
        
        row_attn = F.softmax(row_scores, dim=-1)
        row_attn = self.dropout(row_attn)
        
        # Apply row attention
        row_out = torch.matmul(row_attn, row_v)  # (batch*n_seqs, n_heads, seq_len, head_dim)
        row_out = row_out.view(batch_size, n_seqs, self.n_heads, seq_len, self.head_dim)
        row_out = row_out.transpose(2, 3).reshape(batch_size, n_seqs, seq_len, d_model)
        
        # Column-wise attention (attention across sequences at same position)
        # Transpose to (batch, seq_len, n_seqs, d_model)
        msa_t = msa_repr.transpose(1, 2)
        
        col_q = self.col_q_proj(msa_t).view(batch_size, seq_len, n_seqs, self.n_heads, self.head_dim)
        col_k = self.col_k_proj(msa_t).view(batch_size, seq_len, n_seqs, self.n_heads, self.head_dim)
        col_v = self.col_v_proj(msa_t).view(batch_size, seq_len, n_seqs, self.n_heads, self.head_dim)
        
        # Reshape: (batch*seq_len, n_heads, n_seqs, head_dim)
        col_q = col_q.transpose(2, 3).reshape(batch_size * seq_len, self.n_heads, n_seqs, self.head_dim)
        col_k = col_k.transpose(2, 3).reshape(batch_size * seq_len, self.n_heads, n_seqs, self.head_dim)
        col_v = col_v.transpose(2, 3).reshape(batch_size * seq_len, self.n_heads, n_seqs, self.head_dim)
        
        # Column attention scores
        col_scores = torch.matmul(col_q, col_k.transpose(-2, -1)) * self.scale
        
        # Apply mask for column attention if provided
        if mask is not None:
            mask_t = mask.transpose(1, 2)  # (batch, seq_len, n_seqs)
            mask_expanded = mask_t.reshape(batch_size * seq_len, 1, 1, n_seqs)
            col_scores = col_scores.masked_fill(~mask_expanded, float('-inf'))
        
        col_attn = F.softmax(col_scores, dim=-1)
        col_attn = self.dropout(col_attn)
        
        # Apply column attention
        col_out = torch.matmul(col_attn, col_v)
        col_out = col_out.view(batch_size, seq_len, self.n_heads, n_seqs, self.head_dim)
        col_out = col_out.transpose(2, 3).reshape(batch_size, seq_len, n_seqs, d_model)
        col_out = col_out.transpose(1, 2)  # Back to (batch, n_seqs, seq_len, d_model)
        
        # Combine row and column outputs
        combined = row_out + col_out
        
        return self.out_proj(combined)


class MSATransformerBlock(nn.Module):
    """Transformer block for MSA processing."""
    
    def __init__(self, config: MSAConfig):
        super().__init__()
        self.attention = MSAAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, msa_repr: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention
        attn_out = self.attention(self.norm1(msa_repr), mask)
        msa_repr = msa_repr + attn_out
        
        # Pre-norm feedforward
        ff_out = self.ff(self.norm2(msa_repr))
        msa_repr = msa_repr + ff_out
        
        return msa_repr


class MSAModule(nn.Module):
    """
    MSA processing module that aggregates multiple sequence alignments
    into sequence embeddings.
    """
    
    def __init__(self, config: MSAConfig):
        super().__init__()
        self.config = config
        
        # Token embedding for MSA sequences (same vocab as LM)
        self.token_embed = nn.Embedding(5, config.d_model)  # A, U, G, C, N
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 1, config.max_seq_len, config.d_model) * 0.02
        )
        
        # MSA transformer layers
        self.layers = nn.ModuleList([
            MSATransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Aggregation: average over MSA sequences
        self.aggregation = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Optional: weighted aggregation based on sequence similarity
        self.sequence_weights = nn.Sequential(
            nn.Linear(config.d_model, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, 
                msa_tokens: torch.Tensor,
                sequence_embeddings: torch.Tensor,
                msa_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process MSA and augment sequence embeddings.
        
        Args:
            msa_tokens: (batch_size, n_seqs, seq_len) MSA token indices
            sequence_embeddings: (batch_size, seq_len, d_model) from language model
            msa_mask: Optional (batch_size, n_seqs, seq_len) mask
        
        Returns:
            Augmented sequence embeddings (batch_size, seq_len, d_model)
        """
        batch_size, n_seqs, seq_len = msa_tokens.shape
        device = msa_tokens.device
        
        if not self.config.use_msa or n_seqs <= 1:
            # No MSA or single sequence, return original embeddings
            return sequence_embeddings
        
        # Embed MSA tokens
        msa_repr = self.token_embed(msa_tokens)  # (batch, n_seqs, seq_len, d_model)
        
        # Add positional encoding
        msa_repr = msa_repr + self.pos_encoding[:, :, :seq_len, :]
        msa_repr = self.dropout(msa_repr)
        
        # Apply MSA transformer layers
        for layer in self.layers:
            msa_repr = layer(msa_repr, msa_mask)
        
        # Aggregate MSA representations
        # Option 1: Simple mean (with optional weighting)
        weights = self.sequence_weights(msa_repr)  # (batch, n_seqs, seq_len, 1)
        if msa_mask is not None:
            mask_expanded = msa_mask.unsqueeze(-1).float()
            weights = weights * mask_expanded
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        msa_aggregated = (msa_repr * weights).sum(dim=1)  # (batch, seq_len, d_model)
        msa_aggregated = self.aggregation(msa_aggregated)
        
        # Combine with original sequence embeddings
        # Residual connection to preserve original sequence info
        augmented = sequence_embeddings + msa_aggregated
        
        return augmented
    
    def disable_msa(self):
        """Disable MSA processing (for inference without MSA)."""
        self.config.use_msa = False
    
    def enable_msa(self):
        """Enable MSA processing."""
        self.config.use_msa = True
