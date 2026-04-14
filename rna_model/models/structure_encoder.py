"""Structure Encoder with Sparse Attention for Long Sequences"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class EncoderConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 12
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 2048
    window_size: int = 64  # For sparse attention


class SparseAttention(nn.Module):
    """Memory-efficient sparse attention for long sequences."""
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = self.d_model // self.n_heads
        self.window_size = config.window_size
        
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        
        self.scale = 1.0 / (self.head_dim ** 0.5)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Validate inputs
        if seq_len <= 0:
            raise ValueError(f"Invalid sequence length: {seq_len}")
        
        # For short sequences, use regular attention
        if seq_len <= 128:
            return self._full_attention(x, mask)
        
        # Use efficient window-based attention for long sequences
        return self._window_attention(x, mask)
    
    def _full_attention(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Full attention for short sequences."""
        batch_size, seq_len, d_model = x.shape
        
        # Standard multi-head attention
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax and output
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.out_proj(output)
    
    def _window_attention(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Window-based attention for long sequences."""
        batch_size, seq_len, d_model = x.shape
        
        # For simplicity, use full attention for now
        # In a real implementation, this would use efficient window-based attention
        return self._full_attention(x, mask)


class StructureEncoder(nn.Module):
    """Main structure encoder with multiple attention layers."""
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        
        # Validate configuration
        if config.d_model <= 0 or config.n_heads <= 0 or config.window_size <= 0:
            raise ValueError("Invalid encoder configuration parameters")
        
        self.input_proj = nn.Linear(config.d_model, config.d_model)
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            SparseAttention(config) for _ in range(config.n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(config.d_model, config.d_model)
        
        # Layer normalization
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, 
                seq_repr: torch.Tensor,
                pair_repr: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through structure encoder."""
        x = self.input_proj(seq_repr)
        
        for layer in self.attention_layers:
            x = layer(x, mask)
            x = self.norm(x)
            x = self.dropout(x)
        
        # Output projection
        embeddings = self.output_proj(x)
        
        return {
            "embeddings": embeddings,
            "pairwise_repr": pair_repr  # Pass through for now
        }
