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
    """Window-based sparse attention for long sequences."""
    
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
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply window mask
        window_mask = self._create_window_mask(seq_len, x.device)
        scores = scores.masked_fill(window_mask == 0, -1e9)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)
    
    def _create_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create window-based attention mask."""
        mask = torch.zeros(seq_len, seq_len, device=device)
        
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1
        
        return mask.unsqueeze(0).unsqueeze(0)


class StructureEncoder(nn.Module):
    """Compact structure encoder with sparse attention."""
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        
        self.input_proj = nn.Linear(config.d_model, config.d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.n_layers)
        ])
        
        self.sparse_attention = SparseAttention(config)
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(embeddings)
        
        # Use sparse attention for long sequences
        if x.size(1) > 128:
            x = self.sparse_attention(x, mask)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=~mask if mask is not None else None)
        
        return self.norm(x)
