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
        """Standard full attention for short sequences."""
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)
    
    def _window_attention(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Memory-efficient window-based attention."""
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process windows to avoid O(n^2) memory
        for i in range(seq_len):
            # Define window around position i
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            
            # Extract window queries, keys, values
            q_i = q[:, i:i+1, :, :].transpose(1, 2)  # (batch, heads, 1, head_dim)
            k_window = k[:, start:end, :, :].transpose(1, 2)  # (batch, heads, window, head_dim)
            v_window = v[:, start:end, :, :].transpose(1, 2)  # (batch, heads, window, head_dim)
            
            # Compute attention scores for this window
            scores = torch.matmul(q_i, k_window.transpose(-2, -1)) * self.scale  # (batch, heads, 1, window)
            
            # Apply mask if provided
            if mask is not None:
                window_mask = mask[:, start:end]  # (batch, window)
                window_mask = window_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, window)
                scores = scores.masked_fill(window_mask == 0, -1e9)
            
            # Apply attention
            attn_weights = F.softmax(scores, dim=-1)  # (batch, heads, 1, window)
            context = torch.matmul(attn_weights, v_window)  # (batch, heads, 1, head_dim)
            
            # Store result
            output[:, i:i+1, :, :] = context.transpose(1, 2)  # (batch, 1, heads, head_dim)
        
        # Reshape and project output
        output = output.contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(output)


class StructureEncoder(nn.Module):
    """Compact structure encoder with efficient sparse attention."""
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        
        # Validate configuration
        if config.d_model <= 0 or config.n_heads <= 0 or config.window_size <= 0:
            raise ValueError("Invalid encoder configuration parameters")
        
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
        # Validate input
        if embeddings.dim() != 3:
            raise ValueError(f"Expected 3D embeddings tensor, got {embeddings.dim()}D")
        
        batch_size, seq_len, d_model = embeddings.shape
        if d_model != self.config.d_model:
            raise ValueError(f"Embedding dimension mismatch: expected {self.config.d_model}, got {d_model}")
        
        x = self.input_proj(embeddings)
        
        # Use sparse attention for long sequences
        if seq_len > 128:
            x = self.sparse_attention(x, mask)
        
        # Apply transformer layers with memory management
        for i, layer in enumerate(self.layers):
            try:
                x = layer(x, src_key_padding_mask=~mask if mask is not None else None)
                # Clear cache periodically to prevent memory buildup
                if torch.cuda.is_available() and i % 2 == 0:
                    torch.cuda.empty_cache()
            except Exception as e:
                raise RuntimeError(f"Error in transformer layer {i}: {e}")
        
        return self.norm(x)
