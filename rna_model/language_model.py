"""RNA Language Model with Contact-Aware Pretraining"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class LMConfig:
    vocab_size: int = 5  # A, U, G, C, N
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    d_ff: int = 2048
    max_seq_len: int = 2048
    dropout: float = 0.1
    span_mask_prob: float = 0.15
    contact_head_dim: int = 256


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for RNA sequences."""
    
    def __init__(self, d_model: int, max_seq_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention with rotary embeddings."""
    
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = self.d_model // self.n_heads
        
        assert self.head_dim * self.n_heads == self.d_model
        
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""
    
    def __init__(self, config: LMConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention
        attn_out = self.attention(self.norm1(x), mask)
        x = x + attn_out
        
        # Pre-norm feedforward
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        
        return x


class ContactPredictionHead(nn.Module):
    """Auxiliary contact prediction head for long-range interaction learning."""
    
    def __init__(self, config: LMConfig):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(config.d_model, config.contact_head_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.contact_head_dim, config.contact_head_dim),
            nn.ReLU(),
            nn.Linear(config.contact_head_dim, 1)  # Binary contact prediction
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict pairwise contact probabilities."""
        batch_size, seq_len, d_model = embeddings.shape
        
        # Create pairwise representations
        embeddings_i = embeddings.unsqueeze(2).expand(-1, -1, seq_len, -1)
        embeddings_j = embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
        # Concatenate pairwise features
        pairwise = torch.cat([embeddings_i, embeddings_j], dim=-1)
        
        # Predict contacts
        contacts = self.projection(pairwise).squeeze(-1)
        
        return contacts  # Shape: (batch_size, seq_len, seq_len)


class RNALanguageModel(nn.Module):
    """RNA Language Model with masked span LM and contact prediction objectives."""
    
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings (A, U, G, C, N)
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output heads
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        self.contact_head = ContactPredictionHead(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_span_mask(self, 
                        seq_len: int, 
                        batch_size: int,
                        device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create span masks for masked language modeling."""
        mask = torch.ones(batch_size, seq_len, device=device)
        labels = torch.full((batch_size, seq_len), -100, device=device)
        
        for i in range(batch_size):
            # Number of spans to mask (15% of sequence on average)
            n_spans = max(1, int(seq_len * self.config.span_mask_prob / 10))
            
            for _ in range(n_spans):
                # Random span length (geometric distribution, mean=3)
                span_len = torch.geometric(torch.tensor(0.33), sample=True).item() + 1
                span_len = min(span_len, seq_len // 4)  # Cap at 25% of sequence
                
                # Random start position
                start = torch.randint(0, seq_len - span_len + 1, (1,)).item()
                end = start + span_len
                
                mask[i, start:end] = 0
                labels[i, start:end] = torch.randint(0, self.config.vocab_size, (span_len,))
        
        return mask, labels
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_contacts: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional contact prediction.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            return_contacts: Whether to return contact predictions
            
        Returns:
            Dictionary containing:
            - embeddings: Sequence embeddings
            - logits: LM logits (if training)
            - contacts: Contact predictions (if return_contacts=True)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Token embeddings + positional encoding
        x = self.token_embed(input_ids) * math.sqrt(self.config.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        outputs = {"embeddings": x}
        
        # LM head
        logits = self.lm_head(x)
        outputs["logits"] = logits
        
        # Contact prediction
        if return_contacts:
            contacts = self.contact_head(x)
            outputs["contacts"] = contacts
        
        return outputs
    
    def get_embeddings(self, 
                      input_ids: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get frozen embeddings for downstream tasks."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, return_contacts=False)
            return outputs["embeddings"]


def masked_span_loss(logits: torch.Tensor, 
                    target_ids: torch.Tensor, 
                    mask: torch.Tensor) -> torch.Tensor:
    """Compute masked span language modeling loss."""
    # Only compute loss on masked positions
    active_loss = mask.view(-1) == 0
    active_logits = logits.view(-1, logits.size(-1))[active_loss]
    active_labels = target_ids.view(-1)[active_loss]
    
    loss = F.cross_entropy(active_logits, active_labels, ignore_index=-100)
    return loss


def contact_loss(pred_contacts: torch.Tensor, 
                 true_contacts: torch.Tensor,
                 contact_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute binary cross-entropy contact loss."""
    # Apply sigmoid to get probabilities
    pred_probs = torch.sigmoid(pred_contacts)
    
    if contact_mask is not None:
        # Only compute loss on valid positions
        pred_probs = pred_probs[contact_mask]
        true_contacts = true_contacts[contact_mask]
    
    loss = F.binary_cross_entropy(pred_probs, true_contacts.float())
    return loss
