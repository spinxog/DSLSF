"""Secondary Structure Prediction with Top-K Hypotheses and Pseudoknot Support"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class SSConfig:
    """Configuration for secondary structure predictor."""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 2048
    dropout: float = 0.1
    n_hypotheses: int = 3  # Top-k hypotheses
    contact_bins: int = 64  # Distance bins for contact prediction
    pseudoknot_dim: int = 128


class PairwiseAttention(nn.Module):
    """Pairwise attention for secondary structure prediction."""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert self.head_dim * n_heads == d_model
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, 
                seq_repr: torch.Tensor,
                pair_repr: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            seq_repr: Sequence representations (batch_size, seq_len, d_model)
            pair_repr: Pairwise representations (batch_size, seq_len, seq_len, d_model)
            mask: Optional sequence mask
        """
        batch_size, seq_len, _ = seq_repr.shape
        
        # Expand sequence representations for pairwise attention
        seq_i = seq_repr.unsqueeze(2).expand(-1, -1, seq_len, -1)
        seq_j = seq_repr.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
        # Combine sequence and pairwise representations
        combined = seq_i + seq_j + pair_repr
        
        # Multi-head attention on pairwise features
        q = self.q_proj(combined).view(batch_size, seq_len, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(combined).view(batch_size, seq_len, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(combined).view(batch_size, seq_len, seq_len, self.n_heads, self.head_dim)
        
        # Reshape for attention computation
        q = q.transpose(2, 3).contiguous().view(batch_size * seq_len, self.n_heads, seq_len, self.head_dim)
        k = k.transpose(2, 3).contiguous().view(batch_size * seq_len, self.n_heads, seq_len, self.head_dim)
        v = v.transpose(2, 3).contiguous().view(batch_size * seq_len, self.n_heads, seq_len, self.head_dim)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
            mask_2d = mask_2d.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask_2d == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        
        # Reshape back
        context = context.view(batch_size, seq_len, seq_len, self.n_heads, self.head_dim)
        context = context.transpose(2, 3).contiguous().view(batch_size, seq_len, seq_len, self.d_model)
        
        return self.out_proj(context)


class SSBlock(nn.Module):
    """Secondary structure transformer block."""
    
    def __init__(self, config: SSConfig):
        super().__init__()
        self.pair_attention = PairwiseAttention(config.d_model, config.n_heads)
        self.pair_norm = nn.LayerNorm(config.d_model)
        
        self.seq_attention = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.seq_norm = nn.LayerNorm(config.d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        self.ff_norm = nn.LayerNorm(config.d_model)
    
    def forward(self, 
                seq_repr: torch.Tensor,
                pair_repr: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pairwise attention
        pair_out = self.pair_attention(self.pair_norm(seq_repr), pair_repr, mask)
        pair_repr = pair_repr + pair_out
        
        # Update sequence representation from pairwise
        seq_from_pair = torch.mean(pair_out, dim=2)
        seq_repr = seq_repr + seq_from_pair
        
        # Sequence self-attention
        seq_out, _ = self.seq_attention(self.seq_norm(seq_repr), 
                                       self.seq_norm(seq_repr), 
                                       self.seq_norm(seq_repr),
                                       key_padding_mask=~mask if mask is not None else None)
        seq_repr = seq_repr + seq_out
        
        # Feedforward
        ff_out = self.ff(self.ff_norm(seq_repr))
        seq_repr = seq_repr + ff_out
        
        return seq_repr, pair_repr


class ContactHead(nn.Module):
    """Contact prediction head for base pairing."""
    
    def __init__(self, config: SSConfig):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.contact_bins)
        )
    
    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """Predict contact probabilities for each residue pair."""
        return self.projection(pair_repr)  # Shape: (batch_size, seq_len, seq_len, contact_bins)


class PseudoknotHead(nn.Module):
    """Pseudoknot-specific prediction head."""
    
    def __init__(self, config: SSConfig):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(config.d_model, config.pseudoknot_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.pseudoknot_dim, config.pseudoknot_dim // 2),
            nn.ReLU(),
            nn.Linear(config.pseudoknot_dim // 2, 3)  # No pseudoknot, simple pseudoknot, complex pseudoknot
        )
    
    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """Predict pseudoknot type for each residue pair."""
        return self.projection(pair_repr)  # Shape: (batch_size, seq_len, seq_len, 3)


class SecondaryStructurePredictor(nn.Module):
    """Secondary structure predictor with top-k hypotheses and pseudoknot support."""
    
    def __init__(self, config: SSConfig):
        super().__init__()
        self.config = config
        
        # Input projection from LM embeddings
        self.input_proj = nn.Linear(config.d_model, config.d_model)
        
        # Initialize pairwise representations
        self.pair_init = nn.Parameter(torch.randn(1, 1, 1, config.d_model))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SSBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output heads
        self.contact_head = ContactHead(config)
        self.pseudoknot_head = PseudoknotHead(config)
        
        # Confidence prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, 
                embeddings: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            embeddings: LM embeddings of shape (batch_size, seq_len, d_model)
            mask: Optional sequence mask
            
        Returns:
            Dictionary containing:
            - contact_logits: Contact predictions for each hypothesis
            - pseudoknot_logits: Pseudoknot predictions
            - confidences: Confidence scores for each hypothesis
        """
        batch_size, seq_len, _ = embeddings.shape
        device = embeddings.device
        
        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=device)
        
        # Project input embeddings
        seq_repr = self.input_proj(embeddings)
        seq_repr = self.dropout(seq_repr)
        
        # Initialize pairwise representations
        pair_repr = self.pair_init.expand(batch_size, seq_len, seq_len, -1)
        
        # Apply transformer blocks
        for block in self.blocks:
            seq_repr, pair_repr = block(seq_repr, pair_repr, mask)
        
        # Predict contacts
        contact_logits = self.contact_head(pair_repr)
        
        # Predict pseudoknots
        pseudoknot_logits = self.pseudoknot_head(pair_repr)
        
        outputs = {
            "contact_logits": contact_logits,
            "pseudoknot_logits": pseudoknot_logits,
            "pair_repr": pair_repr
        }
        
        return outputs
    
    def sample_hypotheses(self, 
                         contact_logits: torch.Tensor,
                         pseudoknot_logits: torch.Tensor,
                         temperature: float = 1.0) -> List[Dict[str, torch.Tensor]]:
        """
        Sample top-k secondary structure hypotheses.
        
        Args:
            contact_logits: Contact predictions (batch_size, seq_len, seq_len, contact_bins)
            pseudoknot_logits: Pseudoknot predictions (batch_size, seq_len, seq_len, 3)
            temperature: Sampling temperature
            
        Returns:
            List of hypotheses, each containing:
            - contact_probs: Contact probability matrix
            - pseudoknot_probs: Pseudoknot probability matrix
            - confidence: Overall confidence score
        """
        batch_size, seq_len, _, _ = contact_logits.shape
        hypotheses = []
        
        for b in range(batch_size):
            batch_hypotheses = []
            
            # Convert logits to probabilities
            contact_probs = F.softmax(contact_logits[b] / temperature, dim=-1)
            pseudoknot_probs = F.softmax(pseudoknot_logits[b] / temperature, dim=-1)
            
            # Extract contact probability (sum of pairing bins)
            # Assuming first half of bins represent non-paired, second half paired
            contact_prob = contact_probs[..., self.config.contact_bins//2:].sum(dim=-1)
            
            # Sample different hypotheses using different strategies
            for k in range(self.config.n_hypotheses):
                # Strategy 1: Use raw probabilities
                if k == 0:
                    hyp_contact = contact_prob
                    hyp_pseudoknot = pseudoknot_probs.argmax(dim=-1).float()
                
                # Strategy 2: Thresholded contacts (more conservative)
                elif k == 1:
                    threshold = 0.3 + 0.2 * k / self.config.n_hypotheses
                    hyp_contact = (contact_prob > threshold).float()
                    hyp_pseudoknot = pseudoknot_probs.argmax(dim=-1).float()
                
                # Strategy 3: Stochastic sampling
                else:
                    hyp_contact = torch.bernoulli(contact_prob).float()
                    hyp_pseudoknot = F.one_hot(
                        torch.multinomial(pseudoknot_probs.view(-1, 3), 1).squeeze(-1),
                        num_classes=3
                    ).float().view(seq_len, seq_len, 3).sum(dim=-1)
                
                # Compute confidence
                confidence = (hyp_contact * contact_prob).mean() + \
                           (hyp_pseudoknot * pseudoknot_probs.max(dim=-1)[0]).mean()
                confidence = confidence / 2
                
                batch_hypotheses.append({
                    "contact_probs": hyp_contact,
                    "pseudoknot_probs": hyp_pseudoknot,
                    "confidence": confidence
                })
            
            hypotheses.append(batch_hypotheses)
        
        return hypotheses


def secondary_structure_loss(contact_logits: torch.Tensor,
                             target_contacts: torch.Tensor,
                             target_pseudoknots: torch.Tensor,
                             contact_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute secondary structure loss."""
    # Contact loss (cross-entropy on distance bins)
    batch_size, seq_len, seq_len, contact_bins = contact_logits.shape
    
    # Flatten for loss computation
    contact_flat = contact_logits.view(-1, contact_bins)
    target_flat = target_contacts.view(-1)
    
    if contact_mask is not None:
        contact_flat = contact_flat[contact_mask.view(-1)]
        target_flat = target_flat[contact_mask.view(-1)]
    
    contact_loss = F.cross_entropy(contact_flat, target_flat, ignore_index=-100)
    
    # Pseudoknot loss
    if target_pseudoknots is not None:
        pseudoknot_loss = F.cross_entropy(
            target_pseudoknots.view(-1, 3),
            target_pseudoknots.view(-1),
            ignore_index=-100
        )
        return contact_loss + 0.5 * pseudoknot_loss
    
    return contact_loss
