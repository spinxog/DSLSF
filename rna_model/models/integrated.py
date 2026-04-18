"""Integrated model combining all RNA 3D folding components."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .language_model import RNALanguageModel, LMConfig
from .secondary_structure import SecondaryStructurePredictor
from .structure_encoder import StructureEncoder
from .msa_module import MSAModule, MSAConfig
from ..core.geometry_module import GeometryModule, GeometryConfig
from ..core.sampler import RNASampler, SamplerConfig
from ..core.refinement import GeometryRefiner, RefinementConfig


class IntegratedModel(nn.Module):
    """Integrated model combining all components."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize all components
        self.language_model = RNALanguageModel(config.lm_config)
        
        # MSA module for processing multiple sequence alignments
        # Uses same d_model as language model for easy integration
        self.msa_config = MSAConfig(
            d_model=config.lm_config.d_model,
            n_heads=8,
            n_layers=4,
            use_msa=getattr(config, 'use_msa', True)
        )
        self.msa_module = MSAModule(self.msa_config)
        
        self.secondary_structure = SecondaryStructurePredictor(config.ss_config)
        self.structure_encoder = StructureEncoder(config.encoder_config)
        
        # Projection layer to bridge encoder output (256d) to geometry module (512d)
        self.encoder_to_geometry_proj = nn.Linear(
            config.encoder_config.d_model, 
            config.geometry_config.d_model
        )
        
        self.geometry_module = GeometryModule(config.geometry_config)
        self.sampler = RNASampler(config.sampler_config)
        self.refiner = GeometryRefiner(config.refinement_config)
        
    def forward(self, 
                tokens: torch.Tensor, 
                attention_mask: torch.Tensor = None,
                msa_tokens: Optional[torch.Tensor] = None,
                msa_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Forward pass through all components.
        
        Args:
            tokens: Input sequence tokens (batch_size, seq_len)
            attention_mask: Optional attention mask
            msa_tokens: Optional MSA tokens (batch_size, n_seqs, seq_len)
            msa_mask: Optional MSA mask (batch_size, n_seqs, seq_len)
        
        Returns:
            Dictionary with model outputs
        """
        # Language model
        lm_outputs = self.language_model(tokens, attention_mask)
        embeddings = lm_outputs["embeddings"]
        
        # Augment with MSA if provided
        if msa_tokens is not None and self.msa_config.use_msa:
            embeddings = self.msa_module(msa_tokens, embeddings, msa_mask)
        
        # Secondary structure prediction
        ss_outputs = self.secondary_structure(embeddings)
        
        # Structure encoding - use pair_repr from secondary structure outputs
        struct_outputs = self.structure_encoder(embeddings, ss_outputs["pair_repr"])
        
        # Project encoder output to geometry module dimensions
        seq_len = embeddings.size(1)
        projected_embeddings = self.encoder_to_geometry_proj(struct_outputs["embeddings"])
        
        # Geometry module
        pair_repr = projected_embeddings.unsqueeze(2).expand(-1, -1, seq_len, -1) + \
                   projected_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
        geometry_outputs = self.geometry_module(projected_embeddings, pair_repr)
        
        return {
            "embeddings": embeddings,
            "ss_outputs": ss_outputs,
            "struct_outputs": struct_outputs,
            "geometry_outputs": geometry_outputs,
            "logits": lm_outputs["logits"]
        }
    
    def enable_msa(self):
        """Enable MSA processing."""
        self.msa_module.enable_msa()
    
    def disable_msa(self):
        """Disable MSA processing."""
        self.msa_module.disable_msa()
