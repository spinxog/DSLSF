"""Integrated model combining all RNA 3D folding components."""

import torch
import torch.nn as nn
from typing import Dict, Any

from .language_model import RNALanguageModel, LMConfig
from .secondary_structure import SecondaryStructurePredictor
from .structure_encoder import StructureEncoder
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
        self.secondary_structure = SecondaryStructurePredictor(config.ss_config)
        self.structure_encoder = StructureEncoder(config.encoder_config)
        self.geometry_module = GeometryModule(config.geometry_config)
        self.sampler = RNASampler(config.sampler_config)
        self.refiner = GeometryRefiner(config.refinement_config)
        
    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor = None) -> Dict[str, Any]:
        """Forward pass through all components."""
        # Language model
        lm_outputs = self.language_model(tokens, attention_mask)
        embeddings = lm_outputs["embeddings"]
        
        # Secondary structure prediction
        ss_outputs = self.secondary_structure(embeddings)
        
        # Structure encoding - use pair_repr from secondary structure outputs
        struct_outputs = self.structure_encoder(embeddings, ss_outputs["pair_repr"])
        
        # Geometry module
        seq_len = embeddings.size(1)
        pair_repr = struct_outputs["embeddings"].unsqueeze(2).expand(-1, -1, seq_len, -1) + \
                   struct_outputs["embeddings"].unsqueeze(1).expand(-1, seq_len, -1, -1)
        geometry_outputs = self.geometry_module(struct_outputs["embeddings"], pair_repr)
        
        return {
            "embeddings": embeddings,
            "ss_outputs": ss_outputs,
            "struct_outputs": struct_outputs,
            "geometry_outputs": geometry_outputs,
            "logits": lm_outputs["logits"]
        }
