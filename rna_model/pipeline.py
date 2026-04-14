# RNA 3D Folding Pipeline - Core Architecture

"""RNA 3D Folding Pipeline - Core Architecture"""

import torch
import torch.nn as nn

from .language_model import RNALanguageModel
from .secondary_structure import SecondaryStructurePredictor
from .structure_encoder import StructureEncoder
from .geometry_module import GeometryModule
from .sampler import RNASampler, SamplerConfig
from .refinement import GeometryRefiner
from .config import GlobalConfig, get_config, validate_config
from .logging_config import setup_logger, StructuredLogger
from .data import RNADatasetLoader, RNAStructure
from .training import Trainer, TrainingConfig
from .evaluation import StructureEvaluator, EvaluationMetrics
from .utils import (
    compute_tm_score, compute_rmsd, superimpose_coordinates,
    compute_contact_map, bin_distances, mask_sequence,
    set_seed, clear_cache, memory_usage
)

# Main pipeline class
class RNAFoldingPipeline:
    """Main RNA 3D folding pipeline."""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger("rna_folding")
        
        # Initialize components
        self.language_model = RNALanguageModel(config.lm_config)
        self.secondary_structure = SecondaryStructurePredictor(config.ss_config)
        self.structure_encoder = StructureEncoder(config.encoder_config)
        self.geometry_module = GeometryModule(config.geometry_config)
        self.sampler = RNASampler(config.sampler_config)
        self.refiner = GeometryRefiner(config.refinement_config)
        
        self.logger.info("RNA 3D folding pipeline initialized")
    
    def predict_single_sequence(self, sequence: str, return_all_decoys: bool = False) -> dict:
        """Predict structure for a single RNA sequence."""
        self.logger.info(f"Predicting structure for sequence: {sequence[:20]}...")
        
        try:
            # Tokenize sequence
            tokens = self._tokenize_sequence(sequence)
            
            # Language model forward pass
            lm_outputs = self.language_model(tokens)
            
            # Secondary structure prediction
            ss_outputs = self.secondary_structure(lm_outputs["embeddings"])
            
            # Structure encoding
            struct_outputs = self.structure_encoder(
                lm_outputs["embeddings"], 
                ss_outputs["contacts"]
            )
            
            # Geometry module
            geometry_outputs = self.geometry_module(
                struct_outputs["embeddings"],
                struct_outputs["pairwise_repr"]
            )
            
            # Generate decoys
            decoys = self.sampler.generate_decoys(
                sequence,
                lm_outputs["embeddings"],
                geometry_outputs["coordinates"],
                return_all_decoys=return_all_decoys
            )
            
            # Refinement
            refined_decoys = []
            for decoy in decoys:
                refined = self.refiner.refine_structure(decoy["coordinates"])
                refined_decoys.append({
                    **decoy,
                    "coordinates": refined["coordinates"],
                    "refined": True
                })
            
            result = {
                "sequence": sequence,
                "n_residues": len(sequence),
                "n_decoys": len(refined_decoys),
                "coordinates": refined_decoys[0]["coordinates"] if refined_decoys else None,
                "confidence": 0.8,  # Default confidence
                "success": True,
                "decoys": refined_decoys if return_all_decoys else refined_decocoys[:5]
            }
            
            self.logger.info(f"Successfully predicted structure for {sequence}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to predict structure: {e}")
            return {
                "sequence": sequence,
                "error": str(e),
                "success": False
            }
    
    def _tokenize_sequence(self, sequence: str) -> dict:
        """Tokenize RNA sequence."""
        token_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
        tokens = [token_map.get(nuc, 4) for nucleotide in sequence.upper()]
        return {"tokens": tokens, "length": len(tokens)}
    
    def load_model(self, model_path: str):
        """Load model from checkpoint."""
        try:
            checkpoint = torch.load(model_path, map_location=self.config.device)
            self.language_model.load_state_dict(checkpoint.get('language_model', {}))
            self.secondary_structure.load_state_dict(checkpoint.get('secondary_structure', {}))
            self.structure_encoder.load_state_dict(checkpoint.get('structure_encoder', {}))
            self.geometry_module.load_state_dict(checkpoint.get('geometry_module', {}))
            self.logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

# Configuration class
class PipelineConfig:
    """Configuration for the RNA folding pipeline."""
    
    def __init__(self, device="auto", max_sequence_length=512, mixed_precision=True):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_sequence_length = max_sequence_length
        self.mixed_precision = mixed_precision
        
        # Sub-configurations
        from .language_model import LMConfig
        from .secondary_structure import SSConfig
        from .structure_encoder import EncoderConfig
        from .geometry_module import GeometryConfig
        from .refinement import RefinementConfig
        
        self.lm_config = LMConfig()
        self.ss_config = SSConfig()
        self.encoder_config = EncoderConfig()
        self.geometry_config = GeometryConfig()
        self.sampler_config = SamplerConfig()
        self.refinement_config = RefinementConfig()

# Integrated model class
class IntegratedModel(nn.Module):
    """Integrated model combining all components."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.language_model = RNALanguageModel(config.lm_config)
        self.secondary_structure = SecondaryStructurePredictor(config.ss_config)
        self.structure_encoder = StructureEncoder(config.encoder_config)
        self.geometry_module = GeometryModule(config.geometry_config)
        
    def forward(self, tokens, mask=None, coordinates=None):
        """Forward pass through all components."""
        # Language model
        lm_outputs = self.language_model(tokens, mask)
        
        # Secondary structure
        ss_outputs = self.secondary_structure(lm_outputs["embeddings"])
        
        # Structure encoding
        struct_outputs = self.structure_encoder(
            lm_outputs["embeddings"], 
            ss_outputs["contacts"]
        )
        
        # Geometry module
        geometry_outputs = self.geometry_module(
            struct_outputs["embeddings"],
            struct_outputs["pairwise_repr"]
        )
        
        return {
            "lm_outputs": lm_outputs,
            "ss_outputs": ss_outputs,
            "struct_outputs": struct_outputs,
            "geometry_outputs": geometry_outputs
        }

__all__ = [
    "RNAFoldingPipeline",
    "PipelineConfig", 
    "IntegratedModel",
    "RNALanguageModel",
    "SecondaryStructurePredictor",
    "StructureEncoder",
    "GeometryModule",
    "RNASampler",
    "GeometryRefiner",
    "GlobalConfig",
    "get_config",
    "validate_config",
    "setup_logger",
    "StructuredLogger",
    "RNADatasetLoader",
    "RNAStructure",
    "Trainer",
    "TrainingConfig",
    "StructureEvaluator",
    "EvaluationMetrics",
    "compute_tm_score",
    "compute_rmsd",
    "superimpose_coordinates",
    "compute_contact_map",
    "bin_distances",
    "mask_sequence",
    "set_seed",
    "clear_cache",
    "memory_usage",
]