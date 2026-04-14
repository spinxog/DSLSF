# RNA 3D Folding Pipeline - Core Architecture

"""RNA 3D Folding Pipeline - Core Architecture"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..models.language_model import RNALanguageModel
from ..models.secondary_structure import SecondaryStructurePredictor
from ..models.structure_encoder import StructureEncoder
from .geometry_module import GeometryModule
from .sampler import RNASampler, SamplerConfig
from .refinement import GeometryRefiner
from .config import GlobalConfig, get_config, validate_config
from .logging_config import setup_logging, StructuredLogger
from ..data import DatasetManager, RNAStructure
from ..training import Trainer, TrainingConfig
from ..evaluation import StructureEvaluator, EvaluationMetrics
from .utils import (
    compute_tm_score, compute_rmsd, superimpose_coordinates,
    compute_contact_map, bin_distances, mask_sequence,
    set_seed, clear_cache, memory_usage
)

# Main pipeline class
class RNAFoldingPipeline:
    """Main RNA 3D folding pipeline."""
    
    def __init__(self, config: "PipelineConfig") -> None:
        self.config = config
        self.logger = setup_logging("rna_folding")
        
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
        # Input validation
        if not sequence or not isinstance(sequence, str):
            error_msg = "Invalid sequence: sequence must be a non-empty string"
            self.logger.error(error_msg)
            return {"sequence": sequence, "error": error_msg, "success": False}
        
        if len(sequence) > self.config.max_sequence_length:
            error_msg = f"Sequence too long: {len(sequence)} > {self.config.max_sequence_length}"
            self.logger.error(error_msg)
            return {"sequence": sequence, "error": error_msg, "success": False}
        
        # Validate sequence contains only valid nucleotides
        valid_nucleotides = set('AUGCaugcNn')
        invalid_chars = set(sequence.upper()) - valid_nucleotides
        if invalid_chars:
            error_msg = f"Invalid nucleotides in sequence: {invalid_chars}"
            self.logger.error(error_msg)
            return {"sequence": sequence, "error": error_msg, "success": False}
        
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
            try:
                decoys, metrics = self.sampler.generate_decoys(
                    sequence,
                    lm_outputs["embeddings"],
                    geometry_outputs["coordinates"],
                    return_all_decoys=return_all_decoys
                )
                self.logger.debug(f"Generated {len(decoys)} decoys in {metrics.total_time:.2f}s")
            except Exception as e:
                self.logger.error(f"Error generating decoys: {e}")
                raise RuntimeError(f"Failed to generate decoys for sequence {sequence[:20]}...: {e}")
            
            # Refinement
            refined_decoys = []
            for i, decoy in enumerate(decoys):
                try:
                    refined = self.refiner.refine_structure(decoy["coordinates"])
                    refined_decoys.append({
                        "coordinates": refined["coordinates"],
                        "confidence": refined["loss"],
                        "refined": True,
                        "original_confidence": decoy["confidence"],
                        "decoy_id": decoy["decoy_id"]
                    })
                except Exception as e:
                    self.logger.warning(f"Refinement failed for decoy {i}, using original: {e}")
                    refined_decoys.append({
                        "coordinates": decoy["coordinates"],
                        "confidence": decoy["confidence"],
                        "refined": False,
                        "decoy_id": decoy["decoy_id"]
                    })
            
            result = {
                "sequence": sequence,
                "n_residues": len(sequence),
                "n_decoys": len(refined_decoys),
                "coordinates": refined_decoys[0]["coordinates"] if refined_decoys else None,
                "confidence": 0.8,  # Default confidence
                "success": True,
                "decoys": refined_decoys if return_all_decoys else refined_decoys[:5]
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
    
    def _tokenize_sequence(self, sequence: str) -> Dict[str, int]:
        """Tokenize RNA sequence."""
        token_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
        tokens = [token_map.get(nuc, 4) for nucleotide in sequence.upper()]
        return {"tokens": tokens, "length": len(tokens)}
    
    def predict_batch(self, sequences: List[str], return_all_decoys: bool = False) -> List[Dict[str, Any]]:
        """Predict structures for multiple sequences in batch.
        
        Args:
            sequences: List of RNA sequences to process
            return_all_decoys: Whether to return all decoys or just the best
            
        Returns:
            List of prediction results, one per sequence
        """
        if not sequences:
            return []
        
        self.logger.info(f"Processing batch of {len(sequences)} sequences")
        
        # Filter and validate sequences
        valid_sequences = []
        invalid_results = []
        
        for i, sequence in enumerate(sequences):
            if not sequence or not isinstance(sequence, str):
                invalid_results.append({
                    "sequence": sequence,
                    "error": "Invalid sequence input",
                    "success": False,
                    "batch_index": i
                })
                continue
            
            if len(sequence) > self.config.max_sequence_length:
                invalid_results.append({
                    "sequence": sequence,
                    "error": f"Sequence too long: {len(sequence)} > {self.config.max_sequence_length}",
                    "success": False,
                    "batch_index": i
                })
                continue
            
            valid_sequences.append((i, sequence))
        
        # Process valid sequences in batch
        batch_results = []
        if valid_sequences:
            try:
                # Tokenize all sequences at once
                max_len = max(len(seq) for _, seq in valid_sequences)
                batch_tokens = torch.zeros(len(valid_sequences), max_len, dtype=torch.long, device=self.device)
                
                for batch_idx, (_, sequence) in enumerate(valid_sequences):
                    token_dict = self._tokenize_sequence(sequence)
                    tokens = token_dict["tokens"]
                    batch_tokens[batch_idx, :len(tokens)] = tokens
                
                # Forward pass for batch
                lm_outputs = self.language_model(batch_tokens)
                
                # Process each sequence individually for now (can be further optimized)
                for orig_idx, sequence in valid_sequences:
                    seq_idx = next(i for i, (idx, _) in enumerate(valid_sequences) if idx == orig_idx)
                    seq_tokens = batch_tokens[seq_idx:seq_idx+1]
                    seq_embeddings = {k: v[seq_idx:seq_idx+1] for k, v in lm_outputs.items()}
                    
                    # Rest of pipeline processing
                    ss_outputs = self.secondary_structure(seq_embeddings["embeddings"])
                    struct_outputs = self.structure_encoder(seq_embeddings["embeddings"], ss_outputs["contacts"])
                    geometry_outputs = self.geometry_module(struct_outputs["embeddings"], struct_outputs["pairwise_repr"])
                    
                    # Generate decoys
                    decoys, metrics = self.sampler.generate_decoys(
                        sequence, seq_embeddings["embeddings"], geometry_outputs["coordinates"],
                        return_all_decoys=return_all_decoys
                    )
                    
                    # Refinement
                    refined_decoys = []
                    for decoy in decoys:
                        try:
                            refined = self.refiner.refine_structure(decoy["coordinates"])
                            refined_decoys.append({
                                "coordinates": refined["coordinates"],
                                "confidence": refined["loss"],
                                "refined": True,
                                "original_confidence": decoy["confidence"],
                                "decoy_id": decoy["decoy_id"]
                            })
                        except Exception as e:
                            self.logger.warning(f"Refinement failed, using original: {e}")
                            refined_decoys.append({
                                "coordinates": decoy["coordinates"],
                                "confidence": decoy["confidence"],
                                "refined": False,
                                "decoy_id": decoy["decoy_id"]
                            })
                    
                    batch_results.append({
                        "sequence": sequence,
                        "n_residues": len(sequence),
                        "n_decoys": len(refined_decoys),
                        "coordinates": refined_decoys[0]["coordinates"] if refined_decoys else None,
                        "confidence": 0.8,
                        "success": True,
                        "decoys": refined_decoys if return_all_decoys else refined_decoys[:5],
                        "batch_index": orig_idx,
                        "metrics": metrics
                    })
                    
            except Exception as e:
                self.logger.error(f"Batch processing failed: {e}")
                # Fall back to individual processing
                for orig_idx, sequence in valid_sequences:
                    result = self.predict_single_sequence(sequence, return_all_decoys)
                    result["batch_index"] = orig_idx
                    batch_results.append(result)
        
        # Combine results maintaining original order
        all_results = invalid_results + batch_results
        all_results.sort(key=lambda x: x["batch_index"])
        
        self.logger.info(f"Batch processing complete: {len(all_results)} results")
        return all_results
    
    def load_model(self, model_path: str, device: str = "auto") -> bool:
        """Load model from checkpoint with security validation."""
        # Input validation
        if not model_path or not isinstance(model_path, str):
            raise ValueError("Model path must be a non-empty string")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not model_path.suffix in ['.pth', '.pt']:
            raise ValueError(f"Invalid model file extension: {model_path.suffix}")
        
        # Security check: ensure file is not suspiciously large
        file_size = model_path.stat().st_size
        if file_size > 10 * 1024 * 1024 * 1024:  # 10GB limit
            raise ValueError(f"Model file too large: {file_size / (1024**3):.1f}GB")
        
        try:
            # Load with weights_only=True for security
            checkpoint = torch.load(model_path, map_location=self.config.device, weights_only=True)
            
            # Comprehensive checkpoint validation
            if not isinstance(checkpoint, dict):
                raise ValueError("Checkpoint must be a dictionary")
            
            # Validate checkpoint structure
            required_keys = ['language_model', 'secondary_structure', 'structure_encoder', 'geometry_module']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                raise KeyError(f"Missing required keys in checkpoint: {missing_keys}")
            
            # Validate each model state dict
            for model_name in required_keys:
                state_dict = checkpoint[model_name]
                if not isinstance(state_dict, dict):
                    raise ValueError(f"State dict for {model_name} must be a dictionary")
                
                # Check for suspicious keys (potential security risk)
                suspicious_keys = ['__builtins__', '__import__', 'eval', 'exec', 'compile']
                for key in state_dict.keys():
                    if any(suspicious in key.lower() for suspicious in suspicious_keys):
                        raise ValueError(f"Suspicious key found in {model_name} state dict: {key}")
                
                # Validate tensor shapes and types
                for param_name, param_tensor in state_dict.items():
                    if not isinstance(param_tensor, torch.Tensor):
                        raise ValueError(f"Parameter {param_name} in {model_name} is not a tensor")
                    
                    # Check for reasonable tensor sizes
                    if param_tensor.numel() > 1e9:  # > 1GB tensor
                        raise ValueError(f"Parameter {param_name} in {model_name} is too large: {param_tensor.numel()} elements")
            
            # Load state dictionaries with validation
            self.language_model.load_state_dict(checkpoint['language_model'], strict=True)
            self.secondary_structure.load_state_dict(checkpoint['secondary_structure'], strict=True)
            self.structure_encoder.load_state_dict(checkpoint['structure_encoder'], strict=True)
            self.geometry_module.load_state_dict(checkpoint['geometry_module'], strict=True)
            
            # Set models to eval mode for inference
            self.language_model.eval()
            self.secondary_structure.eval()
            self.structure_encoder.eval()
            self.geometry_module.eval()
            self.sampler.eval()
            self.refiner.eval()
            
            self.logger.info(f"Model loaded successfully from {model_path}")
            
        except torch.serialization.pickle.UnpicklingError as e:
            self.logger.error(f"Corrupted model file: {e}")
            raise ValueError(f"Invalid or corrupted model file: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

# Configuration class
class PipelineConfig:
    """Configuration for the RNA folding pipeline."""
    
    def __init__(self, device="auto", max_sequence_length=512, mixed_precision=True):
        # Validate inputs
        if device not in ["auto", "cpu", "cuda"]:
            raise ValueError(f"Invalid device: {device}. Must be 'auto', 'cpu', or 'cuda'")
        
        if not isinstance(max_sequence_length, int) or max_sequence_length <= 0:
            raise ValueError(f"max_sequence_length must be a positive integer, got {max_sequence_length}")
        
        if max_sequence_length > 10000:
            raise ValueError(f"max_sequence_length too large: {max_sequence_length}. Maximum allowed is 10000")
        
        if not isinstance(mixed_precision, bool):
            raise ValueError(f"mixed_precision must be boolean, got {type(mixed_precision)}")
        
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
    "setup_logging",
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