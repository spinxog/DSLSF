# RNA 3D Folding Pipeline - Core Architecture

"""RNA 3D Folding Pipeline - Core Architecture"""

import os
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
        
        # Set device
        self.device = torch.device(config.device)
        
        # Initialize components
        self.language_model = RNALanguageModel(config.lm_config)
        self.secondary_structure = SecondaryStructurePredictor(config.ss_config)
        self.structure_encoder = StructureEncoder(config.encoder_config)
        self.geometry_module = GeometryModule(config.geometry_config)
        self.sampler = RNASampler(config.sampler_config)
        self.refiner = GeometryRefiner(config.refinement_config)
        
        # Move models to device
        self.language_model.to(self.device)
        self.secondary_structure.to(self.device)
        self.structure_encoder.to(self.device)
        self.geometry_module.to(self.device)
        self.sampler.to(self.device)
        self.refiner.to(self.device)
        
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
            try:
                tokens = self._tokenize_sequence(sequence)
            except Exception as e:
                self.logger.error(f"Tokenization failed: {e}")
                return {"sequence": sequence, "error": f"Tokenization failed: {e}", "success": False}
            
            # Language model forward pass
            try:
                lm_outputs = self.language_model(tokens)
            except torch.cuda.OutOfMemoryError as e:
                self.logger.error(f"GPU out of memory in language model: {e}")
                return {"sequence": sequence, "error": "GPU out of memory", "success": False}
            except Exception as e:
                self.logger.error(f"Language model failed: {e}")
                return {"sequence": sequence, "error": f"Language model failed: {e}", "success": False}
            
            # Secondary structure prediction
            try:
                ss_outputs = self.secondary_structure(lm_outputs["embeddings"])
            except Exception as e:
                self.logger.warning(f"Secondary structure prediction failed: {e}, proceeding without it")
                ss_outputs = {"contacts": torch.zeros(len(sequence), len(sequence))}
            
            # Structure encoding
            try:
                struct_outputs = self.structure_encoder(
                    lm_outputs["embeddings"], 
                    ss_outputs["contacts"]
                )
            except Exception as e:
                self.logger.error(f"Structure encoding failed: {e}")
                return {"sequence": sequence, "error": f"Structure encoding failed: {e}", "success": False}
            
            # Geometry module
            try:
                geometry_outputs = self.geometry_module(
                    struct_outputs["embeddings"],
                    struct_outputs["pairwise_repr"]
                )
            except Exception as e:
                self.logger.error(f"Geometry module failed: {e}")
                return {"sequence": sequence, "error": f"Geometry module failed: {e}", "success": False}
            
            # Generate decoys with diverse temperatures for better exploration
            # Strategy: [low temp for confident, medium temps, high temp for diversity]
            diverse_temperatures = [0.1, 0.3, 0.5, 0.7, 1.0]  # 5 predictions with varied exploration
            try:
                decoys, metrics = self.sampler.generate_decoys(
                    sequence,
                    lm_outputs["embeddings"],
                    geometry_outputs["coordinates"],
                    return_all_decoys=True,  # Always get all 5 for best-of selection
                    temperatures=diverse_temperatures
                )
                self.logger.debug(f"Generated {len(decoys)} decoys with diverse temperatures in {metrics.total_time:.2f}s")
            except torch.cuda.OutOfMemoryError as e:
                self.logger.error(f"GPU out of memory during decoy generation: {e}")
                return {"sequence": sequence, "error": "GPU out of memory during decoy generation", "success": False}
            except Exception as e:
                self.logger.error(f"Error generating decoys: {e}")
                return {"sequence": sequence, "error": f"Failed to generate decoys: {e}", "success": False}
            
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
            
            # Calculate confidence from model outputs
            try:
                confidence = self._calculate_confidence(refined_decoys, geometry_outputs)
            except Exception as e:
                self.logger.warning(f"Confidence calculation failed: {e}, using default")
                confidence = 0.5
            
            result = {
                "sequence": sequence,
                "n_residues": len(sequence),
                "n_decoys": len(refined_decoys),
                "coordinates": refined_decoys[0]["coordinates"] if refined_decoys else None,
                "confidence": confidence,
                "success": True,
                "decoys": refined_decoys if return_all_decoys else refined_decoys[:5]
            }
            
            self.logger.info(f"Successfully predicted structure for {sequence}")
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"GPU out of memory: {e}")
            return {"sequence": sequence, "error": "GPU out of memory", "success": False}
        except Exception as e:
            self.logger.error(f"Unexpected error during prediction: {e}")
            return {
                "sequence": sequence,
                "error": f"Unexpected error: {e}",
                "success": False
            }
    
    def _tokenize_sequence(self, sequence: str) -> Dict[str, int]:
        """Tokenize RNA sequence."""
        token_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
        tokens = [token_map.get(nuc, 4) for nuc in sequence.upper()]
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
        
        # Process valid sequences in optimized batch
        batch_results = []
        if valid_sequences:
            try:
                # Create batch processing data structures
                sequences_data = [(orig_idx, sequence) for orig_idx, sequence in valid_sequences]
                max_len = max(len(seq) for _, seq in sequences_data)
                batch_size = len(sequences_data)
                
                # Batch tokenize all sequences efficiently
                batch_tokens = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
                sequence_lengths = []
                
                for batch_idx, (_, sequence) in enumerate(sequences_data):
                    token_dict = self._tokenize_sequence(sequence)
                    tokens = token_dict["tokens"]
                    batch_tokens[batch_idx, :len(tokens)] = torch.tensor(tokens, device=self.device)
                    sequence_lengths.append(len(tokens))
                
                # Create attention mask for variable length sequences
                attention_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < torch.tensor(sequence_lengths, device=self.device).unsqueeze(1)
                
                # Batch forward pass through language model
                lm_outputs = self.language_model(batch_tokens, attention_mask)
                batch_embeddings = lm_outputs["embeddings"]
                
                # Batch secondary structure prediction
                ss_outputs = self.secondary_structure(batch_embeddings)
                
                # Batch structure encoding
                struct_outputs = self.structure_encoder(batch_embeddings, ss_outputs["contacts"])
                
                # Batch geometry processing
                geometry_outputs = self.geometry_module(struct_outputs["embeddings"], struct_outputs["pairwise_repr"])
                
                # Process individual sequences for sampling and refinement (these are inherently sequential)
                for i, (orig_idx, sequence) in enumerate(sequences_data):
                    seq_len = sequence_lengths[i]
                    seq_embeddings = {k: v[i:i+1, :seq_len] for k, v in lm_outputs.items()}
                    seq_geometry = {k: v[i:i+1, :seq_len] if hasattr(v, 'shape') and len(v.shape) > 1 else v 
                                 for k, v in geometry_outputs.items()}
                    
                    try:
                        # Generate decoys for this sequence
                        decoys, metrics = self.sampler.generate_decoys(
                            sequence, seq_embeddings["embeddings"], seq_geometry["coordinates"],
                            return_all_decoys=return_all_decoys
                        )
                        
                        # Batch refinement for all decoys of this sequence
                        refined_decoys = self._batch_refine_decoys(decoys)
                        
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
                        self.logger.error(f"Failed to process sequence {orig_idx}: {e}")
                        batch_results.append({
                            "sequence": sequence,
                            "error": str(e),
                            "success": False,
                            "batch_index": orig_idx
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
    
    def _calculate_confidence(self, refined_decoys: List[Dict], geometry_outputs: Dict) -> float:
        """Calculate confidence score from model outputs and decoy consistency."""
        if not refined_decoys:
            return 0.0
        
        # Base confidence from geometry module
        base_confidence = 0.0
        if "confidence" in geometry_outputs:
            base_confidence = geometry_outputs["confidence"].mean().item() if hasattr(geometry_outputs["confidence"], 'mean') else 0.8
        else:
            base_confidence = 0.8  # Default fallback
        
        # Adjust based on decoy consistency
        decoy_confidences = [decoy.get("confidence", 0.5) for decoy in refined_decoys]
        if decoy_confidences:
            mean_confidence = sum(decoy_confidences) / len(decoy_confidences)
            confidence_variance = sum((c - mean_confidence) ** 2 for c in decoy_confidences) / len(decoy_confidences)
            
            # Higher variance reduces confidence
            consistency_factor = 1.0 - min(confidence_variance, 1.0)
        else:
            consistency_factor = 0.5
        
        # Adjust based on number of successful refinements
        refined_count = sum(1 for decoy in refined_decoys if decoy.get("refined", False))
        refinement_factor = refined_count / len(refined_decoys) if refined_decoys else 0.5
        
        # Combine factors
        final_confidence = base_confidence * consistency_factor * refinement_factor
        
        # Ensure confidence is in valid range
        return max(0.0, min(1.0, final_confidence))
    
    def _batch_refine_decoys(self, decoys: List[Dict]) -> List[Dict]:
        """Batch refine multiple decoys efficiently."""
        if not decoys:
            return []
        
        refined_decoys = []
        
        # Stack coordinates for batch processing
        coords_list = [decoy["coordinates"] for decoy in decoys]
        stacked_coords = torch.stack(coords_list) if all(isinstance(c, torch.Tensor) for c in coords_list) else None
        
        if stacked_coords is not None:
            try:
                # Batch refinement
                batch_refined = self.refiner.refine_structure(stacked_coords)
                
                # Extract individual results
                for i, decoy in enumerate(decoys):
                    refined_coords = batch_refined["refined_coordinates"][i] if batch_refined["refined_coordinates"].dim() > 0 else batch_refined["refined_coordinates"]
                    refined_decoys.append({
                        "coordinates": refined_coords,
                        "confidence": batch_refined["final_loss"] if isinstance(batch_refined["final_loss"], (int, float)) else batch_refined["final_loss"][i],
                        "refined": True,
                        "original_confidence": decoy["confidence"],
                        "decoy_id": decoy["decoy_id"]
                    })
            except (RuntimeError, torch.cuda.OutOfMemoryError, ValueError) as e:
                self.logger.warning(f"Batch refinement failed ({type(e).__name__}), falling back to individual: {e}")
                # Fall back to individual processing
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
                    except (RuntimeError, ValueError, KeyError) as e2:
                        self.logger.warning(f"Individual refinement failed for decoy {decoy['decoy_id']} ({type(e2).__name__}): {e2}")
                        refined_decoys.append({
                            "coordinates": decoy["coordinates"],
                            "confidence": decoy["confidence"],
                            "refined": False,
                            "decoy_id": decoy["decoy_id"]
                        })
        else:
            # Individual processing for mixed tensor types
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
                except (RuntimeError, ValueError, KeyError, AttributeError) as e:
                    self.logger.warning(f"Refinement failed for decoy {decoy['decoy_id']} ({type(e).__name__}), using original: {e}")
                    refined_decoys.append({
                        "coordinates": decoy["coordinates"],
                        "confidence": decoy["confidence"],
                        "refined": False,
                        "decoy_id": decoy["decoy_id"]
                    })
        
        return refined_decoys
    
    def load_model(self, model_path: str, device: str = "auto") -> bool:
        """Load model from checkpoint with comprehensive security validation."""
        # Input validation
        if not model_path or not isinstance(model_path, str):
            raise ValueError("Model path must be a non-empty string")
        
        # Path traversal protection
        model_path = Path(model_path).resolve()
        try:
            # Ensure path is within expected bounds (prevent directory traversal)
            model_path.relative_to(Path.cwd())
        except ValueError:
            raise ValueError(f"Path traversal detected: {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not model_path.suffix in ['.pth', '.pt']:
            raise ValueError(f"Invalid model file extension: {model_path.suffix}")
        
        # Security check: ensure file is not suspiciously large
        file_size = model_path.stat().st_size
        if file_size > 10 * 1024 * 1024 * 1024:  # 10GB limit
            raise ValueError(f"Model file too large: {file_size / (1024**3):.1f}GB")
        
        # Additional security: check file permissions
        if model_path.is_file() and not os.access(model_path, os.R_OK):
            raise PermissionError(f"No read permission for model file: {model_path}")
        
        try:
            # Enhanced security loading
            try:
                # First attempt with maximum security
                checkpoint = torch.load(model_path, map_location=self.config.device, weights_only=True)
            except (RuntimeError, pickle.UnpicklingError) as e:
                # Fallback with additional validation if weights_only fails
                self.logger.warning(f"weights_only loading failed, attempting with validation: {e}")
                checkpoint = self._secure_load_checkpoint(model_path)
            
            # Comprehensive checkpoint validation
            if not isinstance(checkpoint, dict):
                raise ValueError("Checkpoint must be a dictionary")
            
            # Validate checkpoint structure and metadata
            self._validate_checkpoint_structure(checkpoint)
            
            # Validate each model state dict for security
            self._validate_model_state_dicts(checkpoint)
            
            # Load state dictionaries with error handling
            self._load_model_states(checkpoint)
            
            # Set models to eval mode for inference
            self._set_eval_mode()
            
            self.logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except torch.serialization.pickle.UnpicklingError as e:
            self.logger.error(f"Corrupted model file: {e}")
            raise ValueError(f"Invalid or corrupted model file: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _secure_load_checkpoint(self, model_path: Path) -> dict:
        """Secure checkpoint loading with additional validation."""
        import pickle
        import tempfile
        import hashlib
        
        # Calculate file hash for integrity check
        file_hash = hashlib.sha256()
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)
        
        # Load in a controlled environment
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            # Copy file to temporary location
            import shutil
            shutil.copy2(model_path, temp_file.name)
            
            # Load with restricted pickle environment
            class SafeUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # Only allow safe classes from specific modules
                    allowed_modules = {
                        'torch', 'torch._utils', 'collections', 'builtins',
                        'numpy', 'numpy.core.multiarray'
                    }
                    allowed_classes = {
                        'OrderedDict', 'dict', 'list', 'tuple', 'set', 'frozenset',
                        'int', 'float', 'str', 'bool', 'bytes', 'bytearray'
                    }
                    
                    if module in allowed_modules and name in allowed_classes:
                        return super().find_class(module, name)
                    raise pickle.UnpicklingError(f"Unsafe class {module}.{name}")
                
                def load(self):
                    # Additional validation during loading
                    data = super().load()
                    if not isinstance(data, dict):
                        raise pickle.UnpicklingError("Checkpoint must be a dictionary")
                    return data
            
            try:
                with open(temp_file.name, 'rb') as f:
                    checkpoint = SafeUnpickler(f).load()
                
                # Verify checkpoint structure
                required_keys = {'language_model', 'secondary_structure', 'structure_encoder', 'geometry_module'}
                missing_keys = required_keys - set(checkpoint.keys())
                if missing_keys:
                    raise ValueError(f"Missing required keys in checkpoint: {missing_keys}")
                
                # Validate tensor data types
                for key, value in checkpoint.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if hasattr(sub_value, 'dtype'):
                                # Check for reasonable tensor data types
                                if sub_value.dtype not in [torch.float32, torch.float64, torch.float16, torch.int32, torch.int64]:
                                    raise ValueError(f"Invalid tensor dtype in {key}.{sub_key}: {sub_value.dtype}")
                
                return checkpoint
                
            except Exception as e:
                raise ValueError(f"Failed to load checkpoint securely: {e}")
    
    def _validate_checkpoint_structure(self, checkpoint: dict) -> None:
        """Validate checkpoint structure and metadata."""
        # Check for required keys
        required_keys = ['language_model', 'secondary_structure', 'structure_encoder', 'geometry_module']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise KeyError(f"Missing required keys in checkpoint: {missing_keys}")
        
        # Validate metadata if present
        if 'metadata' in checkpoint:
            metadata = checkpoint['metadata']
            if not isinstance(metadata, dict):
                raise ValueError("Checkpoint metadata must be a dictionary")
            
            # Check version compatibility
            if 'version' in metadata:
                version = metadata['version']
                if not isinstance(version, str):
                    raise ValueError("Checkpoint version must be a string")
        
        # Check for suspicious top-level keys
        suspicious_keys = ['__builtins__', '__import__', 'eval', 'exec', 'compile', '__code__']
        for key in checkpoint.keys():
            if any(suspicious in key.lower() for suspicious in suspicious_keys):
                raise ValueError(f"Suspicious key found in checkpoint: {key}")
    
    def _validate_model_state_dicts(self, checkpoint: dict) -> None:
        """Validate individual model state dictionaries."""
        required_keys = ['language_model', 'secondary_structure', 'structure_encoder', 'geometry_module']
        
        for model_name in required_keys:
            state_dict = checkpoint[model_name]
            if not isinstance(state_dict, dict):
                raise ValueError(f"State dict for {model_name} must be a dictionary")
            
            # Check for suspicious keys
            suspicious_keys = ['__builtins__', '__import__', 'eval', 'exec', 'compile', '__code__']
            for key in state_dict.keys():
                if any(suspicious in key.lower() for suspicious in suspicious_keys):
                    raise ValueError(f"Suspicious key found in {model_name} state dict: {key}")
            
            # Validate tensor properties
            self._validate_tensor_properties(state_dict, model_name)
    
    def _validate_tensor_properties(self, state_dict: dict, model_name: str) -> None:
        """Validate tensor properties in state dict."""
        for param_name, param_tensor in state_dict.items():
            if not isinstance(param_tensor, torch.Tensor):
                raise ValueError(f"Parameter {param_name} in {model_name} is not a tensor")
            
            # Check for reasonable tensor sizes
            if param_tensor.numel() > 1e9:  # > 1GB tensor
                raise ValueError(f"Parameter {param_name} in {model_name} is too large: {param_tensor.numel()} elements")
            
            # Check for NaN or Inf values
            if torch.isnan(param_tensor).any() or torch.isinf(param_tensor).any():
                raise ValueError(f"Parameter {param_name} in {model_name} contains NaN or Inf values")
            
            # Check tensor dtype is reasonable
            valid_dtypes = [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int8, torch.int16, torch.int32, torch.int64]
            if param_tensor.dtype not in valid_dtypes:
                raise ValueError(f"Parameter {param_name} in {model_name} has invalid dtype: {param_tensor.dtype}")
    
    def _load_model_states(self, checkpoint: dict) -> None:
        """Load model states with error handling."""
        required_keys = ['language_model', 'secondary_structure', 'structure_encoder', 'geometry_module']
        
        for model_name in required_keys:
            try:
                model = getattr(self, model_name)
                model.load_state_dict(checkpoint[model_name], strict=True)
            except AttributeError:
                raise ValueError(f"Model {model_name} not found in pipeline")
            except RuntimeError as e:
                raise ValueError(f"Failed to load state dict for {model_name}: {e}")
    
    def _set_eval_mode(self) -> None:
        """Set all models to evaluation mode."""
        models = [self.language_model, self.secondary_structure, self.structure_encoder, 
                 self.geometry_module, self.sampler, self.refiner]
        
        for model in models:
            model.eval()

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