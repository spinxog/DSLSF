"""Complete RNA 3D Folding Pipeline"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import time
import logging
import gc
from contextlib import contextmanager

from .language_model import RNALanguageModel, LMConfig
from .secondary_structure import SecondaryStructurePredictor, SSConfig
from .structure_encoder import StructureEncoder, EncoderConfig
from .geometry_module import GeometryModule, GeometryConfig
from .sampler import RNASampler, SamplerConfig
from .refinement import GeometryRefiner, RefinementConfig


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    # Model configs
    lm_config: LMConfig = field(default_factory=LMConfig)
    ss_config: SSConfig = field(default_factory=SSConfig)
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    geometry_config: GeometryConfig = field(default_factory=GeometryConfig)
    sampler_config: SamplerConfig = field(default_factory=SamplerConfig)
    refinement_config: RefinementConfig = field(default_factory=RefinementConfig)
    
    # Pipeline settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    compile_model: bool = False
    
    # Competition settings
    max_sequence_length: int = 512
    inference_timeout: float = 144.0  # seconds per sequence (8h / 200 sequences)


class IntegratedModel(nn.Module):
    """Integrated model combining all components."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config
        
        # Model components
        self.language_model = RNALanguageModel(config.lm_config)
        self.ss_predictor = SecondaryStructurePredictor(config.ss_config)
        self.structure_encoder = StructureEncoder(config.encoder_config)
        self.geometry_module = GeometryModule(config.geometry_config)
        
        # Confidence head for ranking
        self.confidence_head = nn.Sequential(
            nn.Linear(config.geometry_config.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self,
                embeddings: torch.Tensor,
                ss_contacts: Optional[torch.Tensor] = None,
                ss_pseudoknots: Optional[torch.Tensor] = None,
                msa_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through integrated model."""
        batch_size, seq_len, _ = embeddings.shape
        
        # Encode sequence
        encoded = self.structure_encoder(embeddings)
        
        # Create pairwise representations
        pair_repr = encoded.unsqueeze(2).expand(-1, -1, seq_len, -1) + \
                   encoded.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
        # Add secondary structure information if provided
        if ss_contacts is not None:
            ss_expanded = ss_contacts.unsqueeze(-1).expand(-1, -1, -1, self.config.geometry_config.d_model)
            pair_repr = pair_repr + ss_expanded * 0.1
        
        # Geometry prediction
        geometry_outputs = self.geometry_module(encoded, pair_repr)
        
        # Add confidence prediction
        geometry_outputs["confidence"] = self.confidence_head(encoded)
        
        return geometry_outputs


class RNAFoldingPipeline:
    """Complete RNA 3D folding pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models
        self.model = IntegratedModel(config).to(self.device)
        self.sampler = RNASampler(config.sampler_config).to(self.device)
        self.refiner = GeometryRefiner(config.refinement_config).to(self.device)
        
        # Optional: Fast refiner for competition
        self.fast_refiner = nn.ModuleDict({
            "geometry": self.refiner,
            "simple": nn.Module()  # Placeholder for ultra-fast refiner
        })
        
        # Model compilation for speed (if available)
        if config.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                print("Model compiled for speed optimization")
            except Exception as e:
                print(f"Model compilation failed: {e}")
    
    @contextmanager
    def _memory_context(self):
        """Context manager for memory management."""
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def predict_single_sequence(self,
                               sequence: str,
                               msa_data: Optional[np.ndarray] = None,
                               return_all_decoys: bool = False) -> Dict[str, any]:
        """
        Predict 3D structure for a single RNA sequence.
        
        Args:
            sequence: RNA sequence (A, U, G, C)
            msa_data: Optional MSA data
            return_all_decoys: Whether to return all decoys or just top 5
            
        Returns:
            Dictionary with predictions
        """
        start_time = time.time()
        
        # Validate sequence
        if not sequence or len(sequence) == 0:
            raise ValueError("Empty sequence provided")
        if len(sequence) > self.config.max_sequence_length:
            raise ValueError(f"Sequence too long: {len(sequence)} > {self.config.max_sequence_length}")
        if not all(nucleotide in 'AUGCaugc' for nucleotide in sequence):
            raise ValueError(f"Invalid nucleotides in sequence: {sequence}")
        
        with self._memory_context():
            try:
                # Tokenize sequence
                tokens = self._tokenize_sequence(sequence)
                
                # Get LM embeddings (cached if possible)
                embeddings = self._get_embeddings(tokens)
                
                # Predict secondary structure
                ss_hypotheses = self._predict_secondary_structure(embeddings)
                
                # Prepare MSA features
                msa_features = self._prepare_msa_features(msa_data) if msa_data is not None else None
                
                # Generate diverse decoys
                decoys = self.sampler.sample_decoys(
                    self.model, embeddings, ss_hypotheses, msa_features, self.device
                )
                
                # Cluster and select top 5
                selected_decoys = self.sampler.cluster_and_select(decoys, n_selected=5)
                
                # Refine structures
                refined_decoys = []
                for decoy in selected_decoys:
                    if "coordinates" in decoy:
                        refined = self._refine_structure(decoy)
                        refined_decoys.append(refined)
                    else:
                        refined_decoys.append(decoy)
                
                # Rank by confidence
                ranked_decoys = self._rank_decoys(refined_decoys)
                
                # Format output
                output = self._format_output(ranked_decoys, sequence)
                
                # Check timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > self.config.inference_timeout:
                    logging.warning(f"Inference took {elapsed_time:.2f}s > {self.config.inference_timeout}s")
                
                if return_all_decoys:
                    output["all_decoys"] = decoys
                    output["selected_decoys"] = selected_decoys
                    output["refined_decoys"] = refined_decoys
                    output["elapsed_time"] = elapsed_time
                
                return output
                
            except Exception as e:
                self._handle_prediction_error(sequence, str(e))
    
    def predict_batch(self,
                     sequences: List[str],
                     msa_data: Optional[List[np.ndarray]] = None) -> List[Dict]:
        """Predict structures for multiple sequences."""
        if not sequences:
            return []
            
        results = []
        
        for i, sequence in enumerate(sequences):
            msa = msa_data[i] if msa_data and i < len(msa_data) else None
            
            try:
                result = self.predict_single_sequence(sequence, msa)
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing sequence {i}: {e}")
                # Fail fast - no fallback coordinates in ML
                results.append({
                    "sequence": sequence,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    def _tokenize_sequence(self, sequence: str) -> torch.Tensor:
        """Convert RNA sequence to token IDs."""
        token_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
        
        tokens = []
        for nucleotide in sequence.upper():
            if nucleotide in token_map:
                tokens.append(token_map[nucleotide])
            else:
                tokens.append(token_map['N'])  # Unknown nucleotide
        
        return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
    
    def _get_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        """Get LM embeddings (cached or computed)."""
        try:
            with torch.no_grad():
                outputs = self.model.language_model(tokens)
                embeddings = outputs["embeddings"]
                # Validate tensor shape
                if embeddings.dim() != 3:
                    raise ValueError(f"Expected 3D embeddings tensor, got {embeddings.dim()}D")
                return embeddings
        except Exception as e:
            logging.error(f"Error getting embeddings: {e}")
            # Fail fast - no fallback embeddings in ML
            raise RuntimeError(f"Failed to generate embeddings: {e}")
    
    def _predict_secondary_structure(self, embeddings: torch.Tensor) -> List[Dict]:
        """Predict secondary structure hypotheses."""
        with torch.no_grad():
            outputs = self.model.ss_predictor(embeddings)
            hypotheses = self.model.ss_predictor.sample_hypotheses(
                outputs["contact_logits"], outputs["pseudoknot_logits"]
            )
            return hypotheses[0]  # Return hypotheses for first (and only) batch
    
    def _prepare_msa_features(self, msa_data: np.ndarray) -> torch.Tensor:
        """Prepare MSA features from raw MSA data."""
        try:
            if msa_data is None:
                return None
                
            # Validate input
            if not isinstance(msa_data, np.ndarray):
                raise ValueError("MSA data must be numpy array")
                
            msa_tensor = torch.from_numpy(msa_data).float().to(self.device)
            
            # Add batch dimension if needed
            if msa_tensor.dim() == 3:
                msa_tensor = msa_tensor.unsqueeze(0)
            elif msa_tensor.dim() != 4:
                raise ValueError(f"Unexpected MSA tensor shape: {msa_tensor.shape}")
            
            return msa_tensor
        except Exception as e:
            logging.error(f"Error preparing MSA features: {e}")
            return None
    
    def _refine_structure(self, decoy: Dict) -> Dict:
        """Refine a single decoy structure."""
        refined = decoy.copy()
        
        if "coordinates" in decoy:
            coords = decoy["coordinates"]
            
            # Use fast refiner for competition
            if hasattr(self.fast_refiner['simple'], 'forward'):
                refined_coords = self.fast_refiner['simple'](coords)
            else:
                # Use geometry refiner
                distance_restraints = decoy.get("distance_logits", None)
                if distance_restraints is not None:
                    # Convert distance logits to actual distances
                    distance_probs = torch.softmax(distance_restraints, dim=-1)
                    # Convert to expected distances (simplified)
                    distances = torch.sum(distance_probs * torch.arange(distance_probs.size(-1)), dim=-1)
                    distances = distances * 0.5  # Scale to reasonable Å distances
                else:
                    distances = None
                
                with torch.no_grad():
                    refiner_output = self.refiner(coords, distances)
                    refined_coords = refiner_output["refined_coordinates"]
            
            refined["coordinates"] = refined_coords
            refined["refined"] = True
        
        return refined
    
    def _rank_decoys(self, decoys: List[Dict]) -> List[Dict]:
        """Rank decoys by confidence score."""
        scored_decoys = []
        
        for decoy in decoys:
            confidence = decoy.get("confidence", torch.tensor(0.5))
            if isinstance(confidence, torch.Tensor):
                confidence = confidence.item()
            
            scored_decoys.append((decoy, confidence))
        
        # Sort by confidence (descending)
        scored_decoys.sort(key=lambda x: x[1], reverse=True)
        
        return [decoy for decoy, _ in scored_decoys]
    
    def _handle_prediction_error(self, sequence: str, error_msg: str) -> None:
        """Handle prediction errors by logging and re-raising."""
        logging.error(f"Failed to predict structure for sequence {sequence[:20]}...: {error_msg}")
        # Re-raise the error to fail fast - no fallback coordinates in ML
        raise RuntimeError(f"Structure prediction failed: {error_msg}")
    
    def _format_output(self, decoys: List[Dict], sequence: str) -> Dict:
        """Format output for competition submission."""
        # Extract C1' coordinates (or first atom) for each decoy
        all_coords = []
        
        for decoy in decoys:
            if "coordinates" in decoy:
                coords = decoy["coordinates"][0, :, 0, :].cpu().numpy()  # First atom
                all_coords.append(coords)
        
        if not all_coords:
            raise RuntimeError(f"No valid decoys generated for sequence {sequence[:20]}...")
        
        coordinates = torch.cat(all_coords, dim=0).cpu().numpy()
        
        # Ensure we have exactly 5 decoys
        while len(all_coords) < 5:
            all_coords.append(all_coords[-1].copy())  # Duplicate last
        
        # Format for submission (flatten all coordinates)
        submission_coords = np.concatenate(all_coords[:5])  # Shape: (5 * n_residues, 3)
        
        return {
            "sequence": sequence,
            "coordinates": submission_coords,
            "n_decoys": 5,
            "n_residues": len(sequence),
            "decoys": decoys[:5]  # Keep top 5 decoys with full info
        }
    
    def save_model(self, filepath: str):
        """Save the complete pipeline."""
        checkpoint = {
            "config": self.config,
            "model_state_dict": self.model.state_dict(),
            "sampler_state_dict": self.sampler.state_dict(),
            "refiner_state_dict": self.refiner.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Pipeline saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the complete pipeline."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.sampler.load_state_dict(checkpoint["sampler_state_dict"])
        self.refiner.load_state_dict(checkpoint["refiner_state_dict"])
        
        print(f"Pipeline loaded from {filepath}")
    
    def enable_competition_mode(self):
        """Enable optimizations for competition deployment."""
        try:
            # Enable evaluation mode
            self.model.eval()
            self.sampler.eval()
            self.refiner.eval()
            
            # Disable gradients
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Use mixed precision if available
            if self.config.mixed_precision and torch.cuda.is_available():
                self.model = self.model.half()
                self.sampler = self.sampler.half()
                self.refiner = self.refiner.half()
            
            logging.info("Competition mode enabled")
        except Exception as e:
            logging.error(f"Error enabling competition mode: {e}")
            raise
    
    def estimate_inference_time(self, sequence_length: int) -> float:
        """Estimate inference time for a given sequence length."""
        # Simple heuristic based on sequence length
        base_time = 0.1  # Base time for very short sequences
        scaling_factor = sequence_length / 100.0
        estimated_time = base_time * scaling_factor ** 1.5  # Super-linear scaling
        
        return min(estimated_time, self.config.inference_time)
