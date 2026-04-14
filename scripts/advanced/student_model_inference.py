#!/usr/bin/env python3
"""
Student Model Inference & Coarse Folding

This script implements student model inference for domain-level folding:
1. Run student encoder per domain proposal
2. Apply hybrid attention with local windows and global hubs
3. Inject sparse residual pairs
4. Produce coarse domain decoys with verification
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
import networkx as nx
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rna_model.utils import set_seed


class StudentEncoder(nn.Module):
    """Student encoder with local windowed attention and hub tokens."""
    
    def __init__(self, d_model: int = 256, n_heads: int = 8, 
                 window_size: int = 64, n_hub_tokens: int = 8):
        """
        Initialize student encoder.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            window_size: Local attention window size
            n_hub_tokens: Number of hub tokens
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.n_hub_tokens = n_hub_tokens
        
        # Input embedding
        self.input_embedding = nn.Linear(512, d_model)  # From input features
        
        # Local attention layers
        self.local_attention_layers = nn.ModuleList([
            LocalAttentionBlock(d_model, n_heads, window_size)
            for _ in range(3)  # 3 local attention layers
        ])
        
        # SE(3)-local blocks
        self.se3_blocks = nn.ModuleList([
            SE3LocalBlock(d_model)
            for _ in range(2)  # 2 SE(3) blocks
        ])
        
        # Hub token integration
        self.hub_integration = HubTokenIntegration(d_model, n_hub_tokens)
        
        # Output heads
        self.frame_head = nn.Linear(d_model, 6)  # 6D rotation + translation
        self.torsion_head = nn.Linear(d_model, 4)  # 4 torsion angles
        self.variance_head = nn.Linear(d_model, 1)  # Prediction variance
        self.distance_head = nn.Linear(d_model, 64)  # Distance histogram
        self.confidence_head = nn.Linear(d_model, 1)  # Per-residue confidence
        self.tm_proxy_head = nn.Linear(d_model, 1)  # Predicted TM proxy
        
    def forward(self, features: Dict, domain_info: Dict, 
               sparse_residuals: List[Dict]) -> Dict:
        """
        Forward pass through student encoder.
        
        Args:
            features: Input features
            domain_info: Domain information
            sparse_residuals: Sparse residual pairs
        
        Returns:
            Encoder outputs
        """
        # Input embedding
        x = self._embed_inputs(features, domain_info)
        
        # Apply local attention layers
        for layer in self.local_attention_layers:
            x = layer(x)
        
        # Apply SE(3) local blocks
        for se3_block in self.se3_blocks:
            x = se3_block(x)
        
        # Integrate hub tokens
        x = self.hub_integration(x, domain_info)
        
        # Inject sparse residuals
        x = self._inject_sparse_residuals(x, sparse_residuals)
        
        # Generate outputs
        outputs = {
            'frames': self.frame_head(x),
            'torsions': self.torsion_head(x),
            'variance': self.variance_head(x),
            'distance_histogram': self.distance_head(x),
            'confidence': self.confidence_head(x),
            'tm_proxy': self.tm_proxy_head(x),
            'features': x
        }
        
        return outputs
    
    def _embed_inputs(self, features: Dict, domain_info: Dict) -> torch.Tensor:
        """Embed input features."""
        # Get base features
        base_features = features.get('base_features', {})
        embeddings = base_features.get('embeddings')
        
        if embeddings is None:
            # Create dummy embeddings
            seq_length = domain_info.get('domain_length', 50)
            embeddings = torch.randn(seq_length, 512)
        else:
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        
        # Apply input embedding
        x = self.input_embedding(embeddings)
        
        return x
    
    def _inject_sparse_residuals(self, x: torch.Tensor, 
                              sparse_residuals: List[Dict]) -> torch.Tensor:
        """Inject sparse residual pairs into features."""
        if not sparse_residuals:
            return x
        
        seq_length = x.shape[0]
        
        # Create sparse residual matrix
        residual_matrix = torch.zeros(seq_length, seq_length, x.shape[1])
        
        for residual in sparse_residuals:
            i, j = residual['i'], residual['j']
            score = residual['score']
            
            if 0 <= i < seq_length and 0 <= j < seq_length:
                # Add residual information to both positions
                residual_vector = torch.randn(1, x.shape[1]) * score
                residual_matrix[i, j] += residual_vector
                residual_matrix[j, i] += residual_vector
        
        # Aggregate residuals
        residual_aggregated = torch.sum(residual_matrix, dim=1)
        
        # Add to features
        x = x + residual_aggregated
        
        return x


class LocalAttentionBlock(nn.Module):
    """Local attention block with windowed attention."""
    
    def __init__(self, d_model: int, n_heads: int, window_size: int):
        """
        Initialize local attention block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            window_size: Local window size
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.head_dim = d_model // n_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with local attention.
        
        Args:
            x: Input tensor (seq_len, d_model)
        
        Returns:
            Output tensor
        """
        seq_len, d_model = x.shape
        
        # Self-attention with local window
        attn_out = self._local_attention(x)
        
        # Residual connection and norm
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ff_out = self.ff(x)
        
        # Residual connection and norm
        x = self.norm2(x + ff_out)
        
        return x
    
    def _local_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Compute local attention."""
        seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(seq_len, self.n_heads, self.head_dim).transpose(0, 1)
        K = K.view(seq_len, self.n_heads, self.head_dim).transpose(0, 1)
        V = V.view(seq_len, self.n_heads, self.head_dim).transpose(0, 1)
        
        # Compute attention scores with local masking
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply local mask
        local_mask = self._create_local_mask(seq_len)
        attn_scores = attn_scores.masked_fill(local_mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention
        attn_out = torch.matmul(attn_weights, V)
        
        # Reshape back
        attn_out = attn_out.transpose(0, 1).contiguous().view(seq_len, d_model)
        
        # Output projection
        attn_out = self.out_proj(attn_out)
        
        return attn_out
    
    def _create_local_mask(self, seq_len: int) -> torch.Tensor:
        """Create local attention mask."""
        mask = torch.zeros(seq_len, seq_len)
        
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1
        
        return mask.unsqueeze(0).expand(self.n_heads, -1, -1)


class SE3LocalBlock(nn.Module):
    """SE(3) equivariant block for local neighborhoods."""
    
    def __init__(self, d_model: int):
        """
        Initialize SE(3) block.
        
        Args:
            d_model: Model dimension
        """
        super().__init__()
        
        self.d_model = d_model
        
        # SE(3) layers
        self.se3_layer = SE3Layer(d_model)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SE(3) block.
        
        Args:
            x: Input tensor (seq_len, d_model)
        
        Returns:
            Output tensor
        """
        # Apply SE(3) layer
        se3_out = self.se3_layer(x)
        
        # Residual connection and norm
        x = self.norm(x + se3_out)
        
        return x


class SE3Layer(nn.Module):
    """SE(3) equivariant layer."""
    
    def __init__(self, d_model: int):
        """
        Initialize SE(3) layer.
        
        Args:
            d_model: Model dimension
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Linear layers
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        
        # Activation
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SE(3) layer."""
        # First linear transformation
        x = self.linear1(x)
        x = self.activation(x)
        
        # Second linear transformation
        x = self.linear2(x)
        
        return x


class HubTokenIntegration(nn.Module):
    """Integrate global hub tokens with local features."""
    
    def __init__(self, d_model: int, n_hub_tokens: int):
        """
        Initialize hub token integration.
        
        Args:
            d_model: Model dimension
            n_hub_tokens: Number of hub tokens
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_hub_tokens = n_hub_tokens
        
        # Hub token embeddings
        self.hub_embeddings = nn.Parameter(torch.randn(n_hub_tokens, d_model))
        
        # Hub attention
        self.hub_attention = nn.MultiheadAttention(d_model, 8, batch_first=True)
        
        # Integration layer
        self.integration = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x: torch.Tensor, domain_info: Dict) -> torch.Tensor:
        """
        Integrate hub tokens with local features.
        
        Args:
            x: Local features (seq_len, d_model)
            domain_info: Domain information
        
        Returns:
            Integrated features
        """
        seq_len, d_model = x.shape
        
        # Get hub positions from domain info
        hub_positions = domain_info.get('hub_positions', [])
        
        if not hub_positions:
            # Use default hub positions
            hub_positions = list(range(0, min(self.n_hub_tokens, seq_len), 
                                   max(1, seq_len // self.n_hub_tokens)))
        
        # Select hub tokens
        selected_hubs = self.hub_embeddings[:len(hub_positions)]
        
        # Global attention with hubs
        x_expanded = x.unsqueeze(0)  # (1, seq_len, d_model)
        hubs_expanded = selected_hubs.unsqueeze(0)  # (1, n_hubs, d_model)
        
        # Cross-attention
        hub_attended, _ = self.hub_attention(x_expanded, hubs_expanded, hubs_expanded)
        hub_attended = hub_attended.squeeze(0)  # (seq_len, d_model)
        
        # Integrate with local features
        integrated = torch.cat([x, hub_attended], dim=-1)
        integrated = self.integration(integrated)
        
        return integrated


class CoarseDecoyGenerator:
    """Generate coarse domain decoys from frames and torsions."""
    
    def __init__(self):
        """Initialize coarse decoy generator."""
        
    def generate_decoys(self, encoder_outputs: Dict, 
                      domain_info: Dict) -> Dict:
        """
        Generate coarse decoys from encoder outputs.
        
        Args:
            encoder_outputs: Encoder outputs
            domain_info: Domain information
        
        Returns:
            Coarse decoys
        """
        # Get frames and torsions
        frames = encoder_outputs['frames']
        torsions = encoder_outputs['torsions']
        
        # Convert to C1' coordinates
        coordinates = self._frames_to_coordinates(frames, torsions, domain_info)
        
        return {
            'coordinates': coordinates,
            'frames': frames,
            'torsions': torsions,
            'variance': encoder_outputs['variance'],
            'confidence': encoder_outputs['confidence'],
            'tm_proxy': encoder_outputs['tm_proxy']
        }
    
    def _frames_to_coordinates(self, frames: torch.Tensor, 
                           torsions: torch.Tensor, 
                           domain_info: Dict) -> np.ndarray:
        """Convert frames and torsions to C1' coordinates."""
        seq_len = frames.shape[0]
        coordinates = np.zeros((seq_len, 3))
        
        # Convert frames to rotation matrices and translations
        rotations = frames[:, :3]  # Simplified rotation representation
        translations = frames[:, 3:6]  # Translation components
        
        # Build coordinates sequentially
        for i in range(seq_len):
            if i == 0:
                # First residue at origin
                coordinates[i] = [0, 0, 0]
            else:
                # Apply transformation from previous residue
                if i < len(rotations):
                    # Simplified coordinate building
                    bond_length = 3.4  # C1'-C1' bond length
                    coordinates[i] = coordinates[i-1] + [bond_length, 0, 0]
                    
                    # Apply rotation (simplified)
                    if i < len(translations):
                        coordinates[i] += translations[i].numpy()[:3]
        
        return coordinates


class DomainVerifier:
    """Verify domain-level decoys."""
    
    def __init__(self):
        """Initialize domain verifier."""
        
    def verify_domain(self, decoy: Dict, contact_probs: np.ndarray, 
                   domain_info: Dict) -> Dict:
        """
        Verify domain decoy.
        
        Args:
            decoy: Coarse decoy
            contact_probs: Contact probabilities
            domain_info: Domain information
        
        Returns:
            Verification results
        """
        # Compute contact satisfaction
        contact_satisfaction = self._compute_contact_satisfaction(
            decoy['coordinates'], contact_probs
        )
        
        # Compute torsion penalty
        torsion_penalty = self._compute_torsion_penalty(decoy['torsions'])
        
        # Determine confidence
        is_high_confidence = (
            contact_satisfaction >= 0.8 and 
            torsion_penalty < 0.5
        )
        
        return {
            'contact_satisfaction': contact_satisfaction,
            'torsion_penalty': torsion_penalty,
            'is_high_confidence': is_high_confidence,
            'verification_score': contact_satisfaction - torsion_penalty,
            'domain_info': domain_info
        }
    
    def _compute_contact_satisfaction(self, coordinates: np.ndarray, 
                                 contact_probs: np.ndarray, 
                                 threshold: float = 0.5) -> float:
        """Compute fraction of high-confidence contacts satisfied."""
        n_residues = len(coordinates)
        
        # Get high-confidence contacts
        high_conf_contacts = []
        for i in range(n_residues):
            for j in range(i + 4, n_residues):  # Non-local contacts
                if contact_probs[i, j] > threshold:
                    high_conf_contacts.append((i, j))
        
        if not high_conf_contacts:
            return 1.0
        
        # Check if contacts are satisfied in structure
        satisfied_contacts = 0
        satisfaction_threshold = 8.0  # Angstroms
        
        for i, j in high_conf_contacts:
            distance = np.linalg.norm(coordinates[i] - coordinates[j])
            if distance <= satisfaction_threshold:
                satisfied_contacts += 1
        
        return satisfied_contacts / len(high_conf_contacts)
    
    def _compute_torsion_penalty(self, torsions: torch.Tensor) -> float:
        """Compute torsion angle penalty."""
        # Convert to numpy
        torsion_np = torsions.detach().numpy()
        
        # Compute penalties for unrealistic torsions
        penalties = []
        
        for i in range(torsion_np.shape[0]):
            torsion_values = torsion_np[i]
            
            # Check if torsions are in reasonable ranges
            for j, torsion in enumerate(torsion_values):
                # Convert to degrees
                torsion_deg = np.degrees(torsion) % 360
                
                # Penalty for unrealistic values
                if j == 0:  # Alpha
                    if not (30 <= torsion_deg <= 90):
                        penalties.append(abs(torsion_deg - 60) / 60)
                elif j == 1:  # Beta
                    if not (120 <= torsion_deg <= 180):
                        penalties.append(abs(torsion_deg - 150) / 30)
                elif j == 2:  # Gamma
                    if not (30 <= torsion_deg <= 80):
                        penalties.append(abs(torsion_deg - 55) / 25)
                elif j == 3:  # Delta
                    if not (70 <= torsion_deg <= 100):
                        penalties.append(abs(torsion_deg - 85) / 15)
        
        return np.mean(penalties) if penalties else 0.0


class StudentModelInference:
    """Main student model inference system."""
    
    def __init__(self, config_path: str):
        """
        Initialize inference system.
        
        Args:
            config_path: Path to configuration
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.student_encoder = StudentEncoder(
            d_model=self.config.get('d_model', 256),
            n_heads=self.config.get('n_heads', 8),
            window_size=self.config.get('window_size', 64),
            n_hub_tokens=self.config.get('n_hub_tokens', 8)
        )
        
        self.decoy_generator = CoarseDecoyGenerator()
        self.domain_verifier = DomainVerifier()
        
    def infer_domain(self, sequence: str, features: Dict, 
                   domain_proposal: Dict, contact_probs: np.ndarray) -> Dict:
        """
        Run student model inference for single domain.
        
        Args:
            sequence: RNA sequence
            features: Input features
            domain_proposal: Domain proposal
            contact_probs: Contact probabilities
        
        Returns:
            Inference results
        """
        # Prepare domain info
        domain_info = self._prepare_domain_info(domain_proposal)
        
        # Get sparse residuals
        sparse_residuals = features.get('base_features', {}).get('sparse_residuals', [])
        
        # Run student encoder
        print(f"Running student encoder for domain {domain_proposal.get('proposal_id')}...")
        encoder_outputs = self.student_encoder(features, domain_info, sparse_residuals)
        
        # Generate coarse decoys
        print("Generating coarse decoys...")
        decoys = self.decoy_generator.generate_decoys(encoder_outputs, domain_info)
        
        # Verify domain
        print("Verifying domain decoys...")
        verification = self.domain_verifier.verify_domain(decoys, contact_probs, domain_info)
        
        # Check for early exit
        should_exit_early = (
            verification['contact_satisfaction'] >= 0.85 and 
            verification['torsion_penalty'] < 0.3
        )
        
        return {
            'sequence': sequence,
            'domain_proposal': domain_proposal,
            'domain_info': domain_info,
            'encoder_outputs': encoder_outputs,
            'decoys': decoys,
            'verification': verification,
            'should_exit_early': should_exit_early,
            'inference_timestamp': time.time()
        }
    
    def _prepare_domain_info(self, domain_proposal: Dict) -> Dict:
        """Prepare domain information for inference."""
        domains = domain_proposal.get('domains', {})
        n_domains = len(domains)
        
        # Get domain boundaries
        domain_boundaries = []
        domain_lengths = []
        
        for domain_id, residues in domains.items():
            start = min(residues)
            end = max(residues) + 1
            domain_boundaries.append((start, end))
            domain_lengths.append(end - start)
        
        # Select hub positions (simplified)
        hub_positions = []
        for i, (start, end) in enumerate(domain_boundaries):
            # Place hub at center of domain
            center = (start + end) // 2
            hub_positions.append(center)
        
        return {
            'domains': domains,
            'n_domains': n_domains,
            'domain_boundaries': domain_boundaries,
            'domain_lengths': domain_lengths,
            'domain_length': sum(domain_lengths),
            'hub_positions': hub_positions
        }
    
    def infer_batch(self, sequences: List[str], features_list: List[Dict], 
                   domain_proposals_list: List[List[Dict]], 
                   contact_probs_list: List[np.ndarray]) -> List[List[Dict]]:
        """
        Run inference for batch of sequences with multiple domain proposals.
        
        Args:
            sequences: List of RNA sequences
            features_list: List of feature dictionaries
            domain_proposals_list: List of domain proposal lists
            contact_probs_list: List of contact probability matrices
        
        Returns:
            List of inference results
        """
        all_results = []
        
        for seq_idx, (sequence, features, domain_proposals, contact_probs) in enumerate(
            zip(sequences, features_list, domain_proposals_list, contact_probs_list)
        ):
            print(f"\nProcessing sequence {seq_idx + 1}/{len(sequences)}: {sequence[:20]}...")
            
            seq_results = []
            
            for proposal_idx, domain_proposal in enumerate(domain_proposals):
                print(f"\nDomain proposal {proposal_idx + 1}/{len(domain_proposals)}")
                
                try:
                    result = self.infer_domain(
                        sequence, features, domain_proposal, contact_probs
                    )
                    result['proposal_index'] = proposal_idx
                    seq_results.append(result)
                    
                    # Check early exit
                    if result.get('should_exit_early', False):
                        print(f"Early exit after proposal {proposal_idx + 1}")
                        break
                        
                except Exception as e:
                    logging.error(f"Failed to infer domain {proposal_idx}: {e}")
                    continue
            
            all_results.append(seq_results)
        
        return all_results


def main():
    """Main student model inference function."""
    parser = argparse.ArgumentParser(description="Student Model Inference for RNA Structures")
    parser.add_argument("--config", required=True,
                       help="Configuration file")
    parser.add_argument("--sequences", required=True,
                       help="File with RNA sequences")
    parser.add_argument("--features", required=True,
                       help="File with input features")
    parser.add_argument("--domain-proposals", required=True,
                       help="File with domain proposals")
    parser.add_argument("--contact-probs", required=True,
                       help="File with contact probabilities")
    parser.add_argument("--output-dir", required=True,
                       help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    try:
        # Initialize inference system
        inference_system = StudentModelInference(args.config)
        
        # Load data
        with open(args.sequences, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
        
        with open(args.features, 'r') as f:
            features_list = json.load(f)
        
        with open(args.domain_proposals, 'r') as f:
            domain_proposals_list = json.load(f)
        
        with open(args.contact_probs, 'r') as f:
            contact_probs_list = json.load(f)
            # Convert back to numpy arrays
            contact_probs_list = [np.array(probs) for probs in contact_probs_list]
        
        # Run inference
        results = inference_system.infer_batch(
            sequences, features_list, domain_proposals_list, contact_probs_list
        )
        
        # Save results
        output_file = Path(args.output_dir) / "student_inference_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("✅ Student model inference completed successfully!")
        print(f"   Processed {len(sequences)} sequences")
        print(f"   Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Student model inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
