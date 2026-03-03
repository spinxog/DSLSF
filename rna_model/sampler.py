"""RNA Sampler for Generating Diverse Structural Decoys"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class SamplerConfig:
    n_decoys: int = 20
    temperature_range: Tuple[float, float] = (0.8, 1.2)
    dropout_prob: float = 0.1
    msa_subsample_ratio: float = 0.8
    diversity_weight: float = 0.3


class RNASampler(nn.Module):
    """Fast sampler for generating diverse RNA structure decoys."""
    
    def __init__(self, config: SamplerConfig):
        super().__init__()
        self.config = config
    
    def sample_decoys(self,
                     model: nn.Module,
                     embeddings: torch.Tensor,
                     ss_hypotheses: List[Dict[str, torch.Tensor]],
                     msa_features: Optional[torch.Tensor] = None,
                     device: torch.device = torch.device('cpu')) -> List[Dict[str, torch.Tensor]]:
        """
        Generate diverse structural decoys using multiple sampling strategies.
        
        Args:
            model: Trained structure prediction model
            embeddings: LM embeddings
            ss_hypotheses: List of secondary structure hypotheses
            msa_features: Optional MSA features
            device: Device to run on
            
        Returns:
            List of decoy predictions
        """
        model.eval()
        decoys = []
        batch_size, seq_len, _ = embeddings.shape
        
        # Strategy 1: Different SS hypotheses
        for i, ss_hyp in enumerate(ss_hypotheses):
            if i >= min(self.config.n_decoys // 3, len(ss_hypotheses)):
                break
                
            with torch.no_grad():
                # Sample temperature
                temp = np.random.uniform(*self.config.temperature_range)
                
                # Forward pass with specific SS hypothesis
                outputs = self._forward_with_ss(
                    model, embeddings, ss_hyp, msa_features, temp, device
                )
                
                outputs["strategy"] = "ss_hypothesis"
                outputs["hypothesis_id"] = i
                decoys.append(outputs)
        
        # Strategy 2: MC Dropout
        remaining_decoys = self.config.n_decoys - len(decoys)
        for i in range(remaining_decoys // 2):
            with torch.no_grad():
                # Enable dropout during inference
                model.train()
                
                temp = np.random.uniform(*self.config.temperature_range)
                
                # Use first SS hypothesis with dropout
                ss_hyp = ss_hypotheses[0] if ss_hypotheses else None
                outputs = self._forward_with_ss(
                    model, embeddings, ss_hyp, msa_features, temp, device
                )
                
                outputs["strategy"] = "mc_dropout"
                outputs["hypothesis_id"] = i
                decoys.append(outputs)
        
        # Strategy 3: MSA subsampling (if available)
        if msa_features is not None:
            remaining_decoys = self.config.n_decoys - len(decoys)
            for i in range(remaining_decoys):
                with torch.no_grad():
                    model.eval()
                    
                    # Subsample MSA
                    subsampled_msa = self._subsample_msa(msa_features)
                    
                    temp = np.random.uniform(*self.config.temperature_range)
                    ss_hyp = ss_hypotheses[0] if ss_hypotheses else None
                    
                    outputs = self._forward_with_ss(
                        model, embeddings, ss_hyp, subsampled_msa, temp, device
                    )
                    
                    outputs["strategy"] = "msa_subsample"
                    outputs["hypothesis_id"] = i
                    decoys.append(outputs)
        
        # Fill remaining with temperature variation
        while len(decoys) < self.config.n_decoys:
            with torch.no_grad():
                model.eval()
                
                temp = np.random.uniform(*self.config.temperature_range)
                ss_hyp = ss_hypotheses[0] if ss_hypotheses else None
                
                outputs = self._forward_with_ss(
                    model, embeddings, ss_hyp, msa_features, temp, device
                )
                
                outputs["strategy"] = "temperature_variation"
                outputs["hypothesis_id"] = len(decoys)
                decoys.append(outputs)
        
        return decoys[:self.config.n_decoys]
    
    def _forward_with_ss(self,
                        model: nn.Module,
                        embeddings: torch.Tensor,
                        ss_hyp: Optional[Dict[str, torch.Tensor]],
                        msa_features: Optional[torch.Tensor],
                        temperature: float,
                        device: torch.device) -> Dict[str, torch.Tensor]:
        """Forward pass with specific secondary structure hypothesis."""
        batch_size, seq_len, _ = embeddings.shape
        
        # Prepare inputs
        inputs = {"embeddings": embeddings.to(device)}
        
        if ss_hyp is not None:
            inputs["ss_contacts"] = ss_hyp["contact_probs"].to(device)
            inputs["ss_pseudoknots"] = ss_hyp["pseudoknot_probs"].to(device)
        
        if msa_features is not None:
            inputs["msa_features"] = msa_features.to(device)
        
        # Forward pass
        outputs = model(**inputs)
        
        # Apply temperature to logits if present
        if "distance_logits" in outputs:
            outputs["distance_logits"] = outputs["distance_logits"] / temperature
        
        if "angle_logits" in outputs:
            outputs["angle_logits"] = outputs["angle_logits"] / temperature
        
        if "torsion_logits" in outputs:
            outputs["torsion_logits"] = outputs["torsion_logits"] / temperature
        
        return outputs
    
    def _subsample_msa(self, msa_features: torch.Tensor) -> torch.Tensor:
        """Subsample MSA features."""
        n_sequences = msa_features.size(1)
        n_keep = max(1, int(n_sequences * self.config.msa_subsample_ratio))
        
        # Randomly sample sequences
        indices = torch.randperm(n_sequences)[:n_keep]
        return msa_features[:, indices, :, :]
    
    def cluster_and_select(self,
                          decoys: List[Dict[str, torch.Tensor]],
                          n_selected: int = 5) -> List[Dict[str, torch.Tensor]]:
        """
        Cluster decoys and select diverse representatives.
        
        Args:
            decoys: List of decoy predictions
            n_selected: Number of representatives to select
            
        Returns:
            Selected diverse decoys
        """
        if len(decoys) <= n_selected:
            return decoys
        
        # Extract coordinates for clustering
        coords_list = []
        for decoy in decoys:
            if "coordinates" in decoy:
                # Use C1' coordinates (or first atom) for clustering
                coords = decoy["coordinates"][0, :, 0, :].cpu().numpy()  # (seq_len, 3)
                coords_list.append(coords)
        
        if not coords_list:
            # Fallback: return first n_selected decoys
            return decoys[:n_selected]
        
        # Simple clustering based on RMSD
        selected_indices = self._cluster_by_rmsd(coords_list, n_selected)
        
        return [decoys[i] for i in selected_indices]
    
    def _cluster_by_rmsd(self, coords_list: List[np.ndarray], n_clusters: int) -> List[int]:
        """Cluster structures by RMSD and select representatives."""
        n_structures = len(coords_list)
        
        if n_structures <= n_clusters:
            return list(range(n_structures))
        
        # Compute pairwise RMSD matrix
        rmsd_matrix = np.zeros((n_structures, n_structures))
        
        for i in range(n_structures):
            for j in range(i + 1, n_structures):
                rmsd = self._compute_rmsd(coords_list[i], coords_list[j])
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd
        
        # Greedy clustering: pick most diverse structures
        selected = [0]  # Start with first structure
        candidates = list(range(1, n_structures))
        
        while len(selected) < n_clusters and candidates:
            # Find candidate farthest from all selected
            best_candidate = None
            best_min_distance = -1
            
            for candidate in candidates:
                min_distance = min(rmsd_matrix[candidate, sel] for sel in selected)
                
                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_candidate = candidate
            
            if best_candidate is not None:
                selected.append(best_candidate)
                candidates.remove(best_candidate)
            else:
                break
        
        # Fill remaining with highest confidence if available
        if len(selected) < n_clusters:
            # Sort by confidence (assuming it's available)
            remaining = [i for i in range(n_structures) if i not in selected]
            selected.extend(remaining[:n_clusters - len(selected)])
        
        return selected[:n_clusters]
    
    def _compute_rmsd(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Compute RMSD between two coordinate sets."""
        # Simple RMSD without optimal superposition
        diff = coords1 - coords2
        rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
        return float(rmsd)


class AdvancedSampler(nn.Module):
    """Advanced sampler with diffusion-based generation (research version)."""
    
    def __init__(self, config: SamplerConfig):
        super().__init__()
        self.config = config
        
        # Simple diffusion network (placeholder)
        self.diffusion_net = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # Coordinate updates
        )
    
    def diffuse_sample(self,
                      initial_coords: torch.Tensor,
                      n_steps: int = 50) -> torch.Tensor:
        """Generate structures via diffusion process."""
        coords = initial_coords.clone()
        
        for step in range(n_steps):
            # Add noise
            noise = torch.randn_like(coords) * 0.1
            noisy_coords = coords + noise
            
            # Denoise
            with torch.no_grad():
                delta = self.diffusion_net(noisy_coords.flatten(-2))
                delta = delta.view_as(coords)
                coords = noisy_coords - delta * 0.1
        
        return coords
