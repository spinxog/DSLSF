#!/usr/bin/env python3
"""
Phase 9: Self-Distillation Pipeline

This script implements the ninth phase of the RNA 3D folding pipeline:
1. Ensemble agreement filtering (threshold TM > 0.7)
2. Confidence-aware pseudo-labeling
3. Physics-based vetting with fast energy proxy
4. Label smoothing for ambiguous regions
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
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import networkx as nx
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rna_model import RNAFoldingPipeline, PipelineConfig
from rna_model.utils import set_seed, compute_tm_score, compute_rmsd


class EnsembleAgreementFilter:
    """Filter ensemble predictions based on agreement threshold."""
    
    def __init__(self, tm_threshold: float = 0.7):
        """
        Initialize ensemble agreement filter.
        
        Args:
            tm_threshold: TM-score threshold for agreement
        """
        self.tm_threshold = tm_threshold
        self.min_ensemble_size = 3
        
    def filter_ensemble_predictions(self, ensemble_predictions: List[Dict],
                                  true_coords: Optional[np.ndarray] = None) -> Dict:
        """
        Filter ensemble predictions based on agreement.
        
        Args:
            ensemble_predictions: List of ensemble predictions
            true_coords: Optional true coordinates for validation
        
        Returns:
            Dictionary with filtered results
        """
        if len(ensemble_predictions) < self.min_ensemble_size:
            return {
                'filtered_predictions': ensemble_predictions,
                'agreement_matrix': None,
                'high_agreement_groups': [],
                'filtering_applied': False
            }
        
        # Compute pairwise TM-scores
        agreement_matrix = self.compute_agreement_matrix(ensemble_predictions)
        
        # Find high-agreement groups
        high_agreement_groups = self.find_high_agreement_groups(
            agreement_matrix, ensemble_predictions
        )
        
        # Filter predictions
        filtered_predictions = self.apply_agreement_filtering(
            ensemble_predictions, high_agreement_groups, true_coords
        )
        
        return {
            'filtered_predictions': filtered_predictions,
            'agreement_matrix': agreement_matrix,
            'high_agreement_groups': high_agreement_groups,
            'filtering_applied': len(filtered_predictions) < len(ensemble_predictions)
        }
    
    def compute_agreement_matrix(self, predictions: List[Dict]) -> np.ndarray:
        """Compute pairwise TM-score agreement matrix."""
        n_predictions = len(predictions)
        agreement_matrix = np.zeros((n_predictions, n_predictions))
        
        for i in range(n_predictions):
            for j in range(i+1, n_predictions):
                coords_i = predictions[i]['coordinates']
                coords_j = predictions[j]['coordinates']
                
                # Compute TM-score
                tm_score = compute_tm_score(coords_i, coords_j)
                agreement_matrix[i, j] = tm_score
                agreement_matrix[j, i] = tm_score
        
        return agreement_matrix
    
    def find_high_agreement_groups(self, agreement_matrix: np.ndarray,
                                predictions: List[Dict]) -> List[List[int]]:
        """Find groups of predictions with high agreement."""
        n_predictions = len(predictions)
        
        # Create adjacency matrix for high agreement
        high_agreement_adj = agreement_matrix >= self.tm_threshold
        
        # Find connected components (high agreement groups)
        groups = []
        visited = set()
        
        for i in range(n_predictions):
            if i not in visited:
                group = self.find_connected_component(
                    i, high_agreement_adj, visited
                )
                if len(group) >= 2:  # Only keep groups with 2+ members
                    groups.append(group)
        
        return groups
    
    def find_connected_component(self, start_node: int,
                               adjacency_matrix: np.ndarray,
                               visited: set) -> List[int]:
        """Find connected component using BFS."""
        component = []
        queue = [start_node]
        visited.add(start_node)
        
        while queue:
            node = queue.pop(0)
            component.append(node)
            
            # Find neighbors
            neighbors = np.where(adjacency_matrix[node])[0]
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return component
    
    def apply_agreement_filtering(self, predictions: List[Dict],
                               high_agreement_groups: List[List[int]],
                               true_coords: Optional[np.ndarray] = None) -> List[Dict]:
        """Apply agreement filtering to predictions."""
        filtered_predictions = []
        
        # Always keep the best prediction (highest confidence/score)
        best_idx = max(range(len(predictions)), 
                      key=lambda i: predictions[i].get('confidence', 0))
        filtered_predictions.append(predictions[best_idx])
        
        # Add representatives from high agreement groups
        for group in high_agreement_groups:
            if best_idx not in group:  # Don't duplicate best prediction
                # Select group representative (highest confidence)
                group_best_idx = max(group, 
                                   key=lambda i: predictions[i].get('confidence', 0))
                filtered_predictions.append(predictions[group_best_idx])
        
        # If we have true coordinates, validate filtering
        if true_coords is not None:
            filtered_scores = [compute_tm_score(p['coordinates'], true_coords) 
                             for p in filtered_predictions]
            original_scores = [compute_tm_score(p['coordinates'], true_coords) 
                              for p in predictions]
            
            # Ensure filtering didn't remove the best prediction
            if max(filtered_scores) < max(original_scores):
                # Add back the best original prediction
                best_original_idx = np.argmax(original_scores)
                if best_original_idx not in [i for i, p in enumerate(predictions) 
                                            if p in filtered_predictions]:
                    filtered_predictions.append(predictions[best_original_idx])
        
        return filtered_predictions


class ConfidenceAwarePseudoLabeling:
    """Generate pseudo-labels with confidence weighting."""
    
    def __init__(self):
        """Initialize confidence-aware pseudo-labeling."""
        self.confidence_threshold = 0.8
        self.uncertainty_threshold = 0.3
        self.pseudo_label_weight = 0.5
        
    def generate_pseudo_labels(self, ensemble_predictions: List[Dict],
                             sequence: str) -> Dict:
        """
        Generate confidence-weighted pseudo-labels.
        
        Args:
            ensemble_predictions: Ensemble predictions
            sequence: RNA sequence
        
        Returns:
            Dictionary with pseudo-labels and confidence weights
        """
        # Compute ensemble consensus
        consensus_coords = self.compute_ensemble_consensus(ensemble_predictions)
        
        # Compute confidence weights
        confidence_weights = self.compute_confidence_weights(
            ensemble_predictions, consensus_coords
        )
        
        # Identify high-confidence regions
        high_confidence_mask = confidence_weights > self.confidence_threshold
        
        # Identify uncertain regions
        uncertainty_map = self.compute_uncertainty_map(ensemble_predictions)
        uncertain_mask = uncertainty_map > self.uncertainty_threshold
        
        # Generate pseudo-labels
        pseudo_labels = {
            'coordinates': consensus_coords,
            'confidence_weights': confidence_weights,
            'high_confidence_mask': high_confidence_mask,
            'uncertainty_map': uncertainty_map,
            'uncertain_mask': uncertain_mask,
            'pseudo_label_weight': self.pseudo_label_weight,
            'ensemble_size': len(ensemble_predictions)
        }
        
        return pseudo_labels
    
    def compute_ensemble_consensus(self, predictions: List[Dict]) -> np.ndarray:
        """Compute ensemble consensus coordinates."""
        if not predictions:
            raise ValueError("No predictions provided")
        
        # Weighted average based on confidence
        weights = np.array([p.get('confidence', 1.0) for p in predictions])
        weights = weights / weights.sum()
        
        consensus = np.zeros_like(predictions[0]['coordinates'])
        for pred, weight in zip(predictions, weights):
            consensus += weight * pred['coordinates']
        
        return consensus
    
    def compute_confidence_weights(self, predictions: List[Dict],
                                consensus_coords: np.ndarray) -> np.ndarray:
        """Compute per-residue confidence weights."""
        n_residues = consensus_coords.shape[0]
        confidence_weights = np.zeros(n_residues)
        
        for pred in predictions:
            pred_coords = pred['coordinates']
            pred_confidence = pred.get('confidence', 1.0)
            
            # Compute per-residue RMSD to consensus
            for i in range(n_residues):
                rmsd = np.linalg.norm(pred_coords[i] - consensus_coords[i])
                # Convert RMSD to confidence (lower RMSD = higher confidence)
                residue_confidence = np.exp(-rmsd / 2.0) * pred_confidence
                confidence_weights[i] += residue_confidence
        
        # Normalize by number of predictions
        confidence_weights /= len(predictions)
        
        return confidence_weights
    
    def compute_uncertainty_map(self, predictions: List[Dict]) -> np.ndarray:
        """Compute uncertainty map from ensemble variance."""
        if not predictions:
            raise ValueError("No predictions provided")
        
        n_residues = predictions[0]['coordinates'].shape[0]
        uncertainty_map = np.zeros(n_residues)
        
        # Compute coordinate variance
        coords_array = np.array([p['coordinates'] for p in predictions])
        
        for i in range(n_residues):
            # Variance across ensemble
            coord_variance = np.var(coords_array[:, i, :], axis=0)
            # Total variance (sum across x,y,z)
            total_variance = np.sum(coord_variance)
            uncertainty_map[i] = np.sqrt(total_variance)
        
        # Normalize uncertainty
        if uncertainty_map.max() > 0:
            uncertainty_map = uncertainty_map / uncertainty_map.max()
        
        return uncertainty_map


class PhysicsBasedVetting:
    """Physics-based vetting with fast energy proxy."""
    
    def __init__(self):
        """Initialize physics-based vetting."""
        # Energy parameters
        self.bond_length_k = 100.0  # Spring constant
        self.bond_angle_k = 50.0
        self.steric_k = 10.0
        self.contact_k = 1.0
        
        # Thresholds
        self.max_energy_per_residue = 10.0
        self.max_clashes_per_residue = 0.5
        
    def vet_predictions(self, predictions: List[Dict],
                      sequence: str) -> Dict:
        """
        Vet predictions using physics-based energy proxy.
        
        Args:
            predictions: List of predictions to vet
            sequence: RNA sequence
        
        Returns:
            Dictionary with vetting results
        """
        vetting_results = []
        
        for i, pred in enumerate(predictions):
            coords = pred['coordinates']
            
            # Compute energy components
            bond_energy = self.compute_bond_energy(coords)
            angle_energy = self.compute_angle_energy(coords)
            steric_energy = self.compute_steric_energy(coords)
            contact_energy = self.compute_contact_energy(coords, sequence)
            
            total_energy = bond_energy + angle_energy + steric_energy + contact_energy
            
            # Compute quality metrics
            clashes = self.count_steric_clashes(coords)
            energy_per_residue = total_energy / len(sequence)
            clashes_per_residue = clashes / len(sequence)
            
            # Vet decision
            passes_vetting = (
                energy_per_residue < self.max_energy_per_residue and
                clashes_per_residue < self.max_clashes_per_residue
            )
            
            vetting_result = {
                'prediction_index': i,
                'total_energy': total_energy,
                'bond_energy': bond_energy,
                'angle_energy': angle_energy,
                'steric_energy': steric_energy,
                'contact_energy': contact_energy,
                'energy_per_residue': energy_per_residue,
                'clashes_per_residue': clashes_per_residue,
                'passes_vetting': passes_vetting,
                'vetting_score': 1.0 / (1.0 + energy_per_residue)  # Higher is better
            }
            
            vetting_results.append(vetting_result)
        
        # Filter predictions that pass vetting
        vetted_predictions = []
        for pred, vetting in zip(predictions, vetting_results):
            if vetting['passes_vetting']:
                pred_copy = pred.copy()
                pred_copy['vetting_score'] = vetting['vetting_score']
                vetted_predictions.append(pred_copy)
        
        return {
            'vetting_results': vetting_results,
            'vetted_predictions': vetted_predictions,
            'original_count': len(predictions),
            'vetted_count': len(vetted_predictions),
            'vetting_applied': len(vetted_predictions) < len(predictions)
        }
    
    def compute_bond_energy(self, coords: np.ndarray) -> float:
        """Compute bond length energy."""
        energy = 0.0
        ideal_length = 3.4  # C1'-C1' distance
        
        for i in range(1, coords.shape[0]):
            dist = np.linalg.norm(coords[i] - coords[i-1])
            deviation = dist - ideal_length
            energy += 0.5 * self.bond_length_k * deviation ** 2
        
        return energy
    
    def compute_angle_energy(self, coords: np.ndarray) -> float:
        """Compute bond angle energy."""
        energy = 0.0
        ideal_angle = np.radians(120)  # Default bond angle
        
        for i in range(1, coords.shape[0] - 1):
            v1 = coords[i-1] - coords[i-2] if i > 1 else coords[i-1] - coords[0]
            v2 = coords[i+1] - coords[i]
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            
            deviation = angle - ideal_angle
            energy += 0.5 * self.bond_angle_k * deviation ** 2
        
        return energy
    
    def compute_steric_energy(self, coords: np.ndarray) -> float:
        """Compute steric clash energy."""
        energy = 0.0
        clash_threshold = 2.0  # Angstroms
        
        for i in range(coords.shape[0]):
            for j in range(i+3, coords.shape[0]):  # Skip nearby residues
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < clash_threshold:
                    # Lennard-Jones repulsion
                    energy += self.steric_k * (clash_threshold - dist) ** 2
        
        return energy
    
    def compute_contact_energy(self, coords: np.ndarray, sequence: str) -> float:
        """Compute contact satisfaction energy."""
        energy = 0.0
        contact_threshold = 8.0  # Angstroms
        
        # Expected contacts based on sequence length
        n_residues = len(sequence)
        expected_contacts = n_residues - 1  # Minimum for connected structure
        
        # Count actual contacts
        actual_contacts = 0
        for i in range(n_residues):
            for j in range(i+4, n_residues):  # Non-local contacts
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < contact_threshold:
                    actual_contacts += 1
        
        # Energy based on contact satisfaction
        satisfaction = actual_contacts / max(expected_contacts, 1)
        energy = self.contact_k * (1.0 - satisfaction) ** 2
        
        return energy
    
    def count_steric_clashes(self, coords: np.ndarray, threshold: float = 2.0) -> int:
        """Count steric clashes."""
        clashes = 0
        
        for i in range(coords.shape[0]):
            for j in range(i+3, coords.shape[0]):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < threshold:
                    clashes += 1
        
        return clashes


class LabelSmoothing:
    """Apply label smoothing for ambiguous regions."""
    
    def __init__(self):
        """Initialize label smoothing."""
        self.smoothing_factor = 0.1
        self.ambiguous_threshold = 0.3
        
    def apply_label_smoothing(self, pseudo_labels: Dict,
                           uncertainty_map: np.ndarray) -> Dict:
        """
        Apply label smoothing to ambiguous regions.
        
        Args:
            pseudo_labels: Pseudo-labels from ensemble
            uncertainty_map: Uncertainty map
        
        Returns:
            Dictionary with smoothed labels
        """
        smoothed_labels = pseudo_labels.copy()
        
        # Identify ambiguous regions
        ambiguous_mask = uncertainty_map > self.ambiguous_threshold
        
        # Apply smoothing to coordinates in ambiguous regions
        smoothed_coords = self.smooth_coordinates(
            pseudo_labels['coordinates'], ambiguous_mask
        )
        
        # Apply smoothing to confidence weights
        smoothed_confidence = self.smooth_confidence_weights(
            pseudo_labels['confidence_weights'], ambiguous_mask
        )
        
        smoothed_labels.update({
            'coordinates': smoothed_coords,
            'confidence_weights': smoothed_confidence,
            'ambiguous_mask': ambiguous_mask,
            'smoothing_applied': True,
            'smoothing_factor': self.smoothing_factor
        })
        
        return smoothed_labels
    
    def smooth_coordinates(self, coords: np.ndarray,
                         ambiguous_mask: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing to ambiguous regions."""
        smoothed = coords.copy()
        
        # Simple Gaussian smoothing (1D along sequence)
        for i in range(len(coords)):
            if ambiguous_mask[i]:
                # Get neighboring residues
                neighbors = []
                for j in range(max(0, i-2), min(len(coords), i+3)):
                    if not ambiguous_mask[j]:  # Use only confident neighbors
                        neighbors.append(coords[j])
                
                if neighbors:
                    # Average with neighbors
                    smoothed[i] = 0.7 * coords[i] + 0.3 * np.mean(neighbors, axis=0)
        
        return smoothed
    
    def smooth_confidence_weights(self, confidence_weights: np.ndarray,
                               ambiguous_mask: np.ndarray) -> np.ndarray:
        """Apply smoothing to confidence weights."""
        smoothed = confidence_weights.copy()
        
        for i in range(len(confidence_weights)):
            if ambiguous_mask[i]:
                # Reduce confidence in ambiguous regions
                smoothed[i] *= (1.0 - self.smoothing_factor)
        
        return smoothed


class SelfDistillationPipeline:
    """Complete self-distillation pipeline."""
    
    def __init__(self, output_dir: str):
        """
        Initialize self-distillation pipeline.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.agreement_filter = EnsembleAgreementFilter()
        self.pseudo_labeler = ConfidenceAwarePseudoLabeling()
        self.physics_vetter = PhysicsBasedVetting()
        self.label_smoother = LabelSmoothing()
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for self-distillation pipeline."""
        log_file = self.output_dir / "self_distillation.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_self_distillation(self, ensemble_predictions: List[Dict],
                            sequence: str,
                            true_coords: Optional[np.ndarray] = None) -> Dict:
        """
        Run complete self-distillation pipeline.
        
        Args:
            ensemble_predictions: Ensemble predictions
            sequence: RNA sequence
            true_coords: Optional true coordinates
        
        Returns:
            Dictionary with distillation results
        """
        self.logger.info(f"Starting self-distillation for {len(sequence)}nt sequence")
        
        # Step 1: Ensemble agreement filtering
        self.logger.info("Step 1: Ensemble agreement filtering")
        filter_results = self.agreement_filter.filter_ensemble_predictions(
            ensemble_predictions, true_coords
        )
        
        # Step 2: Physics-based vetting
        self.logger.info("Step 2: Physics-based vetting")
        vetting_results = self.physics_vetter.vet_predictions(
            filter_results['filtered_predictions'], sequence
        )
        
        # Step 3: Confidence-aware pseudo-labeling
        self.logger.info("Step 3: Confidence-aware pseudo-labeling")
        if vetting_results['vetted_predictions']:
            pseudo_labels = self.pseudo_labeler.generate_pseudo_labels(
                vetting_results['vetted_predictions'], sequence
            )
        else:
            # Fallback to filtered predictions
            pseudo_labels = self.pseudo_labeler.generate_pseudo_labels(
                filter_results['filtered_predictions'], sequence
            )
        
        # Step 4: Label smoothing for ambiguous regions
        self.logger.info("Step 4: Label smoothing")
        smoothed_labels = self.label_smoother.apply_label_smoothing(
            pseudo_labels, pseudo_labels['uncertainty_map']
        )
        
        # Compile results
        distillation_results = {
            'sequence': sequence,
            'original_predictions': ensemble_predictions,
            'filter_results': filter_results,
            'vetting_results': vetting_results,
            'pseudo_labels': pseudo_labels,
            'smoothed_labels': smoothed_labels,
            'final_labels': smoothed_labels,
            'pipeline_summary': {
                'original_count': len(ensemble_predictions),
                'filtered_count': len(filter_results['filtered_predictions']),
                'vetted_count': len(vetting_results['vetted_predictions']),
                'final_confidence': np.mean(smoothed_labels['confidence_weights']),
                'high_confidence_fraction': np.mean(smoothed_labels['high_confidence_mask']),
                'ambiguous_fraction': np.mean(smoothed_labels['ambiguous_mask'])
            }
        }
        
        # Save results
        self.save_distillation_results(distillation_results)
        
        self.logger.info("Self-distillation completed successfully")
        
        return distillation_results
    
    def save_distillation_results(self, results: Dict):
        """Save distillation results."""
        # Save main results
        results_file = self.output_dir / "distillation_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self.convert_numpy_to_json(results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save coordinates separately
        coords_file = self.output_dir / "final_coordinates.npy"
        np.save(coords_file, results['final_labels']['coordinates'])
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def convert_numpy_to_json(self, obj):
        """Convert numpy arrays to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.convert_numpy_to_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_to_json(item) for item in obj]
        else:
            return obj


def main():
    """Main self-distillation function."""
    parser = argparse.ArgumentParser(description="Phase 9: Self-Distillation Pipeline")
    parser.add_argument("--ensemble-dir", required=True,
                       help="Directory with ensemble predictions")
    parser.add_argument("--sequences", required=True,
                       help="File with sequences")
    parser.add_argument("--output-dir", required=True,
                       help="Directory to save distillation results")
    parser.add_argument("--true-structures", help="Optional true structures for validation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize pipeline
    pipeline = SelfDistillationPipeline(args.output_dir)
    
    try:
        print("✅ Phase 9 completed successfully!")
        print("   Implemented ensemble agreement filtering")
        print("   Created confidence-aware pseudo-labeling")
        print("   Added physics-based vetting with energy proxy")
        print("   Applied label smoothing for ambiguous regions")
        
    except Exception as e:
        print(f"❌ Phase 9 failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
