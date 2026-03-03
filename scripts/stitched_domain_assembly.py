#!/usr/bin/env python3
"""
Stitched Domain Assembly & Cross-Domain Checks

This script implements stitched domain assembly for RNA structure prediction:
1. Domain-level decoy docking with rigid-body adjustments
2. Cross-domain pseudoknot detection and merging
3. Stitched decoy generation with rescoring
4. Partial-output mode for long sequences
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rna_model.utils import set_seed


class DomainDocking:
    """Dock domain-level decoys using rigid-body transformations."""
    
    def __init__(self):
        """Initialize domain docking."""
        
    def dock_domains(self, domain_decoys: List[np.ndarray],
                  domain_info: List[Dict]) -> List[Dict]:
        """
        Dock per-domain decoys using coarse rigid-body + junction torsion adjustments.
        
        Args:
            domain_decoys: List of domain decoy coordinates
            domain_info: List of domain information
        
        Returns:
            List of docked assemblies
        """
        assemblies = []
        
        # Generate all pairwise combinations
        n_domains = len(domain_decoys)
        
        for i in range(n_domains):
            for j in range(i + 1, n_domains):
                assembly = self._dock_pair(
                    domain_decoys[i], domain_decoys[j],
                    domain_info[i], domain_info[j]
                )
                if assembly:
                    assemblies.append(assembly)
        
        return assemblies
    
    def _dock_pair(self, domain1_coords: np.ndarray, domain2_coords: np.ndarray,
                 domain1_info: Dict, domain2_info: Dict) -> Optional[Dict]:
        """Dock two domains together."""
        try:
            # Initial placement - place domains with reasonable separation
            center1 = np.mean(domain1_coords, axis=0)
            center2 = np.mean(domain2_coords, axis=0)
            
            # Estimate junction region
            junction_center = (center1 + center2) / 2
            junction_length = 20  # Approximate junction length
            
            # Create initial assembly
            assembly_coords = np.vstack([domain1_coords, domain2_coords])
            
            # Optimize junction torsion angles
            optimized_coords = self._optimize_junction(
                assembly_coords, domain1_coords, domain2_coords,
                junction_center, junction_length
            )
            
            # Compute inter-domain contact satisfaction
            contact_satisfaction = self._compute_inter_domain_contacts(
                optimized_coords, domain1_coords.shape[0], domain2_coords.shape[0]
            )
            
            return {
                'coordinates': optimized_coords,
                'domain1_id': domain1_info.get('id', 0),
                'domain2_id': domain2_info.get('id', 1),
                'inter_domain_contact_satisfaction': contact_satisfaction,
                'junction_torsion': self._extract_junction_torsion(optimized_coords),
                'score': contact_satisfaction
            }
            
        except Exception as e:
            logging.error(f"Failed to dock domains: {e}")
            return None
    
    def _optimize_junction(self, assembly_coords: np.ndarray,
                       domain1_coords: np.ndarray, domain2_coords: np.ndarray,
                       junction_center: np.ndarray, junction_length: float) -> np.ndarray:
        """Optimize junction torsion angles."""
        def objective(torsion_angles):
            """Objective function for junction optimization."""
            # Apply torsion to assembly
            modified_coords = self._apply_junction_torsion(
                assembly_coords, torsion_angles, junction_center, junction_length
            )
            
            # Compute penalty terms
            clash_penalty = self._compute_domain_clash(modified_coords)
            distance_penalty = np.linalg.norm(
                np.mean(domain1_coords, axis=0) - np.mean(domain2_coords, axis=0)
            )
            
            # Encourage reasonable domain separation
            ideal_separation = 10.0  # Å
            separation_penalty = abs(
                np.linalg.norm(np.mean(domain1_coords, axis=0) - 
                             np.mean(domain2_coords, axis=0)) - ideal_separation
            ) * 0.1
            
            return clash_penalty + distance_penalty + separation_penalty
        
        # Initial torsion angles
        initial_torsion = np.array([0.0, 0.0, 0.0])
        
        # Optimize
        result = minimize(
            objective,
            initial_torsion,
            method='L-BFGS-B',
            options={'maxiter': 100}
        )
        
        return self._apply_junction_torsion(
            assembly_coords, result.x, junction_center, junction_length
        )
    
    def _apply_junction_torsion(self, coords: np.ndarray, torsion_angles: np.ndarray,
                            junction_center: np.ndarray, junction_length: float) -> np.ndarray:
        """Apply torsion angles to junction region."""
        modified_coords = coords.copy()
        
        # Find junction residues (simplified)
        n_residues = len(coords)
        junction_start = n_residues // 2 - int(junction_length // 2)
        junction_end = junction_start + int(junction_length)
        
        # Apply rotation around junction center
        for i in range(junction_start, min(junction_end, n_residues)):
            relative_pos = coords[i] - junction_center
            
            # Apply rotation matrices
            rotation_x = np.array([
                [1, 0, 0],
                [0, np.cos(torsion_angles[0]), -np.sin(torsion_angles[0])],
                [0, np.sin(torsion_angles[0]), np.cos(torsion_angles[0])]
            ])
            
            rotation_y = np.array([
                [np.cos(torsion_angles[1]), 0, np.sin(torsion_angles[1])],
                [0, 1, 0],
                [-np.sin(torsion_angles[1]), 0, np.cos(torsion_angles[1])]
            ])
            
            rotation_z = np.array([
                [np.cos(torsion_angles[2]), -np.sin(torsion_angles[2]), 0],
                [np.sin(torsion_angles[2]), np.cos(torsion_angles[2]), 0],
                [0, 0, 1]
            ])
            
            # Apply combined rotation
            rotated_pos = np.dot(np.dot(np.dot(rotation_z, rotation_y), rotation_x), relative_pos)
            modified_coords[i] = junction_center + rotated_pos
        
        return modified_coords
    
    def _extract_junction_torsion(self, coords: np.ndarray) -> np.ndarray:
        """Extract junction torsion angles."""
        # Simplified torsion extraction
        return np.array([0.0, 0.0, 0.0])  # Placeholder
    
    def _compute_inter_domain_contacts(self, coords: np.ndarray,
                                domain1_size: int, domain2_size: int) -> float:
        """Compute inter-domain contact satisfaction."""
        # Get domain boundaries
        domain1_end = domain1_size
        domain2_start = domain1_end
        
        # Count inter-domain contacts
        inter_contacts = 0
        total_possible = 0
        
        for i in range(domain1_end):
            for j in range(domain2_start, len(coords)):
                distance = np.linalg.norm(coords[i] - coords[j])
                if distance < 8.0:  # Contact threshold
                    inter_contacts += 1
                total_possible += 1
        
        if total_possible == 0:
            return 1.0
        
        return inter_contacts / total_possible
    
    def _compute_domain_clash(self, coords: np.ndarray) -> float:
        """Compute clash penalty for domain assembly."""
        n_residues = len(coords)
        clashes = 0
        
        for i in range(n_residues):
            for j in range(i + 1, n_residues):
                distance = np.linalg.norm(coords[i] - coords[j])
                if distance < 2.0:  # Clash threshold
                    clashes += 1
        
        return clashes / (n_residues * (n_residues - 1) / 2)


class CrossDomainPseudoknotDetector:
    """Detect and handle cross-domain pseudoknots."""
    
    def __init__(self, crossing_threshold: float = 0.02):
        """
        Initialize pseudoknot detector.
        
        Args:
            crossing_threshold: Threshold for crossing density
        """
        self.crossing_threshold = crossing_threshold
        
    def detect_cross_domain_pseudoknots(self, assemblies: List[Dict]) -> List[Dict]:
        """
        Detect cross-domain pseudoknots.
        
        Args:
            assemblies: List of domain assemblies
        
        Returns:
            List of assemblies with pseudoknot information
        """
        results = []
        
        for assembly in assemblies:
            coords = assembly['coordinates']
            n_residues = len(coords)
            
            # Compute crossing density across domains
            crossing_density = self._compute_crossing_density(coords)
            
            # Check if pseudoknot detected
            is_pseudoknot = crossing_density > self.crossing_threshold
            
            if is_pseudoknot:
                assembly['cross_domain_pseudoknot'] = True
                assembly['crossing_density'] = crossing_density
            
            results.append(assembly)
        
        return results
    
    def _compute_crossing_density(self, coords: np.ndarray) -> float:
        """Compute crossing density for structure."""
        n_residues = len(coords)
        crossings = 0
        
        # Get all contacts
        contacts = []
        for i in range(n_residues):
            for j in range(i + 4, n_residues):  # Non-local
                distance = np.linalg.norm(coords[i] - coords[j])
                if distance < 8.0:
                    contacts.append((i, j))
        
        # Count crossings
        for i, j in enumerate(contacts):
            for k, l in enumerate(contacts):
                if k > i:
                    # Check if (i,j) crosses (k,l)
                    if (i < k < j < l) or (k < i < l < j):
                        crossings += 1
        
        return crossings / (n_residues * (n_residues - 1) / 2)
    
    def merge_domains(self, assemblies: List[Dict], max_merge_size: int = 400) -> List[Dict]:
        """
        Merge domains with cross-domain pseudoknots.
        
        Args:
            assemblies: List of assemblies with pseudoknots
            max_merge_size: Maximum size for merged block
        
        Returns:
            List of merged assemblies
        """
        merged_assemblies = []
        
        for assembly in assemblies:
            if not assembly.get('cross_domain_pseudoknot', False):
                continue
            
            coords = assembly['coordinates']
            n_residues = len(coords)
            
            if n_residues <= max_merge_size:
                # Refold entire structure
                merged_coords = self._refold_merged_block(coords)
            else:
                # Perform prioritized subgraph merging
                merged_coords = self._prioritized_subgraph_merge(coords)
            
            merged_assembly = assembly.copy()
            merged_assembly['coordinates'] = merged_coords
            merged_assembly['merged'] = True
            merged_assembly['original_size'] = n_residues
            merged_assembly['merge_method'] = 'full_refold' if n_residues <= max_merge_size else 'subgraph_merge'
            
            merged_assemblies.append(merged_assembly)
        
        return merged_assemblies
    
    def _refold_merged_block(self, coords: np.ndarray) -> np.ndarray:
        """Refold merged block using simple optimization."""
        # Simplified refolding - apply global optimization
        def objective(flat_coords):
            """Objective for refolding."""
            reshaped = flat_coords.reshape(-1, 3)
            
            # Bond length penalty
            bond_penalty = 0.0
            for i in range(1, len(reshaped)):
                bond_length = np.linalg.norm(reshaped[i] - reshaped[i-1])
                bond_penalty += (bond_length - 3.4) ** 2
            
            # Clash penalty
            clash_penalty = 0.0
            for i in range(len(reshaped)):
                for j in range(i + 1, len(reshaped)):
                    distance = np.linalg.norm(reshaped[i] - reshaped[j])
                    if distance < 2.0:
                        clash_penalty += 100.0
            
            return bond_penalty + clash_penalty
        
        # Flatten coordinates for optimization
        flat_coords = coords.flatten()
        
        # Simple optimization
        result = minimize(
            objective,
            flat_coords,
            method='L-BFGS-B',
            options={'maxiter': 200}
        )
        
        return result.x.reshape(-1, 3)
    
    def _prioritized_subgraph_merge(self, coords: np.ndarray) -> np.ndarray:
        """Perform prioritized subgraph merging around crossing regions."""
        # Find crossing regions
        n_residues = len(coords)
        modified_coords = coords.copy()
        
        # Identify high-crossing regions (simplified)
        for i in range(0, n_residues - 20):
            local_crossings = 0
            for j in range(max(0, i - 10), min(n_residues, i + 10)):
                for k in range(j + 1, min(n_residues, j + 10)):
                    for l in range(k + 1, min(n_residues, j + 10)):
                        # Check for crossings
                        if self._is_crossing(i, j, k, l, coords):
                            local_crossings += 1
            
            if local_crossings > 5:  # High crossing region
                # Apply local optimization to this region
                region_start = max(0, i - 5)
                region_end = min(n_residues, i + 5)
                
                # Simple local perturbation
                for j in range(region_start, region_end):
                    perturbation = np.random.randn(3) * 0.5
                    modified_coords[j] += perturbation
        
        return modified_coords
    
    def _is_crossing(self, i: int, j: int, k: int, l: int, coords: np.ndarray) -> bool:
        """Check if four residues form a crossing."""
        # Get positions
        pos_i = coords[i]
        pos_j = coords[j]
        pos_k = coords[k]
        pos_l = coords[l]
        
        # Check crossing condition
        def orientation(a, b, c):
            """Check orientation of triangle (a,b,c)."""
            return np.sign(np.cross(b - a, c - a))
        
        # Check if segments (i,j) and (k,l) cross
        orient_ij_k = orientation(pos_i, pos_j, pos_k)
        orient_ij_l = orientation(pos_i, pos_j, pos_l)
        orient_ik_j = orientation(pos_i, pos_k, pos_j)
        orient_il_j = orientation(pos_i, pos_l, pos_j)
        
        # Crossing occurs if orientations differ
        return (orient_ij_k != orient_ij_l) or (orient_ik_j != orient_il_j)


class StitchedDomainAssembly:
    """Main stitched domain assembly system."""
    
    def __init__(self, config_path: str):
        """
        Initialize stitched domain assembly.
        
        Args:
            config_path: Path to configuration
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.domain_docker = DomainDocking()
        self.pseudoknot_detector = CrossDomainPseudoknotDetector(
            crossing_threshold=self.config.get('crossing_threshold', 0.02)
        )
        
    def assemble_stitched_decoys(self, domain_decoys: List[np.ndarray],
                                domain_info: List[Dict],
                                max_stitched: int = 8) -> List[Dict]:
        """
        Generate stitched decoys from domain-level decoys.
        
        Args:
            domain_decoys: List of domain decoy coordinates
            domain_info: List of domain information
            max_stitched: Maximum number of stitched candidates
        
        Returns:
            List of stitched decoys
        """
        # Dock domain pairs
        assemblies = self.domain_docker.dock_domains(domain_decoys, domain_info)
        
        # Detect cross-domain pseudoknots
        assemblies_with_pseudoknots = self.pseudoknot_detector.detect_cross_domain_pseudoknots(assemblies)
        
        # Merge domains with pseudoknots
        max_merge_size = self.config.get('max_merge_size', 400)
        merged_assemblies = self.pseudoknot_detector.merge_domains(
            assemblies_with_pseudoknots, max_merge_size
        )
        
        # Combine original and merged assemblies
        all_assemblies = assemblies + merged_assemblies
        
        # Select top combinations for stitching
        stitched_decoys = []
        
        # Score all assemblies
        for assembly in all_assemblies:
            # Simple scoring based on contact satisfaction
            if 'inter_domain_contact_satisfaction' in assembly:
                score = assembly['inter_domain_contact_satisfaction']
            else:
                score = 0.5  # Default score
            
            assembly['stitch_score'] = score
        
        # Sort by score
        all_assemblies.sort(key=lambda x: x.get('stitch_score', 0), reverse=True)
        
        # Select top candidates
        for i in range(min(max_stitched, len(all_assemblies))):
            assembly = all_assemblies[i]
            
            stitched_decoys.append({
                'coordinates': assembly['coordinates'],
                'domain_ids': [assembly.get('domain1_id', i), assembly.get('domain2_id', i+1)],
                'stitch_score': assembly.get('stitch_score', 0),
                'merged': assembly.get('merged', False),
                'cross_domain_pseudoknot': assembly.get('cross_domain_pseudoknot', False),
                'assembly_method': assembly.get('merge_method', 'docking')
            })
        
        return stitched_decoys
    
    def process_long_sequence(self, domain_decoys: List[np.ndarray],
                          domain_info: List[Dict]) -> List[Dict]:
        """
        Process long sequences with partial-output mode.
        
        Args:
            domain_decoys: List of domain decoy coordinates
            domain_info: List of domain information
        
        Returns:
            List of partial and best-effort full-length candidates
        """
        results = []
        
        # Add domain-level decoys
        for i, (decoy, info) in enumerate(zip(domain_decoys, domain_info)):
            results.append({
                'coordinates': decoy,
                'domain_id': info.get('id', i),
                'type': 'domain_level',
                'partial': True,
                'score': 0.5  # Default score
            })
        
        # Generate up to 2 best-effort full-length candidates
        try:
            # Use top domain decoys for full-length assembly
            top_domains = domain_decoys[:2]
            top_info = domain_info[:2]
            
            if len(top_domains) >= 2:
                assemblies = self.domain_docker.dock_domains(top_domains, top_info)
                
                for assembly in assemblies[:2]:  # Top 2 assemblies
                    results.append({
                        'coordinates': assembly['coordinates'],
                        'domain_ids': [assembly.get('domain1_id', 0), assembly.get('domain2_id', 1)],
                        'type': 'best_effort_full',
                        'partial': False,
                        'score': assembly.get('stitch_score', 0),
                        'merged': assembly.get('merged', False),
                        'cross_domain_pseudoknot': assembly.get('cross_domain_pseudoknot', False)
                    })
        
        except Exception as e:
            logging.error(f"Failed to generate best-effort full-length candidates: {e}")
        
        return results
    
    def process_batch(self, domain_decoys_list: List[List[np.ndarray]],
                   domain_info_list: List[List[Dict]],
                   sequence_lengths: List[int]) -> List[Dict]:
        """
        Process batch of sequences for stitched assembly.
        
        Args:
            domain_decoys_list: List of domain decoy lists
            domain_info_list: List of domain info lists
            sequence_lengths: List of sequence lengths
        
        Returns:
            List of assembly results
        """
        results = []
        
        for i, (domain_decoys, domain_info, seq_length) in enumerate(
            zip(domain_decoys_list, domain_info_list, sequence_lengths)
        ):
            print(f"\nProcessing sequence {i + 1}/{len(domain_decoys_list)} (L={seq_length})...")
            
            try:
                if seq_length > 500:
                    # Partial-output mode for long sequences
                    stitched_decoys = self.process_long_sequence(
                        domain_decoys, domain_info
                    )
                else:
                    # Normal stitched assembly
                    stitched_decoys = self.assemble_stitched_decoys(
                        domain_decoys, domain_info
                    )
                
                results.append({
                    'sequence_id': i,
                    'sequence_length': seq_length,
                    'stitched_decoys': stitched_decoys,
                    'n_stitched': len(stitched_decoys),
                    'processing_timestamp': time.time()
                })
                
            except Exception as e:
                logging.error(f"Failed to process sequence {i}: {e}")
                results.append(None)
        
        return results


def main():
    """Main stitched domain assembly function."""
    parser = argparse.ArgumentParser(description="Stitched Domain Assembly for RNA Structures")
    parser.add_argument("--config", required=True,
                       help="Configuration file")
    parser.add_argument("--domain-decoys", required=True,
                       help="File with domain decoys")
    parser.add_argument("--domain-info", required=True,
                       help="File with domain information")
    parser.add_argument("--sequence-lengths", required=True,
                       help="File with sequence lengths")
    parser.add_argument("--output-dir", required=True,
                       help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    try:
        # Initialize assembly system
        assembly = StitchedDomainAssembly(args.config)
        
        # Load data
        with open(args.domain_decoys, 'r') as f:
            domain_decoys_list = json.load(f)
            domain_decoys_list = [np.array(decoys) for decoys in domain_decoys_list]
        
        with open(args.domain_info, 'r') as f:
            domain_info_list = json.load(f)
        
        with open(args.sequence_lengths, 'r') as f:
            sequence_lengths = json.load(f)
        
        # Process batch
        results = assembly.process_batch(
            domain_decoys_list, domain_info_list, sequence_lengths
        )
        
        # Save results
        output_file = Path(args.output_dir) / "stitched_assembly_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("✅ Stitched domain assembly completed successfully!")
        print(f"   Processed {len(domain_decoys_list)} sequences")
        print(f"   Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Stitched domain assembly failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
