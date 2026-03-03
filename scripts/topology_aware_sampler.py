#!/usr/bin/env python3
"""
Topology-Aware Sampler

This script implements topology-aware sampling for RNA structure prediction:
1. Targeted local topology exploration for entangled structures
2. Graph-edit proposal generators with various operators
3. Global topology refinement with parallel tempering
4. Integration with coarse domain decoys
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
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rna_model.utils import set_seed

# Import from part files
from topology_aware_sampler_part1 import LocalTopologyExplorer
from topology_aware_sampler_part2 import GraphEditProposalGenerator, DefaultSampler


class TopologyAwareSampler:
    """Main topology-aware sampling system."""
    
    def __init__(self, config_path: str):
        """
        Initialize topology-aware sampler.
        
        Args:
            config_path: Path to configuration
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.local_explorer = LocalTopologyExplorer(
            max_proposals=self.config.get('max_local_proposals', 20),
            window_size_factor=self.config.get('window_size_factor', 0.4)
        )
        
        self.graph_editor = GraphEditProposalGenerator()
        self.default_sampler = DefaultSampler()
        
    def sample_topology(self, sequence: str, contact_probs: np.ndarray,
                    domain_info: Dict, coarse_decoy: np.ndarray,
                    is_entangled: bool, complexity_score: float) -> Dict:
        """
        Perform topology-aware sampling.
        
        Args:
            sequence: RNA sequence
            contact_probs: Contact probabilities
            domain_info: Domain information
            coarse_decoy: Coarse decoy coordinates
            is_entangled: Whether structure is entangled
            complexity_score: Complexity score
        
        Returns:
            Sampling results
        """
        proposals = []
        
        # Check if we need targeted local exploration
        if is_entangled or complexity_score > 1.5:
            print("Running targeted local topology exploration...")
            local_proposals = self.local_explorer.explore_local_topology(
                sequence, contact_probs, domain_info, coarse_decoy
            )
            proposals.extend(local_proposals)
        
        # Always run graph-edit proposals
        print("Running graph-edit proposal generator...")
        graph_proposals = self.graph_editor.generate_proposals(
            sequence, contact_probs, coarse_decoy
        )
        proposals.extend(graph_proposals)
        
        # Add default sampler proposals for non-entangled sequences
        if not is_entangled and complexity_score <= 1.5:
            print("Running default sampler...")
            default_proposals = self.default_sampler.sample(
                sequence, contact_probs, []  # Empty SS hypotheses for now
            )
            proposals.extend(default_proposals)
        
        # Score and rank proposals
        scored_proposals = self._score_proposals(proposals, contact_probs)
        scored_proposals.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'sequence': sequence,
            'is_entangled': is_entangled,
            'complexity_score': complexity_score,
            'n_proposals': len(proposals),
            'proposals': scored_proposals,
            'best_proposal': scored_proposals[0] if scored_proposals else None,
            'sampling_timestamp': time.time()
        }
    
    def _score_proposals(self, proposals: List[Dict], 
                       contact_probs: np.ndarray) -> List[Dict]:
        """Score topology proposals."""
        for proposal in proposals:
            if 'coordinates' not in proposal:
                proposal['score'] = 0.0
                continue
            
            coords = proposal['coordinates']
            
            # Compute contact satisfaction
            contact_satisfaction = self._compute_contact_satisfaction(
                coords, contact_probs
            )
            
            # Compute geometric quality
            geometric_score = self._compute_geometric_score(coords)
            
            # Combined score
            proposal['score'] = (
                0.6 * contact_satisfaction +
                0.4 * geometric_score
            )
            
            # Add additional metrics
            proposal['contact_satisfaction'] = contact_satisfaction
            proposal['geometric_score'] = geometric_score
        
        return proposals
    
    def _compute_contact_satisfaction(self, coords: np.ndarray,
                                   contact_probs: np.ndarray,
                                   threshold: float = 8.0) -> float:
        """Compute contact satisfaction for proposal."""
        n_residues = len(coords)
        
        # Get high-confidence contacts
        high_conf_contacts = []
        for i in range(n_residues):
            for j in range(i + 4, n_residues):  # Non-local contacts
                if contact_probs[i, j] > 0.7:
                    high_conf_contacts.append((i, j))
        
        if not high_conf_contacts:
            return 1.0
        
        # Count satisfied contacts
        satisfied = 0
        for i, j in high_conf_contacts:
            distance = np.linalg.norm(coords[i] - coords[j])
            if distance <= threshold:
                satisfied += 1
        
        return satisfied / len(high_conf_contacts)
    
    def _compute_geometric_score(self, coords: np.ndarray) -> float:
        """Compute geometric quality score."""
        # Bond length consistency
        n_residues = len(coords)
        
        # Bond length consistency
        bond_lengths = []
        for i in range(1, n_residues):
            bond_length = np.linalg.norm(coords[i] - coords[i-1])
            bond_lengths.append(bond_length)
        
        # Score based on bond length consistency
        if bond_lengths:
            mean_bond = np.mean(bond_lengths)
            std_bond = np.std(bond_lengths)
            
            # Ideal bond length is 3.4 Å
            bond_score = np.exp(-((mean_bond - 3.4) ** 2 / 0.5) * np.exp(-std_bond / 2.0))
        else:
            bond_score = 0.0
        
        # Clashing penalty
        clash_penalty = self._compute_clash_penalty(coords)
        
        # Compactness energy (encourage compact structures)
        center = np.mean(coords, axis=0)
        rg = np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))
        compactness_energy = rg * 0.01
        
        return bond_score - clash_penalty + compactness_energy
    
    def _compute_clash_penalty(self, coords: np.ndarray, 
                           clash_threshold: float = 2.0) -> float:
        """Compute clash penalty."""
        n_residues = len(coords)
        clashes = 0
        
        for i in range(n_residues):
            for j in range(i + 1, n_residues):
                distance = np.linalg.norm(coords[i] - coords[j])
                if distance < clash_threshold:
                    clashes += 1
        
        return clashes / (n_residues * (n_residues - 1) / 2)
    
    def sample_batch(self, sequences: List[str], 
                   contact_probs_list: List[np.ndarray],
                   domain_info_list: List[Dict],
                   coarse_decoys: List[np.ndarray],
                   entanglement_list: List[bool],
                   complexity_scores: List[float]) -> List[Dict]:
        """
        Sample topology for batch of sequences.
        
        Args:
            sequences: List of RNA sequences
            contact_probs_list: List of contact probability matrices
            domain_info_list: List of domain information
            coarse_decoys: List of coarse decoys
            entanglement_list: List of entanglement flags
            complexity_scores: List of complexity scores
        
        Returns:
            List of sampling results
        """
        results = []
        
        for i, (sequence, contact_probs, domain_info, coarse_decoy, 
                 is_entangled, complexity) in enumerate(
            zip(sequences, contact_probs_list, domain_info_list,
                coarse_decoys, entanglement_list, complexity_scores)
        ):
            print(f"\nProcessing sequence {i + 1}/{len(sequences)}: {sequence[:20]}...")
            
            try:
                result = self.sample_topology(
                    sequence, contact_probs, domain_info, coarse_decoy, 
                    is_entangled, complexity
                )
                results.append(result)
                
            except Exception as e:
                logging.error(f"Failed to sample topology for sequence {i}: {e}")
                results.append(None)
        
        return results


def main():
    """Main topology-aware sampling function."""
    parser = argparse.ArgumentParser(description="Topology-Aware Sampling for RNA Structures")
    parser.add_argument("--config", required=True,
                       help="Configuration file")
    parser.add_argument("--sequences", required=True,
                       help="File with RNA sequences")
    parser.add_argument("--contact-probs", required=True,
                       help="File with contact probabilities")
    parser.add_argument("--domain-info", required=True,
                       help="File with domain information")
    parser.add_argument("--coarse-decoys", required=True,
                       help="File with coarse decoys")
    parser.add_argument("--entanglement", required=True,
                       help="File with entanglement flags")
    parser.add_argument("--complexity", required=True,
                       help="File with complexity scores")
    parser.add_argument("--output-dir", required=True,
                       help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    try:
        # Initialize sampler
        sampler = TopologyAwareSampler(args.config)
        
        # Load data
        with open(args.sequences, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
        
        with open(args.contact_probs, 'r') as f:
            contact_probs_list = json.load(f)
            contact_probs_list = [np.array(probs) for probs in contact_probs_list]
        
        with open(args.domain_info, 'r') as f:
            domain_info_list = json.load(f)
        
        with open(args.coarse_decoys, 'r') as f:
            coarse_decoys = json.load(f)
            coarse_decoys = [np.array(decoy) for decoy in coarse_decoys]
        
        with open(args.entanglement, 'r') as f:
            entanglement_list = json.load(f)
        
        with open(args.complexity, 'r') as f:
            complexity_scores = json.load(f)
        
        # Run sampling
        results = sampler.sample_batch(
            sequences, contact_probs_list, domain_info_list,
            coarse_decoys, entanglement_list, complexity_scores
        )
        
        # Save results
        output_file = Path(args.output_dir) / "topology_sampling_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("✅ Topology-aware sampling completed successfully!")
        print(f"   Processed {len(sequences)} sequences")
        print(f"   Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Topology-aware sampling failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
