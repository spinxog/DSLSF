#!/usr/bin/env python3
"""
Parallel Tempering MCMC

This script implements parallel tempering MCMC for RNA structure prediction:
1. Multiple chains with different temperatures
2. Adaptive temperature adjustment
3. Proposal generation with various operators
4. Acceptance tracking and diversity logging
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
from scipy.stats import beta
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rna_model.utils import set_seed


class MCMCState:
    """MCMC state representation."""
    
    def __init__(self, coordinates: np.ndarray, energy: float):
        """
        Initialize MCMC state.
        
        Args:
            coordinates: 3D coordinates
            energy: Energy value
        """
        self.coordinates = coordinates.copy()
        self.energy = energy
        self.step_count = 0
        
    def copy(self):
        """Create a copy of the state."""
        return MCMCState(self.coordinates, self.energy)
    
    def update(self, coordinates: np.ndarray, energy: float):
        """Update state with new coordinates and energy."""
        self.coordinates = coordinates.copy()
        self.energy = energy
        self.step_count += 1


class ProposalGenerator:
    """Generate MCMC proposals."""
    
    def __init__(self):
        """Initialize proposal generator."""
        
    def generate_proposal(self, state: MCMCState, temperature: float) -> Tuple[MCMCState, float]:
        """
        Generate a proposal from current state.
        
        Args:
            state: Current MCMC state
            temperature: Current temperature
        
        Returns:
            New state and proposal log-probability
        """
        # Choose proposal type based on temperature
        if temperature < 1.5:
            # Low temperature: small moves
            return self._small_move(state)
        elif temperature < 3.0:
            # Medium temperature: medium moves
            return self._medium_move(state)
        else:
            # High temperature: large moves
            return self._large_move(state)
    
    def _small_move(self, state: MCMCState) -> Tuple[MCMCState, float]:
        """Generate small move proposal."""
        n_residues = len(state.coordinates)
        
        # Small random perturbation
        perturbation = np.random.randn(n_residues, 3) * 0.2
        
        # Apply to subset of residues
        n_move = min(5, n_residues // 4)
        move_indices = np.random.choice(n_residues, n_move, replace=False)
        
        new_coords = state.coordinates.copy()
        new_coords[move_indices] += perturbation[move_indices]
        
        # Compute new energy (simplified)
        new_energy = self._compute_energy(new_coords)
        
        # Log-probability (symmetric proposal)
        log_prob = 0.0
        
        new_state = MCMCState(new_coords, new_energy)
        
        return new_state, log_prob
    
    def _medium_move(self, state: MCMCState) -> Tuple[MCMCState, float]:
        """Generate medium move proposal."""
        n_residues = len(state.coordinates)
        
        # Medium perturbation with possible rotation
        center = np.mean(state.coordinates, axis=0)
        
        # Random rotation
        angle = np.random.randn() * 0.3
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        # Apply rotation around center
        new_coords = state.coordinates.copy()
        for i in range(n_residues):
            relative_pos = state.coordinates[i] - center
            rotated_pos = np.dot(rotation_matrix, relative_pos)
            new_coords[i] = center + rotated_pos
        
        # Add local perturbation
        n_move = min(10, n_residues // 3)
        move_indices = np.random.choice(n_residues, n_move, replace=False)
        perturbation = np.random.randn(n_move, 3) * 0.5
        new_coords[move_indices] += perturbation
        
        # Compute new energy
        new_energy = self._compute_energy(new_coords)
        
        # Log-probability
        log_prob = 0.0
        
        new_state = MCMCState(new_coords, new_energy)
        
        return new_state, log_prob
    
    def _large_move(self, state: MCMCState) -> Tuple[MCMCState, float]:
        """Generate large move proposal."""
        n_residues = len(state.coordinates)
        
        # Large perturbation with possible domain moves
        if n_residues > 50:
            # Domain-level move
            n_domains = 2
            domain_size = n_residues // n_domains
            
            for d in range(n_domains):
                start = d * domain_size
                end = min((d + 1) * domain_size, n_residues)
                
                # Random rotation and translation for domain
                domain_coords = state.coordinates[start:end]
                domain_center = np.mean(domain_coords, axis=0)
                
                # Large rotation
                angle = np.random.randn() * 0.8
                rotation_matrix = np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                ])
                
                # Apply rotation
                for i in range(start, end):
                    relative_pos = state.coordinates[i] - domain_center
                    rotated_pos = np.dot(rotation_matrix, relative_pos)
                    state.coordinates[i] = domain_center + rotated_pos
                
                # Random translation
                translation = np.random.randn(3) * 2.0
                state.coordinates[start:end] += translation
        else:
            # Global move for small structures
            angle = np.random.randn() * 0.5
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            
            center = np.mean(state.coordinates, axis=0)
            for i in range(n_residues):
                relative_pos = state.coordinates[i] - center
                rotated_pos = np.dot(rotation_matrix, relative_pos)
                state.coordinates[i] = center + rotated_pos
        
        # Compute new energy
        new_energy = self._compute_energy(state.coordinates)
        
        # Log-probability
        log_prob = 0.0
        
        new_state = MCMCState(state.coordinates, new_energy)
        
        return new_state, log_prob
    
    def _compute_energy(self, coordinates: np.ndarray) -> float:
        """Compute energy for coordinates (simplified)."""
        n_residues = len(coordinates)
        
        # Bond length energy
        bond_energy = 0.0
        for i in range(1, n_residues):
            bond_length = np.linalg.norm(coordinates[i] - coordinates[i-1])
            bond_energy += (bond_length - 3.4) ** 2 / 0.1  # Ideal bond length 3.4 Å
        
        # Clash energy
        clash_energy = 0.0
        for i in range(n_residues):
            for j in range(i + 1, n_residues):
                distance = np.linalg.norm(coordinates[i] - coordinates[j])
                if distance < 2.0:  # Clash threshold
                    clash_energy += 100.0 / (distance + 0.1)
        
        # Compactness energy (encourage compact structures)
        center = np.mean(coordinates, axis=0)
        rg = np.sqrt(np.mean(np.sum((coordinates - center) ** 2, axis=1)))
        compactness_energy = rg * 0.01
        
        return bond_energy + clash_energy + compactness_energy


class ParallelTemperingMCMC:
    """Parallel tempering MCMC sampler."""
    
    def __init__(self, n_chains: int = 4, temperatures: List[float] = None,
                 max_steps: int = 500, swap_interval: int = 10):
        """
        Initialize parallel tempering MCMC.
        
        Args:
            n_chains: Number of parallel chains
            temperatures: Temperature for each chain
            max_steps: Maximum steps per chain
            swap_interval: Steps between temperature swaps
        """
        self.n_chains = n_chains
        self.temperatures = temperatures or [1.0, 1.6, 2.6, 4.2]
        self.max_steps = max_steps
        self.swap_interval = swap_interval
        
        self.proposal_generator = ProposalGenerator()
        
        # Target acceptance rates
        self.target_acceptance = [0.2, 0.5]  # [min, max]
        
    def run_mcmc(self, initial_state: MCMCState, 
                 time_budget: float = 240.0) -> Dict:
        """
        Run parallel tempering MCMC.
        
        Args:
            initial_state: Initial MCMC state
            time_budget: Maximum time in seconds
        
        Returns:
            MCMC results
        """
        start_time = time.time()
        
        # Initialize chains
        chains = [initial_state.copy() for _ in range(self.n_chains)]
        chain_energies = [state.energy for state in chains]
        
        # Statistics
        swap_attempts = 0
        swap_acceptances = 0
        acceptance_counts = [0] * self.n_chains
        total_proposals = [0] * self.n_chains
        
        # Track best state
        best_state = initial_state.copy()
        best_energy = initial_state.energy
        
        # Track diversity
        topology_clusters = []
        
        print(f"Running parallel tempering MCMC with {self.n_chains} chains...")
        print(f"Temperatures: {self.temperatures}")
        
        for step in tqdm(range(self.max_steps), desc="MCMC steps"):
            # Check time budget
            if time.time() - start_time > time_budget:
                print(f"Time budget reached after {step} steps")
                break
            
            # Propose moves for each chain
            for chain_idx in range(self.n_chains):
                current_state = chains[chain_idx]
                temperature = self.temperatures[chain_idx]
                
                # Generate proposal
                new_state, log_prob = self.proposal_generator.generate_proposal(
                    current_state, temperature
                )
                
                # Metropolis acceptance
                delta_energy = new_state.energy - current_state.energy
                
                if delta_energy < 0:
                    accept_prob = 1.0
                else:
                    accept_prob = np.exp(-delta_energy / temperature)
                
                if np.random.random() < accept_prob:
                    chains[chain_idx] = new_state
                    chain_energies[chain_idx] = new_state.energy
                    acceptance_counts[chain_idx] += 1
                    
                    # Update best state
                    if new_state.energy < best_energy:
                        best_state = new_state.copy()
                        best_energy = new_state.energy
                
                total_proposals[chain_idx] += 1
            
            # Temperature swaps
            if step % self.swap_interval == 0:
                swap_attempts += 1
                
                for i in range(self.n_chains - 1):
                    for j in range(i + 1, self.n_chains):
                        # Attempt swap between chains i and j
                        if self._attempt_swap(chains, i, j):
                            swap_acceptances += 1
                            
                            # Swap temperatures
                            self.temperatures[i], self.temperatures[j] = \
                                self.temperatures[j], self.temperatures[i]
                            
                            # Swap states
                            chains[i], chains[j] = chains[j], chains[i]
                            chain_energies[i], chain_energies[j] = \
                                chain_energies[j], chain_energies[i]
            
            # Adaptive temperature adjustment
            if step > 0 and step % 50 == 0:
                self._adjust_temperatures(acceptance_counts, total_proposals)
            
            # Track topology diversity
            if step % 100 == 0:
                current_topologies = self._extract_topologies(chains)
                topology_clusters.append(current_topologies)
        
        # Compute final statistics
        final_acceptance_rates = [
            acceptance_counts[i] / max(total_proposals[i], 1) 
            for i in range(self.n_chains)
        ]
        
        swap_acceptance_rate = swap_acceptances / max(swap_attempts, 1)
        
        # Diversity analysis
        diversity_metrics = self._analyze_diversity(topology_clusters)
        
        return {
            'best_state': best_state,
            'best_energy': best_energy,
            'final_chains': chains,
            'final_energies': chain_energies,
            'final_temperatures': self.temperatures,
            'acceptance_rates': final_acceptance_rates,
            'swap_acceptance_rate': swap_acceptance_rate,
            'total_steps': step + 1,
            'diversity_metrics': diversity_metrics,
            'time_used': time.time() - start_time
        }
    
    def _attempt_swap(self, chains: List[MCMCState], i: int, j: int) -> bool:
        """Attempt temperature swap between chains i and j."""
        state_i = chains[i]
        state_j = chains[j]
        
        temp_i = self.temperatures[i]
        temp_j = self.temperatures[j]
        
        # Compute swap acceptance probability
        delta_energy_i = state_j.energy - state_i.energy
        delta_energy_j = state_i.energy - state_j.energy
        
        # Metropolis criterion for swap
        if temp_i == temp_j:
            swap_prob = 0.5
        else:
            swap_prob = min(1.0, np.exp(
                (delta_energy_i / temp_i - delta_energy_j / temp_j) / 2
            ))
        
        return np.random.random() < swap_prob
    
    def _adjust_temperatures(self, acceptance_counts: List[int], 
                          total_proposals: List[int]):
        """Adjust temperatures to maintain target acceptance rates."""
        for i in range(self.n_chains):
            if total_proposals[i] == 0:
                continue
                
            acceptance_rate = acceptance_counts[i] / total_proposals[i]
            target_min, target_max = self.target_acceptance
            
            # Adjust temperature based on acceptance rate
            if acceptance_rate < target_min:
                # Too cold, increase temperature
                self.temperatures[i] *= 1.1
            elif acceptance_rate > target_max:
                # Too hot, decrease temperature
                self.temperatures[i] *= 0.9
            
            # Keep temperature in reasonable range
            self.temperatures[i] = np.clip(self.temperatures[i], 0.5, 10.0)
    
    def _extract_topologies(self, chains: List[MCMCState]) -> List[List[int]]:
        """Extract topology information from chains."""
        topologies = []
        
        for state in chains:
            # Simplified topology extraction (contact map)
            n_residues = len(state.coordinates)
            contact_map = np.zeros((n_residues, n_residues))
            
            # Create contact map
            for i in range(n_residues):
                for j in range(i + 4, n_residues):  # Non-local contacts
                    distance = np.linalg.norm(state.coordinates[i] - state.coordinates[j])
                    if distance < 8.0:  # Contact threshold
                        contact_map[i, j] = 1
                        contact_map[j, i] = 1
            
            # Convert to topology signature
            topology = self._topology_to_signature(contact_map)
            topologies.append(topology)
        
        return topologies
    
    def _topology_to_signature(self, contact_map: np.ndarray) -> int:
        """Convert contact map to topology signature."""
        # Simplified signature based on contact patterns
        n_residues = contact_map.shape[0]
        
        # Count contacts per residue
        contacts_per_residue = np.sum(contact_map, axis=1)
        
        # Create signature (simplified)
        signature = 0
        for i, count in enumerate(contacts_per_residue):
            signature += (count * (i + 1)) % 1000
        
        return signature
    
    def _analyze_diversity(self, topology_clusters: List[List[int]]) -> Dict:
        """Analyze diversity of topologies over time."""
        if not topology_clusters:
            return {'n_clusters': 0, 'diversity_score': 0.0}
        
        # Flatten all topologies
        all_topologies = []
        for cluster in topology_clusters:
            all_topologies.extend(cluster)
        
        # Count unique topologies
        unique_topologies = len(set(all_topologies))
        
        # Diversity score (entropy of topology distribution)
        topology_counts = {}
        for topology in all_topologies:
            topology_counts[topology] = topology_counts.get(topology, 0) + 1
        
        # Compute entropy
        total_topologies = sum(topology_counts.values())
        entropy = 0.0
        for count in topology_counts.values():
            if count > 0:
                p = count / total_topologies
                entropy -= p * np.log(p)
        
        return {
            'n_clusters': len(topology_clusters),
            'n_unique_topologies': unique_topologies,
            'diversity_score': entropy,
            'topology_distribution': topology_counts
        }


class DefaultSampler:
    """Default sampler for non-entangled/low complexity sequences."""
    
    def __init__(self):
        """Initialize default sampler."""
        
    def sample(self, sequence: str, contact_probs: np.ndarray,
              ss_hypotheses: List[Dict]) -> List[Dict]:
        """
        Generate baseline topology proposals.
        
        Args:
            sequence: RNA sequence
            contact_probs: Contact probabilities
            ss_hypotheses: Secondary structure hypotheses
        
        Returns:
            List of proposals
        """
        proposals = []
        
        # Generate combinations of top-3 SS hypotheses
        top_hypotheses = ss_hypotheses[:3]
        
        for i, ss1 in enumerate(top_hypotheses):
            for j, ss2 in enumerate(top_hypotheses[i+1:], i+1):
                for k, ss3 in enumerate(top_hypotheses[j+1:], j+1):
                    # Combine hypotheses
                    combined_contacts = self._combine_ss_hypotheses(
                        [ss1, ss2, ss3]
                    )
                    
                    # Generate coordinates from combined contacts
                    coords = self._coords_from_contacts(
                        combined_contacts, len(sequence)
                    )
                    
                    proposals.append({
                        'coordinates': coords,
                        'method': 'ss_combination',
                        'hypotheses': [i, j+1, k+1],
                        'score': np.random.random(),
                        'description': f'Combined SS hypotheses {i},{j+1},{k+1}'
                    })
        
        # Add MSA subsampling proposals
        for i in range(3):
            msa_coords = self._msa_subsample_coords(sequence, contact_probs)
            proposals.append({
                'coordinates': msa_coords,
                'method': 'msa_subsample',
                'subsample_id': i,
                'score': np.random.random(),
                'description': f'MSA subsample {i}'
            })
        
        # Add MC dropout proposals
        for i in range(3):
            dropout_coords = self._mc_dropout_coords(sequence, contact_probs)
            proposals.append({
                'coordinates': dropout_coords,
                'method': 'mc_dropout',
                'dropout_id': i,
                'score': np.random.random(),
                'description': f'MC dropout {i}'
            })
        
        return proposals
    
    def _combine_ss_hypotheses(self, hypotheses: List[Dict]) -> np.ndarray:
        """Combine multiple SS hypotheses."""
        n_residues = len(hypotheses[0]['pair_probs'])
        combined_contacts = np.zeros((n_residues, n_residues))
        
        # Average contact probabilities
        for hypothesis in hypotheses:
            pair_probs = np.array(hypothesis['pair_probs'])
            combined_contacts += pair_probs
        
        combined_contacts /= len(hypotheses)
        
        return combined_contacts
    
    def _coords_from_contacts(self, contact_probs: np.ndarray, 
                           seq_length: int) -> np.ndarray:
        """Generate coordinates from contact probabilities."""
        coords = np.zeros((seq_length, 3))
        
        # Simple coordinate generation based on contacts
        for i in range(seq_length):
            # Place residues along a line with some variation
            coords[i, 0] = i * 3.4  # Bond length
            
            # Add y variation based on contacts
            contact_sum = np.sum(contact_probs[i, :])
            coords[i, 1] = (contact_sum - np.mean(contact_probs)) * 5.0
            
            # Add z variation
            coords[i, 2] = np.random.randn() * 2.0
        
        return coords
    
    def _msa_subsample_coords(self, sequence: str, 
                           contact_probs: np.ndarray) -> np.ndarray:
        """Generate coordinates with MSA subsampling."""
        seq_length = len(sequence)
        
        # Simulate MSA subsampling
        subsample_factor = np.random.uniform(0.7, 0.9)
        modified_contacts = contact_probs * subsample_factor
        
        return self._coords_from_contacts(modified_contacts, seq_length)
    
    def _mc_dropout_coords(self, sequence: str, 
                        contact_probs: np.ndarray) -> np.ndarray:
        """Generate coordinates with MC dropout."""
        seq_length = len(sequence)
        
        # Simulate MC dropout
        dropout_mask = np.random.random(contact_probs.shape) > 0.3
        modified_contacts = contact_probs * dropout_mask
        
        return self._coords_from_contacts(modified_contacts, seq_length)


def main():
    """Main parallel tempering MCMC function."""
    parser = argparse.ArgumentParser(description="Parallel Tempering MCMC for RNA Structures")
    parser.add_argument("--config", required=True,
                       help="Configuration file")
    parser.add_argument("--initial-state", required=True,
                       help="Initial state file")
    parser.add_argument("--output-dir", required=True,
                       help="Directory to save results")
    parser.add_argument("--time-budget", type=float, default=240.0,
                       help="Time budget in seconds")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    try:
        # Load initial state
        with open(args.initial_state, 'r') as f:
            initial_data = json.load(f)
        
        initial_coords = np.array(initial_data['coordinates'])
        initial_energy = initial_data.get('energy', 1000.0)
        initial_state = MCMCState(initial_coords, initial_energy)
        
        # Initialize MCMC
        mcmc = ParallelTemperingMCMC()
        
        # Run MCMC
        results = mcmc.run_mcmc(initial_state, args.time_budget)
        
        # Save results
        output_file = Path(args.output_dir) / "mcmc_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        results['best_state']['coordinates'] = results['best_state']['coordinates'].tolist()
        for i, chain in enumerate(results['final_chains']):
            results['final_chains'][i]['coordinates'] = chain.coordinates.tolist()
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("✅ Parallel tempering MCMC completed successfully!")
        print(f"   Best energy: {results['best_energy']:.3f}")
        print(f"   Total steps: {results['total_steps']}")
        print(f"   Time used: {results['time_used']:.2f}s")
        print(f"   Swap acceptance rate: {results['swap_acceptance_rate']:.3f}")
        print(f"   Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Parallel tempering MCMC failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
