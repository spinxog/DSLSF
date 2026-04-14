#!/usr/bin/env python3
"""
Phase 4: Advanced Sampling & Refinement

This script implements the fourth phase of the RNA 3D folding pipeline:
1. Topology-aware sampling with graph-edit operators
2. Parallel tempering MCMC with adaptive temperature control
3. Geometric refinement with internal coordinate optimization
4. Ensemble generation and diversity selection
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rna_model import RNAFoldingPipeline, PipelineConfig
from rna_model.sampler import RNASampler, SamplerConfig
from rna_model.refinement import GeometryRefiner, RefinementConfig
from rna_model.utils import set_seed, compute_rmsd, compute_tm_score


class TopologyAwareSampler:
    """Topology-aware RNA structure sampler with graph-edit operations."""
    
    def __init__(self, config: SamplerConfig):
        """
        Initialize topology-aware sampler.
        
        Args:
            config: Sampler configuration
        """
        self.config = config
        self.graph_edit_operators = [
            self.stem_rewire,
            self.junction_split,
            self.stem_swap,
            self.hairpin_insert_delete
        ]
        
    def stem_rewire(self, contact_graph: nx.Graph, sequence: str) -> nx.Graph:
        """Rewire stem connections in the contact graph."""
        new_graph = contact_graph.copy()
        
        # Find stems (simplified: look for consecutive base pairs)
        stems = self.find_stems(contact_graph, sequence)
        
        if len(stems) >= 2:
            # Select two random stems to rewire
            stem1, stem2 = np.random.choice(stems, 2, replace=False)
            
            # Rewire by swapping connections
            self.swap_stem_connections(new_graph, stem1, stem2)
        
        return new_graph
    
    def junction_split(self, contact_graph: nx.Graph, sequence: str) -> nx.Graph:
        """Split multi-way junctions in the contact graph."""
        new_graph = contact_graph.copy()
        
        # Find junctions (nodes with degree > 2)
        junctions = [node for node in new_graph.nodes() if new_graph.degree(node) > 2]
        
        for junction in junctions:
            if np.random.random() < 0.3:  # 30% probability
                self.split_junction(new_graph, junction)
        
        return new_graph
    
    def stem_swap(self, contact_graph: nx.Graph, sequence: str) -> nx.Graph:
        """Swap stems between different regions."""
        new_graph = contact_graph.copy()
        
        stems = self.find_stems(contact_graph, sequence)
        
        if len(stems) >= 2:
            # Select stems to swap
            stem1, stem2 = np.random.choice(stems, 2, replace=False)
            
            # Swap stem positions
            self.swap_stem_positions(new_graph, stem1, stem2, sequence)
        
        return new_graph
    
    def hairpin_insert_delete(self, contact_graph: nx.Graph, sequence: str) -> nx.Graph:
        """Insert or delete hairpin loops."""
        new_graph = contact_graph.copy()
        
        if np.random.random() < 0.5:  # Insert hairpin
            self.insert_hairpin(new_graph, sequence)
        else:  # Delete hairpin
            self.delete_hairpin(new_graph, sequence)
        
        return new_graph
    
    def find_stems(self, contact_graph: nx.Graph, sequence: str) -> List[Tuple[int, int]]:
        """Find stem regions in the contact graph."""
        stems = []
        seq_len = len(sequence)
        
        # Look for consecutive base pairs
        for i in range(seq_len - 3):
            for j in range(i + 4, seq_len):
                if contact_graph.has_edge(i, j):
                    # Check if this is part of a stem
                    if self.is_stem_pair(contact_graph, sequence, i, j):
                        stems.append((i, j))
        
        return stems
    
    def is_stem_pair(self, contact_graph: nx.Graph, sequence: str, i: int, j: int) -> bool:
        """Check if (i,j) is a stem base pair."""
        # Simplified: check if adjacent positions also pair
        if i + 1 < len(sequence) and j - 1 >= 0:
            if contact_graph.has_edge(i + 1, j - 1):
                return True
        return False
    
    def swap_stem_connections(self, graph: nx.Graph, stem1: Tuple[int, int], stem2: Tuple[int, int]):
        """Swap connections between two stems."""
        # Remove existing connections
        if graph.has_edge(stem1[0], stem1[1]):
            graph.remove_edge(stem1[0], stem1[1])
        if graph.has_edge(stem2[0], stem2[1]):
            graph.remove_edge(stem2[0], stem2[1])
        
        # Add swapped connections
        graph.add_edge(stem1[0], stem2[1])
        graph.add_edge(stem2[0], stem1[1])
    
    def split_junction(self, graph: nx.Graph, junction: int):
        """Split a junction node."""
        neighbors = list(graph.neighbors(junction))
        
        if len(neighbors) > 2:
            # Split into two groups
            mid = len(neighbors) // 2
            group1 = neighbors[:mid]
            group2 = neighbors[mid:]
            
            # Remove original connections
            for neighbor in neighbors:
                graph.remove_edge(junction, neighbor)
            
            # Create new junction nodes
            junction1 = max(graph.nodes()) + 1
            junction2 = junction1 + 1
            
            # Add new connections
            for neighbor in group1:
                graph.add_edge(junction1, neighbor)
            for neighbor in group2:
                graph.add_edge(junction2, neighbor)
    
    def swap_stem_positions(self, graph: nx.Graph, stem1: Tuple[int, int], stem2: Tuple[int, int], sequence: str):
        """Swap positions of two stems in the sequence."""
        # Create new sequence with swapped stems
        seq_list = list(sequence)
        
        # Get stem sequences
        stem1_start, stem1_end = stem1
        stem2_start, stem2_end = stem2
        
        # Ensure stems are within bounds
        if (stem1_start >= 0 and stem1_end < len(seq_list) and
            stem2_start >= 0 and stem2_end < len(seq_list)):
            
            # Swap the stem sequences
            stem1_seq = seq_list[stem1_start:stem1_end+1]
            stem2_seq = seq_list[stem2_start:stem2_end+1]
            
            # Perform swap
            seq_list[stem1_start:stem1_end+1] = stem2_seq
            seq_list[stem2_start:stem2_end+1] = stem1_seq
            
            # Update graph connections to reflect sequence change
            self._update_graph_after_swap(graph, stem1, stem2, seq_list)
        
        return ''.join(seq_list)
    
    def _update_graph_after_swap(self, graph: nx.Graph, stem1: Tuple[int, int], 
                                stem2: Tuple[int, int], new_sequence: str):
        """Update graph connections after stem swap."""
        # Remove old stem connections
        if graph.has_edge(stem1[0], stem1[1]):
            graph.remove_edge(stem1[0], stem1[1])
        if graph.has_edge(stem2[0], stem2[1]):
            graph.remove_edge(stem2[0], stem2[1])
        
        # Add new connections based on new sequence
        # This is simplified - in practice would need full re-analysis
        if self._can_pair(new_sequence[stem1[0]], new_sequence[stem1[1]]):
            graph.add_edge(stem1[0], stem1[1])
        if self._can_pair(new_sequence[stem2[0]], new_sequence[stem2[1]]):
            graph.add_edge(stem2[0], stem2[1])
    
    def _can_pair(self, base1: str, base2: str) -> bool:
        """Check if two bases can pair."""
        pairing_rules = {
            'A': ['U', 'W'], 'U': ['A', 'W'],
            'G': ['C', 'U', 'R'], 'C': ['G', 'R'],
            'W': ['A', 'U'], 'R': ['A', 'G']
        }
        return base2 in pairing_rules.get(base1, [])
    
    def insert_hairpin(self, graph: nx.Graph, sequence: str):
        """Insert a hairpin loop."""
        seq_len = len(sequence)
        
        # Find a suitable position for hairpin
        for i in range(seq_len - 10):
            if np.random.random() < 0.1:  # 10% probability per position
                # Create hairpin connections
                graph.add_edge(i, i + 9)  # 10-base hairpin
                break
    
    def delete_hairpin(self, graph: nx.Graph, sequence: str):
        """Delete a hairpin loop."""
        # Find and remove hairpin connections
        edges_to_remove = []
        
        for edge in graph.edges():
            i, j = edge
            if abs(j - i) < 15:  # Short-range potential hairpin
                edges_to_remove.append(edge)
        
        # Remove some hairpin edges
        if edges_to_remove:
            edge_to_remove = np.random.choice(edges_to_remove)
            graph.remove_edge(edge_to_remove[0], edge_to_remove[1])
    
    def apply_random_operator(self, contact_graph: nx.Graph, sequence: str) -> nx.Graph:
        """Apply a random graph-edit operator."""
        operator = np.random.choice(self.graph_edit_operators)
        return operator(contact_graph, sequence)


class ParallelTemperingMCMC:
    """Parallel tempering MCMC for RNA structure sampling."""
    
    def __init__(self, temperatures: List[float], n_chains: int = 4):
        """
        Initialize parallel tempering MCMC.
        
        Args:
            temperatures: List of temperatures for different chains
            n_chains: Number of parallel chains
        """
        self.temperatures = temperatures[:n_chains]
        self.n_chains = n_chains
        self.swap_interval = 10
        self.acceptance_rates = [0.0] * n_chains
        
    def run_mcmc(self, initial_coords: np.ndarray, 
                  energy_function: callable,
                  max_steps: int = 500) -> Tuple[np.ndarray, List[float]]:
        """
        Run parallel tempering MCMC.
        
        Args:
            initial_coords: Initial coordinates (n_residues, 3)
            energy_function: Function to compute energy
            max_steps: Maximum number of MCMC steps
        
        Returns:
            Tuple of (best_coords, energy_history)
        """
        n_residues = initial_coords.shape[0]
        
        # Initialize chains
        chains = [initial_coords.copy() for _ in range(self.n_chains)]
        energies = [energy_function(coords) for coords in chains]
        
        # Track best solution
        best_coords = initial_coords.copy()
        best_energy = min(energies)
        
        energy_history = []
        
        for step in tqdm(range(max_steps), desc="MCMC Sampling"):
            # Propose moves for each chain
            for i in range(self.n_chains):
                # Propose new coordinates
                proposal = self.propose_move(chains[i])
                proposal_energy = energy_function(proposal)
                
                # Metropolis acceptance
                delta_e = proposal_energy - energies[i]
                accept_prob = np.exp(-delta_e / self.temperatures[i])
                
                if np.random.random() < accept_prob:
                    chains[i] = proposal
                    energies[i] = proposal_energy
                    self.acceptance_rates[i] += 1
                
                # Track best
                if proposal_energy < best_energy:
                    best_coords = proposal.copy()
                    best_energy = proposal_energy
            
            # Swap states between chains
            if step % self.swap_interval == 0 and step > 0:
                self.swap_states(chains, energies)
            
            energy_history.append(min(energies))
        
        # Normalize acceptance rates
        for i in range(self.n_chains):
            self.acceptance_rates[i] /= max_steps
        
        return best_coords, energy_history
    
    def propose_move(self, coords: np.ndarray) -> np.ndarray:
        """Propose a move for MCMC."""
        n_residues = coords.shape[0]
        
        # Choose move type
        move_type = np.random.choice(['local', 'global', 'torsion'])
        
        if move_type == 'local':
            # Local perturbation
            i = np.random.randint(0, n_residues)
            perturbation = np.random.normal(0, 0.5, 3)
            new_coords = coords.copy()
            new_coords[i] += perturbation
        
        elif move_type == 'global':
            # Global rotation/translation
            rotation = self.random_rotation_matrix()
            translation = np.random.normal(0, 0.2, 3)
            new_coords = np.dot(coords, rotation.T) + translation
        
        else:  # torsion
            # Torsion angle perturbation
            if n_residues >= 4:
                i = np.random.randint(1, n_residues - 2)
                angle_change = np.random.normal(0, 0.1)
                new_coords = self.apply_torsion(coords, i, angle_change)
            else:
                new_coords = coords.copy()
        
        return new_coords
    
    def random_rotation_matrix(self) -> np.ndarray:
        """Generate a random rotation matrix."""
        # Random axis
        axis = np.random.normal(0, 1, 3)
        axis = axis / np.linalg.norm(axis)
        
        # Random angle
        angle = np.random.normal(0, 0.1)
        
        # Rodrigues' rotation formula
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        K = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
        
        R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
        
        return R
    
    def apply_torsion(self, coords: np.ndarray, i: int, angle_change: float) -> np.ndarray:
        """Apply torsion angle change at position i."""
        new_coords = coords.copy()
        
        if i >= 1 and i < len(coords) - 2:
            # Define rotation axis
            v1 = coords[i] - coords[i-1]
            v2 = coords[i+1] - coords[i]
            axis = np.cross(v1, v2)
            
            if np.linalg.norm(axis) > 1e-6:
                axis = axis / np.linalg.norm(axis)
                
                # Apply rotation to residues after i
                for j in range(i+1, len(coords)):
                    v = coords[j] - coords[i]
                    # Rodrigues' formula
                    cos_a = np.cos(angle_change)
                    sin_a = np.sin(angle_change)
                    v_rot = v * cos_a + np.cross(axis, v) * sin_a + axis * np.dot(axis, v) * (1 - cos_a)
                    new_coords[j] = coords[i] + v_rot
        
        return new_coords
    
    def swap_states(self, chains: List[np.ndarray], energies: List[float]):
        """Swap states between chains."""
        for i in range(self.n_chains - 1):
            j = i + 1
            
            # Calculate swap probability
            delta = (1.0/self.temperatures[i] - 1.0/self.temperatures[j]) * (energies[i] - energies[j])
            swap_prob = min(1.0, np.exp(delta))
            
            if np.random.random() < swap_prob:
                # Swap states
                chains[i], chains[j] = chains[j], chains[i]
                energies[i], energies[j] = energies[j], energies[i]


class GeometricRefiner:
    """Geometric refinement with internal coordinate optimization."""
    
    def __init__(self, config: RefinementConfig):
        """
        Initialize geometric refiner.
        
        Args:
            config: Refinement configuration
        """
        self.config = config
        self.bond_length = 3.4  # C1'-C1' distance in Å
        self.bond_angle = np.radians(120)  # Default bond angle
        
    def refine_structure(self, coords: np.ndarray, 
                      constraints: Optional[Dict] = None) -> np.ndarray:
        """
        Refine RNA structure using internal coordinate optimization.
        
        Args:
            coords: Initial coordinates (n_residues, 3)
            constraints: Optional constraints dictionary
        
        Returns:
            Refined coordinates
        """
        n_residues = coords.shape[0]
        
        # Convert to internal coordinates
        internal_coords = self.cartesian_to_internal(coords)
        
        # Optimize internal coordinates
        optimized_internal = self.optimize_internal_coordinates(
            internal_coords, constraints
        )
        
        # Convert back to Cartesian coordinates
        refined_coords = self.internal_to_cartesian(optimized_internal)
        
        return refined_coords
    
    def cartesian_to_internal(self, coords: np.ndarray) -> np.ndarray:
        """Convert Cartesian to internal coordinates."""
        n_residues = coords.shape[0]
        internal_coords = []
        
        for i in range(n_residues):
            if i == 0:
                # First residue: use absolute coordinates
                internal_coords.append(coords[i])
            elif i == 1:
                # Second residue: bond length and angle
                bond_length = np.linalg.norm(coords[i] - coords[i-1])
                internal_coords.append([bond_length])
            else:
                # Subsequent residues: bond length, bond angle, torsion
                v1 = coords[i-1] - coords[i-2]
                v2 = coords[i] - coords[i-1]
                
                bond_length = np.linalg.norm(v2)
                
                # Bond angle
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                bond_angle = np.arccos(np.clip(cos_angle, -1, 1))
                
                # Torsion angle
                if i < n_residues - 1:
                    v3 = coords[i+1] - coords[i]
                    torsion = self.calculate_torsion(v1, v2, v3)
                else:
                    torsion = 0.0
                
                internal_coords.append([bond_length, bond_angle, torsion])
        
        return np.array(internal_coords, dtype=object)
    
    def internal_to_cartesian(self, internal_coords: np.ndarray) -> np.ndarray:
        """Convert internal coordinates to Cartesian coordinates."""
        n_residues = len(internal_coords)
        coords = np.zeros((n_residues, 3))
        
        for i in range(n_residues):
            if i == 0:
                coords[i] = internal_coords[i]
            elif i == 1:
                # Place second residue
                bond_length = internal_coords[i][0]
                coords[i] = coords[i-1] + np.array([bond_length, 0, 0])
            else:
                # Build from internal coordinates
                bond_length, bond_angle, torsion = internal_coords[i]
                
                # Build transformation matrix
                T = self.build_transformation_matrix(bond_length, bond_angle, torsion)
                
                # Apply transformation
                coords[i] = np.dot(T, coords[i-1])
        
        return coords
    
    def calculate_torsion(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
        """Calculate torsion angle."""
        # Normalize vectors
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        v3_norm = v3 / (np.linalg.norm(v3) + 1e-8)
        
        # Calculate torsion
        n1 = np.cross(v1_norm, v2_norm)
        n2 = np.cross(v2_norm, v3_norm)
        
        cos_torsion = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2) + 1e-8)
        sin_torsion = np.dot(np.cross(n1, n2), v2_norm)
        
        torsion = np.arctan2(sin_torsion, cos_torsion)
        
        return torsion
    
    def build_transformation_matrix(self, bond_length: float, 
                               bond_angle: float, torsion: float) -> np.ndarray:
        """Build transformation matrix from internal coordinates."""
        # Simplified transformation matrix
        cos_a = np.cos(bond_angle)
        sin_a = np.sin(bond_angle)
        cos_t = np.cos(torsion)
        sin_t = np.sin(torsion)
        
        T = np.array([
            [bond_length, 0, 0],
            [bond_length * cos_a, bond_length * sin_a, 0],
            [bond_length * cos_a * cos_t, bond_length * sin_a * sin_t, bond_length * np.sqrt(1 - cos_a**2)]
        ])
        
        return T
    
    def optimize_internal_coordinates(self, internal_coords: np.ndarray, 
                                constraints: Optional[Dict] = None) -> np.ndarray:
        """Optimize internal coordinates using gradient descent."""
        def objective(x):
            # Flatten internal coordinates for optimization
            coords_flat = []
            for i, coord in enumerate(x):
                if i == 0:
                    coords_flat.extend(coord)
                elif i == 1:
                    coords_flat.extend(coord)
                else:
                    coords_flat.extend(coord)
            
            # Convert to Cartesian
            cart_coords = self.internal_to_cartesian(x)
            
            # Compute energy (simplified)
            energy = 0.0
            
            # Bond length penalties
            for i in range(1, len(cart_coords)):
                dist = np.linalg.norm(cart_coords[i] - cart_coords[i-1])
                energy += (dist - self.bond_length) ** 2
            
            # Steric clash penalties
            for i in range(len(cart_coords)):
                for j in range(i+3, len(cart_coords)):  # Skip nearby residues
                    dist = np.linalg.norm(cart_coords[i] - cart_coords[j])
                    if dist < 2.0:  # Clash threshold
                        energy += 100.0 * (2.0 - dist) ** 2
            
            return energy
        
        # Optimize using scipy
        result = minimize(objective, internal_coords, method='L-BFGS-B')
        
        return result.x


class EnsembleGenerator:
    """Generate diverse ensemble of RNA structures."""
    
    def __init__(self, pipeline: RNAFoldingPipeline):
        """
        Initialize ensemble generator.
        
        Args:
            pipeline: RNA folding pipeline
        """
        self.pipeline = pipeline
        self.sampler = TopologyAwareSampler(pipeline.sampler.config)
        self.mcmc = ParallelTemperingMCMC([1.0, 1.6, 2.6, 4.2])
        self.refiner = GeometricRefiner(pipeline.refiner.config)
    
    def generate_ensemble(self, sequence: str, n_decoys: int = 20) -> List[np.ndarray]:
        """
        Generate diverse ensemble of structures.
        
        Args:
            sequence: RNA sequence
            n_decoys: Number of decoys to generate
        
        Returns:
            List of coordinate arrays
        """
        print(f"Generating ensemble of {n_decoys} decoys for sequence length {len(sequence)}")
        
        # Initial prediction
        initial_result = self.pipeline.predict_single_sequence(sequence)
        initial_coords = initial_result['coordinates'].reshape(5, -1, 3)[0]  # First decoy
        
        decoys = [initial_coords]
        
        # Generate diverse decoys using different strategies
        strategies = [
            self.generate_topology_variants,
            self.generate_mcmc_variants,
            self.generate_refinement_variants
        ]
        
        for strategy in strategies:
            strategy_decoys = strategy(sequence, initial_coords, n_decoys // len(strategies))
            decoys.extend(strategy_decoys)
        
        # Select diverse subset
        diverse_decoys = self.select_diverse_decoys(decoys, n_decoys)
        
        return diverse_decoys
    
    def generate_topology_variants(self, sequence: str, 
                               initial_coords: np.ndarray, 
                               n_variants: int) -> List[np.ndarray]:
        """Generate variants using topology-aware sampling."""
        variants = []
        
        # Create initial contact graph
        contact_graph = self.create_contact_graph(initial_coords)
        
        for i in range(n_variants):
            # Apply random graph-edit operations
            modified_graph = contact_graph.copy()
            for _ in range(np.random.randint(1, 4)):
                modified_graph = self.sampler.apply_random_operator(modified_graph, sequence)
            
            # Convert back to coordinates
            variant_coords = self.graph_to_coordinates(modified_graph, initial_coords)
            variants.append(variant_coords)
        
        return variants
    
    def generate_mcmc_variants(self, sequence: str, 
                             initial_coords: np.ndarray, 
                             n_variants: int) -> List[np.ndarray]:
        """Generate variants using parallel tempering MCMC."""
        variants = []
        
        def energy_function(coords):
            # Simplified energy function
            energy = 0.0
            
            # Bond length energy
            for i in range(1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[i-1])
                energy += (dist - 3.4) ** 2
            
            # Steric clash energy
            for i in range(len(coords)):
                for j in range(i+3, len(coords)):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    if dist < 2.0:
                        energy += 100.0 * (2.0 - dist) ** 2
            
            return energy
        
        for i in range(n_variants):
            best_coords, _ = self.mcmc.run_mcmc(initial_coords, energy_function, max_steps=200)
            variants.append(best_coords)
        
        return variants
    
    def generate_refinement_variants(self, sequence: str, 
                                 initial_coords: np.ndarray, 
                                 n_variants: int) -> List[np.ndarray]:
        """Generate variants using geometric refinement."""
        variants = []
        
        for i in range(n_variants):
            # Add small perturbation
            perturbed = initial_coords + np.random.normal(0, 0.1, initial_coords.shape)
            
            # Refine
            refined = self.refiner.refine_structure(perturbed)
            variants.append(refined)
        
        return variants
    
    def create_contact_graph(self, coords: np.ndarray) -> nx.Graph:
        """Create contact graph from coordinates."""
        n_residues = coords.shape[0]
        graph = nx.Graph()
        
        # Add nodes
        for i in range(n_residues):
            graph.add_node(i)
        
        # Add edges for contacts (within 8Å)
        for i in range(n_residues):
            for j in range(i+1, n_residues):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < 8.0:
                    graph.add_edge(i, j, weight=dist)
        
        return graph
    
    def graph_to_coordinates(self, graph: nx.Graph, 
                          reference_coords: np.ndarray) -> np.ndarray:
        """Convert contact graph back to coordinates (simplified)."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated graph embedding
        coords = reference_coords.copy()
        
        # Apply small random perturbations based on graph structure
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            if len(neighbors) > 0:
                # Move towards center of neighbors
                center = np.mean([reference_coords[n] for n in neighbors], axis=0)
                coords[node] = 0.7 * coords[node] + 0.3 * center
        
        return coords
    
    def select_diverse_decoys(self, decoys: List[np.ndarray], n_select: int) -> List[np.ndarray]:
        """Select diverse subset of decoys using clustering."""
        if len(decoys) <= n_select:
            return decoys
        
        # Compute pairwise RMSD matrix
        n_decoys = len(decoys)
        rmsd_matrix = np.zeros((n_decoys, n_decoys))
        
        for i in range(n_decoys):
            for j in range(i+1, n_decoys):
                rmsd = compute_rmsd(decoys[i], decoys[j])
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd
        
        # Select diverse decoys using greedy algorithm
        selected = [0]  # Start with first decoy
        remaining = list(range(1, n_decoys))
        
        while len(selected) < n_select and remaining:
            # Find decoy most different from selected ones
            best_idx = None
            best_min_rmsd = -1
            
            for idx in remaining:
                min_rmsd = min([rmsd_matrix[idx, s] for s in selected])
                if min_rmsd > best_min_rmsd:
                    best_min_rmsd = min_rmsd
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break
        
        return [decoys[i] for i in selected]


def main():
    """Main sampling and refinement function."""
    parser = argparse.ArgumentParser(description="Phase 4: Advanced Sampling & Refinement")
    parser.add_argument("--model-path", required=True,
                       help="Path to fine-tuned model")
    parser.add_argument("--sequence", required=True,
                       help="RNA sequence to process")
    parser.add_argument("--output-dir", required=True,
                       help="Directory to save results")
    parser.add_argument("--n-decoys", type=int, default=20,
                       help="Number of decoys to generate")
    parser.add_argument("--device", default="cuda",
                       help="Device for computation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize pipeline
    config = PipelineConfig(device=args.device)
    pipeline = RNAFoldingPipeline(config)
    
    # Load model
    if args.model_path:
        pipeline.load_model(args.model_path)
    
    # Initialize ensemble generator
    generator = EnsembleGenerator(pipeline)
    
    # Generate ensemble
    try:
        decoys = generator.generate_ensemble(args.sequence, args.n_decoys)
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save decoys
        for i, decoy in enumerate(decoys):
            np.save(output_dir / f"decoy_{i}.npy", decoy)
        
        # Save ensemble summary
        summary = {
            'sequence': args.sequence,
            'n_decoys': len(decoys),
            'decoy_files': [f"decoy_{i}.npy" for i in range(len(decoys))]
        }
        
        with open(output_dir / "ensemble_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("✅ Phase 4 completed successfully!")
        print(f"   Generated {len(decoys)} decoys")
        print(f"   Results saved to {output_dir}")
        
    except Exception as e:
        print(f"❌ Phase 4 failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
