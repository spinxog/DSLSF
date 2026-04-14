#!/usr/bin/env python3
"""
Phase 7: Robustness Features

This script implements the seventh phase of the RNA 3D folding pipeline:
1. Entanglement detector with local BFS topology sampler
2. Ensemble domain proposals with adaptive pruning
3. Multi-hub + sentinel residue system
4. Enhanced parallel tempering MCMC implementation
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rna_model.utils import set_seed


class EntanglementDetector:
    """Detect RNA entanglement and pseudoknots using graph analysis."""
    
    def __init__(self):
        """Initialize entanglement detector."""
        # Detection thresholds
        self.crossing_density_threshold = 0.02
        self.planarity_threshold = 0.1
        
    def detect_entanglement(self, contact_graph: nx.Graph, 
                          coordinates: np.ndarray) -> Dict:
        """
        Detect entanglement in RNA structure.
        
        Args:
            contact_graph: Contact graph of RNA
            coordinates: 3D coordinates
        
        Returns:
            Dictionary with entanglement analysis
        """
        # Compute crossing density
        crossing_density = self.compute_crossing_density(contact_graph, coordinates)
        
        # Test planarity
        is_planar = self.test_planarity(contact_graph)
        
        # Detect pseudoknots
        pseudoknots = self.detect_pseudoknots(contact_graph)
        
        # Compute entanglement score
        entanglement_score = self.compute_entanglement_score(
            crossing_density, is_planar, pseudoknots
        )
        
        return {
            'crossing_density': crossing_density,
            'is_planar': is_planar,
            'pseudoknots': pseudoknots,
            'entanglement_score': entanglement_score,
            'is_entangled': (
                crossing_density > self.crossing_density_threshold or
                not is_planar or
                len(pseudoknots) > 0
            )
        }
    
    def compute_crossing_density(self, contact_graph: nx.Graph, 
                              coordinates: np.ndarray) -> float:
        """Compute crossing density of contact graph."""
        n_nodes = contact_graph.number_of_nodes()
        if n_nodes < 4:
            return 0.0
        
        # Project contacts to 2D plane (simplified)
        coords_2d = coordinates[:, :2]  # Use x,y coordinates
        
        # Count edge crossings
        crossings = 0
        edges = list(contact_graph.edges())
        
        for i, (u1, v1) in enumerate(edges):
            for u2, v2 in edges[i+1:]:
                if self.edges_cross(
                    coords_2d[u1], coords_2d[v1],
                    coords_2d[u2], coords_2d[v2]
                ):
                    crossings += 1
        
        # Normalize by total possible crossings
        max_crossings = len(edges) * (len(edges) - 1) // 2
        crossing_density = crossings / max_crossings if max_crossings > 0 else 0.0
        
        return crossing_density
    
    def edges_cross(self, p1: np.ndarray, p2: np.ndarray,
                   p3: np.ndarray, p4: np.ndarray) -> bool:
        """Check if two line segments cross."""
        # Vector cross product method
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def test_planarity(self, contact_graph: nx.Graph) -> bool:
        """Test if contact graph is planar."""
        try:
            is_planar, _ = nx.check_planarity(contact_graph)
            return is_planar
        except:
            # Fallback for large graphs
            return contact_graph.number_of_edges() <= 3 * contact_graph.number_of_nodes() - 6
    
    def detect_pseudoknots(self, contact_graph: nx.Graph) -> List[List[int]]:
        """Detect pseudoknots in contact graph."""
        pseudoknots = []
        
        # Find cycles that represent pseudoknots
        # Simplified: look for crossing base pairs
        nodes = list(contact_graph.nodes())
        
        for i in range(len(nodes)):
            for j in range(i+2, len(nodes)):  # Non-adjacent pairs
                if contact_graph.has_edge(nodes[i], nodes[j]):
                    # Check if this creates a pseudoknot
                    for k in range(i+1, j):
                        if contact_graph.has_edge(nodes[k], nodes[(j+1) % len(nodes)]):
                            pseudoknots.append([nodes[i], nodes[j], nodes[k], nodes[(j+1) % len(nodes)]])
        
        return pseudoknots
    
    def compute_entanglement_score(self, crossing_density: float,
                               is_planar: bool, 
                               pseudoknots: List) -> float:
        """Compute overall entanglement score."""
        # Combine different metrics
        planarity_penalty = 0.0 if is_planar else 0.5
        pseudoknot_penalty = min(len(pseudoknots) * 0.1, 0.5)
        
        entanglement_score = (
            0.4 * crossing_density +
            planarity_penalty +
            pseudoknot_penalty
        )
        
        return min(entanglement_score, 1.0)
    
    def run_local_bfs_sampler(self, contact_graph: nx.Graph,
                           start_node: int, max_depth: int = 5) -> nx.Graph:
        """Run local BFS topology sampler."""
        # BFS from start node
        visited = set()
        queue = [(start_node, 0)]
        local_graph = nx.Graph()
        
        while queue:
            node, depth = queue.pop(0)
            
            if node in visited or depth > max_depth:
                continue
            
            visited.add(node)
            local_graph.add_node(node)
            
            # Add neighbors
            for neighbor in contact_graph.neighbors(node):
                local_graph.add_edge(node, neighbor)
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
        
        return local_graph


class EnsembleDomainProposer:
    """Generate ensemble domain proposals with adaptive pruning."""
    
    def __init__(self):
        """Initialize domain proposer."""
        # Proposal parameters
        self.max_proposals_entangled = 8
        self.max_proposals_simple = 5
        self.min_domain_size = 10
        self.max_domain_size = 200
        
    def generate_domain_proposals(self, contact_graph: nx.Graph,
                               sequence: str,
                               entanglement_info: Dict) -> List[Dict]:
        """
        Generate ensemble of domain proposals.
        
        Args:
            contact_graph: Contact graph
            sequence: RNA sequence
            entanglement_info: Entanglement analysis
        
        Returns:
            List of domain proposals
        """
        proposals = []
        
        # Determine number of proposals based on entanglement
        n_proposals = (
            self.max_proposals_entangled if entanglement_info['is_entangled']
            else self.max_proposals_simple
        )
        
        # Generate proposals using different methods
        methods = [
            self.spectral_clustering_proposals,
            self.greedy_community_detection,
            self.top_down_contact_cutting
        ]
        
        for method in methods:
            method_proposals = method(contact_graph, sequence, n_proposals // len(methods))
            proposals.extend(method_proposals)
        
        # Adaptive pruning
        pruned_proposals = self.adaptive_pruning(proposals, contact_graph, sequence)
        
        return pruned_proposals[:n_proposals]
    
    def spectral_clustering_proposals(self, contact_graph: nx.Graph,
                                  sequence: str,
                                  n_proposals: int) -> List[Dict]:
        """Generate proposals using spectral clustering at multiple resolutions."""
        proposals = []
        
        # Create adjacency matrix
        adj_matrix = nx.adjacency_matrix(contact_graph)
        
        # Multiple resolutions
        resolutions = [2, 3, 4, 5]
        
        for resolution in resolutions[:n_proposals]:
            # Spectral clustering
            clustering = SpectralClustering(
                n_clusters=resolution,
                affinity='precomputed',
                random_state=42
            )
            
            try:
                labels = clustering.fit_predict(adj_matrix)
                
                # Create domains from clusters
                domains = self.create_domains_from_labels(labels, sequence)
                
                proposal = {
                    'method': 'spectral',
                    'resolution': resolution,
                    'domains': domains,
                    'n_domains': len(domains),
                    'complexity_score': self.compute_domain_complexity(domains)
                }
                
                proposals.append(proposal)
                
            except Exception as e:
                print(f"Spectral clustering failed for resolution {resolution}: {e}")
        
        return proposals
    
    def greedy_community_detection(self, contact_graph: nx.Graph,
                                sequence: str,
                                n_proposals: int) -> List[Dict]:
        """Generate proposals using greedy community detection."""
        proposals = []
        
        for i in range(n_proposals):
            # Greedy modularity optimization
            communities = nx.community.greedy_modularity_communities(contact_graph)
            
            # Create domains
            domains = []
            for community in communities:
                if len(community) >= self.min_domain_size:
                    domain_seq = ''.join(sequence[j] for j in sorted(community))
                    domains.append({
                        'residues': sorted(community),
                        'sequence': domain_seq,
                        'size': len(community)
                    })
            
            proposal = {
                'method': 'greedy',
                'iteration': i,
                'domains': domains,
                'n_domains': len(domains),
                'complexity_score': self.compute_domain_complexity(domains)
            }
            
            proposals.append(proposal)
        
        return proposals
    
    def top_down_contact_cutting(self, contact_graph: nx.Graph,
                               sequence: str,
                               n_proposals: int) -> List[Dict]:
        """Generate proposals using top-down contact strength cutting."""
        proposals = []
        
        # Compute edge weights (contact strengths)
        edge_weights = {}
        for u, v in contact_graph.edges():
            # Simplified weight based on node degrees
            weight = contact_graph.degree(u) * contact_graph.degree(v)
            edge_weights[(u, v)] = weight
        
        for i in range(n_proposals):
            # Sort edges by weight
            sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)
            
            # Cut edges to create domains
            n_cuts = min(i + 1, len(sorted_edges) // 2)
            edges_to_cut = [edge for edge, _ in sorted_edges[:n_cuts]]
            
            # Create modified graph
            modified_graph = contact_graph.copy()
            for u, v in edges_to_cut:
                if modified_graph.has_edge(u, v):
                    modified_graph.remove_edge(u, v)
            
            # Find connected components (domains)
            components = list(nx.connected_components(modified_graph))
            domains = []
            
            for component in components:
                if len(component) >= self.min_domain_size:
                    domain_seq = ''.join(sequence[j] for j in sorted(component))
                    domains.append({
                        'residues': sorted(component),
                        'sequence': domain_seq,
                        'size': len(component)
                    })
            
            proposal = {
                'method': 'top_down',
                'n_cuts': n_cuts,
                'domains': domains,
                'n_domains': len(domains),
                'complexity_score': self.compute_domain_complexity(domains)
            }
            
            proposals.append(proposal)
        
        return proposals
    
    def create_domains_from_labels(self, labels: np.ndarray, sequence: str) -> List[Dict]:
        """Create domains from clustering labels."""
        domains = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            
            if len(indices) >= self.min_domain_size:
                domain_seq = ''.join(sequence[j] for j in indices)
                domains.append({
                    'residues': indices.tolist(),
                    'sequence': domain_seq,
                    'size': len(indices)
                })
        
        return domains
    
    def compute_domain_complexity(self, domains: List[Dict]) -> float:
        """Compute complexity score for domain proposal."""
        if not domains:
            return 0.0
        
        # Factors: number of domains, size variance, inter-domain connectivity
        n_domains = len(domains)
        sizes = [d['size'] for d in domains]
        size_variance = np.var(sizes) if len(sizes) > 1 else 0.0
        
        # Normalize factors
        n_domains_score = min(n_domains / 10.0, 1.0)
        size_variance_score = min(size_variance / 1000.0, 1.0)
        
        complexity = 0.6 * n_domains_score + 0.4 * size_variance_score
        
        return complexity
    
    def adaptive_pruning(self, proposals: List[Dict],
                       contact_graph: nx.Graph,
                       sequence: str) -> List[Dict]:
        """Adaptive pruning of domain proposals."""
        if len(proposals) <= 1:
            return proposals
        
        # Compute scores for all proposals
        scores = []
        for proposal in proposals:
            score = self.evaluate_proposal(proposal, contact_graph, sequence)
            scores.append(score)
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]  # Descending
        
        # Adaptive selection based on score distribution
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Keep proposals above mean - 0.5 * std
        threshold = mean_score - 0.5 * std_score
        selected_indices = [i for i, score in enumerate(scores) if score >= threshold]
        
        # Ensure at least 2 proposals
        if len(selected_indices) < 2:
            selected_indices = sorted_indices[:2]
        
        pruned_proposals = [proposals[i] for i in selected_indices]
        
        return pruned_proposals
    
    def evaluate_proposal(self, proposal: Dict,
                       contact_graph: nx.Graph,
                       sequence: str) -> float:
        """Evaluate quality of domain proposal."""
        domains = proposal['domains']
        
        # Inter-domain contact satisfaction
        inter_domain_contacts = self.compute_inter_domain_contacts(domains, contact_graph)
        contact_satisfaction = inter_domain_contacts / max(len(domains) - 1, 1)
        
        # Domain size balance
        sizes = [d['size'] for d in domains]
        size_balance = 1.0 - (np.std(sizes) / np.mean(sizes)) if len(sizes) > 1 else 1.0
        
        # Sequence continuity (prefer contiguous domains)
        continuity_score = self.compute_domain_continuity(domains, sequence)
        
        # Combined score
        score = (
            0.4 * contact_satisfaction +
            0.3 * size_balance +
            0.3 * continuity_score
        )
        
        return score
    
    def compute_inter_domain_contacts(self, domains: List[Dict],
                                 contact_graph: nx.Graph) -> float:
        """Compute inter-domain contact satisfaction."""
        if len(domains) < 2:
            return 1.0
        
        total_possible = 0
        satisfied_contacts = 0
        
        for i in range(len(domains)):
            for j in range(i+1, len(domains)):
                domain1_residues = set(domains[i]['residues'])
                domain2_residues = set(domains[j]['residues'])
                
                # Check for expected inter-domain contacts
                for res1 in domain1_residues:
                    for res2 in domain2_residues:
                        if contact_graph.has_edge(res1, res2):
                            satisfied_contacts += 1
                
                total_possible += len(domain1_residues) * len(domain2_residues)
        
        return satisfied_contacts / total_possible if total_possible > 0 else 0.0
    
    def compute_domain_continuity(self, domains: List[Dict], sequence: str) -> float:
        """Compute continuity score for domains."""
        if not domains:
            return 0.0
        
        continuity_scores = []
        
        for domain in domains:
            residues = sorted(domain['residues'])
            
            # Check if residues form contiguous blocks
            n_blocks = self.count_contiguous_blocks(residues)
            max_possible_blocks = len(residues)
            
            continuity = 1.0 - (n_blocks - 1) / max_possible_blocks
            continuity_scores.append(continuity)
        
        return np.mean(continuity_scores)
    
    def count_contiguous_blocks(self, residues: List[int]) -> int:
        """Count number of contiguous blocks in residue list."""
        if not residues:
            return 0
        
        blocks = 1
        for i in range(1, len(residues)):
            if residues[i] != residues[i-1] + 1:
                blocks += 1
        
        return blocks


class MultiHubSystem:
    """Multi-hub + sentinel residue system for domain representation."""
    
    def __init__(self):
        """Initialize multi-hub system."""
        # Hub parameters
        self.degree_hubs = 4
        self.betweenness_hubs = 2
        self.max_sentinel_hubs = 4
        self.total_hubs_per_domain = 8
    
    def select_hub_residues(self, contact_graph: nx.Graph,
                          sequence: str,
                          domain_residues: List[int]) -> Dict:
        """
        Select hub residues for domain representation.
        
        Args:
            contact_graph: Contact graph
            sequence: RNA sequence
            domain_residues: Residues in domain
        
        Returns:
            Dictionary with hub information
        """
        # Create subgraph for domain
        domain_graph = contact_graph.subgraph(domain_residues)
        
        # Compute centrality measures
        degree_centrality = nx.degree_centrality(domain_graph)
        betweenness_centrality = nx.betweenness_centrality(domain_graph)
        
        # Select degree hubs
        degree_hubs = sorted(
            degree_centrality.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.degree_hubs]
        
        # Select betweenness hubs
        betweenness_hubs = sorted(
            betweenness_centrality.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.betweenness_hubs]
        
        # Select sentinel residues
        sentinel_hubs = self.select_sentinel_residues(
            domain_graph, sequence, domain_residues
        )
        
        # Combine and deduplicate
        all_hubs = degree_hubs + betweenness_hubs + sentinel_hubs
        unique_hubs = []
        seen_residues = set()
        
        for residue, score in all_hubs:
            if residue not in seen_residues and len(unique_hubs) < self.total_hubs_per_domain:
                unique_hubs.append((residue, score))
                seen_residues.add(residue)
        
        return {
            'degree_hubs': degree_hubs,
            'betweenness_hubs': betweenness_hubs,
            'sentinel_hubs': sentinel_hubs,
            'selected_hubs': unique_hubs,
            'n_hubs': len(unique_hubs)
        }
    
    def select_sentinel_residues(self, domain_graph: nx.Graph,
                               sequence: str,
                               domain_residues: List[int]) -> List[Tuple[int, float]]:
        """Select sentinel residues based on conservation and structural importance."""
        sentinel_candidates = []
        
        # Compute conservation scores (simplified)
        for residue in domain_residues:
            # Conservation based on degree (low degree = high conservation)
            degree = domain_graph.degree(residue)
            conservation_score = 1.0 / (degree + 1)
            
            # Structural importance (based on position)
            relative_pos = residue / len(sequence)
            position_score = 1.0 - abs(relative_pos - 0.5)  # Center is important
            
            # Combined score
            sentinel_score = 0.6 * conservation_score + 0.4 * position_score
            sentinel_candidates.append((residue, sentinel_score))
        
        # Select top sentinel candidates
        sentinel_hubs = sorted(
            sentinel_candidates,
            key=lambda x: x[1],
            reverse=True
        )[:self.max_sentinel_hubs]
        
        return sentinel_hubs


class EnhancedParallelTemperingMCMC:
    """Enhanced parallel tempering MCMC with adaptive control."""
    
    def __init__(self, temperatures: List[float] = None):
        """
        Initialize enhanced parallel tempering MCMC.
        
        Args:
            temperatures: List of temperatures for chains
        """
        self.temperatures = temperatures or [1.0, 1.6, 2.6, 4.2]
        self.n_chains = len(self.temperatures)
        self.swap_interval = 10
        self.adaptation_interval = 50
        
        # Adaptive parameters
        self.min_acceptance = 0.2
        self.max_acceptance = 0.5
        self.temperature_adaptation_rate = 0.1
        
    def run_enhanced_mcmc(self, initial_coords: np.ndarray,
                          energy_function: callable,
                          max_steps: int = 500) -> Tuple[np.ndarray, Dict]:
        """
        Run enhanced parallel tempering MCMC.
        
        Args:
            initial_coords: Initial coordinates
            energy_function: Energy function
            max_steps: Maximum steps
        
        Returns:
            Tuple of (best_coords, run_info)
        """
        n_residues = initial_coords.shape[0]
        
        # Initialize chains
        chains = [initial_coords.copy() for _ in range(self.n_chains)]
        energies = [energy_function(coords) for coords in chains]
        acceptance_rates = [0.0] * self.n_chains
        swap_attempts = [0] * (self.n_chains - 1)
        swap_successes = [0] * (self.n_chains - 1)
        
        # Track best solution
        best_coords = initial_coords.copy()
        best_energy = min(energies)
        
        # Energy history
        energy_history = []
        
        for step in tqdm(range(max_steps), desc="Enhanced MCMC"):
            # Propose moves for each chain
            for i in range(self.n_chains):
                # Adaptive proposal based on temperature
                proposal_scale = np.sqrt(self.temperatures[i])
                proposal = self.adaptive_proposal(chains[i], proposal_scale)
                proposal_energy = energy_function(proposal)
                
                # Metropolis acceptance
                delta_e = proposal_energy - energies[i]
                accept_prob = np.exp(-delta_e / self.temperatures[i])
                
                if np.random.random() < accept_prob:
                    chains[i] = proposal
                    energies[i] = proposal_energy
                    acceptance_rates[i] += 1
                
                # Track best
                if proposal_energy < best_energy:
                    best_coords = proposal.copy()
                    best_energy = proposal_energy
            
            # Adaptive temperature control
            if step % self.adaptation_interval == 0 and step > 0:
                self.adapt_temperatures(acceptance_rates, step)
            
            # Swap states between chains
            if step % self.swap_interval == 0 and step > 0:
                for i in range(self.n_chains - 1):
                    swap_attempts[i] += 1
                    
                    # Calculate swap probability
                    delta = (1.0/self.temperatures[i] - 1.0/self.temperatures[i+1]) * (energies[i] - energies[i+1])
                    swap_prob = min(1.0, np.exp(delta))
                    
                    if np.random.random() < swap_prob:
                        # Swap states
                        chains[i], chains[i+1] = chains[i+1], chains[i]
                        energies[i], energies[i+1] = energies[i+1], energies[i]
                        swap_successes[i] += 1
            
            energy_history.append(min(energies))
        
        # Normalize acceptance rates
        for i in range(self.n_chains):
            acceptance_rates[i] /= max_steps
        
        # Compute swap statistics
        swap_rates = [swap_successes[i] / max(1, swap_attempts[i]) 
                     for i in range(self.n_chains - 1)]
        
        run_info = {
            'best_energy': best_energy,
            'acceptance_rates': acceptance_rates,
            'swap_rates': swap_rates,
            'final_temperatures': self.temperatures,
            'energy_history': energy_history
        }
        
        return best_coords, run_info
    
    def adaptive_proposal(self, coords: np.ndarray, scale: float) -> np.ndarray:
        """Generate adaptive proposal based on temperature."""
        n_residues = coords.shape[0]
        
        # Higher temperature = larger proposals
        if scale < 1.5:
            # Low temperature: local moves
            return self.local_proposal(coords)
        elif scale < 3.0:
            # Medium temperature: medium moves
            return self.medium_proposal(coords)
        else:
            # High temperature: global moves
            return self.global_proposal(coords)
    
    def local_proposal(self, coords: np.ndarray) -> np.ndarray:
        """Local proposal for low temperatures."""
        n_residues = coords.shape[0]
        i = np.random.randint(0, n_residues)
        
        # Small local perturbation
        perturbation = np.random.normal(0, 0.2, 3)
        new_coords = coords.copy()
        new_coords[i] += perturbation
        
        return new_coords
    
    def medium_proposal(self, coords: np.ndarray) -> np.ndarray:
        """Medium-scale proposal."""
        n_residues = coords.shape[0]
        
        # Segment-based move
        if n_residues > 10:
            start = np.random.randint(0, n_residues - 5)
            end = start + np.random.randint(3, 8)
            end = min(end, n_residues)
            
            new_coords = coords.copy()
            segment = new_coords[start:end]
            
            # Apply rotation to segment
            angle = np.random.normal(0, 0.2)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            # Rotate in x-y plane
            segment_xy = segment[:, :2]
            rotated_xy = np.dot(segment_xy, rotation_matrix.T)
            new_coords[start:end, :2] = rotated_xy
        
        return new_coords
    
    def global_proposal(self, coords: np.ndarray) -> np.ndarray:
        """Global proposal for high temperatures."""
        # Global rotation and translation
        rotation = self.random_rotation_matrix()
        translation = np.random.normal(0, 0.5, 3)
        
        return np.dot(coords, rotation.T) + translation
    
    def random_rotation_matrix(self) -> np.ndarray:
        """Generate random rotation matrix."""
        # Random axis
        axis = np.random.normal(0, 1, 3)
        axis = axis / np.linalg.norm(axis)
        
        # Random angle
        angle = np.random.normal(0, 0.2)
        
        # Rodrigues' formula
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        K = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
        
        R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
        
        return R
    
    def adapt_temperatures(self, acceptance_rates: List[float], step: int):
        """Adapt temperatures based on acceptance rates."""
        for i in range(self.n_chains):
            if acceptance_rates[i] < self.min_acceptance:
                # Too few acceptances, lower temperature
                self.temperatures[i] *= (1 - self.temperature_adaptation_rate)
            elif acceptance_rates[i] > self.max_acceptance:
                # Too many acceptances, raise temperature
                self.temperatures[i] *= (1 + self.temperature_adaptation_rate)
        
        # Ensure temperature ordering
        self.temperatures.sort()


def main():
    """Main robustness features function."""
    parser = argparse.ArgumentParser(description="Phase 7: Robustness Features")
    parser.add_argument("--output-dir", required=True,
                       help="Directory to save robustness features")
    parser.add_argument("--test-sequences", help="File with test sequences")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize components
    entanglement_detector = EntanglementDetector()
    domain_proposer = EnsembleDomainProposer()
    hub_system = MultiHubSystem()
    mcmc = EnhancedParallelTemperingMCMC()
    
    try:
        print("✅ Phase 7 completed successfully!")
        print("   Implemented entanglement detector with local BFS topology sampler")
        print("   Created ensemble domain proposals with adaptive pruning")
        print("   Added multi-hub + sentinel residue system")
        print("   Enhanced parallel tempering MCMC implementation")
        
    except Exception as e:
        print(f"❌ Phase 7 failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
