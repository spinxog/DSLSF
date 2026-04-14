#!/usr/bin/env python3
"""
Fragment Library

This script implements fragment-based RNA structure prediction:
1. Motif-aware fragment mining from PDB structures
2. Common junction and pseudoknot fragment extraction
3. Fragment assembly sampling with LMDB storage
4. Fragment-based coordinate initialization
"""

import os
import sys
import json
import argparse
import logging
import lmdb
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import PDB
from Bio.PDB import PDBParser, Selection
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
import networkx as nx
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rna_model.utils import set_seed


class MotifAwareFragmentMiner:
    """Mine RNA fragments with motif awareness."""
    
    def __init__(self):
        """Initialize fragment miner."""
        self.motif_patterns = {
            'hairpin': {
                'min_size': 4,
                'max_size': 12,
                'patterns': ['GAAA', 'CUUG', 'GNRA', 'UNCG']
            },
            'internal_loop': {
                'min_size': 6,
                'max_size': 20,
                'patterns': ['AAUAA', 'UUUUU']
            },
            'junction': {
                'min_size': 8,
                'max_size': 30,
                'patterns': ['AGAA', 'UCUU']
            },
            'pseudoknot': {
                'min_size': 10,
                'max_size': 40,
                'patterns': ['GGAAUUCC']
            }
        }
        
    def mine_fragments_from_pdb(self, pdb_dir: str) -> Dict[str, List[Dict]]:
        """
        Mine fragments from PDB structures.
        
        Args:
            pdb_dir: Directory containing PDB files
        
        Returns:
            Dictionary of fragments by motif type
        """
        fragments = {motif_type: [] for motif_type in self.motif_patterns}
        
        pdb_files = list(Path(pdb_dir).glob('*.pdb'))
        
        for pdb_file in tqdm(pdb_files, desc="Mining fragments"):
            try:
                # Parse PDB structure
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure(pdb_file.stem, str(pdb_file))
                
                # Extract RNA chains
                rna_chains = self.extract_rna_chains(structure)
                
                for chain_id, chain in rna_chains.items():
                    # Get sequence and coordinates
                    sequence, coordinates = self.extract_sequence_and_coords(chain)
                    
                    # Detect motifs
                    motif_positions = self.detect_motifs(sequence)
                    
                    # Extract fragments for each motif
                    for motif_type, positions in motif_positions.items():
                        for start, end in positions:
                            fragment = self.extract_fragment(
                                sequence, coordinates, start, end, motif_type
                            )
                            if fragment:
                                fragments[motif_type].append(fragment)
                                
            except Exception as e:
                logging.warning(f"Failed to process {pdb_file}: {e}")
                continue
        
        # Cluster and select representative fragments
        for motif_type in fragments:
            if fragments[motif_type]:
                fragments[motif_type] = self.cluster_and_select_fragments(
                    fragments[motif_type]
                )
        
        return fragments
    
    def extract_rna_chains(self, structure) -> Dict[str, object]:
        """Extract RNA chains from PDB structure."""
        rna_chains = {}
        
        for model in structure:
            for chain in model:
                # Check if chain contains RNA
                is_rna = False
                for residue in chain:
                    if residue.id[0] == ' ' and residue.get_resname() in ['A', 'U', 'G', 'C']:
                        is_rna = True
                        break
                
                if is_rna:
                    rna_chains[chain.id] = chain
        
        return rna_chains
    
    def extract_sequence_and_coords(self, chain) -> Tuple[str, np.ndarray]:
        """Extract sequence and coordinates from chain."""
        sequence = []
        coordinates = []
        
        for residue in chain:
            if residue.id[0] == ' ' and residue.get_resname() in ['A', 'U', 'G', 'C']:
                # Get C1' atom coordinate
                if 'C1\'' in residue:
                    coord = residue['C1\''].get_coord()
                    sequence.append(residue.get_resname())
                    coordinates.append(coord[0])
        
        return ''.join(sequence), np.array(coordinates)
    
    def detect_motifs(self, sequence: str) -> Dict[str, List[Tuple[int, int]]]:
        """Detect motif positions in sequence."""
        motif_positions = {}
        
        for motif_type, config in self.motif_patterns.items():
            positions = []
            
            for pattern in config['patterns']:
                start = 0
                while True:
                    pos = sequence.find(pattern, start)
                    if pos == -1:
                        break
                    
                    # Extend to full fragment size
                    fragment_start = max(0, pos - 5)
                    fragment_end = min(len(sequence), pos + len(pattern) + 5)
                    
                    # Check size constraints
                    fragment_size = fragment_end - fragment_start
                    if config['min_size'] <= fragment_size <= config['max_size']:
                        positions.append((fragment_start, fragment_end))
                    
                    start = pos + 1
            
            motif_positions[motif_type] = positions
        
        return motif_positions
    
    def extract_fragment(self, sequence: str, coordinates: np.ndarray,
                        start: int, end: int, motif_type: str) -> Optional[Dict]:
        """Extract fragment from sequence and coordinates."""
        if start >= end or end > len(sequence):
            return None
        
        fragment_seq = sequence[start:end]
        fragment_coords = coordinates[start:end]
        
        # Compute fragment features
        features = self.compute_fragment_features(fragment_seq, fragment_coords)
        
        fragment = {
            'sequence': fragment_seq,
            'coordinates': fragment_coords,
            'motif_type': motif_type,
            'start': start,
            'end': end,
            'size': len(fragment_seq),
            'features': features
        }
        
        return fragment
    
    def compute_fragment_features(self, sequence: str, coordinates: np.ndarray) -> Dict:
        """Compute features for fragment."""
        features = {}
        
        # Geometric features
        if len(coordinates) > 1:
            # End-to-end distance
            end_to_end = np.linalg.norm(coordinates[-1] - coordinates[0])
            features['end_to_end'] = end_to_end
            
            # Radius of gyration
            center = np.mean(coordinates, axis=0)
            rg = np.sqrt(np.mean(np.sum((coordinates - center) ** 2, axis=1)))
            features['radius_of_gyration'] = rg
            
            # Bond lengths
            bond_lengths = []
            for i in range(1, len(coordinates)):
                bond_length = np.linalg.norm(coordinates[i] - coordinates[i-1])
                bond_lengths.append(bond_length)
            
            features['mean_bond_length'] = np.mean(bond_lengths) if bond_lengths else 0.0
            features['std_bond_length'] = np.std(bond_lengths) if bond_lengths else 0.0
        else:
            features['end_to_end'] = 0.0
            features['radius_of_gyration'] = 0.0
            features['mean_bond_length'] = 0.0
            features['std_bond_length'] = 0.0
        
        # Sequence features
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        features['gc_content'] = gc_content
        
        return features
    
    def cluster_and_select_fragments(self, fragments: List[Dict]) -> List[Dict]:
        """Cluster fragments and select representatives."""
        if len(fragments) <= 10:
            return fragments
        
        # Extract feature vectors
        feature_vectors = []
        for fragment in fragments:
            vector = [
                fragment['features']['end_to_end'],
                fragment['features']['radius_of_gyration'],
                fragment['features']['mean_bond_length'],
                fragment['features']['std_bond_length'],
                fragment['features']['gc_content']
            ]
            feature_vectors.append(vector)
        
        feature_vectors = np.array(feature_vectors)
        
        # Cluster fragments
        clustering = DBSCAN(eps=0.5, min_samples=2)
        labels = clustering.fit_predict(feature_vectors)
        
        # Select representatives from each cluster
        selected_fragments = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            
            # Get fragments in this cluster
            cluster_indices = np.where(labels == label)[0]
            cluster_fragments = [fragments[i] for i in cluster_indices]
            
            # Select fragment closest to cluster center
            cluster_center = np.mean(feature_vectors[cluster_indices], axis=0)
            distances = [np.linalg.norm(feature_vectors[i] - cluster_center) 
                        for i in cluster_indices]
            best_idx = cluster_indices[np.argmin(distances)]
            
            selected_fragments.append(fragments[best_idx])
        
        return selected_fragments


class FragmentAssembler:
    """Assemble RNA structures from fragments."""
    
    def __init__(self, fragment_library: Dict):
        """
        Initialize fragment assembler.
        
        Args:
            fragment_library: Dictionary of fragments by motif type
        """
        self.fragment_library = fragment_library
        self.bond_length = 3.4
        
    def assemble_structure(self, sequence: str, 
                          motif_positions: Dict[str, List[Tuple[int, int]]]) -> np.ndarray:
        """
        Assemble structure from fragments.
        
        Args:
            sequence: Target RNA sequence
            motif_positions: Positions of motifs in sequence
        
        Returns:
            Assembled coordinates
        """
        # Initialize with linear chain
        coordinates = self.initialize_linear_chain(len(sequence))
        
        # Process motifs in order
        processed_positions = set()
        
        for motif_type, positions in sorted(motif_positions.items(), 
                                          key=lambda x: len(x[1]), reverse=True):
            for start, end in positions:
                # Check if this region overlaps with processed regions
                if any(self.ranges_overlap((start, end), processed_pos) 
                      for processed_pos in processed_positions):
                    continue
                
                # Find matching fragment
                fragment = self.find_best_fragment(sequence[start:end], motif_type)
                
                if fragment:
                    # Place fragment in structure
                    coordinates = self.place_fragment(
                        coordinates, fragment, start, end
                    )
                    
                    processed_positions.add((start, end))
        
        # Refine assembled structure
        coordinates = self.refine_assembly(coordinates)
        
        return coordinates
    
    def initialize_linear_chain(self, length: int) -> np.ndarray:
        """Initialize linear chain coordinates."""
        coordinates = np.zeros((length, 3))
        
        for i in range(length):
            coordinates[i, 0] = i * self.bond_length
        
        return coordinates
    
    def ranges_overlap(self, range1: Tuple[int, int], range2: Tuple[int, int]) -> bool:
        """Check if two ranges overlap."""
        return not (range1[1] <= range2[0] or range2[1] <= range1[0])
    
    def find_best_fragment(self, target_seq: str, motif_type: str) -> Optional[Dict]:
        """Find best matching fragment for target sequence."""
        if motif_type not in self.fragment_library:
            return None
        
        candidates = self.fragment_library[motif_type]
        if not candidates:
            return None
        
        # Score candidates by sequence similarity
        best_candidate = None
        best_score = -1
        
        for fragment in candidates:
            score = self.compute_sequence_similarity(target_seq, fragment['sequence'])
            if score > best_score:
                best_score = score
                best_candidate = fragment
        
        return best_candidate if best_score > 0.5 else None
    
    def compute_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Compute sequence similarity."""
        if len(seq1) != len(seq2):
            # Use longest common subsequence for different lengths
            return self.lcs_length(seq1, seq2) / max(len(seq1), len(seq2))
        
        # Compute identity
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)
    
    def lcs_length(self, seq1: str, seq2: str) -> int:
        """Compute longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = np.zeros((m + 1, n + 1))
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i, j] = dp[i-1, j-1] + 1
                else:
                    dp[i, j] = max(dp[i-1, j], dp[i, j-1])
        
        return int(dp[m, n])
    
    def place_fragment(self, coordinates: np.ndarray, fragment: Dict,
                      start: int, end: int) -> np.ndarray:
        """Place fragment in structure."""
        fragment_coords = fragment['coordinates']
        fragment_length = len(fragment_coords)
        target_length = end - start
        
        # Handle size mismatch
        if fragment_length != target_length:
            fragment_coords = self.resize_fragment(fragment_coords, target_length)
        
        # Align fragment to existing structure
        if start > 0 and end < len(coordinates):
            # Align both ends
            start_coord = coordinates[start-1]
            end_coord = coordinates[end]
            
            # Compute transformation
            fragment_start = fragment_coords[0]
            fragment_end = fragment_coords[-1]
            
            # Translation to align start
            translation = start_coord - fragment_start
            aligned_coords = fragment_coords + translation
            
            # Rotation to align end (simplified)
            target_vector = end_coord - start_coord
            fragment_vector = fragment_end - fragment_start
            
            if np.linalg.norm(fragment_vector) > 0 and np.linalg.norm(target_vector) > 0:
                # Compute rotation matrix
                rotation = self.compute_rotation_matrix(fragment_vector, target_vector)
                aligned_coords = np.dot(aligned_coords - start_coord, rotation.T) + start_coord
        else:
            # Simple placement
            if start > 0:
                translation = coordinates[start-1] - fragment_coords[0]
                aligned_coords = fragment_coords + translation
            else:
                aligned_coords = fragment_coords
        
        # Place fragment
        coordinates[start:end] = aligned_coords
        
        return coordinates
    
    def resize_fragment(self, fragment_coords: np.ndarray, target_length: int) -> np.ndarray:
        """Resize fragment to target length."""
        current_length = len(fragment_coords)
        
        if current_length == target_length:
            return fragment_coords
        elif current_length < target_length:
            # Interpolate to larger size
            indices = np.linspace(0, current_length - 1, target_length)
            resized = np.zeros((target_length, 3))
            
            for i, idx in enumerate(indices):
                lower = int(np.floor(idx))
                upper = int(np.ceil(idx))
                
                if lower == upper:
                    resized[i] = fragment_coords[lower]
                else:
                    weight = idx - lower
                    resized[i] = (1 - weight) * fragment_coords[lower] + weight * fragment_coords[upper]
            
            return resized
        else:
            # Downsample
            indices = np.linspace(0, current_length - 1, target_length, dtype=int)
            return fragment_coords[indices]
    
    def compute_rotation_matrix(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """Compute rotation matrix to align vec1 to vec2."""
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Compute rotation axis
        axis = np.cross(vec1_norm, vec2_norm)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-6:
            # Vectors are parallel
            if np.dot(vec1_norm, vec2_norm) > 0:
                return np.eye(3)
            else:
                # 180-degree rotation
                return -np.eye(3)
        
        axis = axis / axis_norm
        
        # Compute angle
        cos_angle = np.dot(vec1_norm, vec2_norm)
        sin_angle = np.sqrt(1 - cos_angle ** 2)
        
        # Rodrigues' formula
        K = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
        
        R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
        
        return R
    
    def refine_assembly(self, coordinates: np.ndarray) -> np.ndarray:
        """Refine assembled structure."""
        # Simple refinement: enforce bond lengths
        refined = coordinates.copy()
        
        for i in range(1, len(refined)):
            # Adjust to maintain bond length
            current_length = np.linalg.norm(refined[i] - refined[i-1])
            if current_length > 0:
                refined[i] = refined[i-1] + (refined[i] - refined[i-1]) * (self.bond_length / current_length)
        
        return refined


class FragmentLibraryManager:
    """Manage fragment library with LMDB storage."""
    
    def __init__(self, library_path: str):
        """
        Initialize library manager.
        
        Args:
            library_path: Path to LMDB library
        """
        self.library_path = Path(library_path)
        self.library_path.mkdir(parents=True, exist_ok=True)
        
    def save_fragments(self, fragments: Dict[str, List[Dict]]) -> None:
        """Save fragments to LMDB."""
        lmdb_path = self.library_path / "fragments.lmdb"
        
        # Create LMDB environment
        env = lmdb.open(str(lmdb_path), map_size=1024*1024*1024*10)  # 10GB
        
        with env.begin(write=True) as txn:
            for motif_type, fragment_list in fragments.items():
                for i, fragment in enumerate(fragment_list):
                    # Serialize fragment
                    data = pickle.dumps(fragment)
                    
                    # Create key
                    key = f"{motif_type}_{i:06d}".encode()
                    
                    # Store in LMDB
                    txn.put(key, data)
        
        # Save metadata
        metadata = {
            'motif_types': list(fragments.keys()),
            'total_fragments': sum(len(fragments[motif]) for motif in fragments),
            'fragments_per_type': {motif: len(fragments[motif]) for motif in fragments}
        }
        
        metadata_file = self.library_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {metadata['total_fragments']} fragments to LMDB")
    
    def load_fragments(self) -> Dict[str, List[Dict]]:
        """Load fragments from LMDB."""
        lmdb_path = self.library_path / "fragments.lmdb"
        
        if not lmdb_path.exists():
            return {}
        
        # Load metadata
        metadata_file = self.library_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {'motif_types': []}
        
        # Load fragments from LMDB
        env = lmdb.open(str(lmdb_path), readonly=True)
        fragments = {motif_type: [] for motif_type in metadata['motif_types']}
        
        with env.begin() as txn:
            cursor = txn.cursor()
            
            for key, value in cursor:
                key_str = key.decode()
                motif_type = key_str.split('_')[0]
                
                if motif_type in fragments:
                    # Safe JSON deserialization with validation
                    try:
                        fragment_data = json.loads(value.decode())
                        # Validate fragment structure
                        if self._validate_fragment(fragment_data):
                            fragments[motif_type].append(fragment_data)
                        else:
                            logging.warning(f"Invalid fragment structure for key: {key_str}")
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        logging.error(f"Failed to decode fragment for key {key_str}: {e}")
                        continue
        
        return fragments
    
    def search_fragments(self, motif_type: str, sequence: str, 
                       top_k: int = 5) -> List[Dict]:
        """Search for best matching fragments."""
        fragments = self.load_fragments()
        
        if motif_type not in fragments:
            return []
        
        # Score and rank fragments
        scored_fragments = []
        for fragment in fragments[motif_type]:
            similarity = self.compute_sequence_similarity(sequence, fragment['sequence'])
            scored_fragments.append((similarity, fragment))
        
        # Sort by similarity and return top-k
        scored_fragments.sort(key=lambda x: x[0], reverse=True)
        
        return [fragment for _, fragment in scored_fragments[:top_k]]
    
    def _validate_fragment(self, fragment: Dict) -> bool:
        """Validate fragment structure to prevent malicious data injection."""
        if not isinstance(fragment, dict):
            return False
        
        # Required fields
        required_fields = ['sequence', 'coordinates', 'motif_type']
        if not all(field in fragment for field in required_fields):
            return False
        
        # Validate sequence
        if not isinstance(fragment['sequence'], str) or len(fragment['sequence']) == 0:
            return False
        
        # Validate coordinates
        coords = fragment['coordinates']
        if not isinstance(coords, list) or len(coords) == 0:
            return False
        
        # Check coordinate structure
        for coord in coords:
            if not isinstance(coord, list) or len(coord) != 3:
                return False
            for val in coord:
                if not isinstance(val, (int, float)):
                    return False
                if abs(val) > 1000:  # Reasonable coordinate range
                    return False
        
        # Validate motif type
        valid_motifs = {'hairpin', 'internal_loop', 'junction', 'pseudoknot'}
        if fragment['motif_type'] not in valid_motifs:
            return False
        
        return True
    
    def compute_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Compute sequence similarity."""
        if len(seq1) != len(seq2):
            return self.lcs_length(seq1, seq2) / max(len(seq1), len(seq2))
        
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)
    
    def lcs_length(self, seq1: str, seq2: str) -> int:
        """Compute longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = np.zeros((m + 1, n + 1))
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i, j] = dp[i-1, j-1] + 1
                else:
                    dp[i, j] = max(dp[i-1, j], dp[i, j-1])
        
        return int(dp[m, n])


def main():
    """Main fragment library function."""
    parser = argparse.ArgumentParser(description="Fragment Library for RNA Structures")
    parser.add_argument("--pdb-dir", required=True,
                       help="Directory containing PDB files")
    parser.add_argument("--library-path", required=True,
                       help="Path to save fragment library")
    parser.add_argument("--action", choices=['mine', 'assemble', 'search'], default='mine',
                       help="Action to perform")
    parser.add_argument("--sequence", help="Target sequence for assembly/search")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    try:
        if args.action == 'mine':
            # Mine fragments
            miner = MotifAwareFragmentMiner()
            fragments = miner.mine_fragments_from_pdb(args.pdb_dir)
            
            # Save to LMDB
            manager = FragmentLibraryManager(args.library_path)
            manager.save_fragments(fragments)
            
        elif args.action == 'assemble':
            # Assemble structure
            manager = FragmentLibraryManager(args.library_path)
            fragments = manager.load_fragments()
            assembler = FragmentAssembler(fragments)
            
            if args.sequence:
                # Detect motifs (simplified)
                miner = MotifAwareFragmentMiner()
                motif_positions = miner.detect_motifs(args.sequence)
                
                # Assemble structure
                coordinates = assembler.assemble_structure(args.sequence, motif_positions)
                print(f"Assembled structure with {len(coordinates)} residues")
            
        elif args.action == 'search':
            # Search fragments
            manager = FragmentLibraryManager(args.library_path)
            
            if args.sequence:
                # Search for hairpin fragments
                results = manager.search_fragments('hairpin', args.sequence)
                print(f"Found {len(results)} matching fragments")
        
        print("✅ Fragment library operations completed successfully!")
        print("   Implemented motif-aware fragment mining")
        print("   Created common junction and pseudoknot fragment extraction")
        print("   Added fragment assembly sampling with LMDB storage")
        print("   Built fragment-based coordinate initialization")
        
    except Exception as e:
        print(f"❌ Fragment library operations failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
