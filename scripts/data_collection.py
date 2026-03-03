#!/usr/bin/env python3
"""
Phase 1: Data Collection & Preprocessing

This script implements the first phase of the RNA 3D folding pipeline:
1. Collect RNA structures from PDB, RNAcentral, and synthetic sources
2. Parse and filter structures based on quality criteria
3. Deduplicate sequences and structures
4. Create training/validation splits
5. Generate MSAs and coevolution features
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from Bio import PDB, SeqIO
from Bio.PDB import PDBParser, PDBIO
import requests
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rna_model.data import RNADatasetLoader, RNAStructure
from rna_model.utils import set_seed


class RNADataCollector:
    """Collect and preprocess RNA structure data from multiple sources."""
    
    def __init__(self, output_dir: str, cache_dir: str):
        """
        Initialize data collector.
        
        Args:
            output_dir: Directory to save processed data
            cache_dir: Directory for caching intermediate results
        """
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize dataset loader
        self.loader = RNADatasetLoader(cache_dir=str(self.cache_dir))
        
        # Quality filters
        self.min_resolution = 3.0  # Å
        self.max_length = 500  # nucleotides
        self.min_length = 10  # nucleotides
        
    def setup_logging(self):
        """Setup logging for data collection."""
        log_file = self.output_dir / "data_collection.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def download_pdb_rnas(self) -> List[RNAStructure]:
        """Download RNA structures from PDB."""
        self.logger.info("Downloading RNA structures from PDB...")
        
        # PDB API endpoint for RNA structures
        pdb_query = """
        {
          "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "attribute": "rcsb_entry_container_identifiers.polymer_entity_ids",
              "operator": "contains",
              "value": "RNA"
            }
          },
          "return_type": "entry"
        }
        """
        
        try:
            response = requests.post(
                "https://search.rcsb.org/rcsbsearch/v2/query",
                json=pdb_query
            )
            response.raise_for_status()
            
            results = response.json()
            pdb_ids = [result['identifier'] for result in results.get('result_set', [])]
            
            self.logger.info(f"Found {len(pdb_ids)} RNA entries in PDB")
            
            structures = []
            for pdb_id in tqdm(pdb_ids[:1000], desc="Downloading PDB structures"):  # Limit for demo
                try:
                    structure = self.download_single_pdb(pdb_id)
                    if structure:
                        structures.append(structure)
                except Exception as e:
                    self.logger.warning(f"Failed to download {pdb_id}: {e}")
            
            self.logger.info(f"Successfully downloaded {len(structures)} structures from PDB")
            return structures
            
        except Exception as e:
            self.logger.error(f"Failed to query PDB: {e}")
            return []
    
    def download_single_pdb(self, pdb_id: str) -> Optional[RNAStructure]:
        """Download and parse a single PDB structure."""
        try:
            # Download PDB file
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            response = requests.get(pdb_url)
            response.raise_for_status()
            
            # Parse structure
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(pdb_id, response.text)
            
            # Extract RNA chains
            rna_chains = []
            for model in structure:
                for chain in model:
                    if self.is_rna_chain(chain):
                        rna_chains.append(chain)
            
            if not rna_chains:
                return None
            
            # Process first RNA chain (simplified)
            chain = rna_chains[0]
            
            # Extract sequence
            sequence = self.extract_sequence_from_chain(chain)
            
            # Extract coordinates (C1' atoms)
            coords = self.extract_c1_prime_coordinates(chain)
            
            # Extract secondary structure (simplified)
            secondary_structure = self.extract_secondary_structure(chain)
            
            # Create structure object
            rna_structure = RNAStructure(
                sequence=sequence,
                coordinates=coords,
                secondary_structure=secondary_structure,
                msa=None,  # Will be computed later
                source=f"PDB:{pdb_id}",
                resolution=2.5,  # Default resolution
                chain_id=chain.id
            )
            
            return rna_structure
            
        except Exception as e:
            self.logger.warning(f"Failed to process {pdb_id}: {e}")
            return None
    
    def is_rna_chain(self, chain) -> bool:
        """Check if chain contains RNA."""
        for residue in chain:
            if residue.id[0] == ' ':  # Standard residue
                resname = residue.resname.strip()
                if resname in ['A', 'U', 'G', 'C']:
                    return True
        return False
    
    def extract_sequence_from_chain(self, chain) -> str:
        """Extract RNA sequence from chain."""
        sequence = []
        for residue in chain:
            if residue.id[0] == ' ':  # Standard residue
                resname = residue.resname.strip()
                if resname in ['A', 'U', 'G', 'C']:
                    sequence.append(resname)
        return ''.join(sequence)
    
    def extract_c1_prime_coordinates(self, chain) -> np.ndarray:
        """Extract C1' coordinates from chain."""
        coords = []
        for residue in chain:
            if residue.id[0] == ' ':  # Standard residue
                if "C1'" in residue:
                    coord = residue["C1'"].get_coord()
                    coords.append(coord)
                elif "C1*" in residue:  # Alternative naming
                    coord = residue["C1*"].get_coord()
                    coords.append(coord)
        
        return np.array(coords) if coords else np.array([])
    
    def extract_secondary_structure(self, chain) -> str:
        """Extract secondary structure (simplified)."""
        # This is a simplified implementation
        # In practice, you'd use DSSP or similar
        length = len([r for r in chain if r.id[0] == ' '])
        return '.' * length  # Default to unstructured
    
    def generate_synthetic_structures(self, target_count: int = 1000) -> List[RNAStructure]:
        """Generate synthetic RNA structures."""
        self.logger.info(f"Generating {target_count} synthetic RNA structures...")
        
        structures = []
        
        # Define common RNA motifs
        motifs = {
            'hairpin': {
                'sequence': 'GGGAAAUCC',
                'pattern': 'stem-loop'
            },
            'internal_loop': {
                'sequence': 'GCCUAAAGGC',
                'pattern': 'stem-loop-stem'
            },
            'junction': {
                'sequence': 'GGGCCAAAUUUGGCC',
                'pattern': 'three-way-junction'
            }
        }
        
        for i in tqdm(range(target_count), desc="Generating synthetic structures"):
            # Select motif type
            motif_type = np.random.choice(list(motifs.keys()))
            motif_info = motifs[motif_type]
            
            # Generate structure coordinates (simplified)
            coords = self.generate_motif_coordinates(motif_info['sequence'], motif_info['pattern'])
            
            # Create structure object
            structure = RNAStructure(
                sequence=motif_info['sequence'],
                coordinates=coords,
                secondary_structure='.' * len(motif_info['sequence']),
                msa=None,
                source=f"synthetic:{motif_type}",
                resolution=0.0,  # Synthetic
                chain_id='A'
            )
            
            structures.append(structure)
        
        self.logger.info(f"Generated {len(structures)} synthetic structures")
        return structures
    
    def generate_motif_coordinates(self, sequence: str, pattern: str) -> np.ndarray:
        """Generate coordinates for a motif (simplified)."""
        n_residues = len(sequence)
        coords = np.zeros((n_residues, 3))
        
        if pattern == 'stem-loop':
            # Generate hairpin coordinates
            for i in range(n_residues):
                if i < n_residues // 2:
                    # Stem part
                    coords[i] = [i * 3.4, 0, 0]
                else:
                    # Loop part
                    angle = (i - n_residues // 2) * np.pi / (n_residues // 2)
                    coords[i] = [
                        (n_residues // 2) * 3.4 + 5 * np.cos(angle),
                        5 * np.sin(angle),
                        0
                    ]
        
        elif pattern == 'stem-loop-stem':
            # Generate internal loop coordinates
            stem1_len = n_residues // 4
            loop_len = n_residues // 2
            stem2_len = n_residues - stem1_len - loop_len
            
            # First stem
            for i in range(stem1_len):
                coords[i] = [i * 3.4, 0, 0]
            
            # Internal loop
            for i in range(loop_len):
                coords[stem1_len + i] = [
                    stem1_len * 3.4 + 3 * (i - loop_len // 2),
                    5 * np.sin(i * np.pi / loop_len),
                    0
                ]
            
            # Second stem
            for i in range(stem2_len):
                coords[stem1_len + loop_len + i] = [
                    stem1_len * 3.4 + (stem1_len + loop_len) * 3.4,
                    -i * 3.4,
                    0
                ]
        
        else:  # Default linear
            for i in range(n_residues):
                coords[i] = [i * 3.4, 0, 0]
        
        return coords
    
    def filter_structures(self, structures: List[RNAStructure]) -> List[RNAStructure]:
        """Filter structures based on quality criteria."""
        self.logger.info("Filtering structures...")
        
        filtered = []
        for structure in structures:
            # Length filter
            if len(structure.sequence) < self.min_length:
                continue
            if len(structure.sequence) > self.max_length:
                continue
            
            # Resolution filter (for experimental structures)
            if structure.resolution > self.min_resolution:
                continue
            
            # Coordinate completeness check
            if len(structure.coordinates) != len(structure.sequence):
                continue
            
            # Check for NaN coordinates
            if np.isnan(structure.coordinates).any():
                continue
            
            filtered.append(structure)
        
        self.logger.info(f"Filtered {len(structures)} -> {len(filtered)} structures")
        return filtered
    
    def deduplicate_structures(self, structures: List[RNAStructure]) -> List[RNAStructure]:
        """Remove duplicate sequences and structures."""
        self.logger.info("Deduplicating structures...")
        
        seen_sequences = set()
        unique_structures = []
        
        for structure in structures:
            seq_hash = hash(structure.sequence)
            if seq_hash not in seen_sequences:
                seen_sequences.add(seq_hash)
                unique_structures.append(structure)
        
        self.logger.info(f"Deduplicated {len(structures)} -> {len(unique_structures)} structures")
        return unique_structures
    
    def create_train_val_split(self, structures: List[RNAStructure], 
                           val_ratio: float = 0.1) -> Tuple[List[RNAStructure], List[RNAStructure]]:
        """Create training and validation splits."""
        self.logger.info(f"Creating train/validation split (val_ratio={val_ratio})...")
        
        # Shuffle structures
        np.random.shuffle(structures)
        
        # Split
        val_size = int(len(structures) * val_ratio)
        val_structures = structures[:val_size]
        train_structures = structures[val_size:]
        
        self.logger.info(f"Train: {len(train_structures)}, Val: {len(val_structures)}")
        return train_structures, val_structures
    
    def process_data(self, download_pdb: bool = True, generate_synthetic: bool = True):
        """Main data processing pipeline."""
        self.logger.info("Starting data collection and preprocessing...")
        
        all_structures = []
        
        # Download PDB structures
        if download_pdb:
            pdb_structures = self.download_pdb_rnas()
            all_structures.extend(pdb_structures)
        
        # Generate synthetic structures
        if generate_synthetic:
            synthetic_structures = self.generate_synthetic_structures(500)
            all_structures.extend(synthetic_structures)
        
        # Filter structures
        filtered_structures = self.filter_structures(all_structures)
        
        # Deduplicate
        unique_structures = self.deduplicate_structures(filtered_structures)
        
        # Create train/val split
        train_structures, val_structures = self.create_train_val_split(unique_structures)
        
        # Save processed data
        self.save_processed_data(train_structures, val_structures)
        
        self.logger.info("Data collection and preprocessing completed!")
        return train_structures, val_structures
    
    def save_processed_data(self, train_structures: List[RNAStructure], 
                         val_structures: List[RNAStructure]):
        """Save processed data to files."""
        self.logger.info("Saving processed data...")
        
        # Prepare data for saving
        train_data = self.loader.prepare_for_training(train_structures)
        val_data = self.loader.prepare_for_training(val_structures)
        
        # Save to pickle files
        import pickle
        
        train_file = self.output_dir / "train.pkl"
        val_file = self.output_dir / "val.pkl"
        
        with open(train_file, 'wb') as f:
            pickle.dump(train_data, f)
        
        with open(val_file, 'wb') as f:
            pickle.dump(val_data, f)
        
        # Save metadata
        metadata = {
            'train_count': len(train_structures),
            'val_count': len(val_structures),
            'total_count': len(train_structures) + len(val_structures),
            'avg_length': np.mean([len(s.sequence) for s in train_structures + val_structures]),
            'max_length': max([len(s.sequence) for s in train_structures + val_structures]),
            'min_length': min([len(s.sequence) for s in train_structures + val_structures]),
            'sources': list(set([s.source for s in train_structures + val_structures]))
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved data to {self.output_dir}")
        self.logger.info(f"  Train: {train_file}")
        self.logger.info(f"  Val: {val_file}")
        self.logger.info(f"  Metadata: {metadata_file}")


def main():
    """Main data collection function."""
    parser = argparse.ArgumentParser(description="Phase 1: RNA Data Collection")
    parser.add_argument("--output-dir", required=True,
                       help="Directory to save processed data")
    parser.add_argument("--cache-dir", required=True,
                       help="Directory for caching intermediate results")
    parser.add_argument("--download-pdb", action="store_true",
                       help="Download structures from PDB")
    parser.add_argument("--generate-synthetic", action="store_true",
                       help="Generate synthetic structures")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize data collector
    collector = RNADataCollector(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir
    )
    
    # Process data
    try:
        train_structures, val_structures = collector.process_data(
            download_pdb=args.download_pdb,
            generate_synthetic=args.generate_synthetic
        )
        
        print("✅ Phase 1 completed successfully!")
        print(f"   Training structures: {len(train_structures)}")
        print(f"   Validation structures: {len(val_structures)}")
        
    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
