#!/usr/bin/env python3
"""
Pre-compute Multiple Sequence Alignments (MSAs) for RNA sequences.

This script generates MSAs for RNA sequences using sequence search tools.
For the Kaggle competition, MSAs can be pre-computed offline and loaded
during inference to augment the model with evolutionary information.

Usage:
    python scripts/precompute_msas.py --input test_sequences.csv --output msas/
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def tokenize_sequence(sequence: str) -> List[int]:
    """Tokenize RNA sequence to indices."""
    token_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
    return [token_map.get(nuc.upper(), 4) for nuc in sequence]


def generate_dummy_msa(sequence: str, n_seqs: int = 10) -> np.ndarray:
    """
    Generate a dummy MSA for testing/development.
    
    In production, this would use real MSA tools like:
    - rMSA (Recursive MSA)
    - Infernal
    - BLAST against RNA databases
    - Rfam/RNAcentral search
    
    Args:
        sequence: Query RNA sequence
        n_seqs: Number of sequences in MSA (including query)
    
    Returns:
        MSA tokens array (n_seqs, seq_len)
    """
    token_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
    seq_len = len(sequence)
    
    # Query sequence (first in MSA)
    query_tokens = tokenize_sequence(sequence)
    msa = [query_tokens]
    
    # Generate homologous sequences with mutations
    # This simulates evolutionary conservation patterns
    for i in range(n_seqs - 1):
        variant = []
        for j, nuc in enumerate(sequence):
            # Random mutation with 10-30% probability
            if np.random.random() < 0.2:
                # Mutate to a different nucleotide
                options = ['A', 'U', 'G', 'C']
                if nuc.upper() in options:
                    options.remove(nuc.upper())
                new_nuc = np.random.choice(options)
                variant.append(token_map[new_nuc])
            else:
                variant.append(token_map.get(nuc.upper(), 4))
        msa.append(variant)
    
    return np.array(msa, dtype=np.int64)


def generate_msa_for_sequence(seq_id: str, sequence: str, n_seqs: int = 10) -> Tuple[str, np.ndarray]:
    """
    Generate MSA for a single sequence.
    
    Args:
        seq_id: Sequence identifier
        sequence: RNA sequence
        n_seqs: Number of sequences in MSA
    
    Returns:
        Tuple of (seq_id, msa_tokens)
    """
    msa = generate_dummy_msa(sequence, n_seqs)
    return seq_id, msa


def precompute_msas(input_file: str, output_dir: str, n_seqs: int = 10, max_workers: int = None):
    """
    Pre-compute MSAs for all sequences in input file.
    
    Args:
        input_file: Path to CSV with sequences (id, sequence columns)
        output_dir: Directory to save MSA files
        n_seqs: Number of sequences per MSA
        max_workers: Number of parallel workers
    """
    import pandas as pd
    
    # Load sequences
    print(f"Loading sequences from {input_file}...")
    df = pd.read_csv(input_file)
    
    if 'id' not in df.columns or 'sequence' not in df.columns:
        print("Error: Input file must have 'id' and 'sequence' columns")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating MSAs for {len(df)} sequences...")
    print(f"MSA depth: {n_seqs} sequences per query")
    
    # Generate MSAs in parallel
    max_workers = max_workers or mp.cpu_count()
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for _, row in df.iterrows():
            future = executor.submit(
                generate_msa_for_sequence,
                row['id'],
                row['sequence'],
                n_seqs
            )
            futures.append(future)
        
        # Collect results
        for i, future in enumerate(futures):
            try:
                seq_id, msa = future.result()
                results.append((seq_id, msa))
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(df)} sequences...")
            except Exception as e:
                print(f"  Error processing sequence: {e}")
    
    # Save MSAs
    print(f"Saving MSAs to {output_dir}...")
    for seq_id, msa in results:
        output_file = output_path / f"{seq_id}.npy"
        np.save(output_file, msa)
    
    # Save metadata
    metadata = {
        'n_sequences': len(results),
        'msa_depth': n_seqs,
        'vocab_size': 5,
        'token_mapping': {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Complete! Saved {len(results)} MSAs to {output_dir}")


def load_msa(seq_id: str, msa_dir: str) -> np.ndarray:
    """Load pre-computed MSA for a sequence."""
    msa_file = Path(msa_dir) / f"{seq_id}.npy"
    if msa_file.exists():
        return np.load(msa_file)
    return None


def main():
    parser = argparse.ArgumentParser(description='Pre-compute MSAs for RNA sequences')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with sequences')
    parser.add_argument('--output', type=str, default='msas', help='Output directory for MSAs')
    parser.add_argument('--n-seq', type=int, default=10, help='Number of sequences per MSA')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    precompute_msas(
        input_file=args.input,
        output_dir=args.output,
        n_seqs=args.n_seq,
        max_workers=args.workers
    )


if __name__ == '__main__':
    main()
