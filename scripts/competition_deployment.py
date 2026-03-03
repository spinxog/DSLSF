#!/usr/bin/env python3
"""
Phase 5: Competition Deployment

This script implements the fifth phase of the RNA 3D folding pipeline:
1. Precompute embeddings and bundle artifacts
2. Implement adaptive budgeting and runtime optimization
3. Create submission pipeline with validation and formatting
4. Generate competition-ready submission files
"""

import os
import sys
import json
import argparse
import logging
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
import faiss
import lmdb
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rna_model import RNAFoldingPipeline, PipelineConfig
from rna_model.utils import set_seed, memory_usage


class EmbeddingCompressor:
    """Compress and cache LM embeddings for fast retrieval."""
    
    def __init__(self, output_dir: str):
        """
        Initialize embedding compressor.
        
        Args:
            output_dir: Directory to save compressed embeddings
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compression parameters
        self.target_dims = 128
        self.quantization_bits = 8
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for compression."""
        log_file = self.output_dir / "compression.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def compress_embeddings(self, embeddings: np.ndarray, 
                        sequences: List[str]) -> Tuple[np.ndarray, Dict]:
        """
        Compress embeddings using PCA and quantization.
        
        Args:
            embeddings: Original embeddings (n_sequences, seq_len, d_model)
            sequences: Corresponding sequences
        
        Returns:
            Tuple of (compressed_embeddings, metadata)
        """
        self.logger.info(f"Compressing {len(sequences)} embeddings...")
        
        # Flatten embeddings for PCA
        n_seqs, seq_len, d_model = embeddings.shape
        flattened = embeddings.reshape(n_seqs, -1)
        
        # Fit PCA
        pca = PCA(n_components=self.target_dims)
        compressed = pca.fit_transform(flattened)
        
        # Quantize to 8-bit
        min_val, max_val = compressed.min(), compressed.max()
        scale = (max_val - min_val) / 255.0
        quantized = np.round((compressed - min_val) / scale).astype(np.uint8)
        
        # Compute reconstruction error
        reconstructed = quantized.astype(np.float32) * scale + min_val
        reconstruction_error = np.mean((flattened - reconstructed) ** 2)
        
        # Store metadata
        metadata = {
            'pca_components': pca.components_,
            'pca_mean': pca.mean_,
            'min_val': min_val,
            'max_val': max_val,
            'scale': scale,
            'reconstruction_error': reconstruction_error,
            'original_shape': (n_seqs, seq_len, d_model),
            'compressed_shape': quantized.shape
        }
        
        self.logger.info(f"Compression complete. Reconstruction error: {reconstruction_error:.6f}")
        
        return quantized, metadata
    
    def compute_family_clusters(self, embeddings: np.ndarray, 
                             sequences: List[str]) -> Dict:
        """
        Cluster sequences into families using HDBSCAN.
        
        Args:
            embeddings: Compressed embeddings
            sequences: Corresponding sequences
        
        Returns:
            Dictionary with cluster information
        """
        self.logger.info("Computing family clusters...")
        
        # Use sequence length as additional feature
        lengths = np.array([len(seq) for seq in sequences]).reshape(-1, 1)
        
        # Combine embeddings and lengths
        features = np.concatenate([embeddings, lengths], axis=1)
        
        # Cluster using HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=5, min_samples=3)
        cluster_labels = clusterer.fit_predict(features)
        
        # Compute cluster statistics
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = {
                    'sequences': [],
                    'indices': [],
                    'avg_length': 0,
                    'contact_correlation': 0.9  # Default
                }
            
            clusters[label]['sequences'].append(sequences[i])
            clusters[label]['indices'].append(i)
        
        # Compute statistics for each cluster
        for label, cluster_info in clusters.items():
            if label != -1:  # Not noise
                seqs = cluster_info['sequences']
                cluster_info['avg_length'] = np.mean([len(seq) for seq in seqs])
                cluster_info['size'] = len(seqs)
        
        self.logger.info(f"Found {len(clusters)} clusters")
        
        return {
            'clusters': clusters,
            'labels': cluster_labels,
            'n_clusters': len([k for k in clusters.keys() if k != -1])
        }
    
    def build_ann_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build FAISS index for fast nearest neighbor retrieval.
        
        Args:
            embeddings: Compressed embeddings
        
        Returns:
            FAISS index
        """
        self.logger.info("Building ANN index...")
        
        # Normalize embeddings
        embeddings_norm = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings_norm)
        
        # Build index
        n_vectors, dim = embeddings_norm.shape
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(dim),
            dim,
            64,  # nlist
            8,   # M (PQ subquantizers)
            8    # nbits (bits per subquantizer)
        )
        
        # Train index
        index.train(embeddings_norm)
        index.add(embeddings_norm)
        
        self.logger.info(f"Built index with {n_vectors} vectors")
        
        return index
    
    def save_compressed_data(self, compressed_embeddings: np.ndarray,
                          metadata: Dict,
                          cluster_info: Dict,
                          ann_index: faiss.Index):
        """Save compressed data and indices."""
        self.logger.info("Saving compressed data...")
        
        # Save compressed embeddings
        embeddings_file = self.output_dir / "compressed_embeddings.npy"
        np.save(embeddings_file, compressed_embeddings)
        
        # Save metadata
        metadata_file = self.output_dir / "compression_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save cluster information
        cluster_file = self.output_dir / "cluster_info.json"
        with open(cluster_file, 'w') as f:
            json.dump(cluster_info, f, indent=2)
        
        # Save ANN index
        index_file = self.output_dir / "ann_index.faiss"
        faiss.write_index(ann_index, str(index_file))
        
        # Create LMDB for fast access
        self.create_lmdb_database(compressed_embeddings, cluster_info)
        
        self.logger.info(f"Saved compressed data to {self.output_dir}")
    
    def create_lmdb_database(self, embeddings: np.ndarray, cluster_info: Dict):
        """Create LMDB database for fast sequence lookup."""
        self.logger.info("Creating LMDB database...")
        
        lmdb_path = self.output_dir / "embeddings.lmdb"
        env = lmdb.open(str(lmdb_path), map_size=1024*1024*1024*10)  # 10GB
        
        with env.begin(write=True) as txn:
            for i, embedding in enumerate(tqdm(embeddings, desc="Creating LMDB")):
                # Store embedding
                key = f"embedding_{i:08d}".encode()
                value = embedding.tobytes()
                txn.put(key, value)
        
        env.close()
        self.logger.info("LMDB database created")


class AdaptiveBudgetManager:
    """Manage adaptive time budgeting for competition deployment."""
    
    def __init__(self, total_time_limit: float = 8.0 * 3600):
        """
        Initialize budget manager.
        
        Args:
            total_time_limit: Total time limit in seconds
        """
        self.total_time_limit = total_time_limit
        self.start_time = None
        self.processed_sequences = 0
        self.total_sequences = 0
        
        # Budget parameters
        self.min_time_per_sequence = 30.0  # Minimum 30 seconds
        self.max_time_per_sequence = 900.0  # Maximum 15 minutes
        
    def start_timing(self):
        """Start timing the competition run."""
        self.start_time = time.time()
        
    def compute_sequence_budget(self, sequence: str, 
                           complexity_score: float) -> float:
        """
        Compute adaptive time budget for a sequence.
        
        Args:
            sequence: RNA sequence
            complexity_score: Complexity score (0-1)
        
        Returns:
            Time budget in seconds
        """
        # Base budget proportional to sequence length
        length = len(sequence)
        base_budget = 60.0 + length * 0.5
        
        # Adjust for complexity
        complexity_multiplier = 1.0 + complexity_score * 2.0
        
        # Check remaining global time
        if self.start_time:
            elapsed = time.time() - self.start_time
            remaining_time = self.total_time_limit - elapsed
            remaining_sequences = self.total_sequences - self.processed_sequences
            
            if remaining_sequences > 0:
                avg_remaining_budget = remaining_time / remaining_sequences
                max_reasonable_budget = min(base_budget * complexity_multiplier, avg_remaining_budget * 2.0)
            else:
                max_reasonable_budget = base_budget * complexity_multiplier
        else:
            max_reasonable_budget = base_budget * complexity_multiplier
        
        # Apply constraints
        final_budget = np.clip(max_reasonable_budget, 
                              self.min_time_per_sequence, 
                              self.max_time_per_sequence)
        
        return final_budget
    
    def compute_complexity_score(self, sequence: str) -> float:
        """
        Compute complexity score for adaptive budgeting.
        
        Args:
            sequence: RNA sequence
        
        Returns:
            Complexity score (0-1)
        """
        length = len(sequence)
        
        # Count potential junction-forming patterns
        junction_patterns = ['GAAA', 'CUUG', 'GNRA', 'UNCG']
        junction_count = sum(1 for pattern in junction_patterns if pattern in sequence)
        
        # GC content
        gc_content = (sequence.count('G') + sequence.count('C')) / length
        
        # Predicted secondary structure complexity (simplified)
        predicted_stems = min(length // 10, 8)
        predicted_loops = max(predicted_stems - 1, 0)
        
        # Combine features
        complexity = (
            0.3 * (length / 500.0) +
            0.2 * junction_count / 5.0 +
            0.2 * gc_content +
            0.3 * min(predicted_loops / 5.0, 1.0)
        )
        
        return np.clip(complexity, 0.0, 1.0)
    
    def should_use_conservative_mode(self) -> bool:
        """Check if we should enter conservative mode."""
        if not self.start_time:
            return False
        
        elapsed = time.time() - self.start_time
        remaining_time = self.total_time_limit - elapsed
        
        # Enter conservative mode if less than 10% time remaining
        return remaining_time < self.total_time_limit * 0.1


class CompetitionPipeline:
    """Complete competition deployment pipeline."""
    
    def __init__(self, model_path: str, cache_dir: str, output_dir: str):
        """
        Initialize competition pipeline.
        
        Args:
            model_path: Path to trained model
            cache_dir: Directory with cached artifacts
            output_dir: Output directory for submissions
        """
        self.model_path = Path(model_path)
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.pipeline = None
        self.budget_manager = AdaptiveBudgetManager()
        self.compressor = EmbeddingCompressor(str(self.cache_dir / "compressed"))
        
    def setup_logging(self):
        """Setup logging for competition pipeline."""
        log_file = self.output_dir / "competition.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_pipeline(self):
        """Setup RNA folding pipeline."""
        self.logger.info("Setting up RNA folding pipeline...")
        
        # Load configuration
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = PipelineConfig(**config_dict)
        else:
            config = PipelineConfig(device="cuda", mixed_precision=True)
        
        # Initialize pipeline
        self.pipeline = RNAFoldingPipeline(config)
        
        # Load model weights
        if (self.model_path / "best_model_by_tm.pth").exists():
            self.pipeline.load_model(str(self.model_path / "best_model_by_tm.pth"))
        else:
            self.logger.warning("No trained model found, using random weights")
        
        # Enable competition mode
        self.pipeline.enable_competition_mode()
        
        self.logger.info("Pipeline setup complete")
    
    def precompute_embeddings(self, sequences: List[str]):
        """Precompute and compress embeddings for all sequences."""
        self.logger.info(f"Precomputing embeddings for {len(sequences)} sequences...")
        
        all_embeddings = []
        
        # Process in batches
        batch_size = 32
        for i in tqdm(range(0, len(sequences), batch_size), desc="Computing embeddings"):
            batch_sequences = sequences[i:i+batch_size]
            
            # Get embeddings from language model
            with torch.no_grad():
                batch_embeddings = []
                for seq in batch_sequences:
                    result = self.pipeline.model.language_model(seq)
                    embedding = result['embeddings'].cpu().numpy()
                    batch_embeddings.append(embedding)
                
                all_embeddings.extend(batch_embeddings)
        
        # Stack embeddings
        embeddings_array = np.stack(all_embeddings)
        
        # Compress embeddings
        compressed_embeddings, metadata = self.compressor.compress_embeddings(embeddings_array, sequences)
        
        # Compute clusters
        cluster_info = self.compressor.compute_family_clusters(compressed_embeddings, sequences)
        
        # Build ANN index
        ann_index = self.compressor.build_ann_index(compressed_embeddings)
        
        # Save everything
        self.compressor.save_compressed_data(compressed_embeddings, metadata, cluster_info, ann_index)
        
        self.logger.info("Precomputation complete")
    
    def process_competition_batch(self, test_sequences: List[str]) -> Dict:
        """
        Process competition batch with adaptive budgeting.
        
        Args:
            test_sequences: List of test sequences
        
        Returns:
            Dictionary with results and metadata
        """
        self.logger.info(f"Processing competition batch of {len(test_sequences)} sequences...")
        
        # Start timing
        self.budget_manager.start_timing()
        self.budget_manager.total_sequences = len(test_sequences)
        
        # Initialize results
        all_results = []
        submission_data = []
        
        # Process each sequence
        for i, sequence in enumerate(tqdm(test_sequences, desc="Processing sequences")):
            self.budget_manager.processed_sequences = i
            
            # Check if we should use conservative mode
            if self.budget_manager.should_use_conservative_mode():
                self.logger.warning("Entering conservative mode")
                # Could implement simplified processing here
            
            # Compute complexity and budget
            complexity = self.budget_manager.compute_complexity_score(sequence)
            budget = self.budget_manager.compute_sequence_budget(sequence, complexity)
            
            sequence_start_time = time.time()
            
            try:
                # Predict structure
                result = self.pipeline.predict_single_sequence(sequence)
                
                # Add metadata
                result.update({
                    'sequence_id': f"seq_{i}",
                    'complexity_score': complexity,
                    'budget_allocated': budget,
                    'time_used': time.time() - sequence_start_time,
                    'success': True
                })
                
                # Add to submission data
                coords = result['coordinates']
                for residue_idx in range(result['n_residues']):
                    for decoy_idx in range(5):
                        x, y, z = coords[decoy_idx * result['n_residues'] + residue_idx]
                        submission_data.append({
                            'residue_id': f"seq_{i}_{residue_idx}_{decoy_idx + 1}",
                            'x': x,
                            'y': y,
                            'z': z,
                            'sequence_id': f"seq_{i}",
                            'decoy': decoy_idx + 1,
                            'residue_index': residue_idx
                        })
                
            except Exception as e:
                self.logger.error(f"Failed to process sequence {i}: {e}")
                
                # Generate fallback
                fallback_coords = self.generate_fallback_coordinates(len(sequence))
                
                result = {
                    'sequence_id': f"seq_{i}",
                    'sequence': sequence,
                    'coordinates': fallback_coords,
                    'n_decoys': 5,
                    'n_residues': len(sequence),
                    'complexity_score': complexity,
                    'budget_allocated': budget,
                    'time_used': time.time() - sequence_start_time,
                    'success': False,
                    'error': str(e)
                }
            
            all_results.append(result)
        
        # Create submission
        submission = self.create_submission(submission_data)
        
        # Generate report
        report = self.generate_competition_report(all_results)
        
        return {
            'results': all_results,
            'submission': submission,
            'report': report
        }
    
    def generate_fallback_coordinates(self, n_residues: int) -> np.ndarray:
        """Generate fallback coordinates (simple linear chain)."""
        decoys = []
        
        for decoy_idx in range(5):
            coords = np.zeros((n_residues, 3))
            
            for i in range(n_residues):
                coords[i, 0] = i * 3.4  # 3.4Å spacing
                coords[i, 1] = np.sin(i * 0.1 + decoy_idx * 0.2) * 0.5
                coords[i, 2] = np.cos(i * 0.1 + decoy_idx * 0.2) * 0.5
            
            decoys.append(coords)
        
        return np.stack(decoys).reshape(-1, 3)
    
    def create_submission(self, submission_data: List[Dict]) -> Dict:
        """Create competition submission format."""
        import pandas as pd
        
        # Create DataFrame
        df = pd.DataFrame(submission_data)
        
        # Save to CSV
        submission_file = self.output_dir / "submission.csv"
        df.to_csv(submission_file, index=False)
        
        # Create metadata file
        metadata = {
            'total_coordinates': len(submission_data),
            'format': 'RNA_3D_competition',
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_file = self.output_dir / "submission_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'csv_file': str(submission_file),
            'metadata_file': str(metadata_file),
            'total_coordinates': len(submission_data)
        }
    
    def generate_competition_report(self, results: List[Dict]) -> Dict:
        """Generate comprehensive competition report."""
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        if successful > 0:
            avg_time = np.mean([r['time_used'] for r in results if r['success']])
            avg_complexity = np.mean([r['complexity_score'] for r in results if r['success']])
            avg_budget = np.mean([r['budget_allocated'] for r in results if r['success']])
        else:
            avg_time = avg_complexity = avg_budget = 0
        
        # Complexity distribution
        complexities = [r['complexity_score'] for r in results if r['success']]
        complexity_bins = [(0, 0.3), (0.3, 0.6), (0.6, 1.0)]
        complexity_dist = {}
        
        for low, high in complexity_bins:
            count = sum(1 for c in complexities if low <= c < high)
            complexity_dist[f"{low}-{high}"] = count
        
        # Memory usage
        memory = memory_usage()
        
        report = {
            'summary': {
                'total_sequences': len(results),
                'successful': successful,
                'failed': failed,
                'success_rate': successful / len(results) if results else 0,
                'avg_time_per_sequence': avg_time,
                'avg_complexity': avg_complexity,
                'avg_budget': avg_budget
            },
            'complexity_distribution': complexity_dist,
            'memory_usage': memory,
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save report
        report_file = self.output_dir / "competition_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


def main():
    """Main competition deployment function."""
    parser = argparse.ArgumentParser(description="Phase 5: Competition Deployment")
    parser.add_argument("--model-path", required=True,
                       help="Path to trained model")
    parser.add_argument("--cache-dir", required=True,
                       help="Directory for cached artifacts")
    parser.add_argument("--test-file", required=True,
                       help="File with test sequences")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for submission")
    parser.add_argument("--precompute", action="store_true",
                       help="Precompute embeddings for all sequences")
    parser.add_argument("--device", default="cuda",
                       help="Device for computation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize pipeline
    pipeline = CompetitionPipeline(args.model_path, args.cache_dir, args.output_dir)
    
    try:
        # Setup pipeline
        pipeline.setup_pipeline()
        
        if args.precompute:
            # Load all sequences for precomputation
            # This would typically be a large dataset
            sequences = ["AUGC" * 50] * 1000  # Example
            pipeline.precompute_embeddings(sequences)
        
        # Load test sequences
        test_sequences = []
        with open(args.test_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('>'):
                    test_sequences.append(line.upper())
        
        # Process competition batch
        results = pipeline.process_competition_batch(test_sequences)
        
        print("✅ Phase 5 completed successfully!")
        print(f"   Processed {len(test_sequences)} sequences")
        print(f"   Success rate: {results['report']['summary']['success_rate']:.2%}")
        print(f"   Submission saved to: {results['submission']['csv_file']}")
        
    except Exception as e:
        print(f"❌ Phase 5 failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
