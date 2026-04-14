#!/usr/bin/env python3
"""
Competition Submission Script for RNA 3D Folding Pipeline

This script handles the complete competition workflow:
1. Load pre-trained models and cached artifacts
2. Process test sequences with adaptive budgeting
3. Generate 5 decoys per sequence
4. Format submission according to competition requirements
5. Perform sanity checks and validation

Designed for HPC training and notebook deployment.
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import pandas as pd
import gc
from contextlib import contextmanager

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from rna_model import RNAFoldingPipeline, PipelineConfig
from rna_model.utils import set_seed, memory_usage, clear_cache


class CompetitionSubmission:
    """Competition submission handler with adaptive budgeting and monitoring."""
    
    def __init__(self, 
                 model_path: str,
                 cache_dir: str,
                 output_dir: str,
                 device: str = "cuda",
                 time_limit_hours: float = 8.0):
        """
        Initialize competition submission.
        
        Args:
            model_path: Path to trained model weights
            cache_dir: Directory with cached embeddings and artifacts
            output_dir: Directory for submission files
            device: Device to use for inference
            time_limit_hours: Total time limit for competition
        """
        self.model_path = Path(model_path)
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.time_limit_seconds = time_limit_hours * 3600
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize pipeline (will be loaded in setup())
        self.pipeline = None
        self.start_time = None
        
        # Monitoring
        self.global_start_time = None
        self.processed_sequences = 0
        self.total_sequences = 0
        
        # Budget tracking
        self.per_sequence_budgets = {}
        self.global_time_used = 0.0
        
    def setup_logging(self):
        """Setup logging for competition run."""
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
        """Load pipeline with competition optimizations."""
        self.logger.info("Setting up competition pipeline...")
        
        # Load configuration
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = PipelineConfig(**config_dict)
        else:
            # Default competition config
            config = PipelineConfig(
                device=str(self.device),
                mixed_precision=True,
                max_sequence_length=512,
                compile_model=True
            )
        
        # Initialize pipeline
        self.pipeline = RNAFoldingPipeline(config)
        
        # Load trained weights
        weights_path = self.model_path / "competition_model.pth"
        if weights_path.exists():
            self.pipeline.load_model(str(weights_path))
            self.logger.info(f"Loaded model weights from {weights_path}")
        else:
            self.logger.warning(f"Model weights not found at {weights_path}, using random weights")
        
        # Enable competition mode
        self.pipeline.enable_competition_mode()
        self.logger.info("Competition mode enabled")
        
        # Verify cache directory
        if not self.cache_dir.exists():
            self.logger.warning(f"Cache directory {self.cache_dir} not found")
    
    def load_test_sequences(self, test_file: str) -> Tuple[List[str], List[str]]:
        """Load test sequences from file."""
        self.logger.info(f"Loading test sequences from {test_file}")
        
        test_path = Path(test_file)
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        sequences = []
        sequence_ids = []
        
        if test_path.suffix.lower() == '.csv':
            # Load from CSV
            df = pd.read_csv(test_path)
            if 'sequence' in df.columns:
                sequences = df['sequence'].tolist()
                sequence_ids = df.get('id', range(len(sequences))).tolist()
            elif 'seq' in df.columns:
                sequences = df['seq'].tolist()
                sequence_ids = df.get('id', range(len(sequences))).tolist()
            else:
                raise ValueError("CSV must have 'sequence' or 'seq' column")
                
        elif test_path.suffix.lower() in ['.fasta', '.fa', '.fna']:
            # Load from FASTA
            try:
                from Bio import SeqIO
                for record in SeqIO.parse(test_path, "fasta"):
                    sequences.append(str(record.seq).upper())
                    sequence_ids.append(record.id)
            except ImportError:
                self.logger.warning("BioPython not installed, falling back to simple FASTA parsing")
                # Simple FASTA parsing
                with open(test_path, 'r') as f:
                    current_seq = ""
                    current_id = ""
                    for line in f:
                        line = line.strip()
                        if line.startswith('>'):
                            if current_seq:
                                sequences.append(current_seq.upper())
                                sequence_ids.append(current_id or f"seq_{len(sequences)}")
                            current_id = line[1:]
                            current_seq = ""
                        else:
                            current_seq += line
                    if current_seq:
                        sequences.append(current_seq.upper())
                        sequence_ids.append(current_id or f"seq_{len(sequences)}")
        else:
            # Assume plain text file
            with open(test_path, 'r') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line and not line.startswith('>'):
                        sequences.append(line.upper())
                        sequence_ids.append(f"seq_{i}")
        
        self.total_sequences = len(sequences)
        self.logger.info(f"Loaded {self.total_sequences} test sequences")
        
        return sequences, sequence_ids
    
    def compute_sequence_budget(self, sequence: str, complexity_score: float) -> float:
        """Compute adaptive time budget for a sequence."""
        # Base budget proportional to sequence length
        base_budget = 60.0 + len(sequence) * 0.5  # 60s base + 0.5s per nt
        
        # Adjust for complexity
        complexity_multiplier = 1.0 + complexity_score * 2.0  # Up to 3x budget for complex sequences
        
        # Apply global budget constraints
        remaining_global_time = self.time_limit_seconds - self.global_time_used
        remaining_sequences = self.total_sequences - self.processed_sequences
        
        if remaining_sequences > 0:
            avg_remaining_budget = remaining_global_time / remaining_sequences
            max_reasonable_budget = min(base_budget * complexity_multiplier, avg_remaining_budget * 2.0)
        else:
            max_reasonable_budget = base_budget * complexity_multiplier
        
        # Apply per-sequence limits
        final_budget = np.clip(max_reasonable_budget, 30.0, 900.0)  # Min 30s, Max 15min
        
        return final_budget
    
    def compute_complexity_score(self, sequence: str) -> float:
        """Compute complexity score for adaptive budgeting."""
        # Simple heuristic based on sequence composition and length
        length = len(sequence)
        
        # Count potential junction-forming patterns
        junction_patterns = ['GAAA', 'CUUG', 'GNRA', 'UNCG']  # Common tetraloops
        junction_count = sum(1 for pattern in junction_patterns if pattern in sequence)
        
        # GC content (higher GC = more stable, potentially more complex)
        gc_content = (sequence.count('G') + sequence.count('C')) / length
        
        # Predicted secondary structure complexity (simplified)
        # Estimate number of stems and loops
        predicted_stems = min(length // 10, 8)  # Rough estimate
        predicted_loops = predicted_stems - 1
        
        # Combine into complexity score
        complexity = (
            0.3 * (length / 500.0) +  # Length component (normalized to 500nt)
            0.2 * junction_count / 5.0 +  # Junction component
            0.2 * gc_content +  # GC content component
            0.3 * min(predicted_loops / 5.0, 1.0)  # Loop component
        )
        
        return np.clip(complexity, 0.0, 1.0)
    
    @contextmanager
    def _memory_context(self):
        """Context manager for memory management during sequence processing."""
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def process_sequence(self, sequence: str, sequence_id: str) -> Dict:
        """Process a single sequence with adaptive budgeting and memory management."""
        sequence_start_time = time.time()
        
        # Validate input
        if not sequence or len(sequence) == 0:
            raise ValueError(f"Empty sequence provided for {sequence_id}")
        if len(sequence) > 512:  # Max sequence length
            raise ValueError(f"Sequence too long: {len(sequence)} > 512")
        
        # Compute complexity and budget
        complexity = self.compute_complexity_score(sequence)
        budget = self.compute_sequence_budget(sequence, complexity)
        
        self.logger.info(f"Processing {sequence_id} (L={len(sequence)}, complexity={complexity:.2f}, budget={budget:.1f}s)")
        
        with self._memory_context():
            try:
                # Set seed for reproducibility
                set_seed(hash(sequence) % 2**32)
                
                # Predict structure
                result = self.pipeline.predict_single_sequence(
                    sequence,
                    return_all_decoys=False  # Only need top 5 for submission
                )
                
                # Validate result
                if 'coordinates' not in result:
                    raise ValueError("No coordinates in pipeline result")
                
                # Add metadata
                result.update({
                    'sequence_id': sequence_id,
                    'complexity_score': complexity,
                    'budget_allocated': budget,
                    'time_used': time.time() - sequence_start_time,
                    'success': True
                })
                
                self.logger.info(f"Success {sequence_id}: Generated {result['n_decoys']} decoys in {result['time_used']:.1f}s")
                
                return result
                
            except Exception as e:
                # Fail fast - no fallback coordinates in ML
                self.logger.error(f"Failed {sequence_id}: Error - {e}")
                
                return {
                    'sequence_id': sequence_id,
                    'sequence': sequence,
                    'error': str(e),
                    'success': False,
                    'complexity_score': complexity,
                    'budget_allocated': budget,
                    'time_used': time.time() - sequence_start_time
                }
    
    def create_submission_format(self, predictions: List[Dict]) -> np.ndarray:
        """Create submission format from predictions."""
        all_coords = []
        
        for pred in predictions:
            if pred.get('success', False) and 'coordinates' in pred:
                coords = pred["coordinates"]  # Should be (n_decoys * n_residues, 3)
                all_coords.append(coords)
            else:
                raise ValueError(f"Cannot create submission for failed prediction: {pred.get('sequence_id', 'unknown')}")
        
        if not all_coords:
            raise ValueError("No successful predictions to create submission")
        
        return np.concatenate(all_coords, axis=0)
    
    def create_summary(self, results: List[Dict]) -> Dict:
        """Create summary statistics."""
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        summary = {
            'total_sequences': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(results) if results else 0.0,
            'total_time': time.time() - self.global_start_time if self.global_start_time else 0.0,
            'avg_time_per_sequence': np.mean([r['time_used'] for r in results]) if results else 0.0,
            'failed_sequences': [r['sequence_id'] for r in failed]
        }
        
        return summary
    
    def run_competition(self, test_file: str, output_file: str):
        """Run complete competition workflow with robust error handling."""
        try:
            self.global_start_time = time.time()
            
            # Load test sequences
            sequences, sequence_ids = self.load_test_sequences(test_file)
            
            if not sequences:
                raise ValueError("No sequences loaded from test file")
            
            # Process all sequences
            results = []
            failed_sequences = []
            
            for i, (sequence, seq_id) in enumerate(zip(sequences, sequence_ids)):
                try:
                    result = self.process_sequence(sequence, seq_id)
                    results.append(result)
                    
                    if not result.get('success', False):
                        failed_sequences.append(seq_id)
                    
                except Exception as e:
                    self.logger.error(f"Critical error processing {seq_id}: {e}")
                    failed_sequences.append(seq_id)
                    continue
                
                # Update global time tracking
                self.global_time_used = time.time() - self.global_start_time
                self.processed_sequences = i + 1
                
                # Check if we're approaching time limit
                if self.global_time_used > self.time_limit_seconds * 0.9:
                    self.logger.warning(f"Approaching time limit: {self.global_time_used/3600:.1f}h / {self.time_limit_seconds/3600:.1f}h")
                    break
                
                # Log progress
                if (i + 1) % 10 == 0:
                    avg_time = self.global_time_used / (i + 1)
                    estimated_total = avg_time * len(sequences)
                    self.logger.info(f"Progress: {i+1}/{len(sequences)}, Avg time: {avg_time:.1f}s, Est total: {estimated_total/3600:.1f}h")
            
            if not results:
                raise RuntimeError("No sequences processed successfully")
            
            # Create submission
            submission_coords = self.create_submission_format(results)
            
            # Validate submission format
            if submission_coords.size == 0:
                raise ValueError("Empty submission coordinates")
            
            # Save submission
            np.save(output_file, submission_coords)
            
            # Create summary
            summary = self.create_summary(results)
            summary['failed_sequences'] = failed_sequences
            
            with open(output_file.replace('.npy', '_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            
            total_time = time.time() - self.global_start_time
            self.logger.info(f"Competition completed in {total_time/3600:.2f} hours")
            self.logger.info(f"Submission saved to {output_file}")
            self.logger.info(f"Success rate: {summary['success_rate']:.2%} ({summary['successful']}/{summary['total_sequences']})")
            
            if failed_sequences:
                self.logger.warning(f"Failed sequences: {failed_sequences}")
            
        except Exception as e:
            self.logger.error(f"Competition failed: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RNA 3D Folding Competition Submission")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache directory")
    parser.add_argument("--test_file", type=str, required=True, help="Test sequences file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--time_limit", type=float, default=8.0, help="Time limit in hours")
    
    args = parser.parse_args()
    
    # Initialize competition submission
    competition = CompetitionSubmission(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        device=args.device,
        time_limit_hours=args.time_limit
    )
    
    # Setup pipeline
    competition.setup_pipeline()
    
    # Run competition
    output_file = Path(args.output_dir) / "submission_coordinates.npy"
    competition.run_competition(args.test_file, str(output_file))


if __name__ == "__main__":
    main()
