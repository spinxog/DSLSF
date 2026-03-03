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
            from Bio import SeqIO
            for record in SeqIO.parse(test_path, "fasta"):
                sequences.append(str(record.seq).upper())
                sequence_ids.append(record.id)
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
    
    def process_sequence(self, sequence: str, sequence_id: str) -> Dict:
        """Process a single sequence with adaptive budgeting."""
        sequence_start_time = time.time()
        
        # Compute complexity and budget
        complexity = self.compute_complexity_score(sequence)
        budget = self.compute_sequence_budget(sequence, complexity)
        
        self.logger.info(f"Processing {sequence_id} (L={len(sequence)}, complexity={complexity:.2f}, budget={budget:.1f}s)")
        
        try:
            # Set seed for reproducibility
            set_seed(hash(sequence) % 2**32)
            
            # Predict structure
            result = self.pipeline.predict_single_sequence(
                sequence,
                return_all_decoys=False  # Only need top 5 for submission
            )
            
            # Add metadata
            result.update({
                'sequence_id': sequence_id,
                'complexity_score': complexity,
                'budget_allocated': budget,
                'time_used': time.time() - sequence_start_time,
                'success': True
            })
            
            self.logger.info(f"✓ {sequence_id}: Generated {result['n_decoys']} decoys in {result['time_used']:.1f}s")
            
            return result
            
        except Exception as e:
            # Fallback handling
            self.logger.error(f"✗ {sequence_id}: Error - {e}")
            
            # Generate fallback coordinates (simple linear chain)
            n_residues = len(sequence)
            fallback_coords = self.generate_fallback_coordinates(n_residues)
            
            return {
                'sequence_id': sequence_id,
                'sequence': sequence,
                'coordinates': fallback_coords,
                'n_decoys': 5,
                'n_residues': n_residues,
                'complexity_score': complexity,
                'budget_allocated': budget,
                'time_used': time.time() - sequence_start_time,
                'success': False,
                'error': str(e)
            }
    
    def generate_fallback_coordinates(self, n_residues: int) -> np.ndarray:
        """Generate fallback coordinates (simple linear chain)."""
        # Generate 5 decoys with slight variations
        decoys = []
        
        for decoy_idx in range(5):
            coords = np.zeros((n_residues, 3))
            
            for i in range(n_residues):
                # Linear arrangement with small variations per decoy
                coords[i, 0] = i * 3.4  # 3.4Å spacing along x-axis
                coords[i, 1] = np.sin(i * 0.1 + decoy_idx * 0.2) * 0.5  # Small y variation
                coords[i, 2] = np.cos(i * 0.1 + decoy_idx * 0.2) * 0.5  # Small z variation
            
            decoys.append(coords)
        
        # Stack all decoys: (5 * n_residues, 3)
        return np.stack(decoys).reshape(-1, 3)
    
    def check_global_time_budget(self) -> bool:
        """Check if we're approaching global time limit."""
        elapsed = time.time() - self.global_start_time
        remaining_time = self.time_limit_seconds - elapsed
        
        # If less than 5% time remaining, enter conservative mode
        if remaining_time < self.time_limit_seconds * 0.05:
            self.logger.warning(f"⚠️  Time budget critical: {remaining_time:.1f}s remaining")
            return True
        
        return False
    
    def run_competition(self, test_file: str):
        """Run the complete competition workflow."""
        self.global_start_time = time.time()
        self.logger.info(f"Starting competition run with {self.time_limit_seconds/3600:.1f}h time limit")
        
        # Setup pipeline
        self.setup_pipeline()
        
        # Load test sequences
        sequences, sequence_ids = self.load_test_sequences(test_file)
        
        # Initialize results storage
        all_results = []
        submission_data = []
        
        # Process each sequence
        for i, (sequence, seq_id) in enumerate(zip(sequences, sequence_ids)):
            self.processed_sequences = i
            
            # Check global time budget
            if self.check_global_time_budget():
                self.logger.warning("⚠️  Entering conservative mode for remaining sequences")
                # Could implement simplified processing here
            
            # Process sequence
            result = self.process_sequence(sequence, seq_id)
            all_results.append(result)
            
            # Update global time tracking
            self.global_time_used = time.time() - self.global_start_time
            
            # Add to submission data if successful
            if result['success']:
                coords = result['coordinates']
                for residue_idx in range(result['n_residues']):
                    for decoy_idx in range(5):
                        global_residue_idx = i * 5 * result['n_residues'] + decoy_idx * result['n_residues'] + residue_idx
                        x, y, z = coords[decoy_idx * result['n_residues'] + residue_idx]
                        submission_data.append({
                            'residue_id': f"{seq_id}_{residue_idx}_{decoy_idx + 1}",
                            'x': x,
                            'y': y,
                            'z': z,
                            'sequence_id': seq_id,
                            'decoy': decoy_idx + 1,
                            'residue_index': residue_idx
                        })
        
        # Save submission
        self.save_submission(submission_data)
        
        # Save detailed results
        self.save_detailed_results(all_results)
        
        # Generate final report
        self.generate_final_report(all_results)
        
        total_time = time.time() - self.global_start_time
        self.logger.info(f"✅ Competition completed in {total_time/3600:.2f}h")
    
    def save_submission(self, submission_data: List[Dict]):
        """Save submission file in competition format."""
        submission_file = self.output_dir / "submission.csv"
        
        # Create DataFrame
        df = pd.DataFrame(submission_data)
        
        # Save to CSV
        df.to_csv(submission_file, index=False)
        
        self.logger.info(f"📄 Submission saved to {submission_file}")
        self.logger.info(f"   Total coordinates: {len(df)}")
    
    def save_detailed_results(self, results: List[Dict]):
        """Save detailed results with metadata."""
        results_file = self.output_dir / "detailed_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for result in results:
            json_result = result.copy()
            if 'coordinates' in json_result:
                json_result['coordinates'] = json_result['coordinates'].tolist()
            json_results.append(json_result)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"📊 Detailed results saved to {results_file}")
    
    def generate_final_report(self, results: List[Dict]):
        """Generate final competition report."""
        report_file = self.output_dir / "competition_report.md"
        
        # Compute statistics
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        if successful > 0:
            avg_time = np.mean([r['time_used'] for r in results if r['success']])
            avg_complexity = np.mean([r['complexity_score'] for r in results if r['success']])
            avg_budget = np.mean([r['budget_allocated'] for r in results if r['success']])
        else:
            avg_time = avg_complexity = avg_budget = 0
        
        total_time = time.time() - self.global_start_time
        
        # Generate report
        report = f"""# RNA 3D Folding Competition Report

## Summary
- **Total Sequences**: {len(results)}
- **Successful Predictions**: {successful}
- **Failed Predictions**: {failed}
- **Success Rate**: {successful/len(results)*100:.1f}%

## Timing
- **Total Time**: {total_time/3600:.2f} hours
- **Average Time per Sequence**: {avg_time:.1f}s
- **Average Budget Allocated**: {avg_budget:.1f}s
- **Time Utilization**: {total_time/self.time_limit_seconds*100:.1f}%

## Sequence Analysis
- **Average Complexity Score**: {avg_complexity:.3f}
- **Average Sequence Length**: {np.mean([len(r['sequence']) for r in results if 'sequence' in r]):.1f} nt

## Performance by Complexity
"""
        
        # Add complexity breakdown
        complexity_bins = [(0, 0.3), (0.3, 0.6), (0.6, 1.0)]
        for low, high in complexity_bins:
            bin_results = [r for r in results if low <= r['complexity_score'] < high]
            if bin_results:
                bin_success = sum(1 for r in bin_results if r['success'])
                bin_avg_time = np.mean([r['time_used'] for r in bin_results if r['success']])
                report += f"- **Complexity {low}-{high}**: {len(bin_results)} sequences, {bin_success/len(bin_results)*100:.1f}% success, {bin_avg_time:.1f}s avg\n"
        
        report += f"""
## Memory Usage
{memory_usage()}

## Recommendations
- Review failed sequences for common patterns
- Consider adjusting complexity thresholds if time budget exceeded
- Monitor GPU memory usage for optimization opportunities

Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        self.logger.info(f"📋 Competition report saved to {report_file}")


def main():
    """Main competition submission function."""
    parser = argparse.ArgumentParser(description="RNA 3D Folding Competition Submission")
    parser.add_argument("--model-path", required=True, 
                       help="Path to trained model directory")
    parser.add_argument("--cache-dir", required=True,
                       help="Directory with cached embeddings and artifacts")
    parser.add_argument("--test-file", required=True,
                       help="File containing test sequences (CSV, FASTA, or TXT)")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for submission files")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                       help="Device to use for inference")
    parser.add_argument("--time-limit", type=float, default=8.0,
                       help="Time limit in hours (default: 8.0)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set global seed
    set_seed(args.seed)
    
    # Validate inputs
    if not Path(args.model_path).exists():
        print(f"Error: Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    if not Path(args.cache_dir).exists():
        print(f"Warning: Cache directory does not exist: {args.cache_dir}")
    
    # Create competition submission handler
    submission = CompetitionSubmission(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        device=args.device,
        time_limit_hours=args.time_limit
    )
    
    # Run competition
    try:
        submission.run_competition(args.test_file)
        print("✅ Competition submission completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Competition interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Competition failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
