#!/usr/bin/env python3
"""
Command-line interface for structure prediction.

This script provides a command-line interface for predicting RNA 3D structures
using the RNA 3D folding pipeline.
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from rna_model import RNAFoldingPipeline, PipelineConfig
from rna_model.logging_config import setup_logger
from rna_model.config import get_config


def predict_command():
    """Command-line interface for structure prediction."""
    
    parser = argparse.ArgumentParser(
        description="Predict RNA 3D structures using the RNA folding pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input arguments
    parser.add_argument(
        "--sequence", "-s",
        type=str,
        help="RNA sequence to predict structure for"
    )
    
    parser.add_argument(
        "--sequence-file", "-f",
        type=Path,
        help="File containing RNA sequences (one per line)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("predictions"),
        help="Output directory for predictions"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-path", "-m",
        type=Path,
        default=Path("checkpoints/latest.pth"),
        help="Path to trained model weights"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to configuration file"
    )
    
    # Pipeline arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing"
    )
    
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    # Output arguments
    parser.add_argument(
        "--format",
        type=str,
        default="npy",
        choices=["npy", "json", "pdb"],
        help="Output format"
    )
    
    parser.add_argument(
        "--return-all-decoys",
        action="store_true",
        help="Return all decoys instead of just top 5"
    )
    
    # Logging arguments
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to log file"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.sequence and not args.sequence_file:
        parser.error("Either --sequence or --sequence-file must be provided")
    
    # Setup logging
    if not args.quiet:
        logger = setup_logger("rna_predict", args.log_file, args.log_level)
        logger.info("Starting RNA structure prediction")
    
    # Load configuration
    if args.config:
        config = get_config(args.config)
    else:
        config = get_config()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Initialize pipeline
    pipeline_config = PipelineConfig(
        device=device,
        max_sequence_length=args.max_seq_len,
        mixed_precision=True
    )
    
    pipeline = RNAFoldingPipeline(pipeline_config)
    
    # Load model if provided
    if args.model_path and args.model_path.exists():
        pipeline.load_model(str(args.model_path))
        if not args.quiet:
            logger.info(f"Loaded model from {args.model_path}")
    
    # Get sequences
    if args.sequence:
        sequences = [args.sequence]
    else:
        sequences = []
        with open(args.sequence_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('>'):
                    sequences.append(line)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Process sequences
    results = []
    
    for i, sequence in enumerate(sequences):
        if not args.quiet:
            logger.info(f"Processing sequence {i+1}/{len(sequences)}: {sequence}")
        
        try:
            result = pipeline.predict_single_sequence(
                sequence,
                return_all_decoys=args.return_all_decoys
            )
            results.append(result)
            
            if not args.quiet:
                logger.info(f"Successfully predicted structure for {sequence}")
                logger.info(f"Generated {result['n_decoys']} decoys for {result['n_residues']} residues")
            
        except Exception as e:
            if not args.quiet:
                logger.error(f"Failed to predict structure for {sequence}: {e}")
            results.append({"sequence": sequence, "error": str(e), "success": False})
    
    # Save results
    if args.format == "npy":
        # Save as numpy array
        all_coords = []
        for result in results:
            if result.get("success", True) and "coordinates" in result:
                all_coords.append(result["coordinates"])
        
        if all_coords:
            submission_coords = np.concatenate(all_coords, axis=0)
            output_file = args.output / "predictions.npy"
            np.save(output_file, submission_coords)
            
            if not args.quiet:
                logger.info(f"Saved predictions to {output_file}")
                logger.info(f"Shape: {submission_coords.shape}")
    
    elif args.format == "json":
        # Save as JSON
        output_file = args.output / "predictions.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if not args.quiet:
            logger.info(f"Saved predictions to {output_file}")
    
    elif args.format == "pdb":
        # Save as PDB (simplified)
        output_file = args.output / "predictions.pdb"
        with open(output_file, 'w') as f:
            for i, result in enumerate(results):
                if result.get("success", True) and "coordinates" in result:
                    coords = result["coordinates"]
                    sequence = result["sequence"]
                    
                    # Simple PDB format
                    for decoy_idx in range(min(5, result['n_decoys'])):
                        start_idx = decoy_idx * result['n_residues']
                        end_idx = start_idx + result['n_residues']
                        
                        for res_idx, (start, end) in enumerate(zip(range(start_idx, end_idx), range(start_idx + 1, end_idx + 1))):
                            coord = coords[start]
                            f.write(f"ATOM  CA  {res_idx+1:4d}     {coord[0]:8.3f}  {coord[1]:8.3f}  {coord[2]:8.3f}   1.00  0.00  0.00\n")
                        
                        f.write("TER\n")
        
        if not args.quiet:
            logger.info(f"Saved predictions to {output_file}")
    
    # Summary
    successful = sum(1 for r in results if r.get("success", True))
    if not args.quiet:
        logger.info(f"Prediction complete: {successful}/{len(sequences)} successful")
    
    return results


if __name__ == "__main__":
    predict_command()
