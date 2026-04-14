"""Evaluation command for RNA 3D folding pipeline."""

import argparse
import sys
import os
import json
from pathlib import Path
import numpy as np
import torch

from ..pipeline import RNAFoldingPipeline, PipelineConfig
from ..evaluation import StructureEvaluator
from ..data import RNADatasetLoader
from ..logging_config import setup_logger
from ..config import get_config


def evaluate_command():
    """Command-line interface for model evaluation."""
    
    parser = argparse.ArgumentParser(
        description="Evaluate RNA 3D structure predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input arguments
    parser.add_argument(
        "--predictions", "-p",
        type=Path,
        required=True,
        help="Directory containing prediction files"
    )
    
    parser.add_argument(
        "--reference", "-r",
        type=Path,
        help="Directory containing reference structures"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("evaluation_results"),
        help="Output directory for evaluation results"
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
    
    # Evaluation arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for evaluation"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    
    # Output arguments
    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "csv"],
        help="Output format"
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
    
    # Security validation for predictions directory
    if not args.predictions.exists():
        parser.error(f"Predictions directory does not exist: {args.predictions}")
    
    if not args.predictions.is_dir():
        parser.error(f"Predictions path is not a directory: {args.predictions}")
    
    # Check for suspicious paths
    predictions_str = str(args.predictions.resolve())
    suspicious_patterns = ['..', '\\\\', '//', '\0', '|', '<', '>', '"', '*', '?']
    for pattern in suspicious_patterns:
        if pattern in predictions_str:
            parser.error(f"Suspicious path pattern detected in predictions directory: {pattern}")
    
    # Check directory permissions
    if not os.access(args.predictions, os.R_OK):
        parser.error(f"No read permissions for predictions directory: {args.predictions}")
    
    # Validate reference directory if provided
    if args.reference:
        if not args.reference.exists():
            parser.error(f"Reference directory does not exist: {args.reference}")
        
        if not args.reference.is_dir():
            parser.error(f"Reference path is not a directory: {args.reference}")
        
        reference_str = str(args.reference.resolve())
        for pattern in suspicious_patterns:
            if pattern in reference_str:
                parser.error(f"Suspicious path pattern detected in reference directory: {pattern}")
        
        if not os.access(args.reference, os.R_OK):
            parser.error(f"No read permissions for reference directory: {args.reference}")
    
    # Setup logging
    if not args.quiet:
        logger = setup_logger("rna_evaluate", args.log_file, args.log_level)
        logger.info("Starting model evaluation")
    
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
        max_sequence_length=config.max_seq_len,
        mixed_precision=True
    )
    
    pipeline = RNAFoldingPipeline(pipeline_config)
    
    # Load model
    if args.model_path and args.model_path.exists():
        pipeline.load_model(str(args.model_path))
        if not args.quiet:
            logger.info(f"Loaded model from {args.model_path}")
    
    # Initialize evaluator
    evaluator = StructureEvaluator()
    
    # Load predictions
    predictions = []
    for pred_file in args.predictions.glob("*.npy"):
        try:
            coords = np.load(pred_file)
            predictions.append({
                "file": pred_file.name,
                "coordinates": coords
            })
        except Exception as e:
            if not args.quiet:
                logger.warning(f"Failed to load {pred_file}: {e}")
    
    # Load reference structures if provided
    reference_structures = []
    if args.reference:
        data_loader = RNADatasetLoader(
            args.reference,
            cache_dir=Path("cache"),
            max_seq_len=config.max_seq_len
        )
        reference_structures = data_loader.load_all_structures()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Evaluate predictions
    results = []
    
    for i, pred_data in enumerate(predictions):
        if not args.quiet:
            logger.info(f"Evaluating prediction {i+1}/{len(predictions)}: {pred_data['file']}")
        
        try:
            pred_coords = pred_data["coordinates"]
            
            # If reference structures are available, compute metrics
            if reference_structures:
                # Find matching reference structure
                ref_coords = None
                for ref_struct in reference_structures:
                    if ref_struct.sequence == pred_coords.shape[0]:  # Simplified matching
                        ref_coords = ref_struct.coordinates
                        break
                
                if ref_coords is not None:
                    # Compute evaluation metrics
                    metrics = evaluator.evaluate_single_prediction(
                        pred_coords, ref_coords
                    )
                    
                    results.append({
                        "file": pred_data["file"],
                        **metrics
                    })
                else:
                    # No matching reference found
                    results.append({
                        "file": pred_data["file"],
                        "error": "No matching reference structure found",
                        "success": False
                    })
            else:
                # No reference structures provided
                results.append({
                    "file": pred_data["file"],
                    "error": "No reference structures provided",
                    "success": False
                })
                
        except Exception as e:
            if not args.quiet:
                logger.error(f"Failed to evaluate {pred_data['file']}: {e}")
            results.append({
                "file": pred_data["file"],
                "error": str(e),
                "success": False
            })
    
    # Save results
    if args.format == "json":
        output_file = args.output / "evaluation_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if not args.quiet:
            logger.info(f"Saved evaluation results to {output_file}")
    
    elif args.format == "csv":
        output_file = args.output / "evaluation_results.csv"
        with open(output_file, 'w') as f:
            f.write("file,tm_score,rmsd,gdt_ts,gdt_ha,lddt,molprobity_clashscore\n")
            for result in results:
                if result.get("success", False):
                    f.write(f"{result['file']},{result.get('tm_score', 'N/A')},{result.get('rmsd', 'N/A')},")
                else:
                    f.write(f"{result['file']},{result.get('tm_score', 'N/A')},{result.get('rmsd', 'N/A')},")
        
        if not args.quiet:
            logger.info(f"Saved evaluation results to {output_file}")
    
    # Summary
    successful = sum(1 for r in results if r.get("success", False))
    if not args.quiet:
        logger.info(f"Evaluation complete: {successful}/{len(predictions)} successful")
    
    return results


if __name__ == "__main__":
    evaluate_command()