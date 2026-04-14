#!/usr/bin/env python3
"""
Basic evaluation example.

This example demonstrates how to evaluate the RNA 3D folding pipeline
on test data and compute performance metrics.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from rna_model import RNAFoldingPipeline, PipelineConfig
from rna_model.evaluation import StructureEvaluator
from rna_model.logging_config import setup_logger, PerformanceLogger
from rna_model.utils import compute_tm_score, compute_rmsd


def create_test_data():
    """Create test data for evaluation."""
    sequences = [
        "GGGAAAUCC",                    # 9 nt
        "GCCUUGGCAAC",                # 10 nt  
        "AUGCUAAUCGAU",              # 12 nt
        "CGGAUCUCCGAGUCC",          # 14 nt
    ]
    
    # Generate "true" structures (simplified)
    true_structures = []
    for sequence in sequences:
        n_residues = len(sequence)
        # Create a simple helical structure
        coords = np.zeros((n_residues, 3, 3))
        
        for i in range(n_residues):
            # Simple helix along x-axis
            coords[i, 0, 0] = i * 3.4  # x-coordinate
            coords[i, 1, 0] = 5.0 * np.sin(i * 0.5)  # y-coordinate
            coords[i, 2, 0] = 5.0 * np.cos(i * 0.5)  # z-coordinate
        
        true_structures.append(coords)
    
    return sequences, true_structures


def main():
    """Run basic evaluation example."""
    
    # Setup logging
    logger = setup_logger("basic_evaluation", Path("logs"))
    perf_logger = PerformanceLogger(logger)
    
    logger.info("Starting basic evaluation example")
    perf_logger.start_timer("total_evaluation")
    
    # Initialize pipeline
    config = PipelineConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_sequence_length=512,
        mixed_precision=True
    )
    
    pipeline = RNAFoldingPipeline(config)
    logger.info("Pipeline initialized successfully")
    
    # Create test data
    sequences, true_structures = create_test_data()
    logger.info(f"Created test data: {len(sequences)} sequences")
    
    # Initialize evaluator
    evaluator = StructureEvaluator()
    logger.info("Evaluator initialized")
    
    # Predict structures
    predictions = []
    
    for i, sequence in enumerate(sequences):
        logger.info(f"Predicting structure for sequence {i+1}/{len(sequences)}")
        
        try:
            result = pipeline.predict_single_sequence(sequence)
            predictions.append(result)
            
            logger.info(f"Successfully predicted structure for {sequence}")
            
        except Exception as e:
            logger.error(f"Failed to predict structure for {sequence}: {e}")
            predictions.append({"sequence": sequence, "error": str(e), "success": False})
    
    # Evaluate predictions
    logger.info("Starting evaluation")
    
    metrics = {
        "total_sequences": len(sequences),
        "successful_predictions": 0,
        "failed_predictions": 0,
        "tm_scores": [],
        "rmsd_scores": [],
        "average_tm_score": 0.0,
        "average_rmsd": 0.0
    }
    
    for i, (sequence, true_structure, prediction) in enumerate(zip(sequences, true_structures, predictions)):
        if prediction.get("success", True) and "coordinates" in prediction:
            # Extract coordinates (first decoy)
            pred_coords = prediction["coordinates"][:len(sequence)]  # First decoy, first len(seq) residues
            pred_coords = pred_coords.reshape(-1, 3)  # Flatten to (n_residues, 3)
            
            # Compute metrics
            tm_score = compute_tm_score(pred_coords, true_structure)
            rmsd_score = compute_rmsd(pred_coords, true_structure)
            
            metrics["tm_scores"].append(tm_score)
            metrics["rmsd_scores"].append(rmsd_score)
            metrics["successful_predictions"] += 1
            
            logger.info(f"Sequence {i+1} ({sequence}): TM-score = {tm_score:.4f}, RMSD = {rmsd_score:.4f}")
            
        else:
            metrics["failed_predictions"] += 1
            logger.error(f"Sequence {i+1} ({sequence}): Evaluation failed")
    
    # Compute averages
    if metrics["tm_scores"]:
        metrics["average_tm_score"] = np.mean(metrics["tm_scores"])
        metrics["average_rmsd"] = np.mean(metrics["rmsd_scores"])
    
    # Log results
    logger.info("Evaluation completed")
    logger.info(f"Total sequences: {metrics['total_sequences']}")
    logger.info(f"Successful predictions: {metrics['successful_predictions']}")
    logger.info(f"Failed predictions: {metrics['failed_predictions']}")
    logger.info(f"Average TM-score: {metrics['average_tm_score']:.4f}")
    logger.info(f"Average RMSD: {metrics['average_rmsd']:.4f}")
    
    # Save results
    results = {
        "metrics": metrics,
        "predictions": predictions,
        "sequences": sequences
    }
    
    import json
    with open("results/basic_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Results saved to results/basic_evaluation.json")
    
    perf_logger.end_timer("total_evaluation", 
                           sequences=len(sequences),
                           successful=metrics["successful_predictions"])
    
    return metrics


if __name__ == "__main__":
    main()
