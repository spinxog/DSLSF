#!/usr/bin/env python3
"""
Basic structure prediction example.

This example demonstrates how to use the RNA 3D folding pipeline
to predict the structure of a single RNA sequence.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from rna_model import RNAFoldingPipeline, PipelineConfig
from rna_model.logging_config import setup_logger
from pathlib import Path


def main():
    """Run basic structure prediction example."""
    
    # Setup logging
    logger = setup_logger("basic_prediction", Path("logs"))
    logger.info("Starting basic structure prediction example")
    
    # Initialize pipeline
    config = PipelineConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_sequence_length=512,
        mixed_precision=True
    )
    
    pipeline = RNAFoldingPipeline(config)
    logger.info("Pipeline initialized successfully")
    
    # Example sequences
    sequences = [
        "GGGAAAUCC",                    # 9 nt
        "GCCUUGGCAAC",                # 10 nt  
        "AUGCUAAUCGAU",              # 12 nt
        "CGGAUCUCCGAGUCC",          # 14 nt
        "AAUCCGGAAUCCGGAAUCCGG",  # 20 nt
    ]
    
    # Predict structures
    results = []
    
    for i, sequence in enumerate(sequences):
        logger.info(f"Processing sequence {i+1}/{len(sequences)}: {sequence}")
        
        try:
            result = pipeline.predict_single_sequence(sequence)
            results.append(result)
            
            logger.info(f"Successfully predicted structure for {sequence}")
            logger.info(f"Generated {result['n_decoys']} decoys for {result['n_residues']} residues")
            logger.info(f"Coordinates shape: {result['coordinates'].shape}")
            
        except Exception as e:
            logger.error(f"Failed to predict structure for {sequence}: {e}")
            results.append({"sequence": sequence, "error": str(e), "success": False})
    
    # Summary
    successful = sum(1 for r in results if r.get("success", True))
    logger.info(f"Prediction complete: {successful}/{len(sequences)} successful")
    
    # Save results
    import numpy as np
    
    all_coords = []
    for result in results:
        if result.get("success", True) and "coordinates" in result:
            all_coords.append(result["coordinates"])
    
    if all_coords:
        submission_coords = np.concatenate(all_coords, axis=0)
        np.save("results/basic_predictions.npy", submission_coords)
        logger.info(f"Saved predictions to results/basic_predictions.npy")
        logger.info(f"Submission coordinates shape: {submission_coords.shape}")
    
    return results


if __name__ == "__main__":
    main()
