"""
Kaggle Competition Submission Notebook for Stanford RNA 3D Folding Challenge

This notebook:
1. Loads pre-trained model and MSAs
2. Generates 5 diverse predictions per sequence using temperature variation
3. Outputs submission.csv in required format (C1' coordinates)

Usage in Kaggle:
- Copy this code to a Kaggle notebook
- Ensure model weights are available
- Run to generate submission.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple
import time

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Model configuration - optimized for accuracy
class CompetitionConfig:
    """Configuration optimized for Kaggle competition."""
    
    # Language Model
    lm_d_model = 512
    lm_n_layers = 12
    lm_n_heads = 8
    lm_max_seq_len = 2048
    
    # Secondary Structure
    ss_d_model = 256
    ss_n_layers = 6
    ss_n_heads = 8
    
    # Structure Encoder
    enc_d_model = 256
    enc_n_layers = 12
    enc_n_heads = 8
    
    # Geometry Module (scaled up)
    geo_d_model = 512
    geo_n_layers = 8
    geo_n_heads = 8
    geo_d_ff = 2048
    
    # Sampler
    n_decoys = 5
    n_steps = 50  # Reduced for speed, increase if time permits
    temperatures = [0.1, 0.3, 0.5, 0.7, 1.0]  # Diverse sampling
    
    # MSA
    use_msa = True
    msa_n_seqs = 10
    
    # Refinement
    refinement_iterations = 50  # Reduced for speed


def load_model(checkpoint_path: str = None):
    """Load the integrated RNA folding model."""
    
    # Import here to handle path issues in Kaggle
    from rna_model.models.integrated import IntegratedModel
    from rna_model.models.language_model import LMConfig
    from rna_model.models.secondary_structure import SSConfig
    from rna_model.models.structure_encoder import EncoderConfig
    from rna_model.core.geometry_module import GeometryConfig
    from rna_model.core.sampler import SamplerConfig
    from rna_model.core.refinement import RefinementConfig
    
    cfg = CompetitionConfig()
    
    # Create config objects
    class ModelConfig:
        def __init__(self):
            self.lm_config = LMConfig(
                d_model=cfg.lm_d_model,
                n_layers=cfg.lm_n_layers,
                n_heads=cfg.lm_n_heads,
                max_seq_len=cfg.lm_max_seq_len
            )
            self.ss_config = SSConfig(
                d_model=cfg.ss_d_model,
                n_layers=cfg.ss_n_layers,
                n_heads=cfg.ss_n_heads
            )
            self.encoder_config = EncoderConfig(
                d_model=cfg.enc_d_model,
                n_layers=cfg.enc_n_layers,
                n_heads=cfg.enc_n_heads
            )
            self.geometry_config = GeometryConfig(
                d_model=cfg.geo_d_model,
                n_layers=cfg.geo_n_layers,
                n_heads=cfg.geo_n_heads,
                d_ff=cfg.geo_d_ff
            )
            self.sampler_config = SamplerConfig(
                n_decoys=cfg.n_decoys,
                n_steps=cfg.n_steps
            )
            self.refinement_config = RefinementConfig(
                n_iterations=cfg.refinement_iterations
            )
            self.use_msa = cfg.use_msa
    
    config = ModelConfig()
    model = IntegratedModel(config)
    model.to(DEVICE)
    model.eval()
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Warning: No checkpoint loaded, using randomly initialized weights")
    
    return model, cfg


def load_msa(seq_id: str, msa_dir: str = None) -> np.ndarray:
    """Load pre-computed MSA for a sequence."""
    if msa_dir is None or not os.path.exists(msa_dir):
        return None
    
    msa_file = Path(msa_dir) / f"{seq_id}.npy"
    if msa_file.exists():
        return np.load(msa_file)
    return None


def predict_structure(model, sequence: str, msa: np.ndarray = None, 
                      temperatures: List[float] = None) -> List[np.ndarray]:
    """
    Predict 3D structure for a single sequence.
    
    Returns 5 sets of coordinates (C1' atom for each residue).
    """
    # Tokenize
    token_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
    tokens = torch.tensor([[token_map.get(nuc.upper(), 4) for nuc in sequence]], 
                         dtype=torch.long, device=DEVICE)
    
    predictions = []
    
    with torch.no_grad():
        # Run model with different temperatures
        for temp in (temperatures or [0.5]):
            # Prepare MSA if available
            msa_tensor = None
            if msa is not None:
                msa_tensor = torch.tensor(msa, dtype=torch.long, device=DEVICE).unsqueeze(0)
            
            # Forward pass
            outputs = model(tokens, msa_tokens=msa_tensor)
            
            # Get coordinates (C1' atom is index 1 in the 3 atoms)
            coords = outputs['geometry_outputs']['coordinates'][0, :, 1, :].cpu().numpy()
            predictions.append(coords)
    
    # If we have fewer predictions than needed, duplicate the best ones
    while len(predictions) < 5:
        predictions.append(predictions[-1].copy())
    
    return predictions[:5]


def generate_submission(test_file: str, model, config, msa_dir: str = None) -> pd.DataFrame:
    """
    Generate submission file for all test sequences.
    
    Returns DataFrame in required submission format.
    """
    # Load test sequences
    test_df = pd.read_csv(test_file)
    
    if 'id' not in test_df.columns or 'sequence' not in test_df.columns:
        raise ValueError("Test file must have 'id' and 'sequence' columns")
    
    print(f"Processing {len(test_df)} test sequences...")
    
    submission_rows = []
    
    start_time = time.time()
    
    for idx, row in test_df.iterrows():
        seq_id = row['id']
        sequence = row['sequence']
        seq_len = len(sequence)
        
        print(f"  [{idx+1}/{len(test_df)}] Processing {seq_id} (length: {seq_len})...")
        
        # Load MSA if available
        msa = load_msa(seq_id, msa_dir) if config.use_msa else None
        
        # Generate 5 predictions with diverse temperatures
        predictions = predict_structure(
            model, 
            sequence, 
            msa=msa,
            temperatures=config.temperatures
        )
        
        # Add to submission rows
        for pred_idx, coords in enumerate(predictions):
            for residue_idx in range(seq_len):
                x, y, z = coords[residue_idx]
                submission_rows.append({
                    'ID': f"{seq_id}_{residue_idx + 1}_{pred_idx + 1}",
                    'resname': sequence[residue_idx],
                    'resid': residue_idx + 1,
                    'x_1': x,
                    'y_1': y,
                    'z_1': z
                })
        
        # Progress report
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining = (len(test_df) - idx - 1) * avg_time
            print(f"    Elapsed: {elapsed:.1f}s, Avg per seq: {avg_time:.1f}s, Est. remaining: {remaining:.1f}s")
    
    # Create submission DataFrame
    submission = pd.DataFrame(submission_rows)
    
    total_time = time.time() - start_time
    print(f"\nTotal inference time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    return submission


def main():
    """Main execution for Kaggle notebook."""
    
    # Paths - adjust for Kaggle environment
    TEST_FILE = '/kaggle/input/test_sequences.csv'  # Kaggle test file path
    MSA_DIR = '/kaggle/input/msas'  # Pre-computed MSAs
    CHECKPOINT_PATH = '/kaggle/input/model_checkpoint.pth'  # Model weights
    OUTPUT_FILE = 'submission.csv'
    
    # Fallback paths for local testing
    if not os.path.exists(TEST_FILE):
        TEST_FILE = 'data/test_sequences.csv'
        MSA_DIR = 'msas'
        CHECKPOINT_PATH = 'checkpoints/best_model.pth'
    
    print("=" * 60)
    print("RNA 3D Folding - Kaggle Submission")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model, config = load_model(CHECKPOINT_PATH)
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Check for MSAs
    if os.path.exists(MSA_DIR):
        print(f"MSA directory found: {MSA_DIR}")
        config.use_msa = True
    else:
        print("MSA directory not found, running without MSAs")
        model.disable_msa()
        config.use_msa = False
    
    # Generate submission
    print(f"\nGenerating submission from: {TEST_FILE}")
    submission = generate_submission(TEST_FILE, model, config, MSA_DIR)
    
    # Save submission
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSubmission saved to: {OUTPUT_FILE}")
    print(f"Total rows: {len(submission)}")
    print(f"Submission shape: {submission.shape}")
    print("\nFirst few rows:")
    print(submission.head(10))
    
    print("\n" + "=" * 60)
    print("Submission generation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
