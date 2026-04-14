#!/usr/bin/env python3
"""
Basic training example.

This example demonstrates how to train the RNA 3D folding pipeline
on synthetic data.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from rna_model import RNAFoldingPipeline, PipelineConfig, IntegratedModel
from rna_model.training import TrainingConfig, Trainer, RNADataset, RNACollator
from rna_model.data import create_sample_dataset
from rna_model.logging_config import setup_logger, PerformanceLogger
from rna_model.config import get_config


def create_synthetic_dataset(n_sequences: int = 100, max_length: int = 50):
    """Create synthetic training data."""
    sequences = []
    structures = []
    
    # Generate random RNA sequences
    nucleotides = ['A', 'U', 'G', 'C']
    
    for _ in range(n_sequences):
        # Random sequence length
        seq_len = np.random.randint(10, max_length)
        sequence = ''.join(np.random.choice(nucleotides, seq_len))
        sequences.append(sequence)
        
        # Generate random coordinates (simplified)
        n_residues = len(sequence)
        coords = np.random.randn(n_residues, 3, 3) * 10.0  # Random positions
        
        structures.append(coords)
    
    return sequences, structures


def main():
    """Run basic training example."""
    
    # Setup logging
    logger = setup_logger("basic_training", Path("logs"))
    perf_logger = PerformanceLogger(logger)
    
    logger.info("Starting basic training example")
    perf_logger.start_timer("total_training")
    
    # Get configuration
    config = get_config()
    
    # Create synthetic dataset
    logger.info("Creating synthetic dataset")
    sequences, structures = create_synthetic_dataset(n_sequences=50, max_length=30)
    
    # Create dataset
    dataset = RNADataset(sequences, structures)
    collator = RNACollator(max_seq_len=30)
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.DEFAULT_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=2
    )
    
    logger.info(f"Dataset created: {len(dataset)} sequences")
    logger.info(f"Batch size: {config.DEFAULT_BATCH_SIZE}")
    
    # Initialize model
    pipeline_config = PipelineConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_sequence_length=30,
        lm_config=config.__dict__.get("LMConfig", {}),
        geometry_config=config.__dict__.get("GeometryConfig", {})
    )
    
    model = IntegratedModel(pipeline_config)
    if torch.cuda.is_available():
        model = model.cuda()
    
    logger.info("Model initialized")
    perf_logger.log_model_stats(model)
    
    # Setup training
    training_config = TrainingConfig(
        batch_size=config.DEFAULT_BATCH_SIZE,
        learning_rate=config.DEFAULT_LEARNING_RATE,
        max_steps=100,  # Short training for demo
        save_every=50,
        log_every=10,
        checkpoint_dir="checkpoints/basic_training"
    )
    
    trainer = Trainer(model, training_config)
    
    logger.info("Trainer initialized")
    logger.info(f"Learning rate: {training_config.learning_rate}")
    logger.info(f"Max steps: {training_config.max_steps}")
    
    # Training loop
    logger.info("Starting training loop")
    
    for epoch in range(5):  # Short training for demo
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            device = next(model.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Training step
            loss_dict = trainer.train_step(batch)
            epoch_loss += loss_dict["total"]
            n_batches += 1
            
            # Log progress
            if batch_idx % 5 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}, "
                           f"Loss: {loss_dict['total']:.6f}")
        
        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        logger.info(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.6f}")
    
    # Save final model
    logger.info("Saving final model")
    model_path = "checkpoints/basic_training/final_model.pth"
    torch.save(model.state_dict(), model_path)
    
    perf_logger.end_timer("total_training", epochs=5, sequences=len(sequences))
    logger.info("Basic training completed successfully")
    
    return model


if __name__ == "__main__":
    main()
