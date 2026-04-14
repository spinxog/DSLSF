"""Training command for RNA 3D folding pipeline."""

import argparse
import sys
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ..pipeline import RNAFoldingPipeline, PipelineConfig
from ..training import Trainer, TrainingConfig
from ..data import RNADatasetLoader, RNAStructure
from ..logging_config import setup_logger
from ..config import get_config


def train_command():
    """Command-line interface for training."""
    
    parser = argparse.ArgumentParser(
        description="Train RNA 3D folding models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        required=True,
        help="Directory containing training data"
    )
    
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-path", "-m",
        type=Path,
        help="Path to save model checkpoints"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    
    parser    .add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Maximum number of training steps"
    )
    
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Save checkpoint every N steps"
    )
    
    parser.add_argument(
        "--eval-every",
        type=int,
        default=500,
        help="Evaluate every N steps"
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use mixed precision training"
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
    
    # Setup logging
    if not args.quiet:
        logger = setup_logger("rna_train", args.log_file, args.log_level)
        logger.info("Starting RNA model training")
    
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
    
    # Setup training configuration
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        save_every=args.save_every,
        eval_every=args.eval_every,
        mixed_precision=args.mixed_precision,
        checkpoint_dir=args.model_path if args.model_path else Path("checkpoints"),
        log_dir=Path("logs")
    )
    
    # Initialize pipeline
    pipeline_config = PipelineConfig(
        device=device,
        max_sequence_length=config.max_seq_len,
        mixed_precision=args.mixed_precision
    )
    
    pipeline = RNAFoldingPipeline(pipeline_config)
    
    # Load model if resuming
    if args.resume and args.model_path and args.model_path.exists():
        pipeline.load_model(str(args.model_path))
        if not args.quiet:
            logger.info(f"Resumed training from {args.model_path}")
    
    # Load data
    data_loader = RNADatasetLoader(
        data_dir=args.data_dir,
        cache_dir=Path("cache"),
        max_seq_len=config.max_seq_len
    )
    
    # Create train/val split
    structures = data_loader.load_all_structures()
    train_structures, val_structures = data_loader.create_train_val_split(
        structures, val_ratio=args.val_ratio
    )
    
    # Preprocess data
    train_data = data_loader.preprocess_for_training(train_structures)
    val_data = data_loader.preprocess_for_training(val_structures)
    
    # Create data loaders
    train_loader = DataLoader(
        train_data["sequences"],
        train_data["coordinates"],
        train_data["mask"],
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_data["sequences"],
        val_data["coordinates"],
        val_data["mask"],
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=True
    )
    
    # Initialize trainer
    trainer = Trainer(pipeline.model, training_config, device)
    
    if not args.quiet:
        logger.info(f"Training on {len(train_structures)} structures")
        logger.info(f"Validating on {len(val_structures)} structures")
        logger.info(f"Device: {device}")
        logger.info(f"Batch size: {training_config.batch_size}")
        logger.info(f"Learning rate: {training_config.learning_rate}")
        logger.info(f"Max steps: {training_config.max_steps}")
    
    # Training loop
    trainer.train(train_loader, val_loader)
    
    if not args.quiet:
        logger.info("Training completed!")
    
    return trainer


if __name__ == "__main__":
    train_command()