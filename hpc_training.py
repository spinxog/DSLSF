#!/usr/bin/env python3
"""
HPC Training Script for RNA 3D Folding Pipeline

This script is designed for training on HPC clusters with multiple GPUs:
1. Distributed training across multiple GPUs
2. Mixed precision training with gradient accumulation
3. Automatic checkpointing and resumption
4. Comprehensive logging and monitoring
5. Curriculum learning and data augmentation
6. Teacher-student distillation workflow

Usage:
    python hpc_training.py --config config.json --data-dir /path/to/data --output-dir /path/to/output --gpus 0,1,2,3
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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from rna_model import IntegratedModel, PipelineConfig
from rna_model.training import Trainer, TrainingConfig, RNADataset, RNACollator
from rna_model.data import RNADatasetLoader
from rna_model.utils import set_seed, memory_usage, clear_cache


class HPCTrainer:
    """HPC-optimized trainer with distributed training support."""
    
    def __init__(self,
                 config: TrainingConfig,
                 pipeline_config: PipelineConfig,
                 data_dir: str,
                 output_dir: str,
                 gpu_ids: List[int],
                 rank: int = 0,
                 world_size: int = 1):
        """
        Initialize HPC trainer.
        
        Args:
            config: Training configuration
            pipeline_config: Model pipeline configuration
            data_dir: Directory containing training data
            output_dir: Directory for outputs and checkpoints
            gpu_ids: List of GPU IDs to use
            rank: Process rank for distributed training
            world_size: Total number of processes
        """
        self.config = config
        self.pipeline_config = pipeline_config
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.gpu_ids = gpu_ids
        self.rank = rank
        self.world_size = world_size
        
        # Setup directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize distributed training if needed
        self.is_distributed = world_size > 1
        if self.is_distributed:
            self.setup_distributed()
        
        # Setup device
        gpu_index = gpu_ids[rank % len(gpu_ids)]
        self.device = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(self.device)
        
        # Initialize components
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.trainer = None
        self.tensorboard_writer = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
    def setup_logging(self):
        """Setup logging for HPC training."""
        log_file = self.log_dir / f"training_rank_{self.rank}.log"
        logging.basicConfig(
            level=logging.INFO,
            format=f'[Rank {self.rank}] %(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout) if self.rank == 0 else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_distributed(self):
        """Setup distributed training."""
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=self.world_size,
            rank=self.rank
        )
        
        self.logger.info(f"Initialized distributed training: rank {self.rank}/{self.world_size}")
    
    def setup_model(self):
        """Initialize and setup the model."""
        self.logger.info("Setting up model...")
        
        # Create model
        self.model = IntegratedModel(self.pipeline_config)
        
        # Load checkpoint if available
        checkpoint_path = self.get_latest_checkpoint()
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            self.logger.info("Starting from scratch")
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Setup distributed training
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.device.index])
            self.logger.info("Model wrapped with DistributedDataParallel")
        
        # Compile model for speed (if available)
        if hasattr(torch, 'compile') and self.config.compile_model:
            try:
                self.model = torch.compile(self.model)
                self.logger.info("Model compiled with torch.compile")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")
    
    def setup_data(self):
        """Setup data loaders."""
        self.logger.info("Setting up data loaders...")
        
        # Load datasets
        loader = RNADatasetLoader(cache_dir=str(self.data_dir / "cache"))
        
        # Load training data
        train_data = loader.load_dataset(self.data_dir / "train.pkl")
        train_dataset = RNADataset(
            sequences=train_data['sequences'],
            structures=train_data['structures'],
            secondary_structures=train_data.get('secondary_structures'),
            msas=train_data.get('msas')
        )
        
        # Load validation data
        val_data = loader.load_dataset(self.data_dir / "val.pkl")
        val_dataset = RNADataset(
            sequences=val_data['sequences'],
            structures=val_data['structures'],
            secondary_structures=val_data.get('secondary_structures'),
            msas=val_data.get('msas')
        )
        
        # Create data loaders
        collator = RNACollator(max_seq_len=self.pipeline_config.max_sequence_length)
        
        # Adjust batch size for distributed training
        effective_batch_size = self.config.batch_size // self.world_size
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collator,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collator
        )
        
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
        self.logger.info(f"Effective batch size: {effective_batch_size}")
    
    def setup_trainer(self):
        """Setup trainer with HPC optimizations."""
        self.logger.info("Setting up trainer...")
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model.module if self.is_distributed else self.model,
            config=self.config,
            device=self.device
        )
        
        # Setup tensorboard (only on rank 0)
        if self.rank == 0:
            tensorboard_dir = self.log_dir / "tensorboard"
            self.tensorboard_writer = SummaryWriter(tensorboard_dir)
            self.trainer.tensorboard_writer = self.tensorboard_writer
            self.logger.info(f"Tensorboard logging to {tensorboard_dir}")
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("*.pth"))
        if not checkpoints:
            return None
        
        # Sort by modification time
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return latest
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load training checkpoint with proper distributed synchronization."""
        checkpoint_path = Path(checkpoint_path)
        
        # Synchronize all processes before checkpoint loading
        if self.is_distributed:
            dist.barrier()
        
        if not checkpoint_path.exists():
            if self.rank == 0:  # Only log on rank 0 to avoid duplicate messages
                self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            if self.is_distributed:
                dist.barrier()
            return
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                if self.is_distributed:
                    self.model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load training state
            if 'global_step' in checkpoint:
                self.global_step = checkpoint['global_step']
            if 'epoch' in checkpoint:
                self.epoch = checkpoint['epoch']
            if 'best_val_loss' in checkpoint:
                self.best_val_loss = checkpoint['best_val_loss']
            
            # Load optimizer state
            if self.trainer is not None and hasattr(self.trainer, 'optimizer') and 'optimizer_state_dict' in checkpoint:
                try:
                    self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except AttributeError as e:
                    if self.rank == 0:
                        self.logger.warning(f"Failed to load optimizer state: {e}")
            
            # Load scheduler state
            if self.trainer is not None and hasattr(self.trainer, 'scheduler') and 'scheduler_state_dict' in checkpoint:
                try:
                    self.trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except AttributeError as e:
                    if self.rank == 0:
                        self.logger.warning(f"Failed to load scheduler state: {e}")
            
            # Synchronize all processes after checkpoint loading
            if self.is_distributed:
                dist.barrier()
                
        except Exception as e:
            if self.rank == 0:
                self.logger.error(f"Failed to load checkpoint: {e}")
            raise
        finally:
            # Ensure all processes are synchronized
            if self.is_distributed:
                dist.barrier()
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint with proper distributed synchronization."""
        # Synchronize all processes before checkpoint saving
        if self.is_distributed:
            dist.barrier()
        
        # Only save on rank 0, but ensure all processes are synchronized
        if self.rank != 0:
            # Non-rank 0 processes wait for rank 0 to finish
            if self.is_distributed:
                dist.barrier()
            return
        
        try:
            checkpoint = {
                'global_step': self.global_step,
                'epoch': self.epoch,
                'best_val_loss': self.best_val_loss,
                'model_state_dict': self.model.module.state_dict() if self.is_distributed else self.model.state_dict(),
                'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                'scheduler_state_dict': self.trainer.scheduler.state_dict(),
                'config': self.config.__dict__,
                'pipeline_config': self.pipeline_config.__dict__
            }
            
            # Save regular checkpoint
            checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pth"
            torch.save(checkpoint, checkpoint_path)
            
            # Save best model
            if is_best:
                best_path = self.checkpoint_dir / "best_model.pth"
                torch.save(checkpoint, best_path)
                self.logger.info(f"New best model saved (loss: {self.best_val_loss:.6f})")
            
            # Keep only last 5 checkpoints to save space
            self.cleanup_checkpoints()
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise
        finally:
            # Synchronize all processes after checkpoint saving
            if self.is_distributed:
                dist.barrier()

def cleanup_checkpoints(self):
    """Keep only recent checkpoints to save space."""
    checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pth"))
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Keep last 5 + best model
    for checkpoint in checkpoints[5:]:
        try:
            checkpoint.unlink()
            self.logger.debug(f"Removed old checkpoint: {checkpoint}")
        except Exception as e:
            self.logger.warning(f"Failed to remove {checkpoint}: {e}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = []
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Training step
            loss_dict = self.trainer.train_step(batch)
            
            # Accumulate losses
            epoch_losses.append(loss_dict['total'])
            
            # Log progress
            if self.global_step % self.config.log_every == 0 and self.rank == 0:
                self.logger.info(
                    f"Epoch {self.epoch}, Step {self.global_step}, "
                    f"Batch {batch_idx}/{num_batches}, "
                    f"Loss: {loss_dict['total']:.6f}"
                )
                
                # Log to tensorboard
                if self.tensorboard_writer:
                    for key, value in loss_dict.items():
                        self.tensorboard_writer.add_scalar(f'train/{key}', value, self.global_step)
            
            self.global_step += 1
        
        # Average epoch loss
        avg_loss = np.mean(epoch_losses)
        return {'total': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Validation step
                loss_dict = self.trainer.evaluate([batch])
                val_losses.append(loss_dict['total'])
        
        # Average validation loss
        avg_loss = np.mean(val_losses)
        return {'total': avg_loss}
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting HPC training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Distributed: {self.is_distributed} (rank {self.rank}/{self.world_size})")
        
        # Training loop
        start_time = time.time()
        
        try:
            while self.global_step < self.config.max_steps:
                # Train epoch
                train_loss = self.train_epoch()
                
                # Validate
                if self.epoch % self.config.eval_every == 0:
                    val_loss = self.validate()
                    
                    if self.rank == 0:
                        self.logger.info(
                            f"Epoch {self.epoch}: "
                            f"Train Loss: {train_loss['total']:.6f}, "
                            f"Val Loss: {val_loss['total']:.6f}"
                        )
                        
                        # Log to tensorboard
                        if self.tensorboard_writer:
                            self.tensorboard_writer.add_scalar('train/epoch_loss', train_loss['total'], self.epoch)
                            self.tensorboard_writer.add_scalar('val/epoch_loss', val_loss['total'], self.epoch)
                        
                        # Save checkpoint
                        is_best = val_loss['total'] < self.best_val_loss
                        if is_best:
                            self.best_val_loss = val_loss['total']
                        
                        if self.epoch % self.config.save_every == 0 or is_best:
                            self.save_checkpoint(is_best)
                
                self.epoch += 1
                
                # Check memory usage and perform periodic cleanup
                if self.global_step % 1000 == 0 and self.rank == 0:
                    memory = memory_usage()
                    self.logger.info(f"Memory usage: {memory}")
                    
                    # Always perform periodic cache cleanup to prevent memory leaks
                    clear_cache()
                    self.logger.info("Performed periodic GPU cache cleanup")
                    
                    # Additional cleanup if memory is high
                    if 'allocated' in memory and memory['allocated'] > 40:  # 40GB
                        # Force garbage collection
                        import gc
                        gc.collect()
                        clear_cache()
                        self.logger.warning("High memory usage detected, performed aggressive cleanup")
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            # Save final checkpoint
            if self.rank == 0:
                self.save_checkpoint()
                
                # Close tensorboard
                if self.tensorboard_writer:
                    self.tensorboard_writer.close()
                
                total_time = time.time() - start_time
                self.logger.info(f"Training completed in {total_time/3600:.2f} hours")
                self.logger.info(f"Final step: {self.global_step}, Final epoch: {self.epoch}")
    
    def cleanup(self):
        """Cleanup distributed training."""
        if self.is_distributed:
            dist.destroy_process_group()


def setup_environment():
    """Setup environment variables for distributed training."""
    # Set CUDA visible devices
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # May already be set


def main():
    """Main HPC training function."""
    parser = argparse.ArgumentParser(description="HPC Training for RNA 3D Folding")
    parser.add_argument("--config", required=True,
                       help="Path to training configuration JSON file")
    parser.add_argument("--data-dir", required=True,
                       help="Directory containing training data")
    parser.add_argument("--output-dir", required=True,
                       help="Directory for outputs and checkpoints")
    parser.add_argument("--gpus", required=True,
                       help="Comma-separated list of GPU IDs to use")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from latest checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Parse GPU IDs
    gpu_ids = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    world_size = len(gpu_ids)
    
    # Load configurations
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    
    training_config = TrainingConfig(**config_dict.get('training', {}))
    pipeline_config = PipelineConfig(**config_dict.get('pipeline', {}))
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize distributed training
    if world_size > 1:
        mp.spawn(
            train_worker,
            args=(training_config, pipeline_config, args.data_dir, args.output_dir, gpu_ids),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU training
        train_worker(training_config, pipeline_config, args.data_dir, args.output_dir, gpu_ids, 0, 1)


def train_worker(training_config: TrainingConfig,
              pipeline_config: PipelineConfig,
              data_dir: str,
              output_dir: str,
              gpu_ids: List[int],
              rank: int,
              world_size: int):
    """Worker function for distributed training."""
    # Initialize trainer
    trainer = HPCTrainer(
        config=training_config,
        pipeline_config=pipeline_config,
        data_dir=data_dir,
        output_dir=output_dir,
        gpu_ids=gpu_ids,
        rank=rank,
        world_size=world_size
    )
    
    try:
        # Setup and train
        trainer.setup_model()
        trainer.setup_data()
        trainer.setup_trainer()
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
