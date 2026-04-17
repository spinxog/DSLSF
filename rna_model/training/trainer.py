"""Training utilities for RNA 3D folding pipeline."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import time
import sys
import logging
import hashlib
from pathlib import Path

from ..core.constants import MODEL, LOGGING, COMPETITION

from ..models.language_model import RNALanguageModel, masked_span_loss, contact_loss
from ..models.secondary_structure import SecondaryStructurePredictor, secondary_structure_loss
from ..models.integrated import IntegratedModel
from ..core.geometry_module import GeometryModule, geometry_loss, fape_loss
from ..core.utils import set_seed, clear_cache, memory_usage


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    batch_size: int = MODEL.DEFAULT_BATCH_SIZE
    num_workers: int = 4
    pin_memory: bool = True
    
    # Model architecture constants
    DEFAULT_D_MODEL: int = MODEL.DEFAULT_D_MODEL
    DEFAULT_N_LAYERS: int = MODEL.DEFAULT_N_LAYERS
    DEFAULT_N_HEADS: int = MODEL.DEFAULT_N_HEADS
    DEFAULT_D_FF: int = MODEL.DEFAULT_D_FF
    DEFAULT_MAX_SEQ_LEN: int = MODEL.DEFAULT_MAX_SEQ_LEN
    
    # Optimization
    learning_rate: float = MODEL.DEFAULT_LEARNING_RATE
    weight_decay: float = MODEL.DEFAULT_WEIGHT_DECAY
    warmup_steps: int = 1000
    max_steps: int = MODEL.DEFAULT_MAX_STEPS
    gradient_clip_norm: float = 1.0
    
    # Loss weights
    lm_loss_weight: float = 1.0
    ss_loss_weight: float = 1.0
    geometry_loss_weight: float = 2.0
    fape_loss_weight: float = 1.0
    
    # Training schedule
    save_every: int = 1000
    eval_every: int = 500
    log_every: int = 100
    
    # Hardware
    mixed_precision: bool = True
    accumulate_gradients: int = 1
    
    # Paths
    output_dir: str = "./outputs"
    resume_from_checkpoint: Optional[str] = None
    checkpoint_interval: int = 1000  # Save checkpoint every N steps
    max_checkpoints_to_keep: int = 5  # Keep only last N checkpoints
    
    # Memory management constants
    MEMORY_CLEANUP_INTERVAL: int = COMPETITION.MEMORY_CLEANUP_INTERVAL
    MAX_CACHE_SIZE: int = COMPETITION.MAX_CACHE_SIZE
    GPU_MEMORY_THRESHOLD: float = COMPETITION.GPU_MEMORY_THRESHOLD
    memory_threshold_gb: float = 40.0  # Memory threshold for aggressive cleanup
    
    # Competition constants
    DEFAULT_MAX_SEQUENCE_LENGTH: int = COMPETITION.DEFAULT_MAX_SEQUENCE_LENGTH
    DEFAULT_INFERENCE_TIMEOUT: float = COMPETITION.DEFAULT_INFERENCE_TIMEOUT
    DEFAULT_TIME_LIMIT_HOURS: float = COMPETITION.DEFAULT_TIME_LIMIT_HOURS
    
    def validate(self) -> None:
        """Validate training configuration parameters."""
        if self.batch_size <= 0 or self.batch_size > 128:
            raise ValueError(f"batch_size must be between 1 and 128, got {self.batch_size}")
        
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")
        
        if not (1e-6 <= self.learning_rate <= 1e-1):
            raise ValueError(f"learning_rate must be between 1e-6 and 1e-1, got {self.learning_rate}")
        
        if self.weight_decay < 0 or self.weight_decay > 1e-1:
            raise ValueError(f"weight_decay must be between 0 and 1e-1, got {self.weight_decay}")
        
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        
        if self.gradient_clip_norm <= 0:
            raise ValueError(f"gradient_clip_norm must be positive, got {self.gradient_clip_norm}")
        
        if self.accumulate_gradients <= 0:
            raise ValueError(f"accumulate_gradients must be positive, got {self.accumulate_gradients}")
        
        if self.memory_threshold_gb <= 0 or self.memory_threshold_gb > 100:
            raise ValueError(f"memory_threshold_gb must be between 0 and 100, got {self.memory_threshold_gb}")
        
        # Validate loss weights
        if any(x < 0 for x in [self.lm_loss_weight, self.ss_loss_weight, 
                              self.geometry_loss_weight, self.fape_loss_weight]):
            raise ValueError("All loss weights must be non-negative")
        
        # Validate intervals
        if self.checkpoint_interval <= 0 or self.checkpoint_interval > self.max_steps:
            raise ValueError(f"checkpoint_interval must be between 1 and max_steps, got {self.checkpoint_interval}")
        
        if self.save_every <= 0 or self.save_every > self.max_steps:
            raise ValueError(f"save_every must be between 1 and max_steps, got {self.save_every}")
        
        if self.eval_every <= 0 or self.eval_every > self.max_steps:
            raise ValueError(f"eval_every must be between 1 and max_steps, got {self.eval_every}")
        
        if self.log_every <= 0 or self.log_every > self.max_steps:
            raise ValueError(f"log_every must be between 1 and max_steps, got {self.log_every}")
        
        # Validate path
        if not self.output_dir or not isinstance(self.output_dir, str):
            raise ValueError("output_dir must be a non-empty string")
        
        if self.resume_from_checkpoint and not isinstance(self.resume_from_checkpoint, str):
            raise ValueError("resume_from_checkpoint must be a string path")
        
        # Validate checkpoint parameters
        if self.max_checkpoints_to_keep <= 0:
            raise ValueError(f"max_checkpoints_to_keep must be positive, got {self.max_checkpoints_to_keep}")


class CheckpointManager:
    """Manage training checkpoints with comprehensive error handling and validation."""
    
    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = None, 
                 min_disk_space_gb: float = None, checksum_validation: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints or LOGGING.DEFAULT_MAX_CHECKPOINTS
        self.min_disk_space_gb = min_disk_space_gb or LOGGING.MIN_DISK_SPACE_GB
        self.checksum_validation = checksum_validation
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _check_disk_space(self) -> bool:
        """Check if there's enough disk space for checkpoint saving."""
        try:
            import shutil
            stat = shutil.disk_usage(self.checkpoint_dir)
            free_space_gb = stat.free / (1024**3)
            return free_space_gb >= self.min_disk_space_gb
        except Exception as e:
            logging.warning(f"Could not check disk space: {e}")
            return True  # Assume there's enough space if we can't check
    
    def _validate_checkpoint_data(self, checkpoint: Dict[str, Any]) -> bool:
        """Validate checkpoint data integrity."""
        required_keys = {'step', 'epoch', 'model_state_dict', 'optimizer_state_dict', 'timestamp'}
        
        # Check required keys
        if not all(key in checkpoint for key in required_keys):
            missing = required_keys - set(checkpoint.keys())
            logging.error(f"Checkpoint missing required keys: {missing}")
            return False
        
        # Validate data types
        try:
            assert isinstance(checkpoint['step'], int) and checkpoint['step'] >= 0
            assert isinstance(checkpoint['epoch'], int) and checkpoint['epoch'] >= 0
            assert isinstance(checkpoint['loss'], (int, float)) and checkpoint['loss'] >= 0
            assert isinstance(checkpoint['timestamp'], (int, float))
            assert isinstance(checkpoint['model_state_dict'], dict)
            assert isinstance(checkpoint['optimizer_state_dict'], dict)
        except (AssertionError, TypeError, ValueError) as e:
            logging.error(f"Checkpoint data validation failed: {e}")
            return False
        
        return True
    
    def _compute_checksum(self, data: Any) -> str:
        """Compute checksum for checkpoint data."""
        import json
        try:
            # Create a deterministic representation
            if isinstance(data, dict):
                # Sort keys for deterministic ordering
                data_str = json.dumps(data, sort_keys=True, default=str)
            else:
                data_str = str(data)
            return hashlib.sha256(data_str.encode()).hexdigest()
        except Exception as e:
            logging.warning(f"Could not compute checksum: {e}")
            return ""
    
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Any, step: int, epoch: int, loss: float,
                       config: Dict[str, Any]) -> Path:
        """Save training checkpoint with comprehensive error handling."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pth"
        
        # Check disk space first
        if not self._check_disk_space():
            raise RuntimeError(f"Insufficient disk space (minimum {self.min_disk_space_gb}GB required)")
        
        # Prepare checkpoint data
        try:
            checkpoint = {
                'step': step,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': float(loss),  # Ensure float for JSON serialization
                'config': config,
                'timestamp': time.time(),
                'pytorch_version': torch.__version__,
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            }
        except Exception as e:
            raise RuntimeError(f"Failed to prepare checkpoint data: {e}")
        
        # Validate checkpoint data
        if not self._validate_checkpoint_data(checkpoint):
            raise RuntimeError("Checkpoint data validation failed")
        
        # Add checksum if enabled
        if self.checksum_validation:
            checkpoint['checksum'] = self._compute_checksum(checkpoint)
        
        # Save with temporary file and atomic rename
        temp_path = checkpoint_path.with_suffix('.tmp')
        backup_path = checkpoint_path.with_suffix('.bak')
        
        try:
            # Save to temporary file first
            torch.save(checkpoint, temp_path)
            
            # Verify the saved file
            if not temp_path.exists() or temp_path.stat().st_size == 0:
                raise RuntimeError("Temporary checkpoint file is empty or missing")
            
            # Verify we can load it back
            try:
                loaded = torch.load(temp_path, map_location='cpu')
                if not self._validate_checkpoint_data(loaded):
                    raise RuntimeError("Saved checkpoint validation failed")
            except Exception as e:
                raise RuntimeError(f"Saved checkpoint verification failed: {e}")
            
            # Create backup of existing checkpoint if it exists
            if checkpoint_path.exists():
                try:
                    checkpoint_path.rename(backup_path)
                except Exception as e:
                    logging.warning(f"Could not create backup of existing checkpoint: {e}")
            
            # Atomic rename
            temp_path.rename(checkpoint_path)
            
            # Remove backup if successful
            if backup_path.exists():
                backup_path.unlink()
            
            logging.info(f"Successfully saved checkpoint: {checkpoint_path}")
            
        except Exception as e:
            # Cleanup on failure
            for path in [temp_path, backup_path]:
                if path.exists():
                    try:
                        path.unlink()
                    except Exception:
                        pass
            raise RuntimeError(f"Failed to save checkpoint: {e}")
        
        # Clean up old checkpoints
        try:
            self._cleanup_old_checkpoints()
        except Exception as e:
            logging.warning(f"Checkpoint cleanup failed: {e}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Path, validate: bool = True) -> Dict[str, Any]:
        """Load training checkpoint with validation and error recovery."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Try backup if main file is corrupted
        backup_path = checkpoint_path.with_suffix('.bak')
        paths_to_try = [checkpoint_path]
        if backup_path.exists():
            paths_to_try.append(backup_path)
        
        last_error = None
        for path in paths_to_try:
            try:
                # Load checkpoint
                checkpoint = torch.load(path, map_location='cpu')
                
                # Validate if requested
                if validate and not self._validate_checkpoint_data(checkpoint):
                    raise RuntimeError("Checkpoint validation failed")
                
                # Validate checksum if present
                if (validate and self.checksum_validation and 
                    'checksum' in checkpoint):
                    stored_checksum = checkpoint['checksum']
                    computed_checksum = self._compute_checksum(checkpoint)
                    if stored_checksum != computed_checksum:
                        raise RuntimeError("Checkpoint checksum mismatch - data corruption detected")
                
                # If we loaded from backup, restore it
                if path != checkpoint_path:
                    logging.warning(f"Loaded checkpoint from backup: {path}")
                    try:
                        path.rename(checkpoint_path)
                    except Exception as e:
                        logging.warning(f"Could not restore backup checkpoint: {e}")
                
                logging.info(f"Successfully loaded checkpoint: {checkpoint_path}")
                return checkpoint
                
            except Exception as e:
                last_error = e
                logging.warning(f"Failed to load checkpoint from {path}: {e}")
                continue
        
        # If we get here, all attempts failed
        raise RuntimeError(f"Failed to load checkpoint from all sources. Last error: {last_error}")
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint with validation."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pth"))
        if not checkpoints:
            return None
        
        # Sort by modification time and validate
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        for checkpoint in checkpoints:
            try:
                # Quick validation by checking file size and trying to load metadata
                if checkpoint.stat().st_size > 0:
                    # Try to load just the step number without loading full state dict
                    metadata = torch.load(checkpoint, map_location='cpu')
                    if isinstance(metadata, dict) and 'step' in metadata:
                        return checkpoint
            except Exception as e:
                logging.warning(f"Invalid checkpoint {checkpoint}: {e}")
                # Move invalid checkpoint to .corrupted
                corrupted_path = checkpoint.with_suffix('.corrupted')
                try:
                    checkpoint.rename(corrupted_path)
                except Exception:
                    pass
        
        return None
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pth"))
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by modification time and remove oldest
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        removed_count = 0
        for checkpoint in checkpoints[self.max_checkpoints:]:
            try:
                checkpoint.unlink()
                removed_count += 1
                logging.debug(f"Removed old checkpoint: {checkpoint}")
            except Exception as e:
                logging.warning(f"Failed to remove checkpoint {checkpoint}: {e}")
        
        if removed_count > 0:
            logging.info(f"Cleaned up {removed_count} old checkpoints")
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get statistics about checkpoints."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pth"))
        
        total_size = sum(p.stat().st_size for p in checkpoints)
        
        stats = {
            'count': len(checkpoints),
            'max_checkpoints': self.max_checkpoints,
            'total_size_mb': total_size / (1024 * 1024),
            'latest_step': None,
            'oldest_step': None
        }
        
        if checkpoints:
            checkpoints.sort(key=lambda p: p.stat().st_mtime)
            try:
                oldest = torch.load(checkpoints[0], map_location='cpu')
                latest = torch.load(checkpoints[-1], map_location='cpu')
                stats['oldest_step'] = oldest.get('step')
                stats['latest_step'] = latest.get('step')
            except Exception:
                pass
        
        return stats


class RNADataset(Dataset):
    """Dataset for RNA structure prediction."""
    
    def __init__(self,
                 sequences: List[str],
                 structures: List[np.ndarray],
                 secondary_structures: Optional[List[np.ndarray]] = None,
                 msas: Optional[List[np.ndarray]] = None):
        self.sequences = sequences
        self.structures = structures
        self.secondary_structures = secondary_structures
        self.msas = msas
        
        assert len(sequences) == len(structures)
        if secondary_structures is not None:
            assert len(sequences) == len(secondary_structures)
        if msas is not None:
            assert len(sequences) == len(msas)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        item = {
            "sequence": self.sequences[idx],
            "coordinates": self.structures[idx]
        }
        
        if self.secondary_structures is not None:
            item["secondary_structure"] = self.secondary_structures[idx]
        
        if self.msas is not None:
            item["msa"] = self.msas[idx]
        
        return item


class RNACollator:
    """Collator for batching RNA data."""
    
    def __init__(self, max_seq_len: int = MODEL.DEFAULT_MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        
        # Find max sequence length in batch
        max_len = min(max(len(item["sequence"]) for item in batch), self.max_seq_len)
        
        # Tokenize sequences
        sequences = []
        coords = []
        masks = []
        
        for item in batch:
            seq = item["sequence"][:max_len]
            coord = item["coordinates"][:max_len]
            
            # Pad sequences
            padded_seq = seq + "N" * (max_len - len(seq))
            sequences.append(padded_seq)
            
            # Pad coordinates
            padded_coord = np.zeros((max_len, coord.shape[1], 3))
            padded_coord[:len(coord)] = coord
            coords.append(padded_coord)
            
            # Create mask
            mask = [1] * len(seq) + [0] * (max_len - len(seq))
            masks.append(mask)
        
        return {
            "sequences": sequences,
            "coordinates": torch.tensor(np.array(coords), dtype=torch.float32),
            "mask": torch.tensor(masks, dtype=torch.bool)
        }


class Trainer:
    """Trainer for RNA 3D folding pipeline."""
    
    def __init__(self, 
                 model: IntegratedModel,
                 config: TrainingConfig,
                 device: torch.device = torch.device("cuda")):
        # Validate configuration before initialization
        config.validate()
        
        self.model = model
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.max_steps
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Logging
        self.step = 0
        self.best_loss = float('inf')
        
        # Create directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Tokenize sequences
        tokens = self._tokenize_batch(batch["sequences"])
        
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            # Forward pass through language model
            lm_outputs = self.model.language_model(tokens)
            embeddings = lm_outputs["embeddings"]
            
            # Secondary structure prediction
            ss_outputs = self.model.ss_predictor(embeddings)
            
            # Geometry prediction
            encoded = self.model.structure_encoder(embeddings)
            seq_len = embeddings.size(1)
            pair_repr = encoded.unsqueeze(2).expand(-1, -1, seq_len, -1) + \
                       encoded.unsqueeze(1).expand(-1, seq_len, -1, -1)
            geometry_outputs = self.model.geometry_module(encoded, pair_repr)
            
            # Compute losses
            losses = {}
            
            # Language modeling loss
            mask, labels = self.model.language_model.create_span_mask(
                seq_len, tokens.size(0), self.device
            )
            lm_logits = lm_outputs["logits"]
            losses["lm"] = masked_span_loss(lm_logits, labels, mask)
            
            # Secondary structure loss (if targets available)
            if "secondary_structure" in batch:
                ss_targets = batch["secondary_structure"].to(self.device)
                losses["ss"] = secondary_structure_loss(
                    ss_outputs["contact_logits"],
                    ss_targets["contacts"],
                    ss_targets.get("pseudoknots")
                )
            
            # Geometry losses
            coords = batch["coordinates"]
            mask = batch["mask"]
            
            # FAPE loss - compute proper frames from coordinates
            if "coordinates" in geometry_outputs:
                # Compute local frames from predicted coordinates
                pred_coords = geometry_outputs["coordinates"]
                frames = self._compute_local_frames(pred_coords, mask)
                
                # Compute true frames from ground truth coordinates
                true_frames = self._compute_local_frames(coords, mask)
                
                losses["fape"] = fape_loss(
                    pred_coords,
                    frames,
                    coords,
                    true_frames,
                    mask
                )
            
            # Multi-task geometry loss
            losses["geometry"] = geometry_loss(
                geometry_outputs.get("distance_logits"),
                geometry_outputs.get("angle_logits"),
                geometry_outputs.get("torsion_logits"),
                geometry_outputs.get("pucker_logits"),
                None, None, None, None,  # Targets would come from data
                mask
            )
            
            # Total loss
            total_loss = (self.config.lm_loss_weight * losses.get("lm", 0) +
                         self.config.ss_loss_weight * losses.get("ss", 0) +
                         self.config.geometry_loss_weight * losses.get("geometry", 0) +
                         self.config.fape_loss_weight * losses.get("fape", 0))
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            
            if (self.step + 1) % self.config.accumulate_gradients == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            total_loss.backward()
            
            if (self.step + 1) % self.config.accumulate_gradients == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        # Update learning rate
        if (self.step + 1) % self.config.accumulate_gradients == 0:
            self.scheduler.step()
        
        # Convert losses to floats for logging
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        loss_dict["total"] = total_loss.item()
        loss_dict["lr"] = self.optimizer.param_groups[0]["lr"]
        
        return loss_dict
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_losses = {}
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                tokens = self._tokenize_batch(batch["sequences"])
                
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    # Forward pass (no gradient computation)
                    lm_outputs = self.model.language_model(tokens)
                    embeddings = lm_outputs["embeddings"]
                    
                    ss_outputs = self.model.ss_predictor(embeddings)
                    
                    encoded = self.model.structure_encoder(embeddings)
                    seq_len = embeddings.size(1)
                    pair_repr = encoded.unsqueeze(2).expand(-1, -1, seq_len, -1) + \
                               encoded.unsqueeze(1).expand(-1, seq_len, -1, -1)
                    geometry_outputs = self.model.geometry_module(encoded, pair_repr)
                    
                    # Compute validation losses (simplified)
                    losses = {}
                    losses["lm"] = 0.0  # Would need targets
                    losses["ss"] = 0.0  # Would need targets
                    losses["geometry"] = 0.0  # Would need targets
                    
                    total_loss = sum(losses.values())
                
                # Accumulate losses
                for k, v in losses.items():
                    if k not in total_losses:
                        total_losses[k] = 0.0
                    total_losses[k] += v
                
                total_losses["total"] = total_loss
                n_batches += 1
        
        # Average losses
        for k in total_losses:
            total_losses[k] /= n_batches
        
        return total_losses
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop."""
        print(f"Starting training for {self.config.max_steps} steps")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        
        start_time = time.time()
        
        while self.step < self.config.max_steps:
            for batch in train_loader:
                if self.step >= self.config.max_steps:
                    break
                
                # Training step
                loss_dict = self.train_step(batch)
                
                # Logging
                if self.step % self.config.log_every == 0:
                    elapsed = time.time() - start_time
                    memory = memory_usage()
                    
                    print(f"Step {self.step}: {loss_dict}")
                    print(f"Time: {elapsed:.2f}s, Memory: {memory}")
                
                # Evaluation
                if val_loader is not None and self.step % self.config.eval_every == 0:
                    val_losses = self.evaluate(val_loader)
                    print(f"Validation: {val_losses}")
                    
                    # Save best model
                    if val_losses["total"] < self.best_loss:
                        self.best_loss = val_losses["total"]
                        self.save_checkpoint("best_model.pth")
                        print(f"New best model saved: {self.best_loss:.4f}")
                
                # Save checkpoint
                if self.step % self.config.save_every == 0:
                    self.save_checkpoint(f"checkpoint_{self.step}.pth")
                
                self.step += 1
        
        print("Training completed!")
        self.save_checkpoint("final_model.pth")
    
    def _tokenize_batch(self, sequences: List[str]) -> torch.Tensor:
        """Tokenize batch of sequences with proper type validation."""
        if not isinstance(sequences, list):
            raise TypeError(f"Expected list of sequences, got {type(sequences)}")
        
        if not sequences:
            return torch.empty((0, 0), dtype=torch.long, device=self.device)
        
        # Convert all sequences to strings with validation
        string_sequences = []
        for i, seq in enumerate(sequences):
            if isinstance(seq, torch.Tensor):
                # Convert tensor back to string with validation
                if seq.numel() == 0:
                    string_sequences.append("")
                    continue
                try:
                    seq_tokens = seq.cpu().numpy() if seq.device != torch.device('cpu') else seq.numpy()
                    seq_str = ''.join([self._token_to_char(int(token)) for token in seq_tokens])
                    string_sequences.append(seq_str)
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Invalid tensor sequence at index {i}: {e}")
            elif isinstance(seq, str):
                string_sequences.append(seq)
            else:
                # Convert other types to string
                try:
                    seq_str = str(seq)
                    string_sequences.append(seq_str)
                except Exception as e:
                    raise ValueError(f"Cannot convert sequence at index {i} to string: {e}")
        
        # Validate sequences
        for i, seq in enumerate(string_sequences):
            if not isinstance(seq, str):
                raise TypeError(f"Sequence at index {i} is not a string: {type(seq)}")
        
        token_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
        
        # Find maximum sequence length with reasonable limit
        max_len = max(len(seq) for seq in string_sequences)
        if max_len > 10000:  # Reasonable upper limit
            raise ValueError(f"Sequence too long: {max_len} > 10000")
        
        tokens = torch.zeros(len(string_sequences), max_len, dtype=torch.long, device=self.device)
        
        for i, seq in enumerate(string_sequences):
            for j, nucleotide in enumerate(seq.upper()):  # Convert to uppercase
                if j < max_len:
                    tokens[i, j] = token_map.get(nucleotide, token_map['N'])
        
        return tokens
    
    def _token_to_char(self, token: int) -> str:
        """Convert token integer back to character."""
        token_map = {0: 'A', 1: 'U', 2: 'G', 3: 'C', 4: 'N'}
        if token not in token_map:
            raise ValueError(f"Invalid token: {token}")
        return token_map[token]
    
    def _compute_local_frames(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute local coordinate frames from 3D coordinates.
        
        Args:
            coords: (batch_size, seq_len, 3) coordinates
            mask: (batch_size, seq_len) boolean mask
            
        Returns:
            frames: (batch_size, seq_len, 4) quaternion frames
        """
        batch_size, seq_len, _ = coords.shape
        
        # Handle edge cases
        if seq_len < 3:
            # For very short sequences, return identity frames
            frames = torch.zeros(batch_size, seq_len, 4, device=coords.device)
            frames[..., 0] = 1.0  # Default to identity quaternion
            return frames
        
        frames = torch.zeros(batch_size, seq_len, 4, device=coords.device)
        frames[..., 0] = 1.0  # Default to identity quaternion
        
        # Compute local frames for valid residues (skip first and last)
        for i in range(1, seq_len - 1):
            # Check if current, previous, and next residues are valid
            valid_mask = mask[:, i-1] & mask[:, i] & mask[:, i+1]
            
            if valid_mask.any():
                # Get three consecutive points
                p_prev = coords[valid_mask, i-1]
                p_curr = coords[valid_mask, i]
                p_next = coords[valid_mask, i+1]
                
                # Compute local coordinate system
                v1 = p_curr - p_prev
                v2 = p_next - p_curr
                
                # Normalize vectors
                v1_norm = torch.norm(v1, dim=-1, keepdim=True)
                v2_norm = torch.norm(v2, dim=-1, keepdim=True)
                
                # Avoid division by zero
                v1 = v1 / (v1_norm + 1e-8)
                v2 = v2 / (v2_norm + 1e-8)
                
                # Compute local frame using cross product
                z_axis = v1
                x_axis = torch.cross(v1, v2)
                x_axis_norm = torch.norm(x_axis, dim=-1, keepdim=True)
                x_axis = x_axis / (x_axis_norm + 1e-8)
                y_axis = torch.cross(z_axis, x_axis)
                
                # Convert rotation matrix to quaternion
                rotation_matrix = torch.stack([x_axis, y_axis, z_axis], dim=-1)
                quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)
                
                frames[valid_mask, i] = quaternion
        
        return frames
    
    def _rotation_matrix_to_quaternion(self, matrices: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrices to quaternions."""
        # Simple matrix to quaternion conversion
        trace = matrices[..., 0, 0] + matrices[..., 1, 1] + matrices[..., 2, 2]
        
        quaternions = torch.zeros_like(trace).unsqueeze(-1).repeat(1, 4)
        
        # Case 1: trace > 0
        mask = trace > 0
        if mask.any():
            s = 0.5 / torch.sqrt(trace[mask] + 1.0)
            quaternions[mask, 0] = 0.25 / s
            quaternions[mask, 1] = (matrices[mask, 2, 1] - matrices[mask, 1, 2]) * s
            quaternions[mask, 2] = (matrices[mask, 0, 2] - matrices[mask, 2, 0]) * s
            quaternions[mask, 3] = (matrices[mask, 1, 0] - matrices[mask, 0, 1]) * s
        
        # Case 2: trace <= 0, find largest diagonal element
        # (Simplified implementation)
        mask2 = ~mask
        if mask2.any():
            # Use first diagonal element as fallback
            quaternions[mask2, 1] = 1.0  # Default to x-axis rotation
        
        # Normalize quaternions
        norm = torch.norm(quaternions, dim=-1, keepdim=True)
        quaternions = quaternions / (norm + 1e-8)
        
        return quaternions
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        filepath = Path(self.config.checkpoint_dir) / filename
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filename: str):
        """Load training checkpoint."""
        filepath = Path(self.config.checkpoint_dir) / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.step = checkpoint["step"]
        self.best_loss = checkpoint["best_loss"]
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        print(f"Loaded checkpoint from {filename}, step {self.step}")


def create_training_config(overrides: Optional[Dict] = None) -> TrainingConfig:
    """Create training configuration with optional overrides."""
    config = TrainingConfig()
    
    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return config


def train_model(model: IntegratedModel,
                train_dataset: RNADataset,
                val_dataset: Optional[RNADataset] = None,
                config_overrides: Optional[Dict] = None,
                seed: int = 42) -> Trainer:
    """Train the RNA model."""
    # Set seed
    set_seed(seed)
    
    # Create config
    config = create_training_config(config_overrides)
    
    # Create data loaders
    from ..core.constants import MODEL
    collator = RNACollator(max_seq_len=MODEL.DEFAULT_MAX_SEQ_LEN)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collator
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=collator
        )
    
    # Create trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    trainer = Trainer(model, config, device)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    return trainer
