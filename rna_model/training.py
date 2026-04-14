"""Training utilities for RNA 3D folding pipeline."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import time
from pathlib import Path

from .language_model import RNALanguageModel, masked_span_loss, contact_loss
from .secondary_structure import SecondaryStructurePredictor, secondary_structure_loss
from .geometry_module import GeometryModule, geometry_loss, fape_loss
from .pipeline import IntegratedModel, PipelineConfig
from .utils import set_seed, clear_cache, memory_usage


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    max_steps: int = 100000
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
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


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
    
    def __init__(self, max_seq_len: int = 512):
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
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
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
        
        # Convert all sequences to strings if they aren't already
        string_sequences = []
        for seq in sequences:
            if isinstance(seq, torch.Tensor):
                # Convert tensor back to string
                seq = ''.join([['A', 'U', 'G', 'C', 'N'][token.item()] for token in seq])
            elif not isinstance(seq, str):
                seq = str(seq)
            string_sequences.append(seq)
        
        token_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
        
        max_len = max(len(seq) for seq in string_sequences)
        tokens = torch.zeros(len(string_sequences), max_len, dtype=torch.long, device=self.device)
        
        for i, seq in enumerate(string_sequences):
            for j, nucleotide in enumerate(seq):
                if j < max_len and nucleotide in token_map:
                    tokens[i, j] = token_map[nucleotide]
                else:
                    tokens[i, j] = token_map['N']
        
        return tokens
    
    def _compute_local_frames(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute local coordinate frames from 3D coordinates.
        
        Args:
            coords: (batch_size, seq_len, 3) coordinates
            mask: (batch_size, seq_len) boolean mask
            
        Returns:
            frames: (batch_size, seq_len, 4) quaternion frames
        """
        batch_size, seq_len, _ = coords.shape
        frames = torch.zeros(batch_size, seq_len, 4, device=coords.device)
        frames[..., 0] = 1.0  # Default to identity quaternion
        
        # Compute local frames for valid residues
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
        checkpoint = torch.load(filepath, map_location=self.device)
        
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
    collator = RNACollator(max_seq_len=512)
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
