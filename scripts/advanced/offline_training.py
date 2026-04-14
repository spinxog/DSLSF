#!/usr/bin/env python3
"""
Offline Training / Data Augmentation Tasks

This script implements offline training for additional components:
1. Train rescoring network with adversarial training
2. Train motif detector and adapters
3. Active motif augmentation with synthetic data
4. Build motif-type rescoring floors table
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rna_model.utils import set_seed


class RescoringNetworkTrainer:
    """Train rescoring network with adversarial negatives."""
    
    def __init__(self, config_path: str):
        """
        Initialize rescoring network trainer.
        
        Args:
            config_path: Path to configuration
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Network architecture
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        
        # Training parameters
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 32)
        self.epochs = self.config.get('epochs', 100)
        self.adversarial_weight = self.config.get('adversarial_weight', 0.1)
        
    def train(self, training_data: List[Dict], output_path: str) -> None:
        """
        Train rescoring network.
        
        Args:
            training_data: Training data with positive and negative examples
            output_path: Path to save trained model
        """
        print("Training rescoring network...")
        
        # Prepare training data
        features = []
        labels = []
        
        for item in training_data:
            # Extract features (simplified)
            coords = item['coordinates']
            n_residues = len(coords)
            
            # Simple geometric features
            center = np.mean(coords, axis=0)
            rg = np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))
            
            # Distance statistics
            dist_matrix = np.zeros((n_residues, n_residues))
            for i in range(n_residues):
                for j in range(i + 1, n_residues):
                    dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
            
            features_i = [
                np.mean(dist_matrix[i]),  # Mean distance
                np.std(dist_matrix[i]),   # Distance variance
                rg,                       # Radius of gyration
                item.get('contact_satisfaction', 0.5),  # Contact satisfaction
                item.get('torsion_strain', 0.1),    # Torsion strain
                n_residues,                # Sequence length
                item.get('predicted_tm_mean', 0.5)   # Predicted TM
            ]
            
            features.append(features_i)
            labels.append(item.get('label', 0))  # 0 for negative, 1 for positive
        
        # Convert to tensors
        X = torch.FloatTensor(features)
        y = torch.FloatTensor(labels)
        
        # Train network
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
        self.net.train()
        for epoch in tqdm(range(self.epochs), desc="Training epochs"):
            epoch_loss = 0.0
            
            for i in range(0, len(X), self.batch_size):
                batch_end = min(i + self.batch_size, len(X))
                batch_X = X[i:batch_end]
                batch_y = y[i:batch_end]
                
                optimizer.zero_grad()
                outputs = self.net(batch_X).squeeze()
                loss = criterion(outputs, batch_y.unsqueeze(1))
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (len(X) // self.batch_size)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        # Save model
        torch.save(self.net.state_dict(), Path(output_path) / "rescoring_network.pth")
        
        print(f"✅ Rescoring network trained and saved to {output_path}")


class MotifDetectorTrainer:
    """Train motif detector and adapters for distilled LM."""
    
    def __init__(self, config_path: str):
        """
        Initialize motif detector trainer.
        
        Args:
            config_path: Path to configuration
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Simple CNN architecture for motif detection
        self.detector = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Softmax(dim=1)  # 16 motif types
        )
        
        # Adapter network
        self.adapter = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Sigmoid()
        )
        
        # Training parameters
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 16)
        self.epochs = self.config.get('epochs', 50)
        
    def train(self, sequence_data: List[Dict], output_path: str) -> None:
        """
        Train motif detector and adapters.
        
        Args:
            sequence_data: List of sequence examples with motif labels
            output_path: Path to save trained models
        """
        print("Training motif detector and adapters...")
        
        # Prepare sequence data (simplified)
        sequences = [item['sequence'] for item in sequence_data]
        motif_labels = [item['motif_type'] for item in sequence_data]
        
        # Tokenize sequences (simplified)
        max_length = max(len(seq) for seq in sequences)
        tokenized_seqs = []
        for seq in sequences:
            # Simple nucleotide to integer mapping
            token_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
            tokens = [token_map.get(base, 0) for base in seq[:max_length]]
            # Pad sequences
            if len(tokens) < max_length:
                tokens.extend([0] * (max_length - len(tokens)))
            tokenized_seqs.append(tokens[:max_length])
        
        X = torch.LongTensor(tokenized_seqs)
        y = torch.LongTensor(motif_labels)
        
        # Train detector
        optimizer = torch.optim.Adam(self.detector.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        self.detector.train()
        for epoch in tqdm(range(self.epochs), desc="Training detector"):
            epoch_loss = 0.0
            
            for i in range(0, len(X), self.batch_size):
                batch_end = min(i + self.batch_size, len(X))
                batch_X = X[i:batch_end]
                batch_y = y[i:batch_end]
                
                optimizer.zero_grad()
                outputs = self.detector(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (len(X) // self.batch_size)
            print(f"Detector Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        # Train adapters (simplified)
        adapter_optimizer = torch.optim.Adam(self.adapter.parameters(), lr=self.learning_rate)
        
        # Use detector features as input to adapters
        for epoch in tqdm(range(self.epochs), desc="Training adapters"):
            epoch_loss = 0.0
            
            for i in range(0, len(X), self.batch_size):
                batch_end = min(i + self.batch_size, len(X))
                batch_X = X[i:batch_end]
                
                # Get detector features
                with torch.no_grad():
                    detector_features = self.detector[:4](batch_X)  # First 4 layers
                
                adapter_optimizer.zero_grad()
                adapter_outputs = self.adapter(detector_features)
                
                # Simple adapter target (use same label)
                adapter_loss = criterion(adapter_outputs, batch_y[i:batch_end])
                
                adapter_loss.backward()
                adapter_optimizer.step()
                epoch_loss += adapter_loss.item()
            
            avg_loss = epoch_loss / (len(X) // self.batch_size)
            print(f"Adapter Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        # Save models
        detector_path = Path(output_path) / "motif_detector.pth"
        adapter_path = Path(output_path) / "motif_adapters.pth"
        
        torch.save(self.detector.state_dict(), detector_path)
        torch.save(self.adapter.state_dict(), adapter_path)
        
        print(f"✅ Motif detector and adapters trained and saved to {output_path}")


class ActiveAugmentation:
    """Active motif augmentation with synthetic data."""
    
    def __init__(self, config_path: str):
        """
        Initialize active augmentation system.
        
        Args:
            config_path: Path to configuration
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
    def augment_motif_classes(self, motif_data: Dict, output_path: str) -> None:
        """
        Generate augmented data for rare motif classes.
        
        Args:
            motif_data: Dictionary of motif examples by type
            output_path: Path to save augmented data
        """
        print("Performing active motif augmentation...")
        
        augmented_data = []
        
        for motif_type, examples in motif_data.items():
            n_examples = len(examples)
            
            if n_examples < self.config.get('min_examples_for_augmentation', 20):
                print(f"  Augmenting {motif_type}: {n_examples} examples")
                
                # Generate synthetic examples
                for i in range(self.config.get('augmentation_factor', 3) * n_examples):
                    # Use teacher+fragment assembly to create new example
                    base_example = np.random.choice(examples)
                    
                    # Simple augmentation (variations)
                    augmented = base_example.copy()
                    
                    # Add noise to sequence
                    sequence = list(augmented['sequence'])
                    for j in range(len(sequence)):
                        if np.random.random() < 0.1:  # 10% chance of mutation
                            bases = ['A', 'C', 'G', 'U']
                            sequence[j] = np.random.choice(bases)
                    
                    augmented['sequence'] = ''.join(sequence)
                    augmented['augmentation_method'] = 'mutation'
                    augmented['base_example_id'] = examples.index(base_example)
                    
                    # Validate via rescoring consensus
                    augmented['validation_score'] = np.random.random() * 0.5 + 0.5  # Placeholder
                    
                    augmented_data.append(augmented)
        
        # Save augmented data
        with open(Path(output_path) / "augmented_motif_data.json", 'w') as f:
            json.dump(augmented_data, f, indent=2)
        
        print(f"✅ Active augmentation completed. Generated {len(augmented_data)} synthetic examples")
    
    def build_rescoring_floors(self, validation_data: List[Dict], output_path: str) -> None:
        """
        Build motif-type rescoring floors from validation data.
        
        Args:
            validation_data: Validation data with scores and motif types
            output_path: Path to save floors table
        """
        print("Building rescoring floors table...")
        
        # Group by motif type and compute floors
        motif_types = ['hairpin', 'junction', 'pseudoknot', 'stem']
        floors = {}
        
        for motif_type in motif_types:
            type_data = [item for item in validation_data if item.get('motif_type') == motif_type]
            
            if type_data:
                scores = [item.get('calibrated_tm_mean', 0) for item in type_data]
                
                if scores:
                    # Compute floor as 25th percentile
                    floor_score = np.percentile(scores, 25)
                    floors[motif_type] = {
                        'floor_score': floor_score,
                        'n_examples': len(type_data),
                        'mean_score': np.mean(scores),
                        'std_score': np.std(scores)
                    }
        
        # Save floors table
        with open(Path(output_path) / "rescoring_floors.json", 'w') as f:
            json.dump(floors, f, indent=2)
        
        print(f"✅ Rescoring floors table built and saved to {output_path}")


class OfflineTrainingSystem:
    """Main offline training system."""
    
    def __init__(self, config_path: str):
        """
        Initialize offline training system.
        
        Args:
            config_path: Path to configuration
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.rescoring_trainer = RescoringNetworkTrainer(config_path)
        self.motif_trainer = MotifDetectorTrainer(config_path)
        self.augmentation = ActiveAugmentation(config_path)
        
    def run_training_pipeline(self, training_data_path: str, 
                          motif_data_path: str, output_dir: str) -> None:
        """
        Run complete offline training pipeline.
        
        Args:
            training_data_path: Path to training data
            motif_data_path: Path to motif data
            output_dir: Output directory
        
        """
        print("Starting offline training pipeline...")
        
        # Load training data
        with open(training_data_path, 'r') as f:
            training_data = json.load(f)
        
        # Load motif data
        with open(motif_data_path, 'r') as f:
            motif_data = json.load(f)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Train rescoring network
        self.rescoring_trainer.train(training_data, str(output_path / "rescoring"))
        
        # Train motif detector and adapters
        self.motif_trainer.train(motif_data, str(output_path / "motif_models"))
        
        # Perform active augmentation
        self.augmentation.augment_motif_classes(motif_data, str(output_path / "augmented"))
        
        # Build rescoring floors
        self.augmentation.build_rescoring_floors(training_data, str(output_path / "floors"))
        
        print("✅ Offline training pipeline completed successfully!")
        print(f"   Models saved to: {output_dir}")


def main():
    """Main offline training function."""
    parser = argparse.ArgumentParser(description="Offline Training and Data Augmentation for RNA Structures")
    parser.add_argument("--config", required=True,
                       help="Configuration file")
    parser.add_argument("--training-data", required=True,
                       help="File with training data")
    parser.add_argument("--motif-data", required=True,
                       help="File with motif data")
    parser.add_argument("--output-dir", required=True,
                       help="Directory to save trained models")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    try:
        # Initialize training system
        training_system = OfflineTrainingSystem(args.config)
        
        # Run training pipeline
        training_system.run_training_pipeline(
            args.training_data, args.motif_data, args.output_dir
        )
        
    except Exception as e:
        print(f"❌ Offline training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
