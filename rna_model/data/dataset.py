"""Dataset management for RNA 3D folding research."""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from .data import RNAStructure, DataValidator, DataPreprocessor


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    description: str
    version: str
    size: int  # Number of structures
    sequence_length_stats: Dict[str, Any]
    source: str
    license: str
    preprocessing_pipeline: List[str]
    quality_metrics: Dict[str, Any]
    created_at: str
    path: str
    checksum: str


class DatasetManager:
    """Manage datasets for RNA 3D folding research."""
    
    def __init__(self, datasets_dir: str = "datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(exist_ok=True)
        self.registry_file = self.datasets_dir / "registry.json"
        self.registry = self._load_registry()
        
        # Initialize data validator and preprocessor
        self.validator = DataValidator()
        self.preprocessor = DataPreprocessor(self.validator)
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load dataset registry."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save dataset registry."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_dataset(self, structures: List[RNAStructure], name: str, 
                        description: str, source: str, license: str = "Unknown",
                        version: str = "1.0") -> str:
        """Register a new dataset."""
        print(f"Registering dataset: {name}")
        
        # Validate structures
        processed_structures, processing_stats = self.preprocessor.preprocess_dataset(
            structures, normalize_coordinates=True, center_coordinates=True, remove_invalid_atoms=True
        )
        
        if not processed_structures:
            raise ValueError("No valid structures after preprocessing")
        
        # Calculate statistics
        sequence_lengths = [len(s.sequence) for s in processed_structures]
        sequence_stats = {
            'min_length': min(sequence_lengths),
            'max_length': max(sequence_lengths),
            'mean_length': np.mean(sequence_lengths),
            'std_length': np.std(sequence_lengths),
            'median_length': np.median(sequence_lengths)
        }
        
        # Quality metrics
        quality_metrics = {
            'validation_success_rate': processing_stats['successfully_processed'] / processing_stats['total_input'],
            'total_processed': processing_stats['total_input'],
            'validation_errors': processing_stats['validation_errors'],
            'validation_warnings': processing_stats['validation_warnings']
        }
        
        # Create dataset directory
        dataset_id = f"{name.lower().replace(' ', '_')}_{version}"
        dataset_dir = self.datasets_dir / dataset_id
        dataset_dir.mkdir(exist_ok=True)
        
        # Save structures
        dataset_path = dataset_dir / "dataset.json"
        dataset_data = {
            'structures': []
        }
        
        for structure in processed_structures:
            structure_data = {
                'sequence': structure.sequence,
                'coordinates': structure.coordinates.tolist(),
                'atom_names': structure.atom_names,
                'residue_names': structure.residue_names,
                'chain_id': structure.chain_id,
                'metadata': structure.metadata
            }
            dataset_data['structures'].append(structure_data)
        
        with open(dataset_path, 'w') as f:
            json.dump(dataset_data, f)
        
        # Calculate checksum
        try:
            with open(dataset_path, 'rb') as f:
                checksum = hashlib.md5(f.read()).hexdigest()
        except IOError as e:
            raise ValueError(f"Failed to calculate checksum for {dataset_path}: {e}")
        
        # Create dataset info
        dataset_info = DatasetInfo(
            name=name,
            description=description,
            version=version,
            size=len(processed_structures),
            sequence_length_stats=sequence_stats,
            source=source,
            license=license,
            preprocessing_pipeline=processing_stats['preprocessing_steps'],
            quality_metrics=quality_metrics,
            created_at=datetime.now().isoformat(),
            path=str(dataset_path),
            checksum=checksum
        )
        
        # Update registry
        self.registry[dataset_id] = asdict(dataset_info)
        self._save_registry()
        
        print(f"Dataset registered: {dataset_id}")
        print(f"Size: {len(processed_structures)} structures")
        print(f"Validation success rate: {quality_metrics['validation_success_rate']:.2%}")
        
        return dataset_id
    
    def load_dataset(self, dataset_id: str) -> Tuple[List[RNAStructure], DatasetInfo]:
        """Load a dataset."""
        if dataset_id not in self.registry:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        dataset_info = DatasetInfo(**self.registry[dataset_id])
        
        # Load structures
        with open(dataset_info.path, 'r') as f:
            dataset_data = json.load(f)
        
        structures = []
        for structure_data in dataset_data['structures']:
            structure = RNAStructure(
                sequence=structure_data['sequence'],
                coordinates=np.array(structure_data['coordinates']),
                atom_names=structure_data['atom_names'],
                residue_names=structure_data['residue_names'],
                chain_id=structure_data.get('chain_id', 'A'),  # Default to 'A' if not present
                metadata=structure_data.get('metadata', {})
            )
            structures.append(structure)
        
        return structures, dataset_info
    
    def list_datasets(self) -> Dict[str, DatasetInfo]:
        """List all registered datasets."""
        return {k: DatasetInfo(**v) for k, v in self.registry.items()}
    
    def get_dataset_info(self, dataset_id: str) -> DatasetInfo:
        """Get dataset information."""
        if dataset_id not in self.registry:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        return DatasetInfo(**self.registry[dataset_id])
    
    def split_dataset(self, dataset_id: str, train_ratio: float = 0.8, 
                      val_ratio: float = 0.1, test_ratio: float = 0.1,
                      random_seed: int = 42) -> Tuple[List[RNAStructure], List[RNAStructure], List[RNAStructure]]:
        """Split dataset into train/val/test sets."""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        structures, _ = self.load_dataset(dataset_id)
        
        # Shuffle structures
        np.random.seed(random_seed)
        indices = np.random.permutation(len(structures))
        
        n_total = len(structures)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        train_structures = [structures[i] for i in train_indices]
        val_structures = [structures[i] for i in val_indices]
        test_structures = [structures[i] for i in test_indices]
        
        return train_structures, val_structures, test_structures
    
    def create_cross_validation_splits(self, dataset_id: str, n_folds: int = 5, 
                                     random_seed: int = 42) -> List[Tuple[List[RNAStructure], List[RNAStructure]]]:
        """Create cross-validation splits."""
        structures, _ = self.load_dataset(dataset_id)
        
        # Shuffle structures
        np.random.seed(random_seed)
        indices = np.random.permutation(len(structures))
        
        fold_size = len(structures) // n_folds
        splits = []
        
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_folds - 1 else len(structures)
            
            test_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            test_structures = [structures[i] for i in test_indices]
            train_structures = [structures[i] for i in train_indices]
            
            splits.append((train_structures, test_structures))
        
        return splits
    
    def filter_by_length(self, dataset_id: str, min_length: int = None, 
                          max_length: int = None) -> List[RNAStructure]:
        """Filter structures by sequence length."""
        structures, _ = self.load_dataset(dataset_id)
        
        filtered = []
        for structure in structures:
            seq_len = len(structure.sequence)
            if min_length is not None and seq_len < min_length:
                continue
            if max_length is not None and seq_len > max_length:
                continue
            filtered.append(structure)
        
        return filtered
    
    def get_statistics(self, dataset_id: str) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        structures, info = self.get_dataset_info(dataset_id)
        
        sequence_lengths = [len(s.sequence) for s in structures]
        
        # Basic statistics
        stats = {
            'dataset_id': dataset_id,
            'name': info.name,
            'version': info.version,
            'size': len(structures),
            'sequence_length_stats': info.sequence_length_stats,
            'quality_metrics': info.quality_metrics
        }
        
        # Additional statistics
        stats.update({
            'nucleotide_composition': self._calculate_nucleotide_composition(structures),
            'coordinate_statistics': self._calculate_coordinate_statistics(structures),
            'structure_diversity': self._calculate_structure_diversity(structures)
        })
        
        return stats
    
    def _calculate_nucleotide_composition(self, structures: List[RNAStructure]) -> Dict[str, float]:
        """Calculate nucleotide composition."""
        composition = {'A': 0, 'U': 0, 'G': 0, 'C': 0, 'N': 0}
        total_nucleotides = 0
        
        for structure in structures:
            for nuc in structure.sequence.upper():
                if nuc in composition:
                    composition[nuc] += 1
                total_nucleotides += 1
        
        if total_nucleotides > 0:
            for nuc in composition:
                composition[nuc] = composition[nuc] / total_nucleotides
        
        return composition
    
    def _calculate_coordinate_statistics(self, structures: List[RNAStructure]) -> Dict[str, float]:
        """Calculate coordinate statistics."""
        all_coords = []
        
        for structure in structures:
            all_coords.extend(structure.coordinates.flatten())
        
        all_coords = np.array(all_coords)
        
        return {
            'min_coord': float(np.min(all_coords)),
            'max_coord': float(np.max(all_coords)),
            'mean_coord': float(np.mean(all_coords)),
            'std_coord': float(np.std(all_coords)),
            'median_coord': float(np.median(all_coords))
        }
    
    def _calculate_structure_diversity(self, structures: List[RNAStructure]) -> Dict[str, float]:
        """Calculate structure diversity metrics."""
        if len(structures) < 2:
            return {'unique_sequences': 1.0, 'avg_pairwise_rmsd': 0.0}
        
        # Unique sequences
        unique_sequences = len(set(s.sequence for s in structures))
        diversity = unique_sequences / len(structures)
        
        # Average pairwise RMSD (sample for performance)
        from .utils import compute_rmsd
        
        n_samples = min(100, len(structures))
        sample_indices = np.random.choice(len(structures), n_samples, replace=False)
        
        rmsd_values = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                idx1, idx2 = sample_indices[i], sample_indices[j]
                rmsd = compute_rmsd(structures[idx1].coordinates, structures[idx2].coordinates)
                rmsd_values.append(rmsd)
        
        return {
            'unique_sequences': diversity,
            'avg_pairwise_rmsd': np.mean(rmsd_values) if rmsd_values else 0.0,
            'n_pairs_compared': len(rmsd_values)
        }


# Convenience functions for common research workflows
def register_pdb_dataset(pdb_files: List[str], name: str, description: str, 
                        source: str = "PDB files") -> str:
    """Register a dataset from PDB files."""
    from .data import RNADatasetLoader
    
    loader = RNADatasetLoader()
    structures = []
    
    for pdb_file in pdb_files:
        try:
            structure = loader.load_pdb_structure(pdb_file)
            structures.append(structure)
        except Exception as e:
            print(f"Warning: Failed to load {pdb_file}: {e}")
    
    if not structures:
        raise ValueError("No valid structures loaded")
    
    manager = DatasetManager()
    return manager.register_dataset(structures, name, description, source)


def create_train_val_test_split(dataset_id: str, train_ratio: float = 0.8, 
                                val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[str, str, str]:
    """Create train/val/test splits and return split dataset IDs."""
    manager = DatasetManager()
    
    train_structures, val_structures, test_structures = manager.split_dataset(
        dataset_id, train_ratio, val_ratio, test_ratio
    )
    
    # Register splits as separate datasets
    train_id = f"{dataset_id}_train"
    val_id = f"{dataset_id}_val"
    test_id = f"{dataset_id}_test"
    
    info = manager.get_dataset_info(dataset_id)
    
    manager.register_dataset(train_structures, f"{info.name} - Train", 
                          f"Training split of {info.name}", info.source, info.license, info.version)
    manager.register_dataset(val_structures, f"{info.name} - Validation", 
                          f"Validation split of {info.name}", info.source, info.license, info.version)
    manager.register_dataset(test_structures, f"{info.name} - Test", 
                          f"Test split of {info.name}", info.source, info.license, info.version)
    
    return train_id, val_id, test_id
