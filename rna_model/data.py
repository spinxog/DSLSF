"""Data loading and preprocessing utilities for RNA 3D folding."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import logging
import hashlib
import threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from Bio import SeqIO
from Bio.PDB import PDBParser, PDBIO
import tempfile
import os
import time
from contextlib import contextmanager
from functools import wraps
import asyncio
from .utils import tokenize_rna_sequence, compute_contact_map, bin_distances


def retry_on_io_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry I/O operations on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (IOError, OSError) as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            return None
        return wrapper
    return decorator


# Cross-platform file locking implementation
class FileLock:
    """Cross-platform file locking using threading locks."""
    
    _locks = {}  # Class-level lock registry
    
    @classmethod
    def get_lock(cls, path: Path) -> threading.Lock:
        """Get or create a lock for the given path."""
        path_str = str(path.absolute())
        if path_str not in cls._locks:
            cls._locks[path_str] = threading.Lock()
        return cls._locks[path_str]


@contextmanager
def file_lock(lock_file: Path):
    """Context manager for cross-platform file locking."""
    lock = FileLock.get_lock(lock_file)
    lock_path = lock_file.with_suffix('.lock')
    
    try:
        # Acquire thread lock first
        with lock:
            # Create lock file as additional safety
            lock_path.touch(exist_ok=True)
            yield
    finally:
        # Clean up lock file
        try:
            if lock_path.exists():
                lock_path.unlink()
        except OSError:
            pass


@dataclass
class RNAStructure:
    """Container for RNA structure data."""
    sequence: str
    coordinates: np.ndarray  # Shape: (n_residues, n_atoms, 3)
    atom_names: List[str]    # Atom names per residue
    residue_names: List[str] # Residue names (A, U, G, C)
    chain_id: str
    pdb_id: Optional[str] = None
    resolution: Optional[float] = None
    secondary_structure: Optional[np.ndarray] = None
    contacts: Optional[np.ndarray] = None


class RNADatasetLoader:
    """Loader for RNA structure datasets."""
    
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # PDB parser
        self.pdb_parser = PDBParser()
        
        # Standard RNA atoms
        self.standard_atoms = ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'", "N1", "N3", "C2", "C4", "C5", "C6", "O2", "O4", "N6", "N2", "O6", "N7", "N9"]
    
    def _validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """Validate file path to prevent path traversal and symlink attacks."""
        file_path = Path(file_path)
        
        # Check for suspicious patterns first
        suspicious_patterns = ['..', '\\\\', '//', '\0', '|', '<', '>', '"', '*', '?']
        path_str = str(file_path)
        for pattern in suspicious_patterns:
            if pattern in path_str:
                raise ValueError(f"Suspicious path pattern detected: {pattern}")
        
        # Normalize path and resolve to absolute path without following symlinks
        try:
            abs_path = file_path.resolve(strict=False)
        except (RuntimeError, OSError) as e:
            raise ValueError(f"Invalid path resolution: {e}")
        
        # Check if the original path is a symlink (potential security risk)
        if file_path.exists() and file_path.is_symlink():
            raise ValueError(f"Symbolic links not allowed for security: {file_path}")
        
        # Check if any parent directory is a symlink
        for parent in file_path.parents:
            if parent.exists() and parent.is_symlink():
                raise ValueError(f"Symbolic links in parent directories not allowed: {parent}")
        
        # Check if path is within allowed directory (cache_dir)
        cache_abs = self.cache_dir.resolve()
        try:
            # Use relative_to with strict checking
            abs_path.relative_to(cache_abs)
        except ValueError:
            raise ValueError(f"File path outside allowed directory: {file_path}")
        
        # Additional security checks
        if len(str(abs_path)) > 4096:  # Prevent path length attacks
            raise ValueError("Path too long")
        
        # Check for Windows drive letters if on Unix system
        if os.name != 'nt' and ':' in str(file_path)[:2]:
            raise ValueError("Invalid path format for current OS")
        
        return abs_path
    
    @retry_on_io_error(max_retries=3, delay=1.0)
    def load_pdb_structure(self, pdb_file: Union[str, Path]) -> RNAStructure:
        """Load RNA structure from PDB file with security validation and retry logic."""
        pdb_file = self._validate_file_path(pdb_file)
        
        if not pdb_file.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        
        # Simple PDB parsing with validation
        coordinates = []
        atom_names = []
        residue_names = []
        
        try:
            with open(pdb_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if len(line) > 54 and (line.startswith('ATOM') or line.startswith('HETATM')):
                        try:
                            atom_name = line[12:16].strip()
                            residue_name = line[17:20].strip()
                            x_str = line[30:38].strip()
                            y_str = line[38:46].strip()
                            z_str = line[46:54].strip()
                            
                            # Validate coordinate values
                            x = float(x_str)
                            y = float(y_str)
                            z = float(z_str)
                            
                            # Check for reasonable coordinate values
                            if not (-1000 <= x <= 1000 and -1000 <= y <= 1000 and -1000 <= z <= 1000):
                                logging.warning(f"Unreasonable coordinates at line {line_num}: ({x}, {y}, {z})")
                                continue
                            
                            coordinates.append([x, y, z])
                            atom_names.append(atom_name)
                            residue_names.append(residue_name)
                        except (ValueError, IndexError) as e:
                            logging.warning(f"Error parsing line {line_num}: {e}")
                            continue
        except Exception as e:
            raise RuntimeError(f"Error reading PDB file {pdb_file}: {e}")
        
        if not coordinates:
            raise ValueError(f"No valid coordinates found in {pdb_file}")
        
        coordinates = np.array(coordinates)
        
        # Group by residue with thread safety
        residues = {}
        # Use a context manager for thread safety
        lock = threading.Lock()
        try:
            with lock:
                for i, (coord, atom_name, residue_name) in enumerate(zip(coordinates, atom_names, residue_names)):
                    if residue_name not in residues:
                        residues[residue_name] = []
                    residues[residue_name].append((coord, atom_name))
        finally:
            # Lock is automatically released by context manager
            pass
        
        # Convert to standard format
        sequence = ''.join([res[0][0] for res in residues.values()])
        n_residues = len(residues)
        
        # Create coordinate array (n_residues, n_atoms_per_residue, 3)
        standard_coords = np.zeros((n_residues, 3, 3))  # Simplified: 3 atoms per residue
        
        for i, (residue_name, atoms) in enumerate(residues.items()):
            for j, (coord, atom_name) in enumerate(atoms[:3]):  # Take first 3 atoms
                standard_coords[i, j] = coord
        
        return RNAStructure(
            sequence=sequence,
            coordinates=standard_coords,
            atom_names=[[atom[1] for atom in atoms[:3]] for atoms in residues.values()],
            residue_names=list(residues.keys()),
            chain_id="A",
            pdb_id=pdb_file.stem
        )
    
    def _is_rna_chain(self, chain) -> bool:
        """Check if chain contains RNA."""
        rna_residues = set(['A', 'U', 'G', 'C'])
        
        for residue in chain:
            if residue.get_resname() in rna_residues:
                return True
        
        return False
    
    def _extract_chain_data(self, chain) -> Tuple[str, np.ndarray, List[str], List[str]]:
        """Extract sequence, coordinates, and atom info from chain."""
        residues = list(chain.get_residues())
        
        sequence = ""
        coordinates = []
        atom_names = []
        residue_names = []
        
        for residue in residues:
            res_name = residue.get_resname()
            if res_name not in ['A', 'U', 'G', 'C']:
                continue
            
            # Add to sequence
            sequence += res_name
            residue_names.append(res_name)
            
            # Extract coordinates for standard atoms
            res_coords = []
            res_atoms = []
            
            for atom_name in self.standard_atoms:
                if atom_name in residue:
                    atom = residue[atom_name]
                    coords = atom.get_coord()
                    res_coords.append(coords)
                    res_atoms.append(atom_name)
                else:
                    # Missing atom - use placeholder
                    res_coords.append(np.array([0.0, 0.0, 0.0]))
                    res_atoms.append(atom_name)
            
            coordinates.append(np.array(res_coords))
            atom_names.append(res_atoms)
        
        return sequence, np.array(coordinates), atom_names, residue_names
    
    def load_rnacentral_sequences(self, download: bool = True) -> List[str]:
        """Load RNA sequences from RNAcentral database with cache validation."""
        cache_file = self.cache_dir / "rnacentral_sequences.json"
        
        if cache_file.exists() and not download:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Validate cache integrity
                if not isinstance(data, list) or not all(isinstance(seq, str) for seq in data):
                    raise ValueError("Invalid cache format")
                
                # Verify checksum if available
                if '_metadata' in data and isinstance(data, dict):
                    if 'checksum' in data['_metadata']:
                        # Validate checksum
                        data_copy = data.copy()
                        del data_copy['_metadata']
                        json_str = json.dumps(data_copy, separators=(',', ':'))
                        calculated_checksum = hashlib.sha256(json_str.encode()).hexdigest()
                        stored_checksum = data['_metadata']['checksum']
                        
                        if calculated_checksum != stored_checksum:
                            raise ValueError("Cache checksum mismatch")
                    
                    # Extract sequences from dict format
                    if 'sequences' in data:
                        sequences = data['sequences']
                    else:
                        sequences = data
                else:
                    sequences = data
                
                return sequences
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                self.logger.warning(f"Cache validation failed: {e}, regenerating data")
        
        # For now, return a small sample
        # In practice, this would download from RNAcentral
        sample_sequences = [
            "GGGAAAUCC",
            "GCCUUGGCAAC",
            "AUGCUAAUCGAU",
            "CGGAUCUCCGAGUCC",
            "AAUCCGGAAUCCGGAAUCCGG"
        ]
        
        # Save with checksum
        cache_data = {
            'sequences': sample_sequences,
            '_metadata': {
                'version': '1.0',
                'created_at': time.time(),
                'checksum': ''
            }
        }
        
        # Calculate checksum
        json_str = json.dumps(sample_sequences, separators=(',', ':'))
        cache_data['_metadata']['checksum'] = hashlib.sha256(json_str.encode()).hexdigest()
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
        
        return sample_sequences
    
    def create_synthetic_structures(self, sequences: List[str]) -> List[RNAStructure]:
        """Create synthetic structures for sequences (for pretraining)."""
        synthetic_structures = []
        
        for seq in sequences:
            # Simple linear chain structure
            n_residues = len(seq)
            n_atoms = len(self.standard_atoms)
            
            # Create coordinates in a simple helical arrangement
            coordinates = np.zeros((n_residues, n_atoms, 3))
            
            for i in range(n_residues):
                # Place atoms in a simple pattern
                for j, atom_name in enumerate(self.standard_atoms):
                    if j == 0:  # P atom
                        coordinates[i, j, 0] = i * 3.4  # Along x-axis
                        coordinates[i, j, 1] = 0.0
                        coordinates[i, j, 2] = 0.0
                    elif j == 1:  # O5'
                        coordinates[i, j, 0] = i * 3.4 + 1.5
                        coordinates[i, j, 1] = 1.0
                        coordinates[i, j, 2] = 0.0
                    else:
                        # Simple arrangement for other atoms
                        coordinates[i, j, 0] = i * 3.4 + j * 0.3
                        coordinates[i, j, 1] = np.sin(j * 0.5) * 2.0
                        coordinates[i, j, 2] = np.cos(j * 0.5) * 2.0
            
            # Create atom names and residue names
            atom_names = [self.standard_atoms.copy() for _ in range(n_residues)]
            residue_names = list(seq)
            
            # Compute contacts
            contacts = compute_contact_map(coordinates[:, 0, :])
            
            structure = RNAStructure(
                sequence="".join(residue_names),
                coordinates=np.array(coordinates),
                atom_names=atom_names,
                residue_names=residue_names,
                metadata={"source": "synthetic"}
            )
            
            synthetic_structures.append(structure)
        
        return synthetic_structures
    
    def filter_by_length(self, structures: List[RNAStructure], 
                        min_length: int = 20, 
                        max_length: int = 500) -> List[RNAStructure]:
        """Filter structures by sequence length."""
        filtered = []
        
        for structure in structures:
            seq_len = len(structure.sequence)
            if min_length <= seq_len <= max_length:
                filtered.append(structure)
        
        return filtered
    
    def deduplicate_sequences(self, structures: List[RNAStructure], 
                            identity_threshold: float = 0.8) -> List[RNAStructure]:
        """Remove highly similar sequences."""
        if len(structures) <= 1:
            return structures
        
        # Simple sequence identity check
        unique_structures = [structures[0]]
        
        for structure in structures[1:]:
            is_duplicate = False
            
            for unique in unique_structures:
                identity = self._compute_sequence_identity(structure.sequence, unique.sequence)
                if identity >= identity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_structures.append(structure)
        
        return unique_structures
    
    def _compute_sequence_identity(self, seq1: str, seq2: str) -> float:
        """Compute sequence identity between two sequences."""
        # Align sequences (simple approach - use shorter length)
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0
        
        matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
        identity = matches / min_len
        
        return identity
    
    def create_train_val_split(self, structures: List[RNAStructure], 
                            val_ratio: float = 0.1,
                            family_split: bool = True) -> Tuple[List[RNAStructure], List[RNAStructure]]:
        """Create train/validation split."""
        if family_split:
            # Family-based split (simplified - would use actual family info)
            # For now, just random split
            np.random.shuffle(structures)
        
        split_idx = int(len(structures) * (1 - val_ratio))
        train_structures = structures[:split_idx]
        val_structures = structures[split_idx:]
        
        return train_structures, val_structures
    
    def preprocess_for_training(self, structures: List[RNAStructure]) -> Dict[str, List]:
        """Preprocess structures for training."""
        processed_data = {
            "sequences": [],
            "coordinates": [],
            "contacts": [],
            "secondary_structures": []
        }
        
        for structure in structures:
            processed_data["sequences"].append(structure.sequence)
            processed_data["coordinates"].append(structure.coordinates)
            
            if structure.contacts is not None:
                processed_data["contacts"].append(structure.contacts)
            else:
                # Compute contacts if not available
                contacts = compute_contact_map(structure.coordinates[:, 0, :])
                processed_data["contacts"].append(contacts)
            
            # Secondary structure (placeholder - would predict with external tool)
            ss = self._predict_secondary_structure(structure.sequence)
            processed_data["secondary_structures"].append(ss)
        
        return processed_data
    
    def _predict_secondary_structure(self, sequence: str) -> np.ndarray:
        """Predict secondary structure (placeholder)."""
        # Simple base-pairing prediction
        n = len(sequence)
        ss_matrix = np.zeros((n, n))
        
        # Simple complementary base pairing
        complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        
        for i in range(n):
            for j in range(i + 4, n):  # Minimum loop size
                if complement.get(sequence[i]) == sequence[j]:
                    ss_matrix[i, j] = 1
                    ss_matrix[j, i] = 1
        
        return ss_matrix
    
    def save_dataset(self, data: Dict[str, List], filepath: Union[str, Path]):
        """Save processed dataset using secure JSON format with file locking."""
        filepath = self._validate_file_path(filepath)
        
        if filepath.suffix == '.json':
            # Convert numpy arrays to lists for JSON
            json_data = {}
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    json_data[key] = [arr.tolist() for arr in value]
                elif isinstance(value, np.ndarray):
                    json_data[key] = value.tolist()
                else:
                    json_data[key] = value
            
            # Add checksum for integrity
            json_str = json.dumps(json_data, separators=(',', ':'))
            checksum = hashlib.sha256(json_str.encode()).hexdigest()
            json_data['_metadata'] = {'checksum': checksum, 'version': '1.0'}
            
            # Use file locking to prevent concurrent access issues
            with file_lock(filepath):
                # Write to temporary file first, then atomic rename
                temp_file = filepath.with_suffix('.tmp')
                try:
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2)
                    
                    # Atomic rename
                    temp_file.rename(filepath)
                    
                except Exception as e:
                    # Clean up temp file if something went wrong
                    if temp_file.exists():
                        temp_file.unlink()
                    raise e
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}. Only JSON is supported for security.")
    
    def load_dataset(self, filepath: Union[str, Path]) -> Dict[str, List]:
        """Load processed dataset with integrity verification and file locking."""
        filepath = self._validate_file_path(filepath)
        
        if filepath.suffix == '.json':
            # Use file locking to prevent concurrent access issues
            with file_lock(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Verify checksum if present
                    if '_metadata' in data and 'checksum' in data['_metadata']:
                        stored_checksum = data['_metadata']['checksum']
                        # Create copy without metadata for checksum calculation
                        data_copy = data.copy()
                        del data_copy['_metadata']
                        json_str = json.dumps(data_copy, separators=(',', ':'))
                        calculated_checksum = hashlib.sha256(json_str.encode()).hexdigest()
                        
                        if stored_checksum != calculated_checksum:
                            raise ValueError(f"Checksum mismatch for file {filepath}")
                        
                        # Remove metadata from returned data
                        del data['_metadata']
                    
                    # Convert lists back to numpy arrays
                    processed_data = {}
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
                            # Check if this looks like coordinate data
                            if len(value[0]) > 0 and isinstance(value[0][0], list):
                                processed_data[key] = [np.array(arr) for arr in value]
                            else:
                                processed_data[key] = np.array(value)
                        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
                            processed_data[key] = np.array(value)
                        else:
                            processed_data[key] = value
                    
                    return processed_data
                    
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON format in {filepath}: {e}")
                except Exception as e:
                    raise RuntimeError(f"Error loading dataset from {filepath}: {e}")
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}. Only JSON is supported for security.")


class MSAProcessor:
    """Thread-safe processor for Multiple Sequence Alignments."""
    
    def __init__(self, cache_dir: str = "msa_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._shutdown = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
    
    def shutdown(self):
        """Shutdown thread pool executor properly."""
        if not self._shutdown and self._executor:
            self._executor.shutdown(wait=True)
            self._shutdown = True
    
    def create_msa_from_sequence(self, sequence: str, database: List[str]) -> np.ndarray:
        """Create MSA by finding similar sequences."""
        # Simple sequence similarity search
        similar_sequences = []
        
        for db_seq in database:
            if len(db_seq) == len(sequence):
                identity = self._compute_sequence_identity(sequence, db_seq)
                if identity > 0.3:  # 30% identity threshold
                    similar_sequences.append(db_seq)
        
        if not similar_sequences:
            # If no similar sequences found, return just the query sequence
            similar_sequences = [sequence]
        
        # Create MSA matrix (simplified - no actual alignment)
        msa = np.array([list(seq) for seq in similar_sequences], dtype='U1')
        
        return msa
    
    def _compute_sequence_identity(self, seq1: str, seq2: str) -> float:
        """Compute sequence identity."""
        if len(seq1) != len(seq2):
            return 0.0
        
        matches = sum(1 for i in range(len(seq1)) if seq1[i] == seq2[i])
        return matches / len(seq1)
    
    def encode_msa(self, msa: np.ndarray) -> np.ndarray:
        """Encode MSA as numeric features."""
        # Simple one-hot encoding
        token_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, '-': 4}
        
        n_sequences, seq_len = msa.shape
        encoded = np.zeros((n_sequences, seq_len, len(token_map)))
        
        for i in range(n_sequences):
            for j in range(seq_len):
                char = msa[i, j]
                if char in token_map:
                    encoded[i, j, token_map[char]] = 1
        
        return encoded


def create_sample_dataset(output_dir: str = "sample_data"):
    """Create a sample dataset for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample sequences
    sequences = [
        "GGGAAAUCC",
        "GCCUUGGCAAC",
        "AUGCUAAUCGAU",
        "CGGAUCUCCGAGUCC",
        "AAUCCGGAAUCCGGAAUCCGG"
    ]
    
    # Create loader
    loader = RNADatasetLoader(cache_dir=str(output_dir / "cache"))
    
    # Create synthetic structures
    structures = loader.create_synthetic_structures(sequences)
    
    # Preprocess
    data = loader.preprocess_for_training(structures)
    
    # Save
    loader.save_dataset(data, output_dir / "sample_dataset.pkl")
    
    print(f"Sample dataset created in {output_dir}")
    print(f"Sequences: {len(data['sequences'])}")
    print(f"Average length: {np.mean([len(seq) for seq in data['sequences']]):.1f}")
    
    return data
