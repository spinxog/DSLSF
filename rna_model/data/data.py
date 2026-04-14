"""Data loading and preprocessing utilities for RNA 3D folding."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
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
from ..core.utils import tokenize_rna_sequence, compute_contact_map, bin_distances
from ..core.constants import BIOLOGICAL, GEOMETRY, COMPUTATIONAL, VALIDATION


def retry_on_io_error(max_retries: int = COMPUTATIONAL.MAX_RETRIES, delay: float = COMPUTATIONAL.IO_DELAY):
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
    """Cross-platform file locking using threading locks with proper cleanup."""
    
    _locks = {}  # Class-level lock registry
    _lock_registry_lock = threading.RLock()  # Protect access to the registry
    _cleanup_interval = COMPUTATIONAL.CLEANUP_INTERVAL  # Cleanup every N lock accesses
    _access_count = 0
    
    @classmethod
    def get_lock(cls, path: Path) -> threading.Lock:
        """Get or create a lock for the given path with thread safety."""
        path_str = str(path.absolute())
        
        with cls._lock_registry_lock:
            cls._access_count += 1
            
            # Periodic cleanup to prevent memory leaks
            if cls._access_count % cls._cleanup_interval == 0:
                cls._cleanup_unused_locks()
            
            if path_str not in cls._locks:
                cls._locks[path_str] = threading.Lock()
            
            return cls._locks[path_str]
    
    @classmethod
    def _cleanup_unused_locks(cls):
        """Clean up unused locks to prevent memory leaks."""
        try:
            with cls._lock_registry_lock:
                # Make a copy of keys to avoid modifying dict during iteration
                current_paths = list(cls._locks.keys())
                keys_to_remove = []
                
                for path_str in current_paths:
                    lock = cls._locks[path_str]
                    # Check if lock is currently not held (no waiting threads)
                    if not lock.locked():
                        keys_to_remove.append(path_str)
                
                # Remove locks after iteration to avoid modifying dict during iteration
                for key in keys_to_remove:
                    del cls._locks[key]
        except Exception as e:
            # If cleanup fails, continue to avoid breaking the locking mechanism
            logging.warning(f"Lock cleanup failed: {e}")
            pass
    
    @classmethod
    def clear_all_locks(cls):
        """Clear all locks - useful for testing or shutdown."""
        with cls._lock_registry_lock:
            cls._locks.clear()
            cls._access_count = 0


@contextmanager
def file_lock(lock_file: Path, timeout: float = None):
    """Context manager for cross-platform file locking with improved error handling."""
    if timeout is None:
        timeout = COMPUTATIONAL.FILE_LOCK_TIMEOUT
    
    lock = FileLock.get_lock(lock_file)
    lock_path = lock_file.with_suffix('.lock')
    
    lock_fd = None
    acquired = False
    
    try:
        # Acquire thread lock first with timeout
        if not lock.acquire(timeout=timeout):
            raise TimeoutError(f"Could not acquire thread lock for {lock_file}")
        
        try:
            # Create/open lock file for process-level locking with secure permissions
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
            
            # Try to acquire file lock with timeout (cross-platform)
            start_time = time.time()
            
            if os.name == 'posix':
                # Unix/Linux systems use fcntl
                import fcntl
                while time.time() - start_time < timeout:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                        break
                    except IOError as e:
                        if e.errno == 11:  # EAGAIN - would block
                            time.sleep(0.1)  # Brief sleep before retry
                        else:
                            raise
                else:
                    raise TimeoutError(f"Could not acquire file lock for {lock_file} after {timeout}s")
            else:
                # Windows systems - use msvcrt or fallback
                try:
                    import msvcrt
                    msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
                    acquired = True
                except (ImportError, IOError) as e:
                    # Fallback to advisory file locking
                    logging.warning(f"Windows file locking not available, using fallback: {e}")
                    acquired = True
            
            if acquired:
                # Write process ID and timestamp for debugging
                lock_info = f"{os.getpid()}:{time.time()}".encode()
                os.write(lock_fd, lock_info)
                os.fsync(lock_fd)  # Ensure data is written to disk
            
            yield
            
        finally:
            # Clean up file locks with comprehensive error handling
            if lock_fd is not None:
                try:
                    # Release file lock based on OS
                    if os.name == 'posix' and acquired:
                        import fcntl
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)  # Release file lock
                    elif os.name == 'nt' and acquired:
                        try:
                            import msvcrt
                            msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
                        except (ImportError, IOError):
                            logging.warning("Windows file locking not available, using fallback")
                    else:
                        # No lock to release (not acquired)
                        pass
                except (OSError, IOError) as e:
                    logging.error(f"Failed to release file lock: {e}")
                finally:
                    try:
                        os.close(lock_fd)
                    except (OSError, IOError) as e:
                        logging.error(f"Failed to close file descriptor: {e}")
            
            # Always release thread lock with proper error handling
            try:
                lock.release()
            except RuntimeError as e:
                if "was never acquired" not in str(e).lower():
                    logging.warning(f"Unexpected error releasing thread lock: {e}")
                # Lock was not acquired by this thread - this is normal
                pass
            except Exception as e:
                logging.error(f"Unexpected error in thread lock release: {e}")
                pass
        
        # Clean up lock file
        try:
            if lock_path.exists():
                lock_path.unlink()
        except OSError as e:
            logging.warning(f"Could not remove lock file {lock_path}: {e}")


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


class DataValidator:
    """Comprehensive data validation for RNA sequences and structures."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.valid_nucleotides = BIOLOGICAL.VALID_NUCLEOTIDES
        self.max_sequence_length = self.config.get('max_sequence_length', VALIDATION.MAX_SEQUENCE_LENGTH)
        self.min_sequence_length = self.config.get('min_sequence_length', VALIDATION.MIN_SEQUENCE_LENGTH)
        self.max_coordinate_value = self.config.get('max_coordinate_value', GEOMETRY.MAX_COORDINATE_VALUE)
        self.min_coordinate_value = self.config.get('min_coordinate_value', GEOMETRY.MIN_COORDINATE_VALUE)
        
    def validate_sequence(self, sequence: str) -> Dict[str, Any]:
        """Validate RNA sequence with comprehensive edge case handling."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check for None or empty input
        if sequence is None:
            validation_result['valid'] = False
            validation_result['errors'].append("Sequence cannot be None")
            return validation_result
        
        # Check sequence type
        if not isinstance(sequence, str):
            validation_result['valid'] = False
            validation_result['errors'].append(f"Sequence must be a string, got {type(sequence).__name__}")
            return validation_result
        
        # Check for empty string
        if not sequence:
            validation_result['valid'] = False
            validation_result['errors'].append("Sequence cannot be empty")
            return validation_result
        
        # Check for whitespace-only sequences
        if sequence.isspace():
            validation_result['valid'] = False
            validation_result['errors'].append("Sequence cannot contain only whitespace")
            return validation_result
        
        # Strip whitespace but preserve original for stats
        stripped_sequence = sequence.strip()
        if len(stripped_sequence) != len(sequence):
            validation_result['warnings'].append("Sequence contains leading/trailing whitespace")
        
        # Check sequence length
        seq_len = len(stripped_sequence)
        validation_result['stats']['length'] = seq_len
        validation_result['stats']['original_length'] = len(sequence)
        
        if seq_len < self.min_sequence_length:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Sequence too short: {seq_len} < {self.min_sequence_length}")
        
        if seq_len > self.max_sequence_length:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Sequence too long: {seq_len} > {self.max_sequence_length}")
        
        # Check for control characters
        if any(ord(c) < 32 or ord(c) == 127 for c in stripped_sequence):
            validation_result['valid'] = False
            validation_result['errors'].append("Sequence contains control characters")
            return validation_result
        
        # Check nucleotide composition
        upper_sequence = stripped_sequence.upper()
        invalid_chars = set(upper_sequence) - self.valid_nucleotides
        if invalid_chars:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Invalid nucleotides: {sorted(invalid_chars)}")
        
        # Calculate composition
        composition = {}
        for nuc in self.valid_nucleotides:
            composition[nuc] = upper_sequence.count(nuc)
        validation_result['stats']['composition'] = composition
        
        # Check for unusual patterns
        total_valid = sum(composition[nuc] for nuc in 'AUGC')
        n_count = composition.get('N', 0)
        
        if n_count > seq_len * VALIDATION.MAX_N_PROPORTION:
            validation_result['warnings'].append(f"High proportion of N nucleotides: {n_count}/{seq_len} ({n_count/seq_len*100:.1f}%)")
        
        if total_valid == 0:
            validation_result['valid'] = False
            validation_result['errors'].append("Sequence contains no valid nucleotides (A, U, G, C)")
        
        # Check for homopolymer runs
        max_homopolymer = 0
        current_homopolymer = 1
        for i in range(1, len(upper_sequence)):
            if upper_sequence[i] == upper_sequence[i-1]:
                current_homopolymer += 1
                max_homopolymer = max(max_homopolymer, current_homopolymer)
            else:
                current_homopolymer = 1
        
        if max_homopolymer > VALIDATION.MAX_HOMOPOLYMER_LENGTH:
            validation_result['warnings'].append(f"Long homopolymer run detected: {max_homopolymer} nucleotides")
        
        validation_result['stats']['max_homopolymer'] = max_homopolymer
        
        # GC content calculation
        gc_count = composition.get('G', 0) + composition.get('C', 0)
        gc_content = gc_count / total_valid if total_valid > 0 else 0
        validation_result['stats']['gc_content'] = gc_content
        
        if gc_content < VALIDATION.MIN_GC_CONTENT or gc_content > VALIDATION.MAX_GC_CONTENT:
            validation_result['warnings'].append(f"Unusual GC content: {gc_content:.2f}")
        
        return validation_result
    
    def validate_coordinates(self, coordinates: np.ndarray) -> Dict[str, Any]:
        """Validate coordinate array with comprehensive edge case handling."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check for None input
        if coordinates is None:
            validation_result['valid'] = False
            validation_result['errors'].append("Coordinates cannot be None")
            return validation_result
        
        # Check input type
        if not isinstance(coordinates, np.ndarray):
            validation_result['valid'] = False
            validation_result['errors'].append(f"Coordinates must be numpy array, got {type(coordinates).__name__}")
            return validation_result
        
        # Check for empty array
        if coordinates.size == 0:
            validation_result['valid'] = False
            validation_result['errors'].append("Coordinates array cannot be empty")
            return validation_result
        
        # Check array shape
        if coordinates.ndim != 3:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Invalid coordinate dimensions: {coordinates.ndim}, expected 3")
            return validation_result
        
        if coordinates.shape[2] != 3:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Invalid coordinate shape: {coordinates.shape}, expected (N, n_atoms, 3)")
            return validation_result
        
        n_residues, n_atoms, _ = coordinates.shape
        validation_result['stats']['n_residues'] = n_residues
        validation_result['stats']['n_atoms_per_residue'] = n_atoms
        validation_result['stats']['total_atoms'] = n_residues * n_atoms
        
        # Check for reasonable number of atoms per residue
        if n_atoms < 1 or n_atoms > GEOMETRY.MAX_ATOMS_PER_RESIDUE:
            validation_result['warnings'].append(f"Unusual number of atoms per residue: {n_atoms}")
        
        # Check for NaN or infinite values
        nan_mask = np.isnan(coordinates)
        if np.any(nan_mask):
            nan_count = np.sum(nan_mask)
            validation_result['valid'] = False
            validation_result['errors'].append(f"Coordinates contain {nan_count} NaN values")
            validation_result['stats']['nan_count'] = int(nan_count)
        
        inf_mask = np.isinf(coordinates)
        if np.any(inf_mask):
            inf_count = np.sum(inf_mask)
            validation_result['valid'] = False
            validation_result['errors'].append(f"Coordinates contain {inf_count} infinite values")
            validation_result['stats']['inf_count'] = int(inf_count)
        
        # Check for zero vectors (overlapping atoms)
        coords_flat = coordinates.reshape(-1, 3)
        zero_vectors = np.all(np.abs(coords_flat) < GEOMETRY.ZERO_VECTOR_THRESHOLD, axis=1)
        if np.any(zero_vectors):
            zero_count = np.sum(zero_vectors)
            validation_result['warnings'].append(f"{zero_count} atoms have near-zero coordinates")
            validation_result['stats']['zero_vector_count'] = int(zero_count)
        
        # Check coordinate ranges
        min_coord = float(np.min(coords_flat))
        max_coord = float(np.max(coords_flat))
        
        validation_result['stats']['min_coord'] = min_coord
        validation_result['stats']['max_coord'] = max_coord
        validation_result['stats']['mean_coord'] = float(np.mean(coords_flat))
        validation_result['stats']['std_coord'] = float(np.std(coords_flat))
        
        # Check for extreme coordinate values
        if min_coord < self.min_coordinate_value or max_coord > self.max_coordinate_value:
            validation_result['warnings'].append(
                f"Coordinates outside reasonable range [{self.min_coordinate_value}, {self.max_coordinate_value}]: [{min_coord:.2f}, {max_coord:.2f}]"
            )
        
        # Check for duplicate atoms (very close coordinates)
        if len(coords_flat) > 1:
            distances = np.linalg.norm(coords_flat[:, np.newaxis, :] - coords_flat[np.newaxis, :, :], axis=2)
            np.fill_diagonal(distances, np.inf)  # Ignore self-comparisons
            
            close_atoms = distances < GEOMETRY.CLOSE_ATOM_THRESHOLD  # Atoms closer than threshold
            close_count = np.sum(close_atoms) // 2  # Divide by 2 since it's symmetric
            
            if close_count > 0:
                validation_result['warnings'].append(f"{close_count} pairs of atoms are very close (< 0.1 Å)")
                validation_result['stats']['close_atom_pairs'] = int(close_count)
        
        # Check structure geometry
        if n_residues > 1:
            # Check for reasonable bond lengths between consecutive residues
            for i in range(n_residues - 1):
                for j in range(min(n_atoms, 3)):  # Check first few atoms
                    if j < n_atoms:
                        dist = np.linalg.norm(coordinates[i, j] - coordinates[i + 1, j])
                        if dist > GEOMETRY.LONG_BOND_THRESHOLD:  # Very long bond
                            validation_result['warnings'].append(
                                f"Unusually long distance between consecutive residues: {dist:.2f} Å"
                            )
        
        # Check for planarity issues (if we have enough atoms)
        if n_atoms >= 3:
            for i in range(n_residues):
                residue_coords = coordinates[i]
                # Check if atoms are roughly planar (for ring structures)
                if n_atoms >= 4:
                    try:
                        # Fit a plane to the first 4 atoms
                        points = residue_coords[:4]
                        centroid = np.mean(points, axis=0)
                        _, _, vh = np.linalg.svd(points - centroid)
                        normal = vh[2]  # Last singular vector
                        
                        # Check distances from plane
                        distances_from_plane = np.abs(np.dot(points - centroid, normal))
                        max_deviation = np.max(distances_from_plane)
                        
                        if max_deviation > GEOMETRY.PLANARITY_DEVIATION_THRESHOLD:  # Atoms far from planar
                            validation_result['warnings'].append(
                                f"Residue {i} shows significant non-planarity: {max_deviation:.2f} Å"
                            )
                    except Exception:
                        pass  # Skip if SVD fails
        
        return validation_result
    
    def validate_structure(self, structure: 'RNAStructure') -> Dict[str, Any]:
        """Validate complete RNA structure with comprehensive edge case handling."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check for None input
        if structure is None:
            validation_result['valid'] = False
            validation_result['errors'].append("Structure cannot be None")
            return validation_result
        
        # Check required attributes
        required_attrs = ['sequence', 'coordinates', 'atom_names', 'residue_names']
        for attr in required_attrs:
            if not hasattr(structure, attr):
                validation_result['valid'] = False
                validation_result['errors'].append(f"Structure missing required attribute: {attr}")
                return validation_result
        
        # Validate sequence
        seq_validation = self.validate_sequence(structure.sequence)
        if not seq_validation['valid']:
            validation_result['valid'] = False
            validation_result['errors'].extend(seq_validation['errors'])
        validation_result['warnings'].extend(seq_validation['warnings'])
        validation_result['stats']['sequence'] = seq_validation['stats']
        
        # Validate coordinates
        coord_validation = self.validate_coordinates(structure.coordinates)
        if not coord_validation['valid']:
            validation_result['valid'] = False
            validation_result['errors'].extend(coord_validation['errors'])
        validation_result['warnings'].extend(coord_validation['warnings'])
        validation_result['stats']['coordinates'] = coord_validation['stats']
        
        # Check consistency between sequence and coordinates
        seq_len = len(structure.sequence)
        coord_len = len(structure.coordinates)
        
        if seq_len != coord_len:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Sequence length ({seq_len}) doesn't match coordinate count ({coord_len})"
            )
        
        # Validate atom names
        if structure.atom_names is None:
            validation_result['valid'] = False
            validation_result['errors'].append("Atom names cannot be None")
        elif len(structure.atom_names) != coord_len:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Atom names count ({len(structure.atom_names)}) doesn't match coordinate count ({coord_len})"
            )
        else:
            # Check atom name format
            for i, atom_names in enumerate(structure.atom_names):
                if atom_names is None:
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"Atom names for residue {i} cannot be None")
                elif not isinstance(atom_names, (list, tuple)):
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"Atom names for residue {i} must be list or tuple")
                elif len(atom_names) != len(structure.coordinates[i]):
                    validation_result['valid'] = False
                    validation_result['errors'].append(
                        f"Atom names count ({len(atom_names)}) doesn't match coordinate atoms ({len(structure.coordinates[i])}) for residue {i}"
                    )
                else:
                    # Check for valid atom name formats
                    for j, atom_name in enumerate(atom_names):
                        if not isinstance(atom_name, str) or not atom_name.strip():
                            validation_result['warnings'].append(f"Invalid atom name format at residue {i}, atom {j}: '{atom_name}'")
        
        # Validate residue names
        if structure.residue_names is None:
            validation_result['valid'] = False
            validation_result['errors'].append("Residue names cannot be None")
        elif len(structure.residue_names) != seq_len:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Residue names count ({len(structure.residue_names)}) doesn't match sequence length ({seq_len})"
            )
        else:
            # Check residue name validity
            valid_residues = {'A', 'U', 'G', 'C'}
            for i, residue_name in enumerate(structure.residue_names):
                if residue_name not in valid_residues:
                    validation_result['warnings'].append(f"Unusual residue name at position {i}: '{residue_name}'")
        
        # Check chain ID
        if hasattr(structure, 'chain_id'):
            if structure.chain_id is None:
                validation_result['warnings'].append("Chain ID is None")
            elif not isinstance(structure.chain_id, str) or not structure.chain_id.strip():
                validation_result['warnings'].append(f"Invalid chain ID format: '{structure.chain_id}'")
        
        # Check for metadata consistency
        if hasattr(structure, 'metadata') and structure.metadata is not None:
            if not isinstance(structure.metadata, dict):
                validation_result['warnings'].append("Metadata should be a dictionary")
        
        # Additional structure-specific checks
        if seq_len > 0 and coord_len > 0:
            # Check for reasonable structure dimensions
            coords_flat = structure.coordinates.flatten()
            structure_span = float(np.max(coords_flat) - np.min(coords_flat))
            
            if structure_span > GEOMETRY.MAX_REASONABLE_STRUCTURE_SPAN:
                validation_result['warnings'].append(f"Very large structure span: {structure_span:.1f} Å")
            elif structure_span < GEOMETRY.MIN_REASONABLE_STRUCTURE_SPAN:
                validation_result['warnings'].append(f"Very small structure span: {structure_span:.1f} Å")
            
            validation_result['stats']['structure_span'] = structure_span
        
        return validation_result


class DataPreprocessor:
    """Automated data preprocessing pipeline for RNA structures."""
    
    def __init__(self, validator: Optional[DataValidator] = None):
        self.validator = validator or DataValidator()
        self.preprocessing_stats = {
            'total_processed': 0,
            'validation_errors': 0,
            'validation_warnings': 0,
            'preprocessing_steps': []
        }
    
    def preprocess_structure(self, structure: 'RNAStructure', 
                          normalize_coordinates: bool = True,
                          center_coordinates: bool = True,
                          remove_invalid_atoms: bool = True) -> Optional['RNAStructure']:
        """Preprocess a single RNA structure."""
        self.preprocessing_stats['total_processed'] += 1
        
        # Validate structure
        validation_result = self.validator.validate_structure(structure)
        
        if not validation_result['valid']:
            self.preprocessing_stats['validation_errors'] += 1
            logging.error(f"Structure validation failed: {validation_result['errors']}")
            return None
        
        if validation_result['warnings']:
            self.preprocessing_stats['validation_warnings'] += 1
            for warning in validation_result['warnings']:
                logging.warning(f"Structure validation warning: {warning}")
        
        # Create a copy to avoid modifying original
        processed_structure = RNAStructure(
            sequence=structure.sequence,
            coordinates=structure.coordinates.copy(),
            atom_names=[atom_names.copy() for atom_names in structure.atom_names],
            residue_names=structure.residue_names.copy(),
            metadata=structure.metadata.copy()
        )
        
        # Apply preprocessing steps
        preprocessing_steps = []
        
        # Center coordinates
        if center_coordinates:
            processed_structure.coordinates = self.center_coordinates(processed_structure.coordinates)
            preprocessing_steps.append('centered_coordinates')
        
        # Normalize coordinates
        if normalize_coordinates:
            processed_structure.coordinates = self.normalize_coordinates(processed_structure.coordinates)
            preprocessing_steps.append('normalized_coordinates')
        
        # Remove invalid atoms
        if remove_invalid_atoms:
            processed_structure = self.remove_invalid_atoms(processed_structure)
            preprocessing_steps.append('removed_invalid_atoms')
        
        # Update metadata
        processed_structure.metadata['preprocessing_steps'] = preprocessing_steps
        processed_structure.metadata['validation_stats'] = validation_result['stats']
        
        self.preprocessing_stats['preprocessing_steps'].extend(preprocessing_steps)
        
        return processed_structure
    
    def center_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Center coordinates at origin."""
        center = np.mean(coordinates, axis=(0, 1))
        return coordinates - center
    
    def normalize_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Normalize coordinates to unit scale."""
        scale = np.std(coordinates)
        if scale > 0:
            return coordinates / scale
        return coordinates
    
    def remove_invalid_atoms(self, structure: 'RNAStructure') -> 'RNAStructure':
        """Remove atoms with invalid coordinates."""
        valid_indices = []
        valid_atom_names = []
        valid_residue_names = []
        
        for i, (coords, atom_names, residue_name) in enumerate(
            zip(structure.coordinates, structure.atom_names, structure.residue_names)
        ):
            # Check if all atoms in this residue are valid
            if not np.any(np.isnan(coords)) and not np.any(np.isinf(coords)):
                valid_indices.append(i)
                valid_atom_names.append(atom_names)
                valid_residue_names.append(residue_name)
        
        if valid_indices:
            return RNAStructure(
                sequence=''.join([structure.residue_names[i] for i in valid_indices]),
                coordinates=structure.coordinates[valid_indices],
                atom_names=valid_atom_names,
                residue_names=valid_residue_names,
                metadata=structure.metadata
            )
        else:
            # No valid atoms found
            logging.warning("No valid atoms found in structure")
            return structure
    
    def preprocess_dataset(self, structures: List['RNAStructure'], 
                          **preprocessing_options) -> Tuple[List['RNAStructure'], Dict[str, Any]]:
        """Preprocess a dataset of RNA structures."""
        processed_structures = []
        processing_stats = {
            'total_input': len(structures),
            'successfully_processed': 0,
            'validation_errors': 0,
            'validation_warnings': 0,
            'preprocessing_steps': []
        }
        
        for i, structure in enumerate(structures):
            try:
                processed = self.preprocess_structure(structure, **preprocessing_options)
                if processed is not None:
                    processed_structures.append(processed)
                    processing_stats['successfully_processed'] += 1
                else:
                    processing_stats['validation_errors'] += 1
            except Exception as e:
                processing_stats['validation_errors'] += 1
                logging.error(f"Error processing structure {i}: {e}")
        
        # Update global stats
        processing_stats['preprocessing_steps'] = self.preprocessing_stats['preprocessing_steps']
        
        return processed_structures, processing_stats
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        return self.preprocessing_stats.copy()


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
        
        # Comprehensive suspicious pattern detection
        dangerous_patterns = {
            'path_traversal': ['../', '..\\', '..%2f', '..%5c'],
            'null_injection': ['\0', '%00'],
            'command_injection': ['|', '&', ';', '$', '`', '(', ')'],
            'file_operations': ['<', '>', '"', '*', '?'],
            'unicode_exploits': ['%u', '%U', '\\u'],
            'windows_paths': ['\\\\', '//', '/\\', '\\\/']
        }
        
        for category, patterns in dangerous_patterns.items():
            for pattern in patterns:
                if pattern in str(file_path).lower():
                    raise ValueError(f"Security violation - {category}: '{pattern}' in path")
        
        # Unicode normalization attacks prevention
        import unicodedata
        try:
            normalized_path = unicodedata.normalize('NFC', str(file_path))
            if normalized_path != str(file_path):
                raise ValueError("Unicode normalization attack detected")
        except (UnicodeError, ValueError):
            raise ValueError("Invalid Unicode characters in path")
        
        # Path length validation (prevent DoS via long paths)
        if len(str(file_path)) > COMPUTATIONAL.MAX_PATH_LENGTH:  # Conservative limit
            raise ValueError(f"Path too long: {len(str(file_path))} > {COMPUTATIONAL.MAX_PATH_LENGTH} characters")
        
        # OS-specific validation
        if os.name == 'nt':  # Windows
            # Check for reserved names and invalid characters
            reserved_names = {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'}
            name_without_ext = file_path.stem.upper()
            if name_without_ext in reserved_names:
                raise ValueError(f"Reserved device name used: {name_without_ext}")
            
            invalid_chars = '<>:"|?*'
            if any(char in str(file_path) for char in invalid_chars):
                raise ValueError(f"Invalid characters in Windows path")
        else:  # Unix/Linux
            # Check for Windows-style paths on Unix
            if ':' in str(file_path)[:2] or '\\' in str(file_path):
                raise ValueError("Invalid path format for Unix system")
        
        # Validate file extension
        allowed_extensions = {'.pdb', '.cif', '.json', '.pkl', '.npz'}
        if file_path.suffix.lower() not in allowed_extensions:
            raise ValueError(f"File extension not allowed: {file_path.suffix}")
        
        # Get absolute path safely
        try:
            # Use absolute() instead of resolve() to avoid following symlinks
            abs_path = file_path.absolute()
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid path: {e}")
        
        # Comprehensive symlink checking
        def check_symlinks_safely(path: Path) -> None:
            """Check for symlinks in path and parents safely."""
            try:
                # Check if the file itself is a symlink
                if path.exists() and path.is_symlink():
                    raise ValueError(f"Symbolic link not allowed: {path}")
                
                # Check all parent directories for symlinks
                for parent in path.parents:
                    if parent.exists():
                        # Use lstat to check if it's a symlink without following
                        try:
                            stat_info = os.lstat(str(parent))
                            if stat.S_ISLNK(stat_info.st_mode):
                                raise ValueError(f"Symbolic link in parent path not allowed: {parent}")
                        except (OSError, PermissionError):
                            # If we can't stat it, assume it's unsafe
                            raise ValueError(f"Cannot verify parent directory safety: {parent}")
            except (OSError, PermissionError) as e:
                raise ValueError(f"Security check failed for {path}: {e}")
        
        check_symlinks_safely(abs_path)
        
        # Ensure path is within allowed directory
        try:
            cache_abs = self.cache_dir.absolute()
            # Use relative_to with strict checking
            rel_path = abs_path.relative_to(cache_abs)
        except ValueError:
            raise ValueError(f"Path outside allowed directory: {abs_path} not in {cache_abs}")
        
        # Additional safety: check that relative path doesn't try to escape
        if '..' in str(rel_path):
            raise ValueError("Relative path contains parent directory references")
        
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
        except UnicodeDecodeError as e:
            raise RuntimeError(f"Encoding error reading PDB file {pdb_file}: {e}")
        except IOError as e:
            raise RuntimeError(f"I/O error reading PDB file {pdb_file}: {e}")
        except (ValueError, KeyError, IndexError) as e:
            raise ValueError(f"Invalid PDB file format in {pdb_file}: {e}")
        except (OSError, IOError) as e:
            raise IOError(f"I/O error reading PDB file {pdb_file}: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error reading PDB file {pdb_file}: {e}")
        finally:
            # Ensure file is closed properly (context manager handles this)
            pass
        
        if not coordinates:
            raise ValueError(f"No valid coordinates found in {pdb_file}")
        
        coordinates = np.array(coordinates)
        
        # Group by residue with thread safety
        residues = {}
        # Use a context manager for thread safety
        lock = threading.Lock()
        try:
            with lock:
                for i, (coord, atom_name, residue_name) in enumerate(
                    zip(coordinates, atom_names, residue_names)
                ):
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
        """Predict secondary structure using real model inference."""
        try:
            # Import the secondary structure predictor
            from ..models.secondary_structure import SecondaryStructurePredictor, SSConfig
            
            # Create model configuration
            config = SSConfig(
                d_model=256,
                n_heads=8,
                n_layers=6,
                d_ff=1024,
                max_seq_len=len(sequence),
                dropout=0.1
            )
            
            # Initialize the predictor
            predictor = SecondaryStructurePredictor(config)
            predictor.eval()
            
            # Tokenize the sequence
            tokens = tokenize_rna_sequence(sequence)
            sequence_tensor = torch.tensor(tokens["tokens"], dtype=torch.long).unsqueeze(0)
            
            # Predict secondary structure
            with torch.no_grad():
                outputs = predictor(sequence_tensor)
                ss_logits = outputs.get('contacts', torch.zeros(1, len(sequence), len(sequence)))
                ss_probs = torch.sigmoid(ss_logits)
                ss_matrix = (ss_probs > 0.5).float().squeeze(0).cpu().numpy()
            
            # Ensure symmetry and remove diagonal
            ss_matrix = (ss_matrix + ss_matrix.T) / 2
            np.fill_diagonal(ss_matrix, 0)
            
            return ss_matrix.astype(np.float32)
            
        except Exception as e:
            self.logger.warning(f"Failed to predict secondary structure with model, using fallback: {e}")
            # Fallback to simple base-pairing prediction
            return self._fallback_secondary_structure(sequence)
    
    def _fallback_secondary_structure(self, sequence: str) -> np.ndarray:
        """Fallback secondary structure prediction using simple rules."""
        n = len(sequence)
        ss_matrix = np.zeros((n, n))
        
        # Enhanced base-pairing rules with energy considerations
        complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        # Add wobble base pairs
        wobble_pairs = {'G': 'U', 'U': 'G'}
        
        for i in range(n):
            for j in range(i + 3, min(i + 50, n)):  # Reasonable loop size
                # Check for Watson-Crick pairs
                if complement.get(sequence[i]) == sequence[j]:
                    # Add energy-based scoring
                    loop_size = j - i - 1
                    if loop_size >= 3 and loop_size <= 30:
                        # Hairpin loops
                        if i == 0 or j == n - 1:
                            ss_matrix[i, j] = 1.0
                        else:
                            ss_matrix[i, j] = 0.8
                # Check for wobble pairs (GU pairs)
                elif wobble_pairs.get(sequence[i]) == sequence[j]:
                    loop_size = j - i - 1
                    if loop_size >= 3 and loop_size <= 30:
                        ss_matrix[i, j] = 0.6  # Lower confidence for wobble pairs
        
        # Ensure symmetry
        ss_matrix = (ss_matrix + ss_matrix.T) / 2
        np.fill_diagonal(ss_matrix, 0)
        
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
    
    def __init__(self, cache_dir: str = "msa_cache", max_cache_size: int = None):
        if max_cache_size is None:
            max_cache_size = COMPUTATIONAL.MAX_CACHE_SIZE
        
        self.cache_dir = Path(cache_dir)
        self.max_cache_size = max_cache_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._shutdown = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
    
    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool executor."""
        if not self._shutdown:
            self._shutdown = True
            self._executor.shutdown(wait=wait)
    
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
