def generate_fallback_coordinates(self, n_residues: int) -> np.ndarray:
        """Fallback coordinates are not allowed in ML training."""
        raise RuntimeError("Fallback coordinate generation is not allowed. Use real model predictions.")