def _compute_fallback_embedding(self, tokens: List[int], seq_length: int) -> np.ndarray:
        """Compute fallback embedding with sequence-specific features."""
        # Create embedding based on sequence composition
        composition = np.zeros(4)
        for token in tokens: