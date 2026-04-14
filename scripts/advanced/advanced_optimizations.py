if self.model_type == 'real':
                embedding = self._compute_real_embedding(tokens)
            else:
                raise RuntimeError(f"Real embedding model not available for sequence: {seq[:10]}...")