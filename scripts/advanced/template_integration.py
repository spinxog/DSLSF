def _create_template_embeddings(self, template_coords: torch.Tensor) -> torch.Tensor:
        """Create embeddings from template coordinates using geometric features."""
        batch_size = 1
        seq_len = template_coords.shape[0]
        
        # Compute geometric features
        # 1. Distance matrix (pairwise distances between residues)
        distance_matrix = torch.cdist(template_coords, template_coords)
        
        # 2. Local geometry features
        local_features = self._compute_local_geometry_features(template_coords)
        
        # 3. Global shape features
        global_features = self._compute_global_shape_features(template_coords)
        
        # Combine features
        feature_dim = 128
        embeddings = torch.zeros(batch_size, seq_len, feature_dim)
        
        for i in range(seq_len):
            # Distance-based features (average distance to other residues)
            avg_distance = torch.mean(distance_matrix[i])
            embeddings[0, i, :32] = self._encode_distance_feature(avg_distance)
            
            # Local geometry features
            embeddings[0, i, 32:64] = local_features[i]
            
            # Global shape features
            embeddings[0, i, 64:96] = global_features
            
            # Position encoding
            embeddings[0, i, 96:128] = self._encode_position(i, seq_len)
        
        return embeddings
    
    def _encode_distance_feature(self, distance: float) -> torch.Tensor:
        """Encode distance feature."""
        # Use logarithmic encoding for distances
        log_distance = torch.log(distance + 1.0)
        # Normalize to reasonable range
        return torch.clamp(log_distance / 10.0, -1.0, 1.0) * 32
    
    def _compute_local_geometry_features(self, coords: torch.Tensor) -> List[torch.Tensor]:
        """Compute local geometry features for each residue."""
        seq_len = coords.shape[0]
        local_features = []
        
        for i in range(seq_len):
            # Look at local neighborhood (window of 5 residues)
            window_start = max(0, i - 2)
            window_end = min(seq_len, i + 3)
            
            local_coords = coords[window_start:window_end]
            
            if len(local_coords) >= 2:
                # Compute bond angles
                angles = []
                for j in range(len(local_coords) - 1):
                    v1 = local_coords[j] - local_coords[j + 1]
                    v2 = local_coords[j + 2] - local_coords[j + 1] if j + 2 < len(local_coords) else local_coords[0]
                    
                    # Compute angle
                    cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
                    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                    angles.append(torch.acos(cos_angle))
                
                # Average angle
                avg_angle = torch.mean(torch.stack(angles)) if angles else torch.tensor(0.0)
                local_features.append(avg_angle)
            else:
                local_features.append(torch.tensor(0.0))
        
        return local_features
    
    def _compute_global_shape_features(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute global shape features."""
        seq_len = coords.shape[0]
        
        # Principal component analysis of coordinates
        coords_flat = coords.view(seq_len, -1)
        
        # Compute covariance matrix
        cov_matrix = torch.cov(coords_flat.T)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        # Take top 3 eigenvalues (principal components)
        top_eigenvalues = eigenvalues[-3:]  # Sort ascending, take largest
        
        return top_eigenvalues
    
    def _encode_position(self, pos: int, seq_len: int) -> torch.Tensor:
        """Encode position using sinusoidal encoding."""
        # Use sinusoidal position encoding
        encoding = torch.zeros(32)
        
        for i in range(16):
            encoding[2*i] = torch.sin(2.0 * np.pi * pos / seq_len * (i + 1))
            encoding[2*i + 1] = torch.cos(2.0 * np.pi * pos / seq_len * (i + 1))
        
        return encoding