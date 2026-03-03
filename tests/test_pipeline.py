"""Test suite for RNA 3D folding pipeline."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rna_model import RNAFoldingPipeline, PipelineConfig
from rna_model.language_model import RNALanguageModel, LMConfig
from rna_model.secondary_structure import SecondaryStructurePredictor, SSConfig
from rna_model.geometry_module import GeometryModule, GeometryConfig
from rna_model.utils import tokenize_rna_sequence, compute_tm_score


class TestRNALanguageModel:
    """Test RNA language model functionality."""
    
    def test_initialization(self):
        """Test model initialization."""
        config = LMConfig(d_model=256, n_layers=4, n_heads=4)
        model = RNALanguageModel(config)
        
        assert model.config.d_model == 256
        assert model.config.n_layers == 4
        assert model.config.n_heads == 4
        assert len(model.layers) == 4
    
    def test_forward_pass(self):
        """Test forward pass."""
        config = LMConfig(d_model=128, n_layers=2, n_heads=4)
        model = RNALanguageModel(config)
        
        # Create dummy input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 5, (batch_size, seq_len))
        
        # Forward pass
        outputs = model(input_ids)
        
        assert "embeddings" in outputs
        assert "logits" in outputs
        assert outputs["embeddings"].shape == (batch_size, seq_len, 128)
        assert outputs["logits"].shape == (batch_size, seq_len, 5)
    
    def test_contact_prediction(self):
        """Test contact prediction head."""
        config = LMConfig(d_model=128, n_layers=2)
        model = RNALanguageModel(config)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 5, (batch_size, seq_len))
        
        outputs = model(input_ids, return_contacts=True)
        
        assert "contacts" in outputs
        assert outputs["contacts"].shape == (batch_size, seq_len, seq_len)
    
    def test_mask_creation(self):
        """Test span mask creation."""
        config = LMConfig(d_model=128)
        model = RNALanguageModel(config)
        
        batch_size, seq_len = 2, 20
        mask, labels = model.create_span_mask(seq_len, batch_size, torch.device("cpu"))
        
        assert mask.shape == (batch_size, seq_len)
        assert labels.shape == (batch_size, seq_len)
        assert (mask == 0).sum() > 0  # Some tokens should be masked


class TestSecondaryStructure:
    """Test secondary structure predictor."""
    
    def test_initialization(self):
        """Test predictor initialization."""
        config = SSConfig(d_model=128, n_layers=2)
        predictor = SecondaryStructurePredictor(config)
        
        assert predictor.config.d_model == 128
        assert predictor.config.n_layers == 2
        assert predictor.config.n_hypotheses == 3
    
    def test_forward_pass(self):
        """Test forward pass."""
        config = SSConfig(d_model=128, n_layers=2)
        predictor = SecondaryStructurePredictor(config)
        
        batch_size, seq_len, d_model = 2, 10, 128
        embeddings = torch.randn(batch_size, seq_len, d_model)
        
        outputs = predictor(embeddings)
        
        assert "contact_logits" in outputs
        assert "pseudoknot_logits" in outputs
        assert "pair_repr" in outputs
        assert outputs["contact_logits"].shape == (batch_size, seq_len, seq_len, 64)
    
    def test_hypothesis_sampling(self):
        """Test hypothesis sampling."""
        config = SSConfig(d_model=128, n_hypotheses=3)
        predictor = SecondaryStructurePredictor(config)
        
        batch_size, seq_len = 1, 10
        contact_logits = torch.randn(batch_size, seq_len, seq_len, 64)
        pseudoknot_logits = torch.randn(batch_size, seq_len, seq_len, 3)
        
        hypotheses = predictor.sample_hypotheses(contact_logits, pseudoknot_logits)
        
        assert len(hypotheses) == batch_size
        assert len(hypotheses[0]) == 3  # n_hypotheses
        
        for hyp in hypotheses[0]:
            assert "contact_probs" in hyp
            assert "pseudoknot_probs" in hyp
            assert "confidence" in hyp


class TestGeometryModule:
    """Test geometry module."""
    
    def test_initialization(self):
        """Test module initialization."""
        config = GeometryConfig(d_model=128, n_layers=2)
        module = GeometryModule(config)
        
        assert module.config.d_model == 128
        assert module.config.n_layers == 2
        assert len(module.blocks) == 2
    
    def test_forward_pass(self):
        """Test forward pass."""
        config = GeometryConfig(d_model=128, n_layers=2)
        module = GeometryModule(config)
        
        batch_size, seq_len, d_model = 2, 10, 128
        seq_repr = torch.randn(batch_size, seq_len, d_model)
        pair_repr = torch.randn(batch_size, seq_len, seq_len, d_model)
        
        outputs = module(seq_repr, pair_repr)
        
        assert "coordinates" in outputs
        assert "frames" in outputs
        assert "distance_logits" in outputs
        assert "angle_logits" in outputs
        assert "torsion_logits" in outputs
        assert "pucker_logits" in outputs
        assert "confidence" in outputs
        
        # Check coordinate shapes
        coords = outputs["coordinates"]
        assert coords.shape == (batch_size, seq_len, 3, 3)  # n_atoms_per_residue=3


class TestPipeline:
    """Test complete pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        config = PipelineConfig(device="cpu")
        pipeline = RNAFoldingPipeline(config)
        
        assert pipeline.config.device == "cpu"
        assert pipeline.model is not None
        assert pipeline.sampler is not None
        assert pipeline.refiner is not None
    
    def test_single_sequence_prediction(self):
        """Test single sequence prediction."""
        config = PipelineConfig(device="cpu")
        pipeline = RNAFoldingPipeline(config)
        
        sequence = "GGGAAAUCC"
        result = pipeline.predict_single_sequence(sequence)
        
        assert "sequence" in result
        assert "coordinates" in result
        assert "n_decoys" in result
        assert "n_residues" in result
        
        assert result["sequence"] == sequence
        assert result["n_residues"] == len(sequence)
        assert result["n_decoys"] == 5
        
        # Check coordinate shape: (5 * n_residues, 3)
        coords = result["coordinates"]
        expected_shape = (5 * len(sequence), 3)
        assert coords.shape == expected_shape
    
    def test_batch_prediction(self):
        """Test batch prediction."""
        config = PipelineConfig(device="cpu")
        pipeline = RNAFoldingPipeline(config)
        
        sequences = ["GGGAAAUCC", "GCCUUGGCAAC"]
        results = pipeline.predict_batch(sequences)
        
        assert len(results) == len(sequences)
        
        for i, result in enumerate(results):
            assert "sequence" in result
            assert result["sequence"] == sequences[i]
            assert "coordinates" in result
    
    def test_sequence_validation(self):
        """Test sequence validation."""
        config = PipelineConfig(device="cpu")
        pipeline = RNAFoldingPipeline(config)
        
        # Valid sequence
        valid_seq = "AUGC"
        result = pipeline.predict_single_sequence(valid_seq)
        assert "coordinates" in result
        
        # Invalid sequence (too long)
        long_seq = "A" * 1000
        with pytest.raises(ValueError):
            pipeline.predict_single_sequence(long_seq)
    
    def test_competition_mode(self):
        """Test competition mode enabling."""
        config = PipelineConfig(device="cpu")
        pipeline = RNAFoldingPipeline(config)
        
        pipeline.enable_competition_mode()
        
        # Check that model is in eval mode
        assert not pipeline.model.training
        
        # Check that gradients are disabled
        for param in pipeline.model.parameters():
            assert not param.requires_grad


class TestUtils:
    """Test utility functions."""
    
    def test_tokenization(self):
        """Test RNA sequence tokenization."""
        sequence = "AUGC"
        tokens = tokenize_rna_sequence(sequence)
        
        expected = torch.tensor([0, 1, 2, 3])  # A, U, G, C
        assert torch.equal(tokens, expected)
    
    def test_tokenization_with_n(self):
        """Test tokenization with unknown nucleotides."""
        sequence = "AUNGC"
        tokens = tokenize_rna_sequence(sequence)
        
        # N should be tokenized as 4
        expected = torch.tensor([0, 1, 4, 2, 3])
        assert torch.equal(tokens, expected)
    
    def test_tm_score_computation(self):
        """Test TM-score computation."""
        # Identical coordinates should have TM-score = 1
        coords1 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        coords2 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        
        tm_score = compute_tm_score(coords1, coords2)
        assert abs(tm_score - 1.0) < 1e-6
        
        # Different coordinates should have TM-score < 1
        coords3 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        tm_score = compute_tm_score(coords1, coords3)
        assert tm_score < 1.0
        assert tm_score > 0.0


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        config = PipelineConfig(device="cpu")
        pipeline = RNAFoldingPipeline(config)
        
        # Test multiple sequences of varying lengths
        sequences = [
            "AUGC",           # 4 nt
            "GGGAAAUCC",     # 9 nt
            "GCCUUGGCAAC",   # 10 nt
            "AUGCUAAUCGAU",  # 12 nt
        ]
        
        results = pipeline.predict_batch(sequences)
        
        # Check all sequences processed
        assert len(results) == len(sequences)
        
        # Check each result has required fields
        for result in results:
            assert "coordinates" in result
            assert "sequence" in result
            assert "n_decoys" in result
            assert "n_residues" in result
            
            # Check coordinate dimensions
            coords = result["coordinates"]
            expected_coords = 5 * result["n_residues"]
            assert coords.shape == (expected_coords, 3)
            
            # Check no NaN values
            assert not np.isnan(coords).any()
    
    def test_memory_usage(self):
        """Test memory usage monitoring."""
        from rna_model.utils import memory_usage
        
        config = PipelineConfig(device="cpu")
        pipeline = RNAFoldingPipeline(config)
        
        # Process a sequence
        sequence = "GGGAAAUCC"
        pipeline.predict_single_sequence(sequence)
        
        # Memory usage should be available
        memory = memory_usage()
        assert isinstance(memory, dict)
        
        if torch.cuda.is_available():
            assert "allocated" in memory
            assert "cached" in memory
        else:
            assert "cpu_only" in memory


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
