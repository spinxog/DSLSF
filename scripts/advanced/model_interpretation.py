def _load_model(self):
        """Load trained model."""
        try:
            # Load the actual RNA folding model
            from rna_model import RNAFoldingPipeline, PipelineConfig
            
            # Create pipeline configuration
            config = PipelineConfig(
                device="cpu",  # Use CPU for interpretation
                mixed_precision=False  # Disable mixed precision for stability
            )
            
            # Initialize the pipeline
            pipeline = RNAFoldingPipeline(config)
            
            # Load model weights if available
            if self.model_path and Path(self.model_path).exists():
                try:
                    checkpoint = torch.load(self.model_path, map_location='cpu')
                    
                    # Handle different checkpoint formats
                    if 'model_state_dict' in checkpoint:
                        pipeline.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        pipeline.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        pipeline.model.load_state_dict(checkpoint)
                    
                    logging.info(f"Loaded model weights from {self.model_path}")
                except Exception as e:
                    logging.warning(f"Could not load model weights: {e}")
                    logging.info("Using randomly initialized model")
            else:
                logging.info("No model path provided, using randomly initialized model")
            
            # Set model to evaluation mode
            pipeline.model.eval()
            
            return pipeline.model
            
        except ImportError as e:
            logging.error(f"Could not import RNA folding pipeline: {e}")
            # Fallback to simple model if pipeline not available
            return self._create_simple_model()
    
    def _create_simple_model(self):
        """Create simple RNA model as fallback."""
        class SimpleRNAModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(4, 512)  # A, C, G, U
                self.positional_encoding = nn.Embedding(512, 512)
                
                # Transformer layers
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=512,
                    nhead=8,
                    dim_feedforward=2048,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
                
                # Attention mechanism
                self.attention = nn.MultiheadAttention(512, 8, batch_first=True)
                
                # Output layers
                self.output = nn.Linear(512, 1)
                
                # Secondary structure prediction
                self.ss_head = nn.Linear(512, 64)  # Contact prediction
                self.geometry_head = nn.Linear(512, 256)  # Geometry features
                
            def forward(self, x, return_attention=False):
                # Embedding
                seq_len = x.size(1)
                positions = torch.arange(seq_len, device=x.device)
                x = self.embedding(x) + self.positional_encoding(positions)
                
                # Transformer
                x = self.transformer(x)
                
                # Self-attention
                attn_output, attn_weights = self.attention(x, x, x)
                
                # Combine with transformer output
                x = x + attn_output
                
                # Outputs
                output = self.output(x)
                ss_logits = self.ss_head(x)
                geometry = self.geometry_head(x)
                
                result = {
                    'output': output,
                    'ss_logits': ss_logits,
                    'geometry': geometry
                }
                
                if return_attention:
                    result['attention'] = attn_weights
                
                return result
        
        model = SimpleRNAModel()
        model.eval()
        return model