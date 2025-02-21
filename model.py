import torch
import torch.nn as nn

class DiscreteDiffusionSimpleTransformer(nn.Module):
    def __init__(self, model_dim, num_heads, num_layers, vocab_size, seq_length):
        super(DiscreteDiffusionSimpleTransformer, self).__init__()

        assert model_dim % num_heads == 0, "MODEL_DIM must be divisible by NUM_HEADS"
        
        self.embedding = nn.Embedding(2, model_dim)  # Binary input (0 or 1)
        self.position_encoding = nn.Parameter(torch.randn(seq_length, model_dim))  # Learnable positional encoding
        
        self.transformer = nn.Transformer(
            d_model=model_dim, 
            nhead=num_heads, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(model_dim, vocab_size)  # Predict character distribution

    def forward(self, src):
        src = self.embedding(src) + self.position_encoding  # Add positional encoding
        src = self.transformer.encoder(src)
        return self.output_layer(src)  # Shape: [batch_size, seq_length, vocab_size]

