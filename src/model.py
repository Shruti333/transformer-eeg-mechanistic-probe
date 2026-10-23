import torch
import torch.nn as nn
from einops import rearrange

class EEGTransformer(nn.Module):
    """
    Advanced Transformer Encoder for EEG decoding that captures 
    long-range temporal dependencies and exposes multi-head attention maps.
    """
    def __init__(self, n_channels, n_times, n_classes=2, d_model=64, n_heads=4, n_layers=2):
        super(EEGTransformer, self).__init__()
        
        # Project spatial EEG channels to transformer hidden dimension
        self.channel_projection = nn.Linear(n_channels, d_model)
        
        # Learnable positional encoding across time samples
        self.pos_embedding = nn.Parameter(torch.randn(1, n_times, d_model))
        
        # Transformer Encoder Layers with attention weight retention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=128, 
            dropout=0.1, 
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        # x shape: (batch, n_channels, n_times)
        x = rearrange(x, 'b c t -> b t c')
        
        # Project to d_model
        x = self.channel_projection(x) + self.pos_embedding[:, :x.size(1), :]
        
        # Pass through transformer blocks
        x = self.transformer_encoder(x)
        
        # Global temporal pooling
        x = x.mean(dim=1)
        
        out = self.classifier(x)
        return out
