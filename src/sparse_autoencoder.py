import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder (SAE) for mechanistic interpretability of transformer residual streams.
    Disentangles polysemantic representations into sparse, monosemantic features.
    """
    def __init__(self, d_model=64, expansion_factor=4):
        super(SparseAutoencoder, self).__init__()
        self.d_model = d_model
        self.d_hidden = d_model * expansion_factor
        
        # Encoder & Decoder parameters
        self.W_enc = nn.Parameter(torch.randn(d_model, self.d_hidden) / (d_model ** 0.5))
        self.b_enc = nn.Parameter(torch.zeros(self.d_hidden))
        
        self.W_dec = nn.Parameter(torch.randn(self.d_hidden, d_model) / (self.d_hidden ** 0.5))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        
        # Enforce unit norm on decoder columns (standard practice in SAEs)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model] or [N, d_model]
        h_pre = torch.matmul(x, self.W_enc) + self.b_enc
        h = F.relu(h_pre)  # ReLU activation drives sparsity
        
        x_reconstructed = torch.matmul(h, self.W_dec) + self.b_dec
        return x_reconstructed, h

    def compute_loss(self, x, x_reconstructed, h, l1_coefficient=1e-3):
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_reconstructed, x)
        # Sparsity penalty (L1 norm on hidden latent features)
        sparsity_loss = l1_coefficient * torch.sum(torch.abs(h), dim=-1).mean()
        total_loss = recon_loss + sparsity_loss
        return total_loss, recon_loss, sparsity_loss