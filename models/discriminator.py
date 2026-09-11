import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDomainProjectionDiscriminator(nn.Module):
    """
    Multi-Scale PatchGAN Discriminator with Continuous Style Projection
    and Spectral Normalization on all convolutional layers.
    
    Spectral normalization constrains the Lipschitz constant of each conv layer,
    preventing discriminator gradient explosion and ensuring stable GAN training.
    
    Evaluates:
      1. Spatial realism (is the satellite patch realistic remote sensing data?)
      2. Domain conditioning match (does the image texture/physics match the conditioned style z?)
    """
    def __init__(self, in_channels: int = 4, latent_dim: int = 64, dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        sn = nn.utils.spectral_norm  # shorthand
        
        self.conv_layers = nn.Sequential(
            sn(nn.Conv2d(in_channels, dim, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            sn(nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            sn(nn.Conv2d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            sn(nn.Conv2d(dim * 4, dim * 8, kernel_size=4, stride=1, padding=1)),
            nn.InstanceNorm2d(dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Unconditional real/fake score head (spectrally normalized)
        self.unconditional_head = sn(nn.Conv2d(dim * 8, 1, kernel_size=3, stride=1, padding=1))
        
        # Linear projection for continuous style conditioning
        self.style_proj = nn.Linear(latent_dim, dim * 8)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        z: (B, latent_dim) continuous domain style vector
        returns: (B, 1, H', W') patch authenticity logits
        """
        h = self.conv_layers(x)
        out_unconditional = self.unconditional_head(h)
        
        # Style projection: inner product between intermediate features and projected style vector
        z_proj = self.style_proj(z).unsqueeze(2).unsqueeze(3) # (B, dim*8, 1, 1)
        projection_term = (h * z_proj).sum(dim=1, keepdim=True) / (h.size(1) ** 0.5)
        
        return out_unconditional + projection_term
