import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveInstanceNorm2d(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN).
    Modulates normalized spatial features using affine scale (gamma) and bias (beta)
    generated dynamically from continuous domain style embeddings.
    """
    def __init__(self, num_features: int, latent_dim: int):
        super().__init__()
        self.num_features = num_features
        self.fc_gamma = nn.Linear(latent_dim, num_features)
        self.fc_beta = nn.Linear(latent_dim, num_features)
        
        # Initialize gamma to 1 and beta to 0 for stability
        nn.init.ones_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        z: (B, latent_dim)
        """
        gamma = self.fc_gamma(z).unsqueeze(2).unsqueeze(3)
        beta = self.fc_beta(z).unsqueeze(2).unsqueeze(3)
        
        # Instance normalization (zero mean, unit var per channel per sample)
        mean = x.mean(dim=[-2, -1], keepdim=True)
        var = x.var(dim=[-2, -1], keepdim=True, unbiased=False)
        x_norm = (x - mean) / (torch.sqrt(var + 1e-5))
        
        return gamma * x_norm + beta

class AdaINResBlock(nn.Module):
    """
    Residual Block with AdaIN modulation for multi-domain satellite translation.
    """
    def __init__(self, channels: int, latent_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.adain1 = AdaptiveInstanceNorm2d(channels, latent_dim)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.adain2 = AdaptiveInstanceNorm2d(channels, latent_dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.conv1(x)
        h = self.adain1(h, z)
        h = self.act1(h)
        
        h = self.conv2(h)
        h = self.adain2(h, z)
        return residual + h


class UnifiedSatelliteGenerator(nn.Module):
    """
    Single unified generator for arbitrary multi-domain satellite-to-satellite translation.
    Takes source image x (any sensor) + continuous domain style embedding z (target sensor)
    and synthesizes target domain imagery while strictly preserving geospatial content.
    
    Decoder uses bilinear upsampling (instead of nearest) to eliminate checkerboard artifacts.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 4, latent_dim: int = 64, dim: int = 64, num_resblocks: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Content Encoder: Downsamples source satellite image to extract invariant spatial layout
        self.enc_conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Intermediate Bottleneck: AdaIN-modulated ResNet blocks for style transfer
        self.resblocks = nn.ModuleList([
            AdaINResBlock(dim * 4, latent_dim) for _ in range(num_resblocks)
        ])
        
        # Decoder: Multi-Scale AdaIN modulated upsampling layers
        # Bilinear upsampling replaces nearest to prevent checkerboard artifacts
        self.dec_conv1 = nn.Conv2d(dim * 4, dim * 2, kernel_size=3, stride=1, padding=1)
        self.dec_adain1 = AdaptiveInstanceNorm2d(dim * 2, latent_dim)
        self.dec_act1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.dec_conv2 = nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1)
        self.dec_adain2 = AdaptiveInstanceNorm2d(dim, latent_dim)
        self.dec_act2 = nn.LeakyReLU(0.2, inplace=True)
        
        # Output Reconstruction Head
        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(dim, out_channels, kernel_size=7, stride=1, padding=0),
            nn.Tanh() # Maps output pixels to [-1, 1]
        )

    def extract_content(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts domain-invariant content features"""
        h = self.enc_conv1(x)
        h = self.enc_conv2(h)
        h = self.enc_conv3(h)
        return h

    def forward(self, x: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, H, W) Source image
        z_target: (B, latent_dim) Target domain continuous style vector
        returns: (B, out_channels, H, W) Translated image in target sensor domain
        """
        h = self.extract_content(x)
        for resblock in self.resblocks:
            h = resblock(h, z_target)
            
        # Multi-scale decoder with AdaIN modulation at each resolution
        # Bilinear upsampling for artifact-free high-fidelity satellite translations
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = self.dec_conv1(h)
        h = self.dec_adain1(h, z_target)
        h = self.dec_act1(h)
        
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = self.dec_conv2(h)
        h = self.dec_adain2(h, z_target)
        h = self.dec_act2(h)
        
        out = self.out_conv(h)
        return out
