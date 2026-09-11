import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention Block.
    Recalibrates channel-wise feature responses, letting the encoder attend
    to the most discriminative spectral bands (e.g., NIR for vegetation,
    SWIR for soil moisture, B2/B3 for SAR texture).
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * scale


class ContinuousDomainEncoder(nn.Module):
    """
    Self-Supervised Continuous Domain Encoder for Satellite Imagery.
    Extracts continuous domain/sensor style embeddings from unlabeled multi-sensor patches.
    
    Instead of relying on discrete categorical domain labels (like StarGAN v1),
    this encoder maps sensor-specific spectral properties, radiometry, noise profiles,
    and atmospheric effects into a continuous latent style space z in R^d.

    Extended to support optional angle conditioning for multi-angle domains:
    When `angle_deg` is provided in forward(), a small MLP projects the scalar angle
    into a partial latent vector that is concatenated with image-derived style features,
    producing a geometry-aware hybrid embedding.
    
    SE Block after conv3 recalibrates band importance before global statistics pooling.
    """
    def __init__(self, in_channels: int = 4, latent_dim: int = 64, base_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Flexible channel projection (supports multi-spectral, RGB, or SAR)
        self.input_adapter = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Multi-scale feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_dim * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_dim * 4, base_dim * 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # SE Channel Attention: recalibrates spectral band importance before pooling
        self.se = SEBlock(base_dim * 8, reduction=8)

        # Style statistics extraction: Global Channel-Wise Mean + Std Dev pooling
        pooled_feat_dim = (base_dim * 8) * 2
        
        # Optional angle conditioning MLP: scalar angle [0,1] -> latent_dim // 2
        # Fused with image features before the final projection head.
        self.angle_dim = latent_dim // 2
        self.angle_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, self.angle_dim)
        )
        
        # MLP Projection Head to continuous latent sphere
        # Input dim = pooled image features + angle features
        self.projection_head = nn.Sequential(
            nn.Linear(pooled_feat_dim + self.angle_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x: torch.Tensor, normalize: bool = True, angle_deg=None) -> torch.Tensor:
        """
        x: (B, C, H, W)
        angle_deg: Optional acquisition off-nadir angle(s) in degrees [0, 90].
                   - None: no angle info for any sample (sensor domain) -> neutral zero vector.
                   - Python scalar (int/float): same angle applied to every sample in the batch
                     (e.g. evaluating a batch drawn entirely from one angle domain).
                   - Tensor of shape (B,) or (B, 1): per-sample angle. Use a negative value as a
                     sentinel for "no angle" on that row (e.g. a sensor-domain sample mixed into
                     an angle-domain batch under --mode both) — that row gets a neutral zero
                     vector while angle-domain rows in the same batch still get real conditioning.
        returns: (B, latent_dim) style vector
        """
        h = self.input_adapter(x)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)

        # SE channel attention: emphasize most discriminative spectral bands
        h = self.se(h)

        # Calculate first and second order spatial statistics (mean & standard deviation)
        # These moments capture radiometric sensor distributions, haze, and contrast invariant to spatial translation
        mean = h.mean(dim=[-2, -1])
        std = (h.var(dim=[-2, -1], unbiased=False) + 1e-6).sqrt()
        style_feats = torch.cat([mean, std], dim=1)

        # Angle conditioning: project (per-sample) angle to latent_dim//2
        # Normalized to [0, 1] range (90° max) for stable MLP input
        B = x.size(0)
        if angle_deg is None:
            # No angle info at all: contribute zero angle features (neutral)
            angle_feats = torch.zeros(B, self.angle_dim, dtype=x.dtype, device=x.device)
        else:
            if not torch.is_tensor(angle_deg):
                angle_deg = torch.full((B, 1), float(angle_deg), dtype=x.dtype, device=x.device)
            else:
                angle_deg = angle_deg.to(dtype=x.dtype, device=x.device).view(B, 1)
            # Negative sentinel marks rows with no angle (sensor-domain samples in a mixed batch)
            has_angle = (angle_deg >= 0).to(x.dtype)
            angle_norm = angle_deg.clamp(min=0) / 90.0
            angle_feats = self.angle_proj(angle_norm) * has_angle

        # Fuse image style features + angle geometry features
        combined = torch.cat([style_feats, angle_feats], dim=1)

        z = self.projection_head(combined)
        if normalize:
            z = F.normalize(z, p=2, dim=1)
        return z

class ContrastiveDomainLoss(nn.Module):
    """
    InfoNCE-based self-supervised contrastive loss on domain embeddings.
    Treats multiple transformed/augmented patches from the same sensor distribution as positive pairs,
    and patches from differing sensors or distinct scenes as negative pairs.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor:
        """
        z_anchor, z_positive: (B, latent_dim) unit normalized
        """
        batch_size = z_anchor.size(0)
        
        # Cosine similarity matrix between anchor and positive
        sim_matrix = torch.mm(z_anchor, z_positive.t()) / self.temperature
        
        # Positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=z_anchor.device)
        
        loss_a = F.cross_entropy(sim_matrix, labels)
        loss_b = F.cross_entropy(sim_matrix.t(), labels)
        return (loss_a + loss_b) / 2.0
