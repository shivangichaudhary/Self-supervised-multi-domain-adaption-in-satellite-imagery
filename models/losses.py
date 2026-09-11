import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgePreservationLoss(nn.Module):
    """
    Computes gradient / boundary consistency loss using Sobel filters.
    Enforces sharp agricultural parcel edges, roads, and landscape boundaries.
    """
    def __init__(self):
        super().__init__()
        # Sobel kernel for X and Y gradients
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)
        self.l1 = nn.L1Loss()

    def _compute_gradients(self, x: torch.Tensor) -> torch.Tensor:
        # Compute gradient magnitude per channel
        B, C, H, W = x.shape
        x_reshaped = x.view(B * C, 1, H, W)
        gx = F.conv2d(x_reshaped, self.kx, padding=1)
        gy = F.conv2d(x_reshaped, self.ky, padding=1)
        grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return grad.view(B, C, H, W)

    def forward(self, img_fake: torch.Tensor, img_real: torch.Tensor) -> torch.Tensor:
        grad_fake = self._compute_gradients(img_fake)
        grad_real = self._compute_gradients(img_real)
        return self.l1(grad_fake, grad_real)

class MultiDomainLosses(nn.Module):
    """
    Unified multi-objective loss module for self-supervised continuous multi-domain satellite translation.
    Supports both sensor-domain and angle-domain translation losses, including a geometric
    preservation loss and Sobel edge boundary loss.
    """
    def __init__(self, lambda_adv: float = 1.0, lambda_sty: float = 1.0, lambda_cyc: float = 10.0,
                 lambda_sem: float = 2.0, lambda_id: float = 1.0, lambda_geo: float = 5.0, lambda_edge: float = 2.0):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_sty = lambda_sty
        self.lambda_cyc = lambda_cyc
        self.lambda_sem = lambda_sem
        self.lambda_id = lambda_id
        self.lambda_geo = lambda_geo
        self.lambda_edge = lambda_edge
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.edge_loss = EdgePreservationLoss()

    def d_loss(self, d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
        """Least-squares GAN loss for Discriminator"""
        loss_real = torch.mean((d_real - 1.0) ** 2)
        loss_fake = torch.mean(d_fake ** 2)
        return 0.5 * (loss_real + loss_fake)

    def g_adv_loss(self, d_fake: torch.Tensor) -> torch.Tensor:
        """Adversarial loss for Generator"""
        return torch.mean((d_fake - 1.0) ** 2)

    def style_recon_loss(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """Enforces translated image to exhibit target style vector"""
        return self.l1_loss(z_pred, z_target)

    def cycle_consistency_loss(self, x_reconstructed: torch.Tensor, x_original: torch.Tensor) -> torch.Tensor:
        """Guarantees invertibility and prevents mode collapse across domains"""
        return self.l1_loss(x_reconstructed, x_original)

    def semantic_preservation_loss(self, content_fake: torch.Tensor, content_real: torch.Tensor) -> torch.Tensor:
        """
        Penalizes drift in geospatial layout (buildings, roads, forest boundaries).
        Uses intermediate deep content activations.
        """
        return self.l1_loss(content_fake, content_real)

    def identity_loss(self, x_same_domain: torch.Tensor, x_original: torch.Tensor) -> torch.Tensor:
        """Preserves sensor characteristics when conditioned on its own style"""
        return self.l1_loss(x_same_domain, x_original)

    def edge_preservation_loss(self, x_fake: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
        """Enforces crisp parcel boundary and road edge consistency"""
        return self.edge_loss(x_fake, x_real)

    def geometric_preservation_loss(self, content_fake: torch.Tensor, content_real: torch.Tensor,
                                     mask: torch.Tensor = None) -> torch.Tensor:
        """
        Geometric structure preservation loss for angle-domain translation.
        Uses an SSIM-style local-statistics comparison on encoder content features.

        mask: Optional (B,) tensor. When given, only batch rows with mask > 0 (i.e. samples
              where src or tgt is an angle domain) contribute to the loss — needed because a
              batch can mix sensor- and angle-domain samples under --mode both.
        """
        window_size = 7
        padding = window_size // 2

        def local_stats(x):
            mu = F.avg_pool2d(x, kernel_size=window_size, stride=1, padding=padding)
            mu_sq = mu * mu
            sigma_sq = F.avg_pool2d(x * x, kernel_size=window_size, stride=1, padding=padding) - mu_sq
            sigma_sq = sigma_sq.clamp(min=0)
            return mu, sigma_sq

        # SSIM stability constants scale with the actual dynamic range of the content features
        # (arbitrary-scale conv activations, not [0, 1] images) rather than fixed image-domain values.
        data_range = (content_real.amax() - content_real.amin()).clamp(min=1e-3)
        C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2

        mu_f, var_f = local_stats(content_fake)
        mu_r, var_r = local_stats(content_real)
        cov = F.avg_pool2d(content_fake * content_real, kernel_size=window_size, stride=1, padding=padding) - mu_f * mu_r

        ssim_map = ((2 * mu_f * mu_r + C1) * (2 * cov + C2)) / \
                   ((mu_f ** 2 + mu_r ** 2 + C1) * (var_f + var_r + C2))

        per_sample_loss = 1.0 - ssim_map.mean(dim=[1, 2, 3])  # (B,)
        if mask is not None:
            mask = mask.to(per_sample_loss.dtype)
            if mask.sum() > 0:
                return (per_sample_loss * mask).sum() / mask.sum()
            return per_sample_loss.sum() * 0.0
        return per_sample_loss.mean()

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with gamma focusing parameter to overcome extreme class imbalance.
    Supports label smoothing (epsilon) and optional per-class weighting (inverse frequency).
    """
    def __init__(self, gamma: float = 2.0, ignore_index: int = -100, label_smoothing: float = 0.1, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross-entropy with label smoothing and optional class weights
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        valid_mask = (targets != self.ignore_index)
        if valid_mask.sum() > 0:
            return focal_loss[valid_mask].mean()
        return focal_loss.mean()

class SegmentationDiceLoss(nn.Module):
    """
    Combined Class-Weighted Focal Loss (with label smoothing) and Multi-Class Soft Dice Loss
    with void class masking (ignore_index).
    """
    def __init__(self, num_classes: int = 6, ignore_index: int = -100, smooth: float = 1e-5, label_smoothing: float = 0.1, weight: torch.Tensor = None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.focal_loss = FocalLoss(gamma=2.0, ignore_index=ignore_index, label_smoothing=label_smoothing, weight=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, num_classes, H, W)
        targets: (B, H, W)
        """
        focal = self.focal_loss(logits, targets)
        
        # Soft Dice on valid pixels
        probs = F.softmax(logits, dim=1) # (B, C, H, W)
        
        # Mask out void pixels
        if self.ignore_index >= 0:
            valid_mask = (targets != self.ignore_index).unsqueeze(1) # (B, 1, H, W)
            targets_clean = targets.clone()
            targets_clean[targets == self.ignore_index] = 0
            targets_onehot = F.one_hot(targets_clean, self.num_classes).permute(0, 3, 1, 2).float()
            targets_onehot = targets_onehot * valid_mask.float()
            probs = probs * valid_mask.float()
        else:
            targets_onehot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dim=dims)
        cardinality = torch.sum(probs + targets_onehot, dim=dims)
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Only penalize classes that are actually present in the ground truth batch
        present_mask = (targets_onehot.sum(dim=dims) > 0)
        if present_mask.sum() > 0:
            dice_loss = 1.0 - dice_score[present_mask].mean()
        else:
            dice_loss = torch.tensor(0.0, device=logits.device)
        
        return focal + dice_loss
