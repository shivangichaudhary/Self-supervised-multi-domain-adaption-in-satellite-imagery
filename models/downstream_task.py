import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) — used in Domain-Adversarial Neural Networks (DANN).
    Forward pass: identity function.
    Backward pass: reverses gradients by multiplying by -lambda_d.
    This forces the feature extractor to learn domain-invariant representations.
    """
    @staticmethod
    def forward(ctx, x, lambda_d):
        ctx.lambda_d = lambda_d
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_d * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_d: float = 1.0):
        super().__init__()
        self.lambda_d = lambda_d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_d)

    def set_lambda(self, lambda_d: float):
        self.lambda_d = lambda_d


class ResidualDoubleConv(nn.Module):
    """Residual Double Convolution block with skip connection"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

class AttentionGate(nn.Module):
    """
    Attention Gate for U-Net skip connections.
    Suppresses background activations and highlights salient parcel boundaries.
    """
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class ASPPPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        feat = self.conv(self.gap(x))
        return F.interpolate(feat, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    Captures multi-scale contextual features across expansive agricultural fields and narrow parcel boundaries.
    """
    def __init__(self, in_channels: int, out_channels: int, atrous_rates=(1, 2, 4, 8)):
        super().__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        for rate in atrous_rates[1:]:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)

class SatelliteUNet(nn.Module):
    """
    Attention-Augmented Residual U-Net with ASPP Bottleneck and Deep Supervision
    for Satellite Land-Cover Mapping.
    
    Includes:
      - ASPP multi-scale bottleneck for multi-hectare and narrow-strip parcel delineation
      - Attention Gates for skip connections
      - Deep Supervision intermediate auxiliary classifier heads
      - Domain Classifier head via Gradient Reversal Layer (GRL) for unsupervised DAT
    """
    def __init__(self, in_channels: int = 4, num_classes: int = 6, base_channels: int = 32, num_domains: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Encoder with Residual blocks
        self.inc = ResidualDoubleConv(in_channels, base_channels)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualDoubleConv(base_channels, base_channels * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualDoubleConv(base_channels * 2, base_channels * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualDoubleConv(base_channels * 4, base_channels * 8))
        
        # ASPP Multi-Scale Bottleneck
        self.aspp = ASPP(base_channels * 8, base_channels * 8, atrous_rates=(1, 2, 4, 8))

        # Attention Gates for Skip Connections
        self.att3 = AttentionGate(f_g=base_channels * 4, f_l=base_channels * 4, f_int=base_channels * 2)
        self.att2 = AttentionGate(f_g=base_channels * 2, f_l=base_channels * 2, f_int=base_channels)
        self.att1 = AttentionGate(f_g=base_channels, f_l=base_channels, f_int=base_channels // 2)

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.conv_up1 = ResidualDoubleConv(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.conv_up2 = ResidualDoubleConv(base_channels * 4, base_channels * 2)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.conv_up3 = ResidualDoubleConv(base_channels * 2, base_channels)
        
        # Final Segmentation Output
        self.outc = nn.Conv2d(base_channels, num_classes, 1)

        # Deep Supervision Auxiliary Heads
        self.aux_head1 = nn.Conv2d(base_channels * 4, num_classes, 1)  # Scale: H/4, W/4
        self.aux_head2 = nn.Conv2d(base_channels * 2, num_classes, 1)  # Scale: H/2, W/2

        # Domain Adversarial Head (DANN-style with GRL)
        self.grl = GradientReversalLayer(lambda_d=1.0)
        bottleneck_dim = base_channels * 8
        self.domain_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(bottleneck_dim * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_domains)
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """
        x: (B, C, H, W)
        returns: (B, num_classes, H, W) or (main, aux1, aux2)
        """
        H, W = x.shape[2], x.shape[3]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Multi-Scale ASPP Bottleneck
        x4 = self.aspp(x4)

        d3 = self.up1(x4)
        x3_att = self.att3(g=d3, x=x3)
        d3 = torch.cat([d3, x3_att], dim=1)
        d3 = self.conv_up1(d3)
        
        d2 = self.up2(d3)
        x2_att = self.att2(g=d2, x=x2)
        d2 = torch.cat([d2, x2_att], dim=1)
        d2 = self.conv_up2(d2)
        
        d1 = self.up3(d2)
        x1_att = self.att1(g=d1, x=x1)
        d1 = torch.cat([d1, x1_att], dim=1)
        d1 = self.conv_up3(d1)
        
        logits = self.outc(d1)

        if return_aux and self.training:
            aux1 = F.interpolate(self.aux_head1(d3), size=(H, W), mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux_head2(d2), size=(H, W), mode='bilinear', align_corners=False)
            return logits, aux1, aux2

        return logits

    def forward_with_domain(self, x: torch.Tensor, return_aux: bool = False):
        """
        Forward pass that also returns domain classifier logits and optional deep supervision aux heads.
        Used during DAT training to compute the domain adversarial loss.
        """
        H, W = x.shape[2], x.shape[3]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Multi-Scale ASPP Bottleneck
        x4 = self.aspp(x4)

        # Domain adversarial branch from bottleneck features
        x4_reversed = self.grl(x4)
        domain_logits = self.domain_classifier(x4_reversed)
        
        d3 = self.up1(x4)
        x3_att = self.att3(g=d3, x=x3)
        d3 = torch.cat([d3, x3_att], dim=1)
        d3 = self.conv_up1(d3)
        
        d2 = self.up2(d3)
        x2_att = self.att2(g=d2, x=x2)
        d2 = torch.cat([d2, x2_att], dim=1)
        d2 = self.conv_up2(d2)
        
        d1 = self.up3(d2)
        x1_att = self.att1(g=d1, x=x1)
        d1 = torch.cat([d1, x1_att], dim=1)
        d1 = self.conv_up3(d1)
        
        logits = self.outc(d1)

        if return_aux and self.training:
            aux1 = F.interpolate(self.aux_head1(d3), size=(H, W), mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux_head2(d2), size=(H, W), mode='bilinear', align_corners=False)
            return logits, domain_logits, aux1, aux2

        return logits, domain_logits
