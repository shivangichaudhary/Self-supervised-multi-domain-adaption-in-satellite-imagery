import torch

from models.losses import (
    EdgePreservationLoss,
    MultiDomainLosses,
    FocalLoss,
    SegmentationDiceLoss,
)


def test_edge_preservation_loss_zero_for_identical_images():
    loss_fn = EdgePreservationLoss()
    x = torch.randn(2, 4, 16, 16)
    loss = loss_fn(x, x)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_multidomain_losses_basic_sanity():
    losses = MultiDomainLosses()
    d_real = torch.randn(2, 1, 8, 8)
    d_fake = torch.randn(2, 1, 8, 8)
    assert torch.isfinite(losses.d_loss(d_real, d_fake))
    assert torch.isfinite(losses.g_adv_loss(d_fake))

    a = torch.randn(2, 4, 16, 16)
    b = torch.randn(2, 4, 16, 16)
    assert torch.isclose(losses.cycle_consistency_loss(a, a), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(losses.identity_loss(a, a), torch.tensor(0.0), atol=1e-6)


def test_geometric_preservation_loss_zero_for_identical_content():
    losses = MultiDomainLosses()
    content = torch.randn(3, 8, 16, 16)
    loss = losses.geometric_preservation_loss(content, content)
    assert loss.item() < 1e-4


def test_geometric_preservation_loss_respects_per_sample_mask():
    """
    Regression test: geometric_preservation_loss must be computable per-sample and
    averaged only over masked rows, since a training batch can mix angle-domain and
    sensor-domain samples (--mode both) and the loss should only apply to the former.
    """
    losses = MultiDomainLosses()
    torch.manual_seed(0)
    content_fake = torch.randn(4, 8, 16, 16)
    content_real = torch.randn(4, 8, 16, 16)

    mask_all = torch.ones(4)
    mask_none = torch.zeros(4)

    loss_all = losses.geometric_preservation_loss(content_fake, content_real, mask=mask_all)
    loss_unmasked = losses.geometric_preservation_loss(content_fake, content_real)
    assert torch.isclose(loss_all, loss_unmasked, atol=1e-5)

    loss_none = losses.geometric_preservation_loss(content_fake, content_real, mask=mask_none)
    assert loss_none.item() == 0.0

    # Masking to a subset should approximate computing the loss on just that subset (small
    # numerical differences are expected since the SSIM data_range constant is derived from
    # whichever tensor is passed in — 4 rows vs. the 2-row slice — not from the mask itself).
    mask_partial = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss_partial = losses.geometric_preservation_loss(content_fake, content_real, mask=mask_partial)
    loss_subset = losses.geometric_preservation_loss(content_fake[[0, 2]], content_real[[0, 2]])
    assert torch.isclose(loss_partial, loss_subset, atol=1e-2)


def test_focal_loss_ignores_ignore_index():
    loss_fn = FocalLoss(gamma=2.0, ignore_index=-100, label_smoothing=0.0)
    logits = torch.randn(2, 3, 4, 4)
    targets = torch.full((2, 4, 4), -100, dtype=torch.long)
    targets[0, 0, 0] = 1  # one valid pixel
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss)


def test_segmentation_dice_loss_shapes_and_finiteness():
    loss_fn = SegmentationDiceLoss(num_classes=6, ignore_index=-100)
    logits = torch.randn(2, 6, 16, 16)
    targets = torch.randint(0, 6, (2, 16, 16))
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
