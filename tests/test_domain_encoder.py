import torch

from models.domain_encoder import ContinuousDomainEncoder


def _make_encoder():
    return ContinuousDomainEncoder(in_channels=4, latent_dim=16, base_dim=8)


def test_forward_shape_without_angle():
    enc = _make_encoder()
    x = torch.randn(3, 4, 32, 32)
    z = enc(x)
    assert z.shape == (3, 16)


def test_forward_scalar_angle_broadcasts_to_batch():
    enc = _make_encoder()
    x = torch.randn(3, 4, 32, 32)
    z = enc(x, angle_deg=30.0)
    assert z.shape == (3, 16)


def test_forward_per_sample_angle_tensor_with_sentinel():
    """Mixed batch: some rows are angle-domain samples, some are sensor-domain (-1 sentinel)."""
    enc = _make_encoder()
    x = torch.randn(4, 4, 32, 32)
    angle = torch.tensor([-1.0, 0.0, 15.0, 45.0])
    z = enc(x, angle_deg=angle)
    assert z.shape == (4, 16)
    assert torch.isfinite(z).all()


def test_angle_proj_receives_gradient_only_from_angle_rows():
    """
    Regression test for the bug where angle_proj never received gradients during training
    because every training call site passed angle_deg=None. With a per-sample tensor, only
    rows with a non-negative (real) angle should backprop into angle_proj's parameters.
    """
    enc = _make_encoder()
    x = torch.randn(2, 4, 16, 16)

    # All rows sentinel (no angle) -> angle_proj must get zero gradient.
    angle_none = torch.tensor([-1.0, -1.0])
    enc.zero_grad()
    z = enc(x, angle_deg=angle_none)
    z.sum().backward()
    grad_none = enc.angle_proj[0].weight.grad.abs().sum().item()
    assert grad_none == 0.0

    # One real angle row present -> angle_proj must get non-zero gradient.
    angle_mixed = torch.tensor([-1.0, 20.0])
    enc.zero_grad()
    z = enc(x, angle_deg=angle_mixed)
    z.sum().backward()
    grad_mixed = enc.angle_proj[0].weight.grad.abs().sum().item()
    assert grad_mixed > 0.0


def test_angle_deg_none_matches_all_sentinel_tensor():
    """angle_deg=None should be equivalent to every row being the -1 'no angle' sentinel."""
    enc = _make_encoder()
    enc.eval()
    x = torch.randn(3, 4, 16, 16)
    with torch.no_grad():
        z_none = enc(x, angle_deg=None, normalize=False)
        z_sentinel = enc(x, angle_deg=torch.tensor([-1.0, -1.0, -1.0]), normalize=False)
    assert torch.allclose(z_none, z_sentinel, atol=1e-6)
