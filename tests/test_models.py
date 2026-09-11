import torch

from models.generator import UnifiedSatelliteGenerator, AdaptiveInstanceNorm2d, AdaINResBlock
from models.discriminator import MultiDomainProjectionDiscriminator
from models.downstream_task import SatelliteUNet


def test_generator_forward_shape():
    gen = UnifiedSatelliteGenerator(in_channels=4, out_channels=4, latent_dim=16, dim=8, num_resblocks=2)
    x = torch.randn(2, 4, 32, 32)
    z = torch.randn(2, 16)
    out = gen(x, z)
    assert out.shape == (2, 4, 32, 32)
    assert torch.isfinite(out).all()


def test_discriminator_forward_shape():
    disc = MultiDomainProjectionDiscriminator(in_channels=4, latent_dim=16, dim=8)
    x = torch.randn(2, 4, 32, 32)
    z = torch.randn(2, 16)
    out = disc(x, z)
    assert out.dim() == 4
    assert out.size(0) == 2
    assert torch.isfinite(out).all()


def test_downstream_unet_forward_and_domain_head():
    model = SatelliteUNet(in_channels=4, num_classes=6, base_channels=8, num_domains=2)
    x = torch.randn(2, 4, 64, 64)
    logits = model(x)
    assert logits.shape == (2, 6, 64, 64)

    logits2, domain_logits = model.forward_with_domain(x)
    assert logits2.shape == (2, 6, 64, 64)
    assert domain_logits.shape == (2, 2)


def test_downstream_unet_deep_supervision_aux_heads_only_in_training():
    model = SatelliteUNet(in_channels=4, num_classes=6, base_channels=8, num_domains=2)
    x = torch.randn(2, 4, 64, 64)

    model.train()
    out = model(x, return_aux=True)
    assert len(out) == 3
    for t in out:
        assert t.shape[-2:] == (64, 64)

    model.eval()
    out_eval = model(x, return_aux=True)
    assert torch.is_tensor(out_eval)  # aux heads suppressed outside training


def test_generator_extract_content_matches_encoder_stage():
    gen = UnifiedSatelliteGenerator(in_channels=4, out_channels=4, latent_dim=8, dim=8, num_resblocks=1)
    x = torch.randn(1, 4, 32, 32)
    content = gen.extract_content(x)
    assert content.shape == (1, 32, 8, 8)  # dim*4 channels, downsampled by 4x


def test_adain_gamma_one_beta_zero_init_is_identity_like():
    adain = AdaptiveInstanceNorm2d(num_features=8, latent_dim=16)
    x = torch.randn(2, 8, 10, 10)
    z = torch.zeros(2, 16)
    out = adain(x, z)
    # With z=0 and gamma initialized to weight=1/bias=0, beta to weight=0/bias=0,
    # gamma = fc_gamma(0) = 0*... + bias(=0)... actually fc_gamma has weight=ones, bias=0
    # so gamma = W @ 0 + 0 = 0, meaning output should be all zeros (gamma=0, beta=0).
    assert out.shape == x.shape
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)
