import torch


def batch_forward(model, imgs, chunk_size=8, **kwargs):
    """Runs model(imgs, **kwargs) in chunks to bound peak memory on CPU/GPU."""
    if imgs.size(0) <= chunk_size:
        return model(imgs, **kwargs)
    out = []
    for i in range(0, imgs.size(0), chunk_size):
        out.append(model(imgs[i:i + chunk_size], **kwargs))
    return torch.cat(out, dim=0)


def batch_generator(gen, imgs, z, chunk_size=8):
    """Runs gen(imgs, z) in chunks to bound peak memory on CPU/GPU."""
    if imgs.size(0) <= chunk_size:
        return gen(imgs, z)
    out = []
    for i in range(0, imgs.size(0), chunk_size):
        out.append(gen(imgs[i:i + chunk_size], z[i:i + chunk_size]))
    return torch.cat(out, dim=0)


def predict_with_tta(model, imgs, chunk_size=8):
    """
    Test-Time Augmentation (TTA).
    Averages predicted softmax probability distributions across original,
    horizontal-flip, and vertical-flip orientations for robust boundary inference.
    Batched in chunks to avoid memory bottlenecks and thread thrashing on CPU.

    Any prediction compared against a TTA'd prediction must also go through this
    function — comparing a plain forward pass against a TTA'd one attributes part
    of the accuracy gap to TTA itself rather than to whatever is actually being compared.
    """
    if imgs.size(0) > chunk_size:
        chunks = []
        for i in range(0, imgs.size(0), chunk_size):
            chunks.append(predict_with_tta(model, imgs[i:i + chunk_size], chunk_size))
        return torch.cat(chunks, dim=0)

    with torch.no_grad():
        # 1. Original orientation
        probs_orig = torch.softmax(model(imgs), dim=1)

        # 2. Horizontal flip orientation
        imgs_h = torch.flip(imgs, dims=[-1])
        probs_h = torch.flip(torch.softmax(model(imgs_h), dim=1), dims=[-1])

        # 3. Vertical flip orientation
        imgs_v = torch.flip(imgs, dims=[-2])
        probs_v = torch.flip(torch.softmax(model(imgs_v), dim=1), dims=[-2])

        probs_tta = (probs_orig + probs_h + probs_v) / 3.0
    return probs_tta


def dual_path_confidence_ensemble(probs_pixel: torch.Tensor, probs_feat: torch.Tensor) -> torch.Tensor:
    """
    Adaptive confidence-weighted ensemble between the pixel-space (generator-translated
    input) prediction and the feature-space (native-domain, DANN-adapted) prediction:
    whichever path is more confident on a given pixel is weighted more heavily.
    """
    conf_pixel = probs_pixel.max(dim=1, keepdim=True).values
    conf_feat = probs_feat.max(dim=1, keepdim=True).values
    w_pixel = conf_pixel / (conf_pixel + conf_feat + 1e-6)
    w_feat = 1.0 - w_pixel
    return w_pixel * probs_pixel + w_feat * probs_feat
