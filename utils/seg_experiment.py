"""Shared helpers for the Objective 3 and 4 experiments (PASTIS-HD, in-memory tensors, GPU)."""
import glob
import os
import numpy as np
import torch
import torch.nn.functional as F
from models.downstream_task import SatelliteUNet
from models.losses import SegmentationDiceLoss
from models.domain_encoder import ContinuousDomainEncoder
from models.generator import UnifiedSatelliteGenerator
from utils.metrics import compute_segmentation_metrics
from utils.seed import set_seed

SENSORS = ["sentinel2", "spot6", "sentinel1_sar"]
IGNORE_INDEX = 19      # PASTIS "void" label
NUM_CLASSES = 20


# ---------------------------------------------------------------- data
def load_domain(root, domain, split, split_ratio=0.8):
    """Loads all patches of one sensor for a split. Same sorted-ID 80/20 block split as
    MultiSensorSatelliteDataset, so train/held-out membership matches Objectives 1 and 2.
    Returns images (N,4,H,W) float32 in [-1,1] and masks (N,H,W) int64."""
    cache = os.path.join(root, "_cache", f"{domain}_{split}_{split_ratio}.pt")
    if os.path.exists(cache):
        X, M = torch.load(cache, weights_only=True)
        return X, M
    paths = sorted(glob.glob(os.path.join(root, domain, "images", "*.pt")))
    cut = int(len(paths) * split_ratio)
    paths = paths[:cut] if split == "train" else paths[cut:]
    X = torch.stack([torch.load(p, weights_only=True) for p in paths]).float()
    M = torch.stack([torch.load(p.replace("images", "masks"), weights_only=True) for p in paths]).long()
    os.makedirs(os.path.dirname(cache), exist_ok=True)   # reading ~3,600 small files takes minutes; cache them
    torch.save((X, M), cache)
    return X, M


def load_models(output_dir, cfg, device):
    enc = ContinuousDomainEncoder(in_channels=cfg["domain_encoder"]["in_channels"],
                                  latent_dim=cfg["domain_encoder"]["latent_dim"]).to(device)
    enc.load_state_dict(torch.load(os.path.join(output_dir, "domain_encoder.pth"), map_location=device, weights_only=True))
    enc.eval()
    return enc


def build_generator(cfg, device):
    g = cfg["generator"]
    return UnifiedSatelliteGenerator(in_channels=g["in_channels"], out_channels=g["out_channels"],
                                     latent_dim=g["latent_dim"], dim=g["dim"],
                                     num_resblocks=g["num_resblocks"]).to(device)


def load_generator(path, cfg, device):
    g = build_generator(cfg, device)
    g.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    g.eval()
    return g


@torch.no_grad()
def canonical_styles(encoder, train_data, device, max_patches=300):
    """Mean (L2-normalised) encoder embedding of each sensor's training patches."""
    out = {}
    for d, X in train_data.items():
        zs = [encoder(X[i:i + 50].to(device)) for i in range(0, min(len(X), max_patches), 50)]
        out[d] = F.normalize(torch.cat(zs).mean(0, keepdim=True), p=2, dim=1)
    return out


@torch.no_grad()
def translate(generator, X, z, device, chunk=32):
    """Translates X (N,4,H,W) into the style z (1,D); returns a CPU tensor."""
    out = []
    for i in range(0, len(X), chunk):
        xb = X[i:i + chunk].to(device)
        out.append(generator(xb, z.expand(len(xb), -1)).cpu())
    return torch.cat(out)


# ---------------------------------------------------------------- classical (non-learned) baselines
def meanstd_match(X, ref_stats_from, tgt_stats_from):
    """Per-channel mean/std match: statistics of `tgt_stats_from` are mapped onto those of
    `ref_stats_from` (both unlabeled training images). Returns X transformed, clamped to [-1,1]."""
    mu_t = tgt_stats_from.mean(dim=(0, 2, 3), keepdim=True)
    sd_t = tgt_stats_from.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    mu_r = ref_stats_from.mean(dim=(0, 2, 3), keepdim=True)
    sd_r = ref_stats_from.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    return ((X - mu_t) / sd_t * sd_r + mu_r).clamp(-1, 1)


def hist_match(X, ref_stats_from, tgt_stats_from, n_q=512):
    """Per-channel quantile (histogram) matching of tgt-sensor images onto the ref sensor's
    intensity distribution, using only unlabeled training pixels."""
    q = np.linspace(0, 1, n_q)
    out = np.empty_like(X.numpy())
    Xn = X.numpy()
    for c in range(X.shape[1]):
        t = np.quantile(tgt_stats_from[:, c].numpy().ravel()[::7], q)
        r = np.quantile(ref_stats_from[:, c].numpy().ravel()[::7], q)
        t = np.maximum.accumulate(t + np.arange(n_q) * 1e-9)   # strictly non-decreasing for interp
        out[:, c] = np.interp(Xn[:, c], t, r)
    return torch.from_numpy(out).float().clamp(-1, 1)


# ---------------------------------------------------------------- segmentation training / inference
def _augment(x, m):
    """Per-sample dihedral augmentation (flips + transpose) on GPU tensors."""
    B = x.size(0)
    for dim in (-1, -2):
        f = torch.rand(B, device=x.device) < 0.5
        x = torch.where(f.view(B, 1, 1, 1), x.flip(dim), x)
        m = torch.where(f.view(B, 1, 1), m.flip(dim), m)
    t = torch.rand(B, device=x.device) < 0.5
    x = torch.where(t.view(B, 1, 1, 1), x.transpose(-1, -2), x)
    m = torch.where(t.view(B, 1, 1), m.transpose(-1, -2), m)
    return x, m


def _class_weights(M, device):
    counts = torch.bincount(M[M != IGNORE_INDEX].flatten(), minlength=NUM_CLASSES).float()
    valid = counts > 0
    med = counts[valid].median()
    w = torch.ones(NUM_CLASSES)
    w[valid] = torch.sqrt(med / (counts[valid] + 1e-4))
    w = (w / w.mean()).clamp(0.5, 2.5)
    return w.to(device)


def train_segmenter(X, M, device, seed, epochs=30, lr=1e-3, batch_size=16, uda=None, log=print,
                    steps_per_epoch=75):
    """Trains the land-cover U-Net on labeled images (X, M) with a fixed budget of
    epochs x steps_per_epoch optimizer steps, so that models trained on larger (e.g.
    translation-augmented) sets get the same number of updates as the baseline.

    No target labels and no held-out data are used. No checkpoint selection happens on the
    held-out set: the final-epoch weights are returned.

    uda: optional dict(target_X, generator, z_src) -> also applies the existing DANN (gradient
         reversal on translated target patches) + entropy-minimisation recipe using unlabeled
         target patches only.
    """
    set_seed(seed)
    model = SatelliteUNet(in_channels=4, num_classes=NUM_CLASSES, num_domains=2).to(device)
    crit = SegmentationDiceLoss(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX,
                                label_smoothing=0.01, weight=_class_weights(M, device)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warm = max(1, epochs // 5)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / warm if e < warm else
        max(1e-5 / lr, 0.5 * (1 + np.cos(np.pi * (e - warm) / max(1, epochs - warm)))))
    N = len(X)
    Xg, Mg = X.to(device), M.to(device)
    if uda is not None:
        tX = uda["target_X"].to(device)
        dom_ce = torch.nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        model.train()
        if uda is not None:
            p = (ep - 1) / max(1, epochs - 1)
            model.grl.set_lambda(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
        perm = torch.randperm(N, device=device)
        tot = 0.0
        nb = 0
        for i in range(0, min(N - batch_size + 1, steps_per_epoch * batch_size), batch_size):
            idx = perm[i:i + batch_size]
            xb, mb = _augment(Xg[idx], Mg[idx])
            if uda is None:
                logits, a1, a2 = model(xb, return_aux=True)
                loss = crit(logits, mb) + 0.3 * crit(a1, mb) + 0.3 * crit(a2, mb)
            else:
                logits, dom_s, a1, a2 = model.forward_with_domain(xb, return_aux=True)
                loss = crit(logits, mb) + 0.3 * crit(a1, mb) + 0.3 * crit(a2, mb)
                ub = max(1, batch_size // 2)   # smaller target batch keeps the three passes inside 4 GB
                tb = tX[torch.randint(0, len(tX), (ub,), device=device)]
                with torch.no_grad():
                    tb_adapt = uda["generator"](tb, uda["z_src"].expand(ub, -1))
                _, dom_t = model.forward_with_domain(tb_adapt)
                loss = loss + 0.1 * (dom_ce(dom_s, torch.zeros(batch_size, dtype=torch.long, device=device)) +
                                     dom_ce(dom_t, torch.ones(ub, dtype=torch.long, device=device)))
                probs = F.softmax(model(tb), dim=1)
                ent = -(probs * torch.log(probs + 1e-7)).sum(1).mean()
                mean_p = probs.mean(dim=(0, 2, 3))
                div = (mean_p * torch.log(mean_p + 1e-7)).sum()
                loss = loss + 0.05 * ent + 0.05 * div
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            log(f"      epoch {ep:02d}/{epochs}  loss {tot / nb:.4f}")
    model.eval()
    return model


@torch.no_grad()
def predict_probs(model, X, device, tta=True, chunk=32):
    """Softmax probabilities (CPU). TTA = mean over identity, h-flip and v-flip."""
    model.eval()
    out = []
    for i in range(0, len(X), chunk):
        xb = X[i:i + chunk].to(device)
        p = torch.softmax(model(xb), 1)
        if tta:
            p = p + torch.softmax(model(xb.flip(-1)), 1).flip(-1) + torch.softmax(model(xb.flip(-2)), 1).flip(-2)
            p = p / 3.0
        out.append(p.cpu())
    return torch.cat(out)


def seg_metrics(probs_or_preds, M):
    preds = probs_or_preds.argmax(1) if probs_or_preds.dim() == 4 else probs_or_preds
    m = compute_segmentation_metrics(preds, M, NUM_CLASSES, ignore_index=IGNORE_INDEX)
    return {"mIoU": 100 * m["mIoU"], "acc": 100 * m["overall_accuracy"], "f1": 100 * m["macro_f1"]}


def bootstrap_gain(preds_a, preds_b, M, n_boot=300, seed=0):
    """95% CI of mIoU(a) - mIoU(b), resampling patches. Uses one confusion matrix per patch."""
    def per_patch_hist(preds):
        h = []
        for i in range(len(M)):
            t = M[i].flatten()
            ok = (t != IGNORE_INDEX) & (t < NUM_CLASSES)
            h.append(torch.bincount(NUM_CLASSES * t[ok] + preds[i].flatten()[ok],
                                    minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES).float())
        return torch.stack(h)

    def miou(hist):
        tp = torch.diag(hist)
        den = hist.sum(0) + hist.sum(1) - tp
        act = (den > 0)
        act[IGNORE_INDEX] = False
        return 100 * (tp[act] / den[act]).mean().item()

    ha, hb = per_patch_hist(preds_a), per_patch_hist(preds_b)
    g = torch.Generator().manual_seed(seed)
    diffs = []
    for _ in range(n_boot):
        idx = torch.randint(0, len(M), (len(M),), generator=g)
        diffs.append(miou(ha[idx].sum(0)) - miou(hb[idx].sum(0)))
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ---------------------------------------------------------------- generator fidelity
def _sobel_mag(x):
    kx = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]).view(1, 1, 3, 3).to(x.device)
    g = x.mean(1, keepdim=True)
    gx, gy = F.conv2d(g, kx, padding=1), F.conv2d(g, kx.transpose(-1, -2), padding=1)
    return (gx ** 2 + gy ** 2).sqrt()


def _corr(a, b):
    a, b = a.flatten(1), b.flatten(1)
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)).mean().item()


def _ssim01(a, b, win=7):
    """Mean SSIM over a full batch (images in [0,1]); stride-1 uniform window."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pool = lambda t: F.avg_pool2d(t, win, stride=1, padding=win // 2)
    mu_a, mu_b = pool(a), pool(b)
    va, vb, cab = pool(a * a) - mu_a ** 2, pool(b * b) - mu_b ** 2, pool(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * cab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return s.mean().item()


def fidelity(pred, real_tgt, src, tgt_mean):
    """Compares a translation against the aligned real target patch (all in [-1,1]).
    Reports PSNR / SSIM (on [0,1] rescaled images) and Sobel edge correlation, for the
    translation and for two trivial references: copying the source and a constant image
    equal to the target sensor's mean pixel."""
    def block(x):
        a, b = (x + 1) / 2, (real_tgt + 1) / 2
        mse = ((a - b) ** 2).mean().item()
        return {"psnr": 10 * np.log10(1.0 / max(mse, 1e-12)), "ssim": _ssim01(a, b),
                "edge_corr": _corr(_sobel_mag(x), _sobel_mag(real_tgt))}
    const = tgt_mean.view(1, -1, 1, 1).expand_as(real_tgt)
    return {"translation": block(pred), "copy_source": block(src), "constant_target_mean": block(const)}
