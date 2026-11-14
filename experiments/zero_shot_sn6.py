"""Objective 3 (zero-shot part): generators trained on PASTIS-HD applied to SpaceNet 6 (Rotterdam).

Neither SpaceNet 6 sensor (Maxar optical, Capella SAR), nor the region, nor the optical-SAR pairing was seen
in training. Optical patches are translated into the SAR style and SAR patches into the optical style,
and compared with the aligned real patch of the other sensor. Two ways of specifying the target style:
  canonical  the PASTIS style code of the nearest sensor (SPOT 6 for optical, Sentinel-1 for SAR)
  exemplar   the encoder embedding of unlabeled SpaceNet 6 target patches (new-sensor style, no labels, no pairs)
Fidelity metrics are the same as in the leave-one-pair-out experiment (PSNR, SSIM, Sobel edge correlation),
next to two trivial references (copy the source, constant target-mean image). The same spot6<->SAR pair is
also scored on the PASTIS held-out patches so the SpaceNet 6 numbers can be read against an in-distribution one.
Per-patch paired differences (translation minus reference) get a bootstrap 95% CI.
"""
import argparse
import glob
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import load_config, apply_dataset
from utils.seg_experiment import load_domain, load_models, load_generator, canonical_styles, translate, fidelity

OPT, SAR = "spot6", "sentinel1_sar"
METRICS = ("psnr", "ssim", "edge_corr")


def load_patches(root, domain):
    paths = sorted(glob.glob(os.path.join(root, domain, "images", "*.pt")))
    return torch.stack([torch.load(p, weights_only=True) for p in paths]).float()


@torch.no_grad()
def exemplar_style(encoder, X, device, n=300):
    z = torch.cat([encoder(X[i:i + 50].to(device)) for i in range(0, min(len(X), n), 50)]).mean(0, keepdim=True)
    return torch.nn.functional.normalize(z, p=2, dim=1)


def per_patch(pred, real, src, mean, device):
    rows = [fidelity(pred[i:i + 1].to(device), real[i:i + 1].to(device), src[i:i + 1].to(device), mean) for i in range(len(real))]
    return {k: {m: np.array([r[k][m] for r in rows]) for m in METRICS} for k in rows[0]}


def summarize(pp, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(pp["translation"]["psnr"])
    idx = rng.integers(0, n, size=(n_boot, n))
    out = {"n_patches": n}
    for k, v in pp.items():
        out[k] = {m: float(v[m].mean()) for m in METRICS}
    for ref in ("copy_source", "constant_target_mean"):
        out[f"translation_minus_{ref}"] = {}
        for m in METRICS:
            d = pp["translation"][m] - pp[ref][m]
            boots = d[idx].mean(1)
            out[f"translation_minus_{ref}"][m] = {"mean": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sn6", default="data/spacenet6_dataset", help="folder made by data/convert_spacenet6.py")
    ap.add_argument("--enc_dir", default="outputs_pastis_fixed")
    ap.add_argument("--gens", nargs="+", default=["published=outputs_pastis_fixed/generator.pth"],
                    help="name=path pairs of generator checkpoints")
    ap.add_argument("--out", default="results/objective3/zero_shot_sn6.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_pastis_ref", action="store_true")
    args = ap.parse_args()

    dev = args.device
    cfg = apply_dataset(load_config(None), "pastis")
    root = cfg["data"]["data_dir"]
    enc = load_models(args.enc_dir, cfg, dev)
    Xo, Xs = load_patches(args.sn6, OPT), load_patches(args.sn6, SAR)
    assert len(Xo) == len(Xs) and len(Xo) > 0
    print(f"[*] SpaceNet 6 patches: {len(Xo)}  device={dev}", flush=True)
    tr = {d: load_domain(root, d, "train")[0] for d in (OPT, SAR)}
    z_can = canonical_styles(enc, tr, dev)
    z_ex = {OPT: exemplar_style(enc, Xo, dev), SAR: exemplar_style(enc, Xs, dev)}
    sn6 = {OPT: Xo, SAR: Xs}
    mean_sn6 = {d: sn6[d].mean(dim=(0, 2, 3)).to(dev) for d in sn6}
    mean_pastis = {d: tr[d].mean(dim=(0, 2, 3)).to(dev) for d in tr}
    va = {d: load_domain(root, d, "val")[0] for d in (OPT, SAR)} if not args.no_pastis_ref else None

    results = {"n_patches": len(Xo), "sn6_dir": args.sn6}
    for spec in args.gens:
        name, path = spec.split("=", 1)
        g = load_generator(path, cfg, dev)
        R = results[name] = {}
        for src, tgt in ((OPT, SAR), (SAR, OPT)):
            key = f"{src}->{tgt}"
            for style_name, z in (("canonical", z_can[tgt]), ("exemplar", z_ex[tgt])):
                pred = translate(g, sn6[src], z, dev)
                R[f"sn6 {key} [{style_name} style]"] = summarize(per_patch(pred, sn6[tgt], sn6[src], mean_sn6[tgt], dev))
            if va is not None:
                pred = translate(g, va[src], z_can[tgt], dev)
                R[f"pastis_heldout {key} [canonical style]"] = summarize(per_patch(pred, va[tgt], va[src], mean_pastis[tgt], dev))
        for k, v in R.items():
            t, c, m = v["translation"], v["copy_source"], v["constant_target_mean"]
            print(f"  {name:10s} {k:48s} n={v['n_patches']:3d} PSNR {t['psnr']:.2f} (copy {c['psnr']:.2f}, mean {m['psnr']:.2f}) "
                  f"SSIM {t['ssim']:.3f} (copy {c['ssim']:.3f}, mean {m['ssim']:.3f}) edge {t['edge_corr']:.3f} (copy {c['edge_corr']:.3f})", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print("[*] saved", args.out, flush=True)


if __name__ == "__main__":
    main()
