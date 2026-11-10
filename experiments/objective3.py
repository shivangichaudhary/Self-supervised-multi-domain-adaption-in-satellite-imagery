"""Objective 3: leave-one-pair-out generalization of the conditional generator.

For each unordered sensor pair {A, B} the generator is retrained from scratch with both ordered
pairs A->B and B->A removed from training (identity pairs and every other pair stay). It is then
asked to perform exactly the translation it never saw. The reference is a generator trained with
the same recipe and seed on all pairs ("full"). Reported per held-out pair:
  * fidelity of A->B / B->A against the aligned real target patch (PSNR, SSIM, edge correlation),
    next to two trivial references (copy the source, constant target-mean image);
  * downstream effect for pairs that involve Sentinel-2 (the labeled sensor): test-time translation
    of the target image to S2 style, and training on S2 translated to the target style.
The domain encoder is label-free and shared (checkpoint from Objective 1); it saw every sensor's
images individually but no pairing information.
"""
import argparse
import copy
import json
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import load_config, apply_dataset
from train_generator import train_generator
from utils.seg_experiment import (SENSORS, load_domain, load_models, load_generator, canonical_styles, translate,
                                  train_segmenter, predict_probs, seg_metrics, fidelity)

S2 = "sentinel2"
PAIRS = [("sentinel2", "spot6"), ("sentinel2", "sentinel1_sar"), ("spot6", "sentinel1_sar")]
ORDERED = [(a, b) for a in SENSORS for b in SENSORS if a != b]


def cfg_name(pair):
    return "full" if pair is None else f"hold_{pair[0]}_{pair[1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gen_epochs", type=int, default=12)
    ap.add_argument("--seg_epochs", type=int, default=20)
    ap.add_argument("--enc_dir", default="outputs_pastis_fixed", help="folder holding domain_encoder.pth")
    ap.add_argument("--out", default="results/objective3")
    ap.add_argument("--no_downstream", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg0 = apply_dataset(load_config(None), "pastis")
    cfg0["generator"]["epochs"] = args.gen_epochs
    root = cfg0["data"]["data_dir"]
    os.makedirs(args.out, exist_ok=True)
    tr = {d: load_domain(root, d, "train") for d in SENSORS}
    va = {d: load_domain(root, d, "val") for d in SENSORS}
    enc = load_models(args.enc_dir, cfg0, dev)
    z = canonical_styles(enc, {d: tr[d][0] for d in SENSORS}, dev)
    tgt_mean = {d: tr[d][0].mean(dim=(0, 2, 3)).to(dev) for d in SENSORS}
    mpath = os.path.join(args.out, "metrics.json")
    results = json.load(open(mpath)) if os.path.exists(mpath) else {}

    for seed in args.seeds:
        R = results.setdefault(str(seed), {})
        print(f"\n===== seed {seed} =====", flush=True)
        gens = {}
        for pair in [None] + PAIRS:
            name = cfg_name(pair)
            ck = os.path.join(args.out, "ckpt", f"seed{seed}_{name}")
            if not os.path.exists(os.path.join(ck, "generator.pth")):
                cfg = copy.deepcopy(cfg0)
                cfg["seed"] = seed
                cfg["output_dir"] = ck
                cfg["generator"]["excluded_pairs"] = [] if pair is None else [list(pair), [pair[1], pair[0]]]
                t0 = time.time()
                print(f"\n--- training generator: seed {seed}, {name}", flush=True)
                train_generator(cfg, encoder=enc, device=dev)
                print(f"--- done in {time.time() - t0:.0f}s", flush=True)
            else:
                print(f"--- reusing {ck}", flush=True)
            gens[name] = load_generator(os.path.join(ck, "generator.pth"), cfg0, dev)

        # ---- fidelity on all six ordered pairs, for every generator
        for name, g in gens.items():
            F_ = R.setdefault(name, {}).setdefault("fidelity", {})
            for s, t in ORDERED:
                pred = translate(g, va[s][0], z[t], dev)
                F_[f"{s}->{t}"] = fidelity(pred.to(dev), va[t][0].to(dev), va[s][0].to(dev), tgt_mean[t])
        json.dump(results, open(mpath, "w"), indent=1)

        if args.no_downstream:
            continue

        # ---- downstream effect (pairs with the labeled sensor, Sentinel-2)
        def fit(label, X, M):
            t1 = time.time()
            m = train_segmenter(X, M, dev, seed, epochs=args.seg_epochs, log=lambda s: None)
            print(f"  [train] {label} {time.time() - t1:.0f}s", flush=True)
            return m

        m0 = fit("source_only_s2", tr[S2][0], tr[S2][1])
        base = {}
        for t in ("spot6", "sentinel1_sar"):
            base[t] = seg_metrics(predict_probs(m0, va[t][0], dev), va[t][1])
        R["downstream_source_only"] = base
        for name, g in gens.items():
            D = R[name].setdefault("downstream", {})
            for t in ("spot6", "sentinel1_sar"):
                held = name == f"hold_{S2}_{t}"
                xin = translate(g, va[t][0], z[S2], dev)
                D[f"test_time_{t}->s2"] = seg_metrics(predict_probs(m0, xin, dev), va[t][1])
                if name == "full" or held:
                    Xt = translate(g, tr[S2][0], z[t], dev)
                    m = fit(f"{name}: train on S2->{t}", Xt, tr[S2][1])
                    D[f"train_s2->{t}"] = seg_metrics(predict_probs(m, va[t][0], dev), va[t][1])
                    del m
            print(f"  {name}: {json.dumps(D)}", flush=True)
        del m0
        json.dump(results, open(mpath, "w"), indent=1)

    print("\n[*] saved", mpath, flush=True)


if __name__ == "__main__":
    main()
