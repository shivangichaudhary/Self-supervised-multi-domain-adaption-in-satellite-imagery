"""Objective 4: does adapting across sensors improve downstream land-cover segmentation?

Protocol (PASTIS-HD, 19 crop/land classes + void, labels from the S2 train split only):
  * The segmentation U-Net is trained with Sentinel-2 labels. No SPOT/SAR labels are used by
    any adapted variant. The 300 held-out patches of each sensor are touched only for testing.
  * Variants compared on each target sensor (SPOT, SAR) and on S2 itself:
      source-only            S2 model applied directly
      meanstd / histmatch    classical per-channel radiometric matching of the target image to S2
      gen (test-time)        generator translates the target image to the S2 style
      gen + raw ensemble     the original "dual-path" combination of the two
      train on gen (S->T)    S2 train images translated to the target style by the generator
      train on stats (S->T)  same, with the classical mean/std transform instead of the generator
      train on S2 + gen(all) one model trained on S2 plus its translations into every target style
      DANN + entropy (UDA)   existing feature-space recipe, unlabeled target patches only
      target-supervised      upper bound: trained with the target sensor's own labels
  * Everything is repeated for several seeds (segmenter init/order/augmentation).
"""
import argparse
import json
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import load_config, apply_dataset
from utils.seed import set_seed
from utils.seg_experiment import (SENSORS, load_domain, load_models, load_generator, canonical_styles, translate,
                                  meanstd_match, hist_match, train_segmenter, predict_probs, seg_metrics,
                                  bootstrap_gain)

TARGETS = ["spot6", "sentinel1_sar"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--ckpt_dir", default="outputs_pastis_fixed")
    ap.add_argument("--out", default="results/objective4")
    ap.add_argument("--skip_uda", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    cfg = apply_dataset(load_config(None), "pastis")
    root = cfg["data"]["data_dir"]
    print(f"[*] device={dev} seeds={args.seeds} epochs={args.epochs}", flush=True)

    tr = {d: load_domain(root, d, "train") for d in SENSORS}
    va = {d: load_domain(root, d, "val") for d in SENSORS}
    print({d: (tuple(tr[d][0].shape), tuple(va[d][0].shape)) for d in SENSORS}, flush=True)
    Xtr = {d: tr[d][0] for d in SENSORS}
    Mtr = {d: tr[d][1] for d in SENSORS}

    enc = load_models(args.ckpt_dir, cfg, dev)
    gen = load_generator(os.path.join(args.ckpt_dir, "generator.pth"), cfg, dev)
    z = canonical_styles(enc, Xtr, dev)
    S2 = "sentinel2"

    # ---- seed-independent data: translations and classical matching
    t0 = time.time()
    test_in = {}    # target -> variant -> images fed to the S2-trained model at test time
    for t in TARGETS:
        Xv = va[t][0]
        test_in[t] = {
            "source_only": Xv,
            "meanstd": meanstd_match(Xv, Xtr[S2], Xtr[t]),
            "histmatch": hist_match(Xv, Xtr[S2], Xtr[t]),
            "gen_test_time": translate(gen, Xv, z[S2], dev),
        }
    train_sets = {}  # name -> (X, M)
    for t in TARGETS:
        train_sets[f"train_gen_{t}"] = (translate(gen, Xtr[S2], z[t], dev), Mtr[S2])
        train_sets[f"train_stats_{t}"] = (meanstd_match(Xtr[S2], Xtr[t], Xtr[S2]), Mtr[S2])
    train_sets["train_s2_plus_gen_all"] = (
        torch.cat([Xtr[S2]] + [train_sets[f"train_gen_{t}"][0] for t in TARGETS]),
        torch.cat([Mtr[S2]] * (1 + len(TARGETS))))
    print(f"[*] prepared translated/matched data in {time.time() - t0:.0f}s", flush=True)

    results = {}
    keep_preds = {}   # seed-0 predictions for bootstrap CIs
    for seed in args.seeds:
        R = results[str(seed)] = {t: {} for t in TARGETS + [S2]}
        P = {}
        print(f"\n===== seed {seed} =====", flush=True)
        log = lambda s: print(s, flush=True)

        def fit(name, X, M, **kw):
            t1 = time.time()
            log(f"  [train] {name} ({len(X)} imgs)")
            m = train_segmenter(X, M, dev, seed, epochs=args.epochs, log=log, **kw)
            log(f"  [train] {name} done in {time.time() - t1:.0f}s")
            return m

        # 1. source-only S2 model + test-time variants
        m0 = fit("source_only_s2", Xtr[S2], Mtr[S2])
        p_s2 = predict_probs(m0, va[S2][0], dev)
        R[S2]["source_only"] = seg_metrics(p_s2, va[S2][1])
        log(f"  S2 in-domain: {R[S2]['source_only']}")
        for t in TARGETS:
            Mv = va[t][1]
            probs = {k: predict_probs(m0, v, dev) for k, v in test_in[t].items()}
            for k, p in probs.items():
                R[t][k] = seg_metrics(p, Mv)
            ens = (probs["gen_test_time"] + probs["source_only"])
            R[t]["gen_raw_ensemble"] = seg_metrics(ens, Mv)
            P[(t, "source_only")] = probs["source_only"].argmax(1)
            P[(t, "gen_test_time")] = probs["gen_test_time"].argmax(1)
            P[(t, "gen_raw_ensemble")] = ens.argmax(1)
            P[(t, "meanstd")] = probs["meanstd"].argmax(1)
            P[(t, "histmatch")] = probs["histmatch"].argmax(1)
            log(f"  {t}: " + " | ".join(f"{k} {v['mIoU']:.2f}" for k, v in R[t].items()))
            del probs, ens
        del m0

        # 2. train on translated / matched source, test on native target
        for t in TARGETS:
            for tag, key in (("train_gen", f"train_gen_{t}"), ("train_stats", f"train_stats_{t}")):
                m = fit(f"{key}", *train_sets[key])
                p = predict_probs(m, va[t][0], dev)
                R[t][tag] = seg_metrics(p, va[t][1])
                P[(t, tag)] = p.argmax(1)
                log(f"  {t} {tag}: {R[t][tag]}")
                del m, p
        m = fit("train_s2_plus_gen_all", *train_sets["train_s2_plus_gen_all"])
        for t in TARGETS + [S2]:
            p = predict_probs(m, va[t][0], dev)
            R[t]["train_s2_plus_gen_all"] = seg_metrics(p, va[t][1])
            P[(t, "train_s2_plus_gen_all")] = p.argmax(1)
            log(f"  {t} train_s2_plus_gen_all: {R[t]['train_s2_plus_gen_all']}")
        del m

        # 3. existing DANN + entropy recipe (unlabeled target patches only)
        if not args.skip_uda:
            tgt_unl = torch.cat([Xtr[t] for t in TARGETS])
            m = fit("uda_dann_entropy", Xtr[S2], Mtr[S2],
                    uda={"target_X": tgt_unl, "generator": gen, "z_src": z[S2]})
            for t in TARGETS:
                p = predict_probs(m, va[t][0], dev)
                R[t]["uda_dann_entropy"] = seg_metrics(p, va[t][1])
                P[(t, "uda_dann_entropy")] = p.argmax(1)
                log(f"  {t} uda: {R[t]['uda_dann_entropy']}")
            del m

        # 4. target-supervised upper bound (uses the target sensor's own training labels)
        for t in TARGETS:
            m = fit(f"supervised_{t}", Xtr[t], Mtr[t])
            p = predict_probs(m, va[t][0], dev)
            R[t]["target_supervised"] = seg_metrics(p, va[t][1])
            P[(t, "target_supervised")] = p.argmax(1)
            log(f"  {t} supervised: {R[t]['target_supervised']}")
            del m, p

        if seed == args.seeds[0]:
            keep_preds = P
        json.dump(results, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)

    # bootstrap CIs of mIoU gain over source-only (first seed, resampling the 300 patches)
    ci = {}
    for t in TARGETS:
        base = keep_preds[(t, "source_only")]
        ci[t] = {}
        for (tt, k), pr in keep_preds.items():
            if tt == t and k != "source_only":
                ci[t][k] = bootstrap_gain(pr, base, va[t][1])
    json.dump({"metrics": results, "bootstrap_gain_vs_source_only_ci95_seed0": ci},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
    print("\n[*] saved", os.path.join(args.out, "metrics.json"), flush=True)


if __name__ == "__main__":
    main()
