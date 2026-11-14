"""Reference numbers for the Objective 4 table (CPU only, no training).

* majority-class predictor: predicts the most frequent training class (chosen from the S2 *training* masks) for
  every held-out pixel; this is the floor any segmentation result has to be read against;
* prediction collapse of each variant (seed 0): how many classes it predicts and the share of the top class.

    python experiments/obj4_baselines.py --data_dir data/france_dataset_season
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.seg_experiment import load_domain, seg_metrics, NUM_CLASSES

TARGETS = ["spot6", "sentinel1_sar"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/france_dataset_season")
    ap.add_argument("--out_dir", default="results/objective4")
    args = ap.parse_args()

    _, Mtr = load_domain(args.data_dir, "sentinel2", "train")
    counts = torch.bincount(Mtr[Mtr < NUM_CLASSES].flatten(), minlength=NUM_CLASSES)
    maj = int(counts.argmax())
    out = {"majority_class": maj, "majority_share_of_train_pixels_pct": 100 * counts[maj].item() / counts.sum().item(),
           "majority_predictor": {}, "collapse_seed0": {}}

    preds = torch.load(os.path.join(args.out_dir, "seed0_preds.pt"), weights_only=False)
    for t in TARGETS:
        _, Mv = load_domain(args.data_dir, t, "val")
        out["majority_predictor"][t] = seg_metrics(torch.full_like(Mv, maj), Mv)
        out["collapse_seed0"][t] = {}
        for (tt, k), p in preds.items():
            if tt != t:
                continue
            h = torch.bincount(p.flatten(), minlength=NUM_CLASSES).float()
            out["collapse_seed0"][t][k] = {"classes_predicted": int((h > 0).sum()),
                                           "top_class": int(h.argmax()), "top_class_share_pct": 100 * h.max().item() / h.sum().item()}
    json.dump(out, open(os.path.join(args.out_dir, "baselines.json"), "w"), indent=1)
    print(json.dumps(out["majority_predictor"], indent=1))
    print("[*] saved", os.path.join(args.out_dir, "baselines.json"))


if __name__ == "__main__":
    main()
