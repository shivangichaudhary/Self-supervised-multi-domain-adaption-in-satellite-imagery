"""Growing-season variant of the converted PASTIS-HD dataset, used for the Objective 4 segmentation experiments.

Why: data/convert_real_datasets.py takes the *first* acquisition of each Sentinel-2 / Sentinel-1 time series. In
PASTIS-HD that is 24 Sep 2018 for every patch (after harvest, before the new crops are visible), so land-cover
labels are close to unreadable from that image (in-domain mIoU of a U-Net stays near 10%) and no adaptation
effect can be measured against such a baseline.

What: Sentinel-2 and Sentinel-1A inputs are re-extracted as the per-pixel median over all acquisitions between
WINDOW_START and WINDOW_END (2019-05-15 .. 2019-07-15, about 8 Sentinel-2 and 9 Sentinel-1A dates). The median
suppresses clouds (the Sentinel-2 series is not cloud-filtered). The window was fixed before any adaptation
result was seen. Patch order, the train/held-out split, the SPOT images and the masks are identical to the
original dataset (files are copied), and the channel/normalisation conventions are unchanged.

Not touched: data/france_dataset (the Objective 1-3 data and the published results).

    python data/convert_pastis_season.py --pastis data/PASTIS-HD --orig data/france_dataset --out data/france_dataset_season
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import torch

WINDOW_START, WINDOW_END = 20190515, 20190715
MAX_PATCHES = 1500


def common_patch_ids(pastis, n):
    """Same ordering as process_pastis_hd(): sorted string IDs present in S2, S1A, SPOT and TARGET."""
    def ids(pattern, strip):
        return {os.path.splitext(os.path.basename(p))[0].replace(strip, "") for p in glob.glob(os.path.join(pastis, pattern))}
    s2, s1 = ids("DATA_S2/S2_*.npy", "S2_"), ids("DATA_S1A/S1A_*.npy", "S1A_")
    spot = {os.path.splitext(os.path.basename(p))[0].split("_")[-1] for p in glob.glob(os.path.join(pastis, "**/*SPOT*.tif*"), recursive=True)}
    tgt = ids("ANNOTATIONS/TARGET_*.npy", "TARGET_")
    return sorted(s2 & s1 & spot & tgt)[:n]


def window_indices(dates):
    """dates: {'0': 20180924, ...} -> time indices inside the window, in order."""
    idx = [int(k) for k, v in sorted(dates.items(), key=lambda kv: int(kv[0])) if WINDOW_START <= v <= WINDOW_END]
    if not idx:
        raise ValueError("no acquisition inside the window")
    return idx


def s2_tensor(path, idx):
    x = np.load(path, mmap_mode="r")[idx][:, [2, 1, 0, 6]].astype(np.float32)   # R, G, B, NIR
    x = np.median(x, axis=0)
    return torch.clamp(torch.from_numpy(x) / 5000.0 * 2.0 - 1.0, -1.0, 1.0)


def s1_tensor(path, idx):
    x = np.median(np.load(path, mmap_mode="r")[idx][:, :2].astype(np.float32), axis=0)   # VV, VH in dB
    vv, vh = x[0:1], x[1:2]
    x = np.concatenate([vv, vh, vv - vh, 0.5 * (vv + vh)], axis=0)
    return torch.clamp(torch.from_numpy(x), -30.0, 0.0) / 15.0 + 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pastis", default="data/PASTIS-HD")
    ap.add_argument("--orig", default="data/france_dataset")
    ap.add_argument("--out", default="data/france_dataset_season")
    ap.add_argument("--max_patches", type=int, default=MAX_PATCHES)
    args = ap.parse_args()

    meta = {str(f["properties"]["ID_PATCH"]): f["properties"] for f in json.load(open(os.path.join(args.pastis, "metadata.geojson")))["features"]}
    pids = common_patch_ids(args.pastis, args.max_patches)
    print(f"[*] {len(pids)} patches; window {WINDOW_START}..{WINDOW_END}", flush=True)
    for d in ("sentinel2", "spot6", "sentinel1_sar"):
        os.makedirs(os.path.join(args.out, d, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.out, d, "masks"), exist_ok=True)

    n_s2, n_s1 = [], []
    for i, pid in enumerate(pids):
        name = f"france_patch_{i:05d}.pt"
        i2, i1 = window_indices(meta[pid]["dates-S2"]), window_indices(meta[pid]["dates-S1A"])
        n_s2.append(len(i2)); n_s1.append(len(i1))
        torch.save(s2_tensor(os.path.join(args.pastis, "DATA_S2", f"S2_{pid}.npy"), i2), os.path.join(args.out, "sentinel2", "images", name))
        torch.save(s1_tensor(os.path.join(args.pastis, "DATA_S1A", f"S1A_{pid}.npy"), i1), os.path.join(args.out, "sentinel1_sar", "images", name))
        for d, sub in (("spot6", "images"), ("sentinel2", "masks"), ("spot6", "masks"), ("sentinel1_sar", "masks")):   # unchanged files
            shutil.copyfile(os.path.join(args.orig, d, sub, name), os.path.join(args.out, d, sub, name))
        if (i + 1) % 100 == 0 or i + 1 == len(pids):
            print(f"  {i + 1}/{len(pids)}", flush=True)

    # sanity: the original dataset's patch i must be the first acquisition of the same patch id
    first = lambda path, fn: fn(path, [0])
    chk = [0, len(pids) // 2, len(pids) - 1]
    for i in chk:
        a = first(os.path.join(args.pastis, "DATA_S2", f"S2_{pids[i]}.npy"), s2_tensor)
        b = torch.load(os.path.join(args.orig, "sentinel2", "images", f"france_patch_{i:05d}.pt"), weights_only=True)
        assert torch.allclose(a, b, atol=1e-6), f"patch order mismatch at {i}"
    json.dump({"window": [WINDOW_START, WINDOW_END], "patches": len(pids), "s2_dates_per_patch_min_med_max":
               [int(np.min(n_s2)), int(np.median(n_s2)), int(np.max(n_s2))], "s1a_dates_per_patch_min_med_max":
               [int(np.min(n_s1)), int(np.median(n_s1)), int(np.max(n_s1))], "pids": pids},
              open(os.path.join(args.out, "season.json"), "w"))
    open(os.path.join(args.out, "READY"), "w").write("ok\n")
    print(f"[OK] wrote {args.out}; S2 dates/patch (min/med/max) {min(n_s2)}/{int(np.median(n_s2))}/{max(n_s2)}")


if __name__ == "__main__":
    main()
