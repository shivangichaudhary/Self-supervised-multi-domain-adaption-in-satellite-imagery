"""Convert SpaceNet 6 (Rotterdam) tiles into the 128x128x4 [-1, 1] format used for PASTIS-HD.

SpaceNet 6 pairs 0.5 m Maxar pan-sharpened optical tiles (PS-RGB) with Capella X-band quad-pol SAR
tiles (SAR-Intensity: HH, HV, VH, VV) on the same 900x900 grid. Neither sensor, nor the region, is in
the PASTIS training set, so this data is only used for zero-shot evaluation of generators trained on PASTIS.

Channel conventions follow data/convert_real_datasets.py so the PASTIS-trained models see the same layout:
  optical ("spot6" domain)       R, G, B, and the (R+G)/2 NIR proxy, uint8 / 255 * 2 - 1
  SAR ("sentinel1_sar" domain)   VV, VH, VV-VH, mean(VV, VH) in dB, clip(-30, 0 dB) / 15 + 1

Things worth stating in a write-up:
  * Tiles carry a black no-data band (a rotated footprint, identical in both sensors). Only windows that are
    fully valid after a small erosion are kept; each window is `--crop` native pixels square and is resized
    to `--size`. The default crop of 512 px (256 m) gives a 2 m ground sampling distance, against 10 m for the
    PASTIS patches: the scale gap cannot be closed with 900 px tiles and is a confounder of the zero-shot test.
  * SAR-Intensity is stored on a log-like scale (median ~25, p99 ~45, skewness ~0.5), not in calibrated sigma0.
    It is mapped to the PASTIS Sentinel-1 dB range by an affine fit per polarisation that aligns the 5th and
    95th percentiles of the valid pixels to the PASTIS values. This is a fixed two-number calibration per
    polarisation (saved to calibration.json), not a per-image adjustment. Capella is X-band, Sentinel-1 C-band,
    so the SAR domains differ physically as well as radiometrically.
  * Averaging while downsampling is done on the stored (log) values; any bias is absorbed by the calibration.
  * The files are read with a minimal uncompressed-TIFF reader so no GDAL/rasterio dependency is needed.
"""
import argparse
import glob
import json
import os
import struct

import numpy as np
import torch
from scipy import ndimage as ndi

_KIND = {1: "u", 2: "i", 3: "f"}
_TYPE_FMT = {1: ("B", 1), 3: ("H", 2), 4: ("I", 4), 8: ("h", 2), 9: ("i", 4), 11: ("f", 4), 12: ("d", 8)}

# 5th / 95th percentile of the PASTIS-HD Sentinel-1 training patches, in dB (measured on data/france_dataset)
PASTIS_DB_P5_P95 = {"VV": (-16.9, -6.6), "VH": (-24.2, -12.8)}
SN6_BANDS = {"VV": 3, "VH": 2}   # SAR-Intensity band order is HH, HV, VH, VV


def read_tiff(path):
    """Read an uncompressed, strip-organised baseline TIFF as a (C, H, W) array."""
    data = open(path, "rb").read()
    bo = "<" if data[:2] == b"II" else ">"
    if struct.unpack(bo + "H", data[2:4])[0] != 42:
        raise ValueError(f"{path}: not a classic TIFF")
    ifd = struct.unpack(bo + "I", data[4:8])[0]
    n_entries = struct.unpack(bo + "H", data[ifd:ifd + 2])[0]
    tags = {}
    for i in range(n_entries):
        e = data[ifd + 2 + 12 * i: ifd + 14 + 12 * i]
        tag, typ, cnt = struct.unpack(bo + "HHI", e[:8])
        if typ not in _TYPE_FMT:
            continue
        fmt, size = _TYPE_FMT[typ]
        raw = e[8:8 + size * cnt] if size * cnt <= 4 else data[struct.unpack(bo + "I", e[8:12])[0]:][:size * cnt]
        v = struct.unpack(bo + f"{cnt}{fmt}", raw)
        tags[tag] = v if cnt > 1 else v[0]
    as_tuple = lambda v: v if isinstance(v, tuple) else (v,)
    if tags.get(259, 1) != 1 or 322 in tags or 273 not in tags:
        raise ValueError(f"{path}: only uncompressed strip TIFFs are supported")
    w, h = tags[256], tags[257]
    spp = tags.get(277, 1)
    bits = as_tuple(tags[258])[0]
    kind = _KIND[as_tuple(tags.get(339, 1))[0]]
    dt = np.dtype(f"{bo}{kind}{bits // 8}")
    offs, cnts = as_tuple(tags[273]), as_tuple(tags[279])
    arr = np.frombuffer(b"".join(data[o:o + c] for o, c in zip(offs, cnts)), dtype=dt)
    if tags.get(284, 1) == 1:   # chunky: interleaved samples
        return arr.reshape(h, w, spp).transpose(2, 0, 1).astype(dt.newbyteorder("="))
    return arr.reshape(spp, h, w).astype(dt.newbyteorder("="))   # planar: one plane after another


def _resize(a, size, mode):
    """(C, H, W) array -> (C, size, size); 'area' box-averages, 'bilinear' interpolates with antialiasing."""
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None]
    kw = {"mode": "area"} if mode == "area" else {"mode": "bilinear", "align_corners": False, "antialias": True}
    return torch.nn.functional.interpolate(t, size=(size, size), **kw)[0].numpy()


def valid_mask(opt, sar, erode=8):
    """Pixels that carry data in both sensors, shrunk by `erode` px to avoid the footprint edge."""
    v = (opt.max(0) > 0) & (sar.max(0) > 0)
    return ndi.binary_erosion(v, iterations=erode) if erode else v


def find_windows(valid, crop, step=32):
    """Greedy, non-overlapping, fully valid crop x crop windows (top-left corners)."""
    h, w = valid.shape
    ii = np.pad(valid.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    taken, out = np.zeros_like(valid, dtype=bool), []
    for y in range(0, h - crop + 1, step):
        for x in range(0, w - crop + 1, step):
            if ii[y + crop, x + crop] - ii[y, x + crop] - ii[y + crop, x] + ii[y, x] == crop * crop and not taken[y:y + crop, x:x + crop].any():
                taken[y:y + crop, x:x + crop] = True
                out.append((y, x))
    return out


def optical_to_tensor(rgb_uint8, size):
    x = _resize(rgb_uint8.astype(np.float32), size, "bilinear")
    x = np.concatenate([x, 0.5 * x[0:1] + 0.5 * x[1:2]], axis=0)       # NIR proxy, as for SPOT in PASTIS
    return torch.clamp(torch.from_numpy(x) / 255.0 * 2.0 - 1.0, -1.0, 1.0)


def fit_calibration(valid_samples):
    """valid_samples: {'VV': 1-D array, 'VH': 1-D array} of stored SAR values on valid pixels -> affine maps to dB."""
    cal = {}
    for pol, vals in valid_samples.items():
        lo, hi = np.percentile(vals, [5, 95])
        t_lo, t_hi = PASTIS_DB_P5_P95[pol]
        a = (t_hi - t_lo) / (hi - lo)
        cal[pol] = {"a": float(a), "b": float(t_lo - a * lo), "sn6_p5": float(lo), "sn6_p95": float(hi)}
    return cal


def sar_to_tensor(intensity, size, cal):
    """(4, H, W) SAR-Intensity crop -> VV, VH, VV-VH, mean in dB mapped to [-1, 1] (PASTIS convention)."""
    s = _resize(intensity[[SN6_BANDS["VV"], SN6_BANDS["VH"]]].astype(np.float32), size, "area")
    vv = cal["VV"]["a"] * s[0:1] + cal["VV"]["b"]
    vh = cal["VH"]["a"] * s[1:2] + cal["VH"]["b"]
    x = np.concatenate([vv, vh, vv - vh, 0.5 * (vv + vh)], axis=0)
    return torch.clamp(torch.from_numpy(x), -30.0, 0.0) / 15.0 + 1.0


def tile_id(path):
    return os.path.basename(path).split("_")[-1].replace(".tif", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing PS-RGB/ and SAR-Intensity/ (the AOI_11_Rotterdam folder)")
    ap.add_argument("--out", default="data/spacenet6_dataset")
    ap.add_argument("--size", type=int, default=128, help="output patch size")
    ap.add_argument("--crop", type=int, default=512, help="native-pixel window resized to --size (512 px = 256 m -> 2 m GSD)")
    ap.add_argument("--max_tiles", type=int, default=0, help="0 = all tiles")
    ap.add_argument("--calib_tiles", type=int, default=200, help="tiles used to fit the SAR calibration")
    args = ap.parse_args()

    opt = {tile_id(p): p for p in glob.glob(os.path.join(args.src, "PS-RGB", "*.tif"))}
    sar = {tile_id(p): p for p in glob.glob(os.path.join(args.src, "SAR-Intensity", "*.tif"))}
    ids = sorted(set(opt) & set(sar), key=int)
    if args.max_tiles:
        ids = ids[:args.max_tiles]
    print(f"[*] {len(ids)} tiles with both optical and SAR", flush=True)
    for d in ("spot6", "sentinel1_sar"):
        os.makedirs(os.path.join(args.out, d, "images"), exist_ok=True)

    # pass 1: windows and SAR calibration
    rng = np.random.default_rng(0)
    plans, samples = {}, {"VV": [], "VH": []}
    for k, t in enumerate(ids):
        o, s = read_tiff(opt[t]), read_tiff(sar[t])
        v = valid_mask(o, s)
        plans[t] = find_windows(v, args.crop)
        if k < args.calib_tiles and v.any():
            for pol in samples:
                px = s[SN6_BANDS[pol]][v]
                samples[pol].append(rng.choice(px, size=min(20000, px.size), replace=False))
    cal = fit_calibration({p: np.concatenate(v) for p, v in samples.items()})
    json.dump({"pastis_reference_db_p5_p95": PASTIS_DB_P5_P95, "fit": cal, "crop": args.crop, "size": args.size},
              open(os.path.join(args.out, "calibration.json"), "w"), indent=1)
    print("[*] SAR calibration:", {p: {k: round(v, 3) for k, v in c.items()} for p, c in cal.items()}, flush=True)

    # pass 2: write patches
    meta, n = [], 0
    for t in ids:
        if not plans[t]:
            continue
        o, s = read_tiff(opt[t]), read_tiff(sar[t])
        for (y, x) in plans[t]:
            name = f"sn6_{int(t):05d}_{y}_{x}.pt"
            torch.save(optical_to_tensor(o[:, y:y + args.crop, x:x + args.crop], args.size),
                       os.path.join(args.out, "spot6", "images", name))
            torch.save(sar_to_tensor(s[:, y:y + args.crop, x:x + args.crop], args.size, cal),
                       os.path.join(args.out, "sentinel1_sar", "images", name))
            meta.append({"patch": name, "tile": t, "y": y, "x": x})
            n += 1
    json.dump(meta, open(os.path.join(args.out, "patches.json"), "w"), indent=0)
    print(f"[OK] wrote {n} paired patches from {sum(bool(v) for v in plans.values())}/{len(ids)} tiles to {args.out}")


if __name__ == "__main__":
    main()
