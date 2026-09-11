"""
Converter / Adapter for Real France Benchmark Datasets (PASTIS-HD & MSC-France)
==============================================================================
Converts raw PASTIS-HD and MSC-France downloads into unified domain folders
ready for training:
  data/france_dataset/
    ├── sentinel2/       (images/, masks/)
    ├── spot6/           (images/, masks/)
    ├── sentinel1_sar/   (images/, masks/)
    └── landsat8/        (images/, masks/)
"""

import os
import glob
import torch
import numpy as np
from PIL import Image

def process_pastis_hd(pastis_dir: str, output_dir: str, max_patches: int = 1000):
    """
    Parses PASTIS-HD files and exports aligned Sentinel-2, SPOT-6/7, and Sentinel-1 patches.
    Matches across modalities using common patch IDs.
    """
    print(f"\n[*] Processing PASTIS-HD from {pastis_dir} -> {output_dir}")
    os.makedirs(os.path.join(output_dir, "sentinel2", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "sentinel2", "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "spot6", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "spot6", "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "sentinel1_sar", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "sentinel1_sar", "masks"), exist_ok=True)
    
    # Collect files mapped by patch ID
    s2_files = glob.glob(os.path.join(pastis_dir, "DATA_S2", "S2_*.npy"))
    s2_dict = {}
    for p in s2_files:
        pid = os.path.splitext(os.path.basename(p))[0].replace("S2_", "")
        s2_dict[pid] = p

    s1_files = glob.glob(os.path.join(pastis_dir, "DATA_S1A", "S1A_*.npy"))
    s1_dict = {}
    for p in s1_files:
        pid = os.path.splitext(os.path.basename(p))[0].replace("S1A_", "")
        s1_dict[pid] = p

    spot_files = glob.glob(os.path.join(pastis_dir, "**/*SPOT*.tif*"), recursive=True)
    if not spot_files:
        spot_files = glob.glob(os.path.join(pastis_dir, "**/*VHR*.npy"), recursive=True)
    spot_dict = {}
    for p in spot_files:
        base = os.path.splitext(os.path.basename(p))[0]
        # ID is usually the last component after underscore
        pid = base.split("_")[-1]
        spot_dict[pid] = p

    target_files = glob.glob(os.path.join(pastis_dir, "ANNOTATIONS", "TARGET_*.npy"))
    target_dict = {}
    for p in target_files:
        pid = os.path.splitext(os.path.basename(p))[0].replace("TARGET_", "")
        target_dict[pid] = p

    # Find common IDs across available modalities
    common_pids = sorted(list(set(s2_dict.keys()) & set(s1_dict.keys()) & set(spot_dict.keys()) & set(target_dict.keys())))
    print(f"[*] Found {len(common_pids)} fully aligned multi-modal patches (S2 + SPOT + S1 + Labels).")
    
    total = min(len(common_pids), max_patches)
    count = 0
    for idx in range(total):
        pid = common_pids[idx]
        base_name = f"france_patch_{idx:05d}.pt"
        
        # 1. Sentinel-2: Shape (T, 10, 128, 128) -> Extract B4(Red, idx 2), B3(Green, idx 1), B2(Blue, idx 0), B8(NIR, idx 6)
        s2_raw = np.load(s2_dict[pid])
        if s2_raw.ndim == 4:
            s2_raw = s2_raw[0] # First temporal acquisition
        
        # Select RGB (B4, B3, B2) + NIR (B8)
        if s2_raw.shape[0] >= 7:
            # Indices: Red=2, Green=1, Blue=0, NIR=6
            s2_4c = s2_raw[[2, 1, 0, 6]].astype(np.float32)
        else:
            s2_4c = s2_raw[:4].astype(np.float32)
            
        s2_tensor = torch.tensor(s2_4c, dtype=torch.float32)
        s2_tensor = (s2_tensor / 5000.0) * 2.0 - 1.0
        s2_tensor = torch.clamp(s2_tensor, -1.0, 1.0)
        torch.save(s2_tensor, os.path.join(output_dir, "sentinel2", "images", base_name))

        # 2. SPOT-6/7 (VHR Optical): 1280x1280 -> Resize to 128x128 and pad to 4 channels
        p_spot = spot_dict[pid]
        if p_spot.endswith('.tif') or p_spot.endswith('.tiff'):
            im = Image.open(p_spot)
            im_resized = im.resize((128, 128), Image.Resampling.BILINEAR)
            vhr_np = np.array(im_resized, dtype=np.float32) # (128, 128, 3)
            if vhr_np.ndim == 3:
                vhr_np = np.transpose(vhr_np, (2, 0, 1)) # (3, 128, 128)
            elif vhr_np.ndim == 2:
                vhr_np = np.stack([vhr_np] * 3, axis=0)
            # Add synthetic NIR band from Red/Green ratio or intensity
            if vhr_np.shape[0] == 3:
                nir_approx = (vhr_np[0:1] * 0.5 + vhr_np[1:2] * 0.5)
                vhr_np = np.concatenate([vhr_np, nir_approx], axis=0)
            vhr_tensor = torch.tensor(vhr_np[:4], dtype=torch.float32)
            vhr_tensor = (vhr_tensor / 255.0) * 2.0 - 1.0
            vhr_tensor = torch.clamp(vhr_tensor, -1.0, 1.0)
        else:
            vhr_data = np.load(p_spot)
            if vhr_data.ndim == 4:
                vhr_data = vhr_data[0]
            vhr_tensor = torch.tensor(vhr_data[:4], dtype=torch.float32)
            vhr_tensor = (vhr_tensor / 255.0) * 2.0 - 1.0
            vhr_tensor = torch.clamp(vhr_tensor, -1.0, 1.0)
        torch.save(vhr_tensor, os.path.join(output_dir, "spot6", "images", base_name))

        # 3. Sentinel-1 SAR (VV, VH) -> 4-channel (VV, VH, Ratio, Avg) normalized
        s1_raw = np.load(s1_dict[pid])
        if s1_raw.ndim == 4:
            s1_raw = s1_raw[0]
        s1_np = s1_raw[:2].astype(np.float32)
        ratio = s1_np[0:1] - s1_np[1:2]
        avg = (s1_np[0:1] + s1_np[1:2]) / 2.0
        s1_4c = np.concatenate([s1_np, ratio, avg], axis=0)
        s1_tensor = torch.tensor(s1_4c, dtype=torch.float32)
        # dB normalization into [-1, 1]
        s1_tensor = torch.clamp(s1_tensor, -30.0, 0.0) / 15.0 + 1.0
        torch.save(s1_tensor, os.path.join(output_dir, "sentinel1_sar", "images", base_name))

        # 4. Land-Cover Semantic Segmentation Mask (Channel 0 of TARGET)
        target_raw = np.load(target_dict[pid])
        if target_raw.ndim == 3:
            mask_data = target_raw[0]
        else:
            mask_data = target_raw
        mask_tensor = torch.tensor(mask_data, dtype=torch.long)
        torch.save(mask_tensor, os.path.join(output_dir, "sentinel2", "masks", base_name))
        torch.save(mask_tensor, os.path.join(output_dir, "spot6", "masks", base_name))
        torch.save(mask_tensor, os.path.join(output_dir, "sentinel1_sar", "masks", base_name))

        count += 1
        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"  --> Converted {idx + 1}/{total} patches...", flush=True)

    print(f"[OK] Converted {count} PASTIS-HD patches into {output_dir}\n")

def _load_tif_as_4channel(path: str) -> torch.Tensor:
    """Loads a georeferenced tile as a (4, 128, 128) tensor normalized to [-1, 1]."""
    im = Image.open(path)
    im_resized = im.resize((128, 128), Image.Resampling.BILINEAR)
    arr = np.array(im_resized, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=0)
    else:
        arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
    if arr.shape[0] == 3:
        nir_approx = (arr[0:1] * 0.5 + arr[1:2] * 0.5)
        arr = np.concatenate([arr, nir_approx], axis=0)
    tensor = torch.tensor(arr[:4], dtype=torch.float32)
    tensor = (tensor / 255.0) * 2.0 - 1.0
    return torch.clamp(tensor, -1.0, 1.0)


def process_msc_france(msc_dir: str, output_dir: str, max_triplets: int = 1500):
    """
    Parses MSC-France triplet files (Sentinel-2, Landsat-8, SPOT-6).

    NOTE: MSC-France doesn't ship an explicit triplet-id in its filenames the way PASTIS-HD
    does, so triplets are matched by sorted filename order within each sensor's file list.
    This assumes the three glob results are already co-registered 1:1 in that order (true for
    the X-STARS release layout at the time of writing) — if the downloaded copy is organized
    differently, verify a few triplets by eye before trusting the output.
    """
    print(f"\n[*] Processing MSC-France from {msc_dir} -> {output_dir}")
    os.makedirs(os.path.join(output_dir, "landsat8", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "sentinel2", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "spot6", "images"), exist_ok=True)

    l8_files = sorted(glob.glob(os.path.join(msc_dir, "**/*l8*.tif*"), recursive=True))
    s2_files = sorted(glob.glob(os.path.join(msc_dir, "**/*s2*.tif*"), recursive=True))
    spot_files = sorted(glob.glob(os.path.join(msc_dir, "**/*spot*.tif*"), recursive=True))

    count = min(len(l8_files), len(s2_files), len(spot_files), max_triplets)
    print(f"[*] Found {len(l8_files)} Landsat-8, {len(s2_files)} Sentinel-2, {len(spot_files)} SPOT-6 files.")
    if count == 0:
        print("[!] No aligned triplets found (need matching *l8*, *s2*, *spot* tif files) — nothing to convert.")
        return
    print(f"[*] Converting {count} aligned triplets...")

    for idx in range(count):
        base_name = f"msc_patch_{idx:05d}.pt"
        torch.save(_load_tif_as_4channel(l8_files[idx]), os.path.join(output_dir, "landsat8", "images", base_name))
        torch.save(_load_tif_as_4channel(s2_files[idx]), os.path.join(output_dir, "sentinel2", "images", base_name))
        torch.save(_load_tif_as_4channel(spot_files[idx]), os.path.join(output_dir, "spot6", "images", base_name))
        if (idx + 1) % 50 == 0 or (idx + 1) == count:
            print(f"  --> Converted {idx + 1}/{count} triplets...", flush=True)

    print(f"[OK] Converted {count} MSC-France triplets into {output_dir}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pastis_dir", default="data/PASTIS-HD")
    parser.add_argument("--msc_dir", default="data/MSC-France")
    parser.add_argument("--output_dir", default="data/france_dataset")
    parser.add_argument("--max_patches", type=int, default=300)
    args = parser.parse_args()

    if os.path.exists(args.pastis_dir):
        process_pastis_hd(args.pastis_dir, args.output_dir, max_patches=args.max_patches)
    if os.path.exists(args.msc_dir):
        process_msc_france(args.msc_dir, args.output_dir)

