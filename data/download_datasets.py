"""
Dataset Downloader & Preprocessor for PASTIS-HD and MSC-France
=============================================================
This script provides automated downloading and preprocessing for:
1. PASTIS-HD: Multi-modal agricultural benchmark (Sentinel-1, Sentinel-2, SPOT 6-7) across 2,433 patches in France.
2. MSC-France: Multi-sensor triplets (Sentinel-2 10m, Landsat-8 30m, SPOT-6 1.5m) across 4,496 scenes in France.

Sources:
- PASTIS-HD:
    * Hugging Face: https://huggingface.co/datasets/IGNF/PASTIS-HD
    * Zenodo (PASTIS Base): https://zenodo.org/records/5012357 (21.6 GB)
    * Zenodo (PASTIS-HD VHR): https://zenodo.org/records/7845347 (14.4 GB)
- MSC-France (X-STARS):
    * GitHub: https://github.com/VMarsocci/X-STARS
    * EOTDL Platform: https://www.eotdl.com/datasets/MSC-France
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
import shutil
from tqdm import tqdm

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

PASTIS_HD_HF_REPO = "IGNF/PASTIS-HD"
PASTIS_ZENODO_BASE = "https://zenodo.org/records/5012357/files/PASTIS.zip"
PASTIS_HD_ZENODO_VHR = "https://zenodo.org/records/7845347/files/PASTIS-HD.zip"

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url: str, output_path: str):
    """Download a file with progress bar."""
    print(f"[*] Downloading {url} -> {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_pastis_hd_huggingface(target_dir: str):
    """
    Downloads PASTIS-HD from Hugging Face datasets hub.
    Requires: pip install huggingface_hub
    """
    print(f"\n=======================================================")
    print(f" Downloading PASTIS-HD from Hugging Face Hub...        ")
    print(f"=======================================================")
    try:
        from huggingface_hub import snapshot_download
        out_dir = os.path.join(target_dir, "PASTIS-HD")
        os.makedirs(out_dir, exist_ok=True)
        snapshot_download(
            repo_id=PASTIS_HD_HF_REPO,
            repo_type="dataset",
            local_dir=out_dir,
            local_dir_use_symlinks=False
        )
        print(f"[✓] PASTIS-HD downloaded successfully to: {out_dir}")
    except ImportError:
        print("[!] huggingface_hub is not installed.")
        print("[*] Install via: pip install huggingface_hub")
        print("[*] Falling back to direct Zenodo download instructions...")
        download_pastis_hd_zenodo(target_dir)

def download_pastis_hd_zenodo(target_dir: str):
    """Download base PASTIS and PASTIS-HD VHR companion from Zenodo."""
    out_dir = os.path.join(target_dir, "PASTIS-HD")
    os.makedirs(out_dir, exist_ok=True)
    
    print("\nZenodo Official Archive Links:")
    print(f"  1. PASTIS Base (S1 + S2 + Labels): {PASTIS_ZENODO_BASE}")
    print(f"  2. PASTIS-HD (SPOT 6-7 VHR):       {PASTIS_HD_ZENODO_VHR}")
    print(f"\nDownloading archives into: {out_dir}")
    
    # Base
    base_zip = os.path.join(out_dir, "PASTIS_base.zip")
    if not os.path.exists(base_zip):
        download_url(PASTIS_ZENODO_BASE, base_zip)
        print("[*] Extracting PASTIS base...")
        with zipfile.ZipFile(base_zip, 'r') as zf:
            zf.extractall(out_dir)
            
    # VHR companion
    vhr_zip = os.path.join(out_dir, "PASTIS_HD_vhr.zip")
    if not os.path.exists(vhr_zip):
        download_url(PASTIS_HD_ZENODO_VHR, vhr_zip)
        print("[*] Extracting PASTIS-HD VHR...")
        with zipfile.ZipFile(vhr_zip, 'r') as zf:
            zf.extractall(out_dir)

def download_msc_france(target_dir: str):
    """
    Downloads MSC-France (Sentinel-2, Landsat-8, SPOT-6 triplets).
    """
    print(f"\n=======================================================")
    print(f" Downloading MSC-France (4,496 Triplets)...            ")
    print(f"=======================================================")
    out_dir = os.path.join(target_dir, "MSC-France")
    os.makedirs(out_dir, exist_ok=True)
    
    print("MSC-France is distributed through EOTDL & X-STARS:")
    print("  Official Repository: https://github.com/VMarsocci/X-STARS")
    print("  EOTDL Dataset:       https://www.eotdl.com/datasets/MSC-France")
    print("\nYou can install EOTDL and download directly with:")
    print("  pip install eotdl")
    print("  eotdl dataset download -d MSC-France -p data/MSC-France")
    
    try:
        import subprocess
        print("\n[*] Checking for eotdl CLI tool...")
        subprocess.run(["eotdl", "dataset", "download", "-d", "MSC-France", "-p", out_dir], check=True)
        print(f"[✓] MSC-France downloaded successfully to: {out_dir}")
    except Exception as e:
        print(f"[*] Note: Manual/EOTDL authentication may be required: {e}")
        print(f"[*] Target folder prepared at: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PASTIS-HD and MSC-France Datasets")
    parser.add_argument("--dataset", choices=["all", "pastis_hd", "msc_france"], default="all",
                        help="Which dataset to download (default: all)")
    parser.add_argument("--source", choices=["huggingface", "zenodo"], default="huggingface",
                        help="Download source for PASTIS-HD")
    args = parser.parse_args()

    target_dir = DATA_DIR

    if args.dataset in ["all", "pastis_hd"]:
        if args.source == "huggingface":
            download_pastis_hd_huggingface(target_dir)
        else:
            download_pastis_hd_zenodo(target_dir)

    if args.dataset in ["all", "msc_france"]:
        download_msc_france(target_dir)
