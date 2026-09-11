import os
import glob
import torch
import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def _apply_augmentation(img: torch.Tensor, mask: torch.Tensor):
    """
    Satellite-safe augmentation for (C, H, W) image and (H, W) mask.
    Applied identically to image and mask to maintain pixel-label alignment.
    Augmentations:
      - RandomHorizontalFlip (p=0.5)
      - RandomVerticalFlip   (p=0.5)
      - Random 90° Rotation  (k=0,1,2,3)
      - Per-channel Gaussian noise injection (sigma ~ U[0, 0.01])
    """
    # Random horizontal flip
    if random.random() > 0.5:
        img = torch.flip(img, dims=[-1])
        mask = torch.flip(mask, dims=[-1])

    # Random vertical flip
    if random.random() > 0.5:
        img = torch.flip(img, dims=[-2])
        mask = torch.flip(mask, dims=[-2])

    # Random 90° rotation (k in {0, 1, 2, 3})
    k = random.randint(0, 3)
    if k > 0:
        img = torch.rot90(img, k=k, dims=[-2, -1])
        mask = torch.rot90(mask.unsqueeze(0), k=k, dims=[-2, -1]).squeeze(0)

    # Per-channel Gaussian noise (simulates sensor radiometric noise)
    sigma = random.uniform(0.0, 0.01)
    noise = torch.randn_like(img) * sigma
    img = (img + noise).clamp(-1.0, 1.0)

    return img, mask


class MultiSensorSatelliteDataset(Dataset):
    """
    Unified dataset supporting multi-sensor domain loading (Sentinel-2, Landsat-8, SPOT-6, SAR).
    Provides samples for:
      1. Self-supervised domain encoder training (unlabeled patches + style augmentations)
      2. Multi-domain conditional translation training (source image + target domain exemplar)
      3. Downstream semantic segmentation benchmarking
    
    For training split, applies random flips, 90-degree rotations, and Gaussian noise augmentation
    to effectively multiply dataset size and enforce spatially-invariant feature learning.
    """
    def __init__(self, root_dir: str, domains=None, split: str = "train", split_ratio: float = 0.8, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.augment = (split == "train")  # augmentation only during training
        
        if domains is None:
            self.domains = ["sentinel2", "landsat8", "spot6", "sentinel1_sar"]
        else:
            self.domains = domains

        # Domain -> off-nadir angle in degrees, for domains that are angle domains.
        # Sensor domains simply aren't in this map (see .get(..., -1) below).
        from data.synthetic_generator import SyntheticSatelliteGenerator
        self.angle_degrees = SyntheticSatelliteGenerator.ANGLE_DEGREES

        self.domain_to_idx = {d: i for i, d in enumerate(self.domains)}
        self.samples_by_domain = {d: [] for d in self.domains}
        self.all_samples = []

        for domain in self.domains:
            img_dir = os.path.join(root_dir, domain, "images")
            mask_dir = os.path.join(root_dir, domain, "masks")
            if not os.path.exists(img_dir):
                continue
            
            img_paths = sorted(glob.glob(os.path.join(img_dir, "*.pt")))
            split_idx = int(len(img_paths) * split_ratio)
            
            if split == "train":
                selected_paths = img_paths[:split_idx]
            else:
                selected_paths = img_paths[split_idx:]

            for p in selected_paths:
                base = os.path.basename(p)
                m_path = os.path.join(mask_dir, base)
                has_mask = os.path.exists(m_path)
                item = {
                    "image_path": p,
                    "mask_path": m_path if has_mask else None,
                    "domain": domain,
                    "domain_idx": self.domain_to_idx[domain]
                }
                self.samples_by_domain[domain].append(item)
                self.all_samples.append(item)

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        src_item = self.all_samples[idx]
        src_img = torch.load(src_item["image_path"], weights_only=True)
        src_mask = torch.load(src_item["mask_path"], weights_only=True) if src_item["mask_path"] else torch.zeros((src_img.shape[1], src_img.shape[2]), dtype=torch.long)
        
        if self.transform:
            src_img, src_mask = self.transform(src_img, src_mask)

        # Apply augmentation to training data (image+mask synchronized)
        if self.augment:
            src_img, src_mask = _apply_augmentation(src_img, src_mask)
            
        # Sample a random target domain and an exemplar from that domain (for style conditioning)
        target_domain = random.choice(self.domains)
        tgt_items = self.samples_by_domain[target_domain]
        tgt_item = random.choice(tgt_items) if len(tgt_items) > 0 else src_item
        
        tgt_img = torch.load(tgt_item["image_path"], weights_only=True)
        if self.transform:
            tgt_img = self.transform(tgt_img)

        return {
            "src_img": src_img,
            "src_mask": src_mask,
            "src_domain": src_item["domain"],
            "src_domain_idx": src_item["domain_idx"],
            # -1.0 sentinel means "not an angle domain" — see ContinuousDomainEncoder.forward
            "src_angle_deg": float(self.angle_degrees.get(src_item["domain"], -1)),
            "tgt_img": tgt_img,
            "tgt_domain": target_domain,
            "tgt_domain_idx": self.domain_to_idx[target_domain],
            "tgt_angle_deg": float(self.angle_degrees.get(target_domain, -1))
        }

def get_dataloaders(root_dir: str, domains=None, batch_size: int = 8, num_workers: int = 0):
    from data.transforms import SatelliteTransforms
    train_transform = SatelliteTransforms(is_train=True)
    val_transform = SatelliteTransforms(is_train=False)
    
    train_set = MultiSensorSatelliteDataset(root_dir, domains=domains, split="train", transform=train_transform)
    val_set = MultiSensorSatelliteDataset(root_dir, domains=domains, split="val", transform=val_transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader


