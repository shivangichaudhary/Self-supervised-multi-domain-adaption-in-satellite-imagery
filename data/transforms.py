import torch
import torchvision.transforms.functional as TF
import random

class SatelliteTransforms:
    """
    Data augmentations tailored for multi-sensor satellite imagery.
    Operates on (C, H, W) tensors.
    """
    def __init__(self, is_train: bool = True, crop_size: int = 128):
        self.is_train = is_train
        self.crop_size = crop_size

    def __call__(self, img: torch.Tensor, mask: torch.Tensor = None):
        """
        img: (C, H, W) float tensor
        mask: (H, W) long tensor (optional)
        """
        if not self.is_train:
            return (img, mask) if mask is not None else img

        # Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            if mask is not None:
                mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.5:
            img = TF.vflip(img)
            if mask is not None:
                mask = TF.vflip(mask)

        # Random 90 degree rotation
        rot = random.choice([0, 1, 2, 3])
        if rot > 0:
            img = torch.rot90(img, rot, [-2, -1])
            if mask is not None:
                mask = torch.rot90(mask, rot, [-2, -1])

        return (img, mask) if mask is not None else img
