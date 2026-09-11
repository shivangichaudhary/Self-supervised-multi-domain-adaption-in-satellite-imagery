import torch
import numpy as np

def compute_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = -100):
    """
    preds: (N, H, W) integer class predictions
    targets: (N, H, W) integer ground truth classes
    """
    mask = (targets >= 0) & (targets < num_classes)
    if ignore_index >= 0:
        mask = mask & (targets != ignore_index)
        
    hist = torch.bincount(
        num_classes * targets[mask].to(torch.int64) + preds[mask].to(torch.int64),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist

def compute_segmentation_metrics(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = -100):
    """
    Computes mIoU, per-class IoU, Overall Accuracy, and Macro F1 score.
    """
    hist = compute_confusion_matrix(preds, targets, num_classes, ignore_index=ignore_index).float()
    
    tp = torch.diag(hist)
    fp = hist.sum(dim=0) - tp
    fn = hist.sum(dim=1) - tp
    
    # Exclude ignore_index class from mIoU computation
    valid_classes = torch.ones(num_classes, dtype=torch.bool)
    if 0 <= ignore_index < num_classes:
        valid_classes[ignore_index] = False
    
    denominator = tp + fp + fn
    active_mask = (denominator > 0) & valid_classes.to(denominator.device)
    
    iou = torch.where(denominator > 0, tp / denominator, torch.zeros_like(denominator))
    miou = iou[active_mask].mean().item() if active_mask.any() else 0.0
    
    precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
    recall = torch.where(tp + fn > 0, tp / (tp + fn), torch.zeros_like(tp))
    
    f1 = torch.where(precision + recall > 0, 2 * (precision * recall) / (precision + recall), torch.zeros_like(precision))
    macro_f1 = f1[active_mask].mean().item() if active_mask.any() else 0.0
    
    total_correct = tp[valid_classes.to(tp.device)].sum().item()
    total_pixels = hist[valid_classes.to(hist.device), :].sum().item()
    overall_accuracy = (total_correct / total_pixels) if total_pixels > 0 else 0.0
    
    return {
        "mIoU": miou,
        "overall_accuracy": overall_accuracy,
        "macro_f1": macro_f1,
        "per_class_iou": iou.cpu().numpy().tolist()
    }

def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0):
    """
    Computes Peak Signal-to-Noise Ratio between two images.
    img1, img2: tensors in range [0, 1] or [-1, 1]
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(torch.tensor(data_range) / torch.sqrt(mse)).item()

def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11):
    """
    Computes mean structural similarity index (SSIM) between batches of images.
    img1, img2: (B, C, H, W)
    """
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    
    if img1.size(0) > 8:
        img1 = img1[:8]
        img2 = img2[:8]
    
    mu1 = torch.nn.functional.avg_pool2d(img1, window_size, stride=2, padding=window_size//2)
    mu2 = torch.nn.functional.avg_pool2d(img2, window_size, stride=2, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = torch.nn.functional.avg_pool2d(img1 * img1, window_size, stride=2, padding=window_size//2) - mu1_sq
    sigma2_sq = torch.nn.functional.avg_pool2d(img2 * img2, window_size, stride=2, padding=window_size//2) - mu2_sq
    sigma12 = torch.nn.functional.avg_pool2d(img1 * img2, window_size, stride=2, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()
