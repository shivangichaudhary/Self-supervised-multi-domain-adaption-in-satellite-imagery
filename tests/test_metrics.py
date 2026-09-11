import torch

from utils.metrics import compute_segmentation_metrics, compute_psnr, compute_ssim


def test_perfect_prediction_gives_100_percent():
    preds = torch.tensor([[[0, 1], [1, 0]]])
    targets = preds.clone()
    metrics = compute_segmentation_metrics(preds, targets, num_classes=2)
    assert metrics["mIoU"] == 1.0
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_ignore_index_excluded_from_metrics():
    preds = torch.tensor([[[0, 1], [1, 1]]])
    targets = torch.tensor([[[0, -100], [1, 1]]])
    metrics = compute_segmentation_metrics(preds, targets, num_classes=2, ignore_index=-100)
    # Only 3 valid pixels, all correctly predicted -> perfect score despite the ignored pixel
    assert metrics["overall_accuracy"] == 1.0


def test_compute_psnr_infinite_for_identical_images():
    img = torch.rand(1, 4, 8, 8)
    assert compute_psnr(img, img) == float("inf")


def test_compute_ssim_close_to_one_for_identical_images():
    img = torch.rand(2, 4, 16, 16)
    ssim = compute_ssim(img, img)
    assert ssim > 0.99
