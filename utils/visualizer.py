import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.cm as cm

# Default fallback color map for 6 basic classes
COLOR_MAP = [
    [0.7, 0.7, 0.7],  # Background: Gray
    [0.1, 0.7, 0.2],  # Vegetation: Deep Green
    [0.1, 0.4, 0.9],  # Water: Blue
    [0.9, 0.2, 0.2],  # Urban: Red
    [0.9, 0.8, 0.1],  # Agriculture: Yellow
    [0.6, 0.4, 0.2],  # Bare Soil: Brown
]

def tensor_to_display_image(img_tensor: torch.Tensor, domain: str = "sentinel2"):
    """
    Converts a (C, H, W) normalized tensor [-1, 1] to a (H, W, 3) numpy image [0, 1] for visualization.
    """
    img = img_tensor.detach().cpu().float().numpy()
    img = (img + 1.0) / 2.0
    img = np.clip(img, 0.0, 1.0)
    
    if img.shape[0] >= 3:
        # Standard RGB (Channels 0, 1, 2)
        rgb = np.transpose(img[:3], (1, 2, 0))
    elif img.shape[0] == 2:
        # SAR VV, VH -> Composite RGB (VV, VH, VV/VH)
        vv = img[0]
        vh = img[1]
        ratio = np.clip(vv / (vh + 1e-3), 0.0, 1.0)
        rgb = np.stack([vv, vh, ratio], axis=-1)
    else:
        # Grayscale to RGB
        rgb = np.repeat(img[0, :, :, np.newaxis], 3, axis=-1)
        
    return rgb

def mask_to_color(mask_tensor: torch.Tensor, num_classes: int = None):
    """
    Converts (H, W) class index mask to (H, W, 3) RGB color overlay.
    Supports arbitrary number of classes using tab20 colormap.
    """
    mask = mask_tensor.detach().cpu().numpy().astype(int)
    H, W = mask.shape
    color_img = np.zeros((H, W, 3), dtype=np.float32)
    
    unique_classes = np.unique(mask)
    max_cls = int(mask.max())
    
    if max_cls < len(COLOR_MAP):
        palette = COLOR_MAP
    else:
        # Use tab20 / viridis for multi-class benchmarks (e.g. PASTIS 20 classes)
        cmap = cm.get_cmap('tab20', max(20, max_cls + 1))
        palette = [cmap(i)[:3] for i in range(max(20, max_cls + 1))]
        palette[0] = [0.2, 0.2, 0.2] # Class 0 Background as dark gray
    
    for cls_idx in unique_classes:
        if cls_idx >= 0:
            color = palette[cls_idx % len(palette)]
            match = (mask == cls_idx)
            color_img[match] = color
        
    return color_img

def plot_translation_matrix(source_imgs, translations_dict, domain_names, save_path: str):
    """
    Plots a multi-sensor translation matrix showing each source adapted to all target domains.
    """
    N = len(source_imgs)
    num_domains = len(domain_names)
    
    fig, axes = plt.subplots(N, num_domains + 1, figsize=(3 * (num_domains + 1), 3 * N))
    if N == 1:
        axes = np.expand_dims(axes, 0)
        
    for i in range(N):
        # Column 0: Original Source
        axes[i, 0].imshow(tensor_to_display_image(source_imgs[i]))
        axes[i, 0].set_title(f"Source Sample {i+1}", fontsize=10, fontweight="bold")
        axes[i, 0].axis("off")
        
        # Subsequent columns: Translated to Domain j
        for j, d_name in enumerate(domain_names):
            translated_img = translations_dict[d_name][i]
            axes[i, j + 1].imshow(tensor_to_display_image(translated_img, domain=d_name))
            axes[i, j + 1].set_title(f"-> {d_name}", fontsize=9)
            axes[i, j + 1].axis("off")
            
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved translation matrix to {save_path}")

def plot_continuous_interpolation(source_img, style_interpolations, domain_a_name, domain_b_name, alphas, save_path: str):
    """
    Plots smooth continuous interpolation between two distinct sensor style embeddings.
    """
    K = len(style_interpolations)
    fig, axes = plt.subplots(1, K + 1, figsize=(3 * (K + 1), 3.2))
    
    # Source
    axes[0].imshow(tensor_to_display_image(source_img))
    axes[0].set_title("Source Image", fontsize=10, fontweight="bold")
    axes[0].axis("off")
    
    for k, (img_k, alpha) in enumerate(zip(style_interpolations, alphas)):
        axes[k + 1].imshow(tensor_to_display_image(img_k))
        axes[k + 1].set_title(f"α={alpha:.2f}\n{domain_a_name} -> {domain_b_name}", fontsize=8)
        axes[k + 1].axis("off")
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved continuous style interpolation to {save_path}")

def plot_downstream_segmentation_comparison(src_img, tgt_img, gt_mask, pred_no_adapt, pred_adapted, save_path: str):
    """
    Visualizes task-driven downstream segmentation gains:
    [Target Image] | [Ground Truth Mask] | [Pred: No Adaptation (Fails)] | [Pred: Self-Supervised Adapted (Accurate)]
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(tensor_to_display_image(tgt_img))
    axes[0].set_title("Target Sensor Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(mask_to_color(gt_mask))
    axes[1].set_title("Ground Truth Mask", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    
    axes[2].imshow(mask_to_color(pred_no_adapt))
    axes[2].set_title("Source Model (No DA)\n[Degraded by Domain Shift]", fontsize=11, color="darkred")
    axes[2].axis("off")
    
    axes[3].imshow(mask_to_color(pred_adapted))
    axes[3].set_title("Proposed Adapted Model\n[Accurate Land-Cover Alignment]", fontsize=11, color="darkgreen", fontweight="bold")
    axes[3].axis("off")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved downstream segmentation comparison to {save_path}")

def plot_angle_interpolation(source_img, angle_imgs, angle_labels, save_path: str):
    """
    Visualizes smooth geometric transitions from nadir to oblique acquisition angles.

    Renders a horizontal strip where each column is the same scene rendered at a
    progressively larger off-nadir angle. Annotations highlight key geometric effects:
    building lean, GSD blur, and brightness fall-off.

    source_img       : (C, H, W) tensor — the nadir reference image
    angle_imgs       : list of (C, H, W) tensors — one per angle step
    angle_labels     : list of str — e.g. ['nadir_0', 'offnadir_15', 'offnadir_30', 'oblique_45']
    save_path        : output PNG path
    """
    N = len(angle_imgs)
    fig, axes = plt.subplots(1, N + 1, figsize=(3.2 * (N + 1), 4.0))

    # Column 0: original nadir reference
    axes[0].imshow(tensor_to_display_image(source_img))
    axes[0].set_title("Source\n(Nadir Reference)", fontsize=10, fontweight="bold", color="#1a1a2e")
    axes[0].axis("off")

    # Angle effect descriptions for annotation
    angle_notes = {
        "nadir_0":      "0° — No distortion",
        "offnadir_15":  "15° — Mild blur + lean",
        "offnadir_30":  "30° — Foreshortening visible",
        "oblique_45":   "45° — Heavy layover + blur",
    }

    cmap_colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]

    for k, (img_k, label) in enumerate(zip(angle_imgs, angle_labels)):
        col = k + 1
        axes[col].imshow(tensor_to_display_image(img_k))
        note = angle_notes.get(label, label)
        color = cmap_colors[k % len(cmap_colors)]
        axes[col].set_title(f"{note}", fontsize=9, color=color, fontweight="bold")
        axes[col].axis("off")

    fig.suptitle("Multi-Angle View Synthesis: Nadir → Oblique Geometric Transitions",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved angle interpolation plot to {save_path}")

