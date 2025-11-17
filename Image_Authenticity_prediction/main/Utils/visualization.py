"""
### Utility functions for plotting data and images.
"""

import os
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import cv2  # OpenCV for image processing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns
import torch
from matplotlib.figure import Figure
from .logger import info, warn, error, debug

def visualize_similarity_matrix(
    results: Mapping[str, Any],
    model_names: Sequence[str],
    metric: str = 'cosine',
    stat: str = 'mean',
    figsize: Tuple[int, int] = (12, 8),
    cmap: Union[str, colors.Colormap] = 'coolwarm',
    annotate: bool = True,
    plot_upper_triangle: bool = False,
) -> Figure:  # If True, will now plot LOWER triangle
    """
    Visualize similarity matrix between models.
    If plot_upper_triangle is True, it will display the LOWER triangle.
    Color scales are automatically fixed based on the metric.
    """
    if metric not in results['summary']:
        raise ValueError(f"Metric '{metric}' not found in results. Available metrics: {list(results['summary'].keys())}")

    n_models = len(model_names)
    similarity_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        if metric == 'mse' or metric == 'emd':
            similarity_matrix[i, i] = 0.0
        else:
            similarity_matrix[i, i] = 1.0

    for pair, metrics_data in results['summary'][metric].items():
        model_indices = pair.split('_vs_')
        if len(model_indices) == 2 and model_indices[0].isdigit() and model_indices[1].isdigit():
            i, j = map(int, model_indices)
            if i < n_models and j < n_models:
                similarity_matrix[i, j] = metrics_data[stat]
                similarity_matrix[j, i] = metrics_data[stat]
            else:
                warn(f"Model indices {i}, {j} from pair '{pair}' are out of bounds. Skipping.")
        else:
            warn(f"Could not parse indices from pair '{pair}'. Skipping.")

    plt.figure(figsize=figsize)
    current_cmap = cmap
    custom_vmin, custom_vmax = None, None
    cbar_label = ""

    if metric in ['cosine', 'correlation', 'ssim']:
        custom_vmin, custom_vmax = -1.0, 1.0
        cbar_label = f"{metric.capitalize()} Similarity"
        if cmap == 'coolwarm_r': current_cmap = 'coolwarm'
    elif metric == 'mse':
        custom_vmin = 0.0
        if cmap == 'coolwarm': current_cmap = 'coolwarm_r'
        cbar_label = "Mean Squared Error"
    elif metric == 'emd':
        custom_vmin = 0.0
        if cmap == 'coolwarm': current_cmap = 'coolwarm_r'
        cbar_label = "Earth Mover's Distance (Wasserstein)"
    else:
        cbar_label = f"{metric.capitalize()} {stat.capitalize()}"

    mask: Optional[np.ndarray] = None
    if plot_upper_triangle: # Parameter now implies plotting ONLY ONE triangle
        # To plot the LOWER triangle, we mask the UPPER triangle.
        # k=1 means exclude the diagonal from the mask.
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)

    ax = sns.heatmap(
        similarity_matrix,
        annot=annotate,
        fmt=".2f",
        cmap=current_cmap,
        xticklabels=model_names,
        yticklabels=model_names,
        cbar_kws={"label": cbar_label},
        mask=mask,
        vmin=custom_vmin,
        vmax=custom_vmax
    )
    title_metric_name = metric.upper() if metric in ['mse', 'emd'] else metric.capitalize()
    plt.title(f"{title_metric_name} {stat.capitalize()} Between Heatmaps of Different Models")
    plt.tight_layout()
    return plt.gcf()

def denormalize_image(
    tensor: torch.Tensor,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Denormalizes an image tensor."""
    if tensor.dim() != 3:
        raise ValueError(f"Input tensor must have 3 dimensions (C, H, W), but got {tensor.dim()}")
    
    mean_seq = list(mean)
    std_seq = list(std)
    mean_used: Sequence[float] = mean_seq
    std_used: Sequence[float] = std_seq
    if tensor.shape[0] != len(mean_seq) or tensor.shape[0] != len(std_seq):
        if tensor.shape[0] == 1: # Grayscale
            warn("Denormalizing grayscale with potentially RGB stats. Using first value of mean/std.")
            mean_used = [float(mean_seq[0])] if mean_seq else [0.0]
            std_used = [float(std_seq[0])] if std_seq else [1.0]
        else:
            raise ValueError(
                f"Channel mismatch: Tensor has {tensor.shape[0]} channels, mean has {len(mean_seq)}, std has {len(std_seq)}"
            )

    mean_t = torch.as_tensor(mean_used, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std_t = torch.as_tensor(std_used, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)

    denormalized_tensor = tensor * std_t + mean_t
    return torch.clamp(denormalized_tensor, 0., 1.)

def visualize_and_save_saliency(
    image_tensor: torch.Tensor,
    saliency_map: np.ndarray,
    output_dir: str,
    filename_prefix: str,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    overlay_alpha: float = 0.5,
    cmap_name: str = 'bwr',
) -> None:
    """
    Visualizes saliency map, creates an overlay, and saves images.
    """
    if image_tensor.is_cuda:
        debug("image_tensor provided to visualize_and_save_saliency is on CUDA, moving to CPU.")
        image_tensor = image_tensor.cpu()

    # Create a sub-folder for each image's visualizations
    image_specific_output_dir = os.path.join(output_dir, filename_prefix)
    os.makedirs(image_specific_output_dir, exist_ok=True)
    
    temp_dir = os.path.join(image_specific_output_dir, 'temp_heatmap_cache') 
    os.makedirs(temp_dir, exist_ok=True)

    if saliency_map.ndim != 2:
        error(
            "Saliency map has unexpected dimensions %s. Expected (H, W). Skipping visualization.",
            saliency_map.shape,
        )
        return
    saliency_map = np.clip(saliency_map, 0.0, 1.0)
    
    NUMPY_DIR = os.path.join(output_dir, 'numpy_saliency_maps') # Centralized numpy maps
    os.makedirs(NUMPY_DIR, exist_ok=True)
    np.save(os.path.join(NUMPY_DIR, f"{filename_prefix}_saliency_map.npy"), saliency_map)

    try:
        img_denorm_tensor = denormalize_image(image_tensor, mean, std)
    except ValueError as e:
        error(f"Error during denormalization for {filename_prefix}: {e}. Skipping visualization.")
        return

    img_np = img_denorm_tensor.numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0.0, 1.0)
    img_uint8 = (img_np * 255).astype(np.uint8)

    if img_uint8.shape[2] == 1:
        img_display = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    elif img_uint8.shape[2] == 3:
        img_display = img_uint8
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    else:
        error(
            "Unexpected number of channels (%s) for %s. Skipping visualization.",
            img_uint8.shape[2],
            filename_prefix,
        )
        return

    orig_save_path = os.path.join(image_specific_output_dir, f"{filename_prefix}_original.png")
    plt.figure(figsize=(img_display.shape[1]/100, img_display.shape[0]/100), dpi=100) # Match size
    plt.imshow(img_display)
    plt.axis('off'); plt.title("Original Image")
    plt.savefig(orig_save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    try:
        cmap = cm.get_cmap(cmap_name)
    except ValueError:
        warn(f"Colormap '{cmap_name}' not found. Using default 'viridis'.")
        cmap = cm.get_cmap('viridis')
    norm = colors.Normalize(vmin=0, vmax=1)

    heatmap_save_path = os.path.join(image_specific_output_dir, f"{filename_prefix}_heatmap_{cmap_name}.png")
    plt.figure(figsize=(saliency_map.shape[1]/100, saliency_map.shape[0]/100), dpi=100) # Match size
    plt.imshow(saliency_map, cmap=cmap, norm=norm)
    plt.colorbar(label=f'Saliency')
    plt.title(f"Saliency Heatmap ({cmap_name})"); plt.axis('off')
    plt.savefig(heatmap_save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    temp_heatmap_path = os.path.join(temp_dir, f"{filename_prefix}_temp_heatmap_for_overlay.png")
    fig_width_inches = img_display.shape[1] / 100.0
    fig_height_inches = img_display.shape[0] / 100.0
    plt.figure(figsize=(fig_width_inches, fig_height_inches), dpi=100)
    plt.imshow(saliency_map, cmap=cmap, norm=norm); plt.axis('off')
    plt.savefig(temp_heatmap_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

    colored_heatmap_bgr = cv2.imread(temp_heatmap_path)
    if os.path.exists(temp_heatmap_path): os.remove(temp_heatmap_path)
    if os.path.exists(temp_dir) and not os.listdir(temp_dir): 
        try: os.rmdir(temp_dir)
        except OSError: pass # Might fail if another process/thread is accessing

    if colored_heatmap_bgr is None:
        error(
            "Could not read temporary heatmap file for %s: %s. Skipping overlay.",
            filename_prefix,
            temp_heatmap_path,
        )
        return

    if colored_heatmap_bgr.shape[:2] != img_bgr.shape[:2]:
        debug(
            "Resizing heatmap from %s to %s for %s",
            colored_heatmap_bgr.shape[:2],
            img_bgr.shape[:2],
            filename_prefix,
        )
        colored_heatmap_bgr = cv2.resize(
            colored_heatmap_bgr,
            (img_bgr.shape[1], img_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    overlay = cv2.addWeighted(img_bgr, 1.0 - overlay_alpha, colored_heatmap_bgr, overlay_alpha, 0.0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    overlay_save_path = os.path.join(image_specific_output_dir, f"{filename_prefix}_overlay_{cmap_name}.png")
    success = cv2.imwrite(overlay_save_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    if not success:
        error("cv2.imwrite failed for overlay %s. Trying plt.savefig.", overlay_save_path)
        plt.figure(figsize=(overlay_rgb.shape[1]/100, overlay_rgb.shape[0]/100), dpi=100)
        plt.imshow(overlay_rgb); plt.axis('off'); plt.title(f"Saliency Overlay ({cmap_name})")
        plt.savefig(overlay_save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
