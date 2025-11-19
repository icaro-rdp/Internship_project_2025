"""Utility functions for comparisons between heatmaps."""

from itertools import combinations
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, TypedDict, Union, cast

import cv2
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from scipy.stats import pearsonr, wasserstein_distance
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import paired_cosine_distances
from skimage.metrics import structural_similarity as ssim

from .logger import warn


MetricName = str
PairKey = str
UniformedHeatmaps = np.ndarray

class MetricSummary(TypedDict):
    mean: float
    std: float
    min: float
    max: float
    median: float

PerImageResults = Dict[MetricName, Dict[PairKey, np.ndarray]]
SummaryResults = Dict[MetricName, Dict[PairKey, MetricSummary]]

class ComparisonResults(TypedDict):
    per_image: PerImageResults
    summary: SummaryResults


def uniform_heatmaps(
    heatmap_array: np.ndarray,
    height: int = 224,
    width: int = 224,
    num_images: Optional[int] = None,
) -> np.ndarray:
    """Reshape heatmaps to a common spatial resolution.
    
    Args:
        heatmap_array (np.ndarray): Array of shape (B, H, W) containing heatmaps.
        height (int): Target height for resizing.
        width (int): Target width for resizing.
        num_images (Optional[int]): If provided, limits the number of heatmaps processed.    
    Returns:
        np.ndarray: Resized heatmaps of shape (B', height, width), where B' is min(B, num_images).
    """
    if heatmap_array.ndim != 3:
        raise ValueError("Input heatmap array must be 3D (B, height, width)")

    if num_images is not None:
        if num_images <= 0:
            raise ValueError("num_images must be a positive integer.")
        if num_images > heatmap_array.shape[0]:
            raise ValueError(
                f"num_images ({num_images}) cannot be greater than the number of available images ({heatmap_array.shape[0]})."
            )
        heatmap_array = heatmap_array[:num_images]

    n_images, h, w = heatmap_array.shape
    target_hw = (height, width)

    if (h, w) == target_hw:
        return heatmap_array

    resized_heatmaps = np.zeros((n_images, height, width), dtype=heatmap_array.dtype)
    resize_shape = (width, height)

    for idx in range(n_images):
        current_heatmap = heatmap_array[idx]
        if current_heatmap.ndim != 2:
            raise ValueError(
                f"Each heatmap in the array must be 2D. Found heatmap at index {idx} with shape {current_heatmap.shape}"
            )
        resized_heatmaps[idx] = cv2.resize(current_heatmap, resize_shape, interpolation=cv2.INTER_LINEAR)

    return resized_heatmaps


def compare_heatmaps(
    heatmap_arrays: Sequence[np.ndarray],
    metrics: Sequence[str] = ("mse", "correlation", "cosine", "ssim", "emd"),
) -> ComparisonResults:
    """Compare similarity between multiple heatmap collections.
    
    Args:
        heatmap_arrays (Sequence[np.ndarray]): List of heatmap arrays to compare. Each array should have shape (B, H, W).
        metrics (Sequence[str]): List of metrics to compute. Supported metrics: "mse", "correlation", "cosine", "ssim", "emd".
    Returns:
        ComparisonResults: Dictionary containing per-image and summary similarity results.
    """
    if not heatmap_arrays:
        raise ValueError("heatmap_arrays must contain at least one array.")

    metrics_list = list(dict.fromkeys(metrics))
    if not metrics_list:
        raise ValueError("metrics must contain at least one entry.")

    reference_shape = heatmap_arrays[0].shape
    n_images = reference_shape[0]

    for idx, array in enumerate(heatmap_arrays):
        if array.shape != reference_shape:
            raise ValueError(f"Array {idx} has shape {array.shape}, expected {reference_shape}")

    # Flatten arrays once for vectorized operations (Shape: N_images x Flattened_Dim)
    flat_arrays = [arr.reshape(n_images, -1) for arr in heatmap_arrays]

    per_image: PerImageResults = {metric: {} for metric in metrics_list}
    array_pairs = list(combinations(range(len(heatmap_arrays)), 2))

    compute_mse = "mse" in metrics_list
    compute_corr = "correlation" in metrics_list
    compute_cosine = "cosine" in metrics_list
    compute_ssim = "ssim" in metrics_list
    compute_emd = "emd" in metrics_list

    for i, j in array_pairs:
        pair_key = f"{i}_vs_{j}"
        
        flat1 = flat_arrays[i]
        flat2 = flat_arrays[j]

        if compute_mse:
            # Calculate mean across the feature dimension (axis 1)
            per_image["mse"][pair_key] = np.mean((flat1 - flat2) ** 2, axis=1)

        if compute_cosine:
            # paired_cosine_distances returns distance (0=same, 2=opposite). Convert to similarity.
            dists = paired_cosine_distances(flat1, flat2)
            per_image["cosine"][pair_key] = 1.0 - dists

        if compute_corr:
            corrs = np.zeros(n_images, dtype=float)
            for k in range(n_images):
                corrs[k] = pearsonr(flat1[k], flat2[k])[0]
            per_image["correlation"][pair_key] = corrs

        if compute_ssim:
            ssims = np.zeros(n_images, dtype=float)
            for k in range(n_images):
                # Use data_range=2.0 because heatmaps are normalized between -1 and 1
                ssims[k] = ssim(
                    heatmap_arrays[i][k], 
                    heatmap_arrays[j][k], 
                    data_range=2.0 
                )
            per_image["ssim"][pair_key] = ssims

        if compute_emd:
            emds = np.zeros(n_images, dtype=float)
            for k in range(n_images):
                # Shift values to be strictly positive for valid distribution comparison
                dist1 = flat1[k] + 1.0001 
                dist2 = flat2[k] + 1.0001
                emds[k] = wasserstein_distance(dist1, dist2)
            per_image["emd"][pair_key] = emds

    summary: SummaryResults = {metric: {} for metric in metrics_list}
    for metric in metrics_list:
        metric_results = per_image.get(metric, {})
        for pair_key, values in metric_results.items():
            summary[metric][pair_key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }

    return {"per_image": per_image, "summary": summary}


def visualize_similarity_matrix(
    results: ComparisonResults,
    model_names: Sequence[str],
    metric: str = "cosine",
    stat: str = "mean",
    figsize: Tuple[float, float] = (12, 8),
    cmap: Union[str, colors.Colormap] = "coolwarm",
    annotate: bool = True,
    plot_upper_triangle: bool = False,
) -> Figure:
    """Render a similarity matrix between models and return the figure.
    
    Args:
        results (Mapping[str, Any]): Results dictionary from compare_heatmaps.
        model_names (Sequence[str]): List of model names corresponding to heatmap arrays.
        metric (str): Metric to visualize. Supported: "mse", "correlation", "cosine", "ssim", "emd".
        stat (str): Statistic to visualize from summary. Supported: "mean", "std", "min", "max", "median".
        figsize (Tuple[float, float]): Figure size for the plot.
        cmap (Union[str, colors.Colormap]): Colormap for the heatmap.
        annotate (bool): Whether to annotate cells with values.
        plot_upper_triangle (bool): If True, masks the lower triangle of the matrix.
    Returns:
        Figure: Matplotlib figure containing the similarity matrix heatmap.
    """

    if "summary" not in results:
        raise KeyError("results must contain a 'summary' key produced by compare_heatmaps")

    summary_section = cast(Mapping[str, Mapping[str, Mapping[str, float]]], results["summary"])
    if metric not in summary_section:
        raise ValueError(
            f"Metric '{metric}' not found in results. Available metrics: {list(summary_section.keys())}"
        )

    n_models = len(model_names)
    similarity_matrix = np.zeros((n_models, n_models), dtype=float)

    diagonal_value = 0.0 if metric in ("mse", "emd") else 1.0
    np.fill_diagonal(similarity_matrix, diagonal_value)

    for pair, metrics_data in summary_section[metric].items():
        model_indices = pair.split("_vs_")
        if len(model_indices) == 2 and all(index.isdigit() for index in model_indices):
            i, j = map(int, model_indices)
            if i < n_models and j < n_models:
                stat_value = metrics_data.get(stat)
                if stat_value is None:
                    raise KeyError(f"Statistic '{stat}' not found for pair '{pair}' in metric '{metric}'")
                similarity_matrix[i, j] = float(stat_value)
                similarity_matrix[j, i] = float(stat_value)
            else:
                warn("Model indices %s, %s from pair '%s' are out of bounds. Skipping.", i, j, pair)
        else:
            warn("Could not parse indices from pair '%s'. Skipping.", pair)

    current_cmap: Union[str, colors.Colormap] = cmap
    custom_vmin: Optional[float]
    custom_vmax: Optional[float]
    custom_vmin = custom_vmax = None
    cbar_label = ""

    if metric in ["cosine", "correlation", "ssim"]:
        custom_vmin, custom_vmax = -1.0, 1.0
        cbar_label = f"{metric.capitalize()} Similarity"
        if cmap == "coolwarm_r":
            current_cmap = "coolwarm"
    elif metric == "mse":
        custom_vmin = 0.0
        cbar_label = "Mean Squared Error"
        if cmap == "coolwarm":
            current_cmap = "coolwarm_r"
    elif metric == "emd":
        custom_vmin = 0.0
        cbar_label = "Earth Mover's Distance (Wasserstein)"
        if cmap == "coolwarm":
            current_cmap = "coolwarm_r"
    else:
        cbar_label = f"{metric.capitalize()} {stat.capitalize()}"

    mask: Optional[np.ndarray] = None
    if plot_upper_triangle:
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        similarity_matrix,
        annot=annotate,
        fmt=".2f",
        cmap=current_cmap,
        xticklabels=model_names,
        yticklabels=model_names,
        cbar_kws={"label": cbar_label},
        mask=mask,
        vmin=custom_vmin,
        vmax=custom_vmax,
        ax=ax,
    )

    title_metric_name = metric.upper() if metric in ["mse", "emd"] else metric.capitalize()
    ax.set_title(f"{title_metric_name} {stat.capitalize()} Between Heatmaps of Different Models")
    fig.tight_layout()
    return fig


def visualize_similarity_distribution(
    results: ComparisonResults,
    metric: str = "correlation",
    figsize: Tuple[float, float] = (14, 6),
    bins: int = 40,
    color: str = "steelblue",
) -> Figure:
    """Plot histogram of inter-model agreement and consensus-vs-controversy scatter.
    
    The figure consists of two subplots:
    1. Histogram: Distribution of the mean agreement score per image.
    2. Scatter Plot: Mean Agreement vs. Standard Deviation (Controversy) per image.
    
    Args:
        results (Mapping[str, Any]): Results dictionary from compare_heatmaps.
        metric (str): Metric to visualize. Supported: "mse", "correlation", "cosine", "ssim", "emd".
        figsize (Tuple[float, float]): Figure size for the plot.
        bins (int): Number of bins for the histogram.
        color (str): Color for the histogram bars.
        
    Returns:
        Figure: Matplotlib figure containing the plots.
    """
    if "per_image" not in results:
        raise KeyError("results must contain a 'per_image' key produced by compare_heatmaps")
    
    per_image_section = cast(Mapping[str, Mapping[str, np.ndarray]], results["per_image"])
    if metric not in per_image_section:
        raise ValueError(
            f"Metric '{metric}' not found in results. Available metrics: {list(per_image_section.keys())}"
        )
    
    metric_data = per_image_section[metric]
    
    # Stack comparisons into (n_pairs, n_images)
    stacked_comparisons = np.stack(list(metric_data.values()), axis=0)
    
    # Calculate statistics across pairs for each image
    agreement_means = np.mean(stacked_comparisons, axis=0)
    agreement_stds = np.std(stacked_comparisons, axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]})
    
    # Subplot 1: Histogram of means
    ax1.hist(agreement_means, bins=bins, alpha=0.7, edgecolor='black', color=color)
    ax1.axvline(np.mean(agreement_means), color='red', linestyle='--', linewidth=2, 
               label=f'Global Mean: {np.mean(agreement_means):.3f}')
    
    # Adjust labels for -1 to 1 range metrics
    if metric in ["correlation", "cosine"]:
        ax1.set_xlim(-1.1, 1.1)
        xlabel_text = f'Average {metric.capitalize()} (-1=Opposite, 1=Same)'
    else:
        metric_display = metric.upper() if metric in ("mse", "emd", "ssim") else metric.capitalize()
        xlabel_text = f'Average {metric_display} Score'
        
    ax1.set_xlabel(xlabel_text)
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Distribution of Inter-Model Agreement')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')

    # Subplot 2: Scatter of Mean vs Std
    sns.scatterplot(x=agreement_means, y=agreement_stds, ax=ax2, alpha=0.6)
    ax2.set_xlabel('Mean Agreement Score')
    ax2.set_ylabel('Standard Deviation (Controversy)')
    ax2.set_title('Consensus vs. Controversy Analysis')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig