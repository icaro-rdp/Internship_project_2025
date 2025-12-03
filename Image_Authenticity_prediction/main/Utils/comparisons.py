"""Utility functions for comparisons between heatmaps."""

from itertools import combinations
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, TypedDict, Union, cast

import cv2
import numpy as np
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
        resized_heatmaps[idx] = cv2.resize(
            current_heatmap, resize_shape, interpolation=cv2.INTER_LINEAR
        )

    return resized_heatmaps


def top_percent_iou(imgA: np.ndarray, imgB: np.ndarray, p: float = 0.1) -> float:
    """
    Compute the IoU between the sets of top-p% pixels in two images.
    """
    N = imgA.size
    if imgB.size != N:
        raise ValueError("Input images must have the same number of pixels.")
    if not (0 < p < 1):
        raise ValueError("Parameter p must be in the range (0, 1).")

    # Flatten both images
    A = imgA.ravel()
    B = imgB.ravel()

    # Number of pixels to keep (top p%)
    k = int(np.ceil(p * N))

    # ----- Get kth LARGEST value in A -----
    kth_index_A = N - k  # index of kth largest (0-based)
    part_A = np.partition(A, kth_index_A)  # partitioned array
    TA = part_A[kth_index_A]  # threshold for image A

    # ----- Get kth LARGEST value in B -----
    kth_index_B = N - k
    part_B = np.partition(B, kth_index_B)
    TB = part_B[kth_index_B]

    # Create binary masks for top-p% pixels
    maskA = A >= TA
    maskB = B >= TB

    # Compute IoU
    intersection = np.logical_and(maskA, maskB).sum()
    union = np.logical_or(maskA, maskB).sum()

    return float(intersection / union) if union > 0 else 0.0


def compare_heatmaps(
    heatmap_arrays: Sequence[np.ndarray],
    metrics: Sequence[str] = ("mse", "correlation", "cosine", "ssim", "emd"),
    iou_threshold: float = 0.1,
) -> ComparisonResults:
    """Compare similarity between multiple heatmap collections.

    Args:
        heatmap_arrays (Sequence[np.ndarray]): List of heatmap arrays to compare. Each array should have shape (B, H, W).
        metrics (Sequence[str]): List of metrics to compute. Supported metrics: "mse", "correlation", "cosine", "ssim", "emd", "top_percent_iou".
        iou_threshold (float): Threshold p for top_percent_iou metric (default 0.1 for 10%).
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
            raise ValueError(
                f"Array {idx} has shape {array.shape}, expected {reference_shape}"
            )

    # Flatten arrays once for vectorized operations (Shape: N_images x Flattened_Dim)
    flat_arrays = [arr.reshape(n_images, -1) for arr in heatmap_arrays]

    per_image: PerImageResults = {metric: {} for metric in metrics_list}
    array_pairs = list(combinations(range(len(heatmap_arrays)), 2))

    compute_mse = "mse" in metrics_list
    compute_corr = "correlation" in metrics_list
    compute_cosine = "cosine" in metrics_list
    compute_ssim = "ssim" in metrics_list
    compute_emd = "emd" in metrics_list

    # Identify IoU metrics to compute
    iou_metrics = [m for m in metrics_list if m.startswith("top_percent_iou")]

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
                    heatmap_arrays[i][k], heatmap_arrays[j][k], data_range=2.0
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

        for metric in iou_metrics:
            # Determine threshold p
            if metric == "top_percent_iou":
                p = iou_threshold
            else:
                # Parse "top_percent_iou_5" -> 0.05, "top_percent_iou_15" -> 0.15
                try:
                    suffix = metric.split("_")[-1]
                    p = float(suffix) / 100.0
                except ValueError:
                    warn(f"Could not parse IoU threshold from metric name: {metric}")
                    continue

            ious = np.zeros(n_images, dtype=float)
            for k in range(n_images):
                ious[k] = top_percent_iou(
                    heatmap_arrays[i][k], heatmap_arrays[j][k], p=p
                )
            per_image[metric][pair_key] = ious

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
