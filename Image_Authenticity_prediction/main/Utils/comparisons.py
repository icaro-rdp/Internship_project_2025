"""Utility functions for comparisons between heatmaps."""

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import TypedDict

import cv2
import numpy as np
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import paired_cosine_distances

from Image_Authenticity_prediction.main.Utils.logger import warn

MetricName = str
PairKey = str
UniformedHeatmaps = np.ndarray


class MetricSummary(TypedDict):
    mean: float
    std: float
    min: float
    max: float
    median: float


PerImageResults = dict[MetricName, dict[PairKey, np.ndarray]]
SummaryResults = dict[MetricName, dict[PairKey, MetricSummary]]


class ComparisonResults(TypedDict):
    per_image: PerImageResults
    summary: SummaryResults


def uniform_heatmaps(
    heatmap_array: np.ndarray,
    height: int = 224,
    width: int = 224,
    num_images: int | None = None,
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


def topk_superpixels_by_value(beta: np.ndarray, k: int) -> np.ndarray:
    """Return superpixel indices of the largest ``k`` (most positive) values.

    Args:
        beta: 1D superpixel coefficient vector of shape ``(K,)``.
        k: Number of top positive superpixels to select.

    Returns:
        1D integer index array of length ``min(k, beta.size)`` sorted descending.
    """
    flat = np.asarray(beta, dtype=np.float64).ravel()
    k_eff = max(1, min(int(k), flat.size))
    idx = np.argpartition(flat, -k_eff)[-k_eff:]
    return idx[np.argsort(flat[idx])[::-1]]


def bottomk_superpixels_by_value(beta: np.ndarray, k: int) -> np.ndarray:
    """Return superpixel indices of the smallest ``k`` (most negative) values.

    Args:
        beta: 1D superpixel coefficient vector of shape ``(K,)``.
        k: Number of bottom negative superpixels to select.

    Returns:
        1D integer index array of length ``min(k, beta.size)`` sorted ascending.
    """
    flat = np.asarray(beta, dtype=np.float64).ravel()
    k_eff = max(1, min(int(k), flat.size))
    idx = np.argpartition(flat, k_eff - 1)[:k_eff]
    return idx[np.argsort(flat[idx])]


def superpixel_set_iou(
    idx_a: Sequence[int] | np.ndarray, idx_b: Sequence[int] | np.ndarray
) -> float:
    """Compute Intersection-over-Union (IoU) between two sets of superpixel IDs.

    Args:
        idx_a: First collection of superpixel integer indices.
        idx_b: Second collection of superpixel integer indices.

    Returns:
        Set IoU in ``[0.0, 1.0]``, or ``np.nan`` if both sets are empty.
    """
    set_a = {int(x) for x in idx_a}
    set_b = {int(x) for x in idx_b}
    if not set_a and not set_b:
        return float("nan")
    return float(len(set_a & set_b) / len(set_a | set_b))


def compare_lime_betas(
    beta1: np.ndarray,
    beta2: np.ndarray,
    k_list: Sequence[int] = (5, 15, 25),
) -> dict[str, float]:
    """Compare two 1D LIME superpixel coefficient vectors on a shared tessellation.

    Args:
        beta1: First 1D superpixel $\\beta$ array of shape ``(K,)``.
        beta2: Second 1D superpixel $\\beta$ array of shape ``(K,)``.
        k_list: Superpixel counts $K$ for top-positive and bottom-negative IoU.

    Returns:
        Dictionary with ``correlation`` (Pearson), ``spearman``, and
        ``top_percent_iou_{k}`` / ``bot_percent_iou_{k}`` for each ``k`` in
        ``k_list``.
    """
    b1 = np.asarray(beta1, dtype=np.float64).ravel()
    b2 = np.asarray(beta2, dtype=np.float64).ravel()
    if (
        b1.size == 0
        or b1.size != b2.size
        or not (np.isfinite(b1).all() and np.isfinite(b2).all())
    ):
        out: dict[str, float] = {"correlation": float("nan"), "spearman": float("nan")}
        for k in k_list:
            out[f"top_percent_iou_{k}"] = float("nan")
            out[f"bot_percent_iou_{k}"] = float("nan")
        return out

    p_val = (
        float(pearsonr(b1, b2)[0])
        if np.std(b1) > 1e-12 and np.std(b2) > 1e-12
        else float("nan")
    )
    s_res = spearmanr(b1, b2, nan_policy="omit").correlation
    s_val = float(s_res) if s_res is not None else float("nan")

    out = {"correlation": p_val, "spearman": s_val}
    for k in k_list:
        k_eff = min(int(k), b1.size)
        pos1 = topk_superpixels_by_value(b1, k_eff)
        pos2 = topk_superpixels_by_value(b2, k_eff)
        neg1 = bottomk_superpixels_by_value(b1, k_eff)
        neg2 = bottomk_superpixels_by_value(b2, k_eff)
        out[f"top_percent_iou_{k}"] = superpixel_set_iou(pos1, pos2)
        out[f"bot_percent_iou_{k}"] = superpixel_set_iou(neg1, neg2)
    return out


def compare_lime_beta_collections(
    beta_collections: Sequence[Mapping[str, np.ndarray]],
    k_list: Sequence[int] = (5, 15, 25),
) -> ComparisonResults:
    """Compare multiple per-image 1D superpixel $\\beta$ collections across images.

    Matches the exact metric definitions of ``LimeAnalyzeConsistency.py`` (Paper
    Table 5 and Figure 3C) while returning a ``ComparisonResults`` dictionary
    compatible with the repository's plotting and summary utilities.

    Args:
        beta_collections: Sequence of mappings (one per variant or prototype),
            each mapping image keys (``img_0000``, ...) to 1D ``(K,)`` arrays.
        k_list: Superpixel counts $K$ for top-positive and bottom-negative IoU.

    Returns:
        ``ComparisonResults`` containing ``per_image`` and ``summary`` dicts for
        ``correlation`` (Pearson), ``spearman``, and ``top_percent_iou_{k}``.
    """
    if not beta_collections:
        raise ValueError("beta_collections must contain at least one mapping.")

    common_keys = sorted(
        set.intersection(*(set(col.keys()) for col in beta_collections))
    )
    n_images = len(common_keys)
    metrics_list = (
        ["correlation", "spearman"]
        + [f"top_percent_iou_{k}" for k in k_list]
        + [f"bot_percent_iou_{k}" for k in k_list]
    )

    per_image: PerImageResults = {m: {} for m in metrics_list}
    pairs = list(combinations(range(len(beta_collections)), 2))

    for i, j in pairs:
        pair_key = f"{i}_vs_{j}"
        for m in metrics_list:
            per_image[m][pair_key] = np.full(n_images, np.nan, dtype=float)

        for img_idx, key in enumerate(common_keys):
            res = compare_lime_betas(
                beta_collections[i][key],
                beta_collections[j][key],
                k_list=k_list,
            )
            for m, val in res.items():
                per_image[m][pair_key][img_idx] = val

    summary: SummaryResults = {m: {} for m in metrics_list}
    for m in metrics_list:
        for pair_key, values in per_image[m].items():
            finite = values[np.isfinite(values)]
            if finite.size > 0:
                summary[m][pair_key] = {
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "median": float(np.median(finite)),
                }
            else:
                summary[m][pair_key] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "median": float("nan"),
                }

    return {"per_image": per_image, "summary": summary}
