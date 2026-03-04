"""
### Utility functions for plotting data and images.
"""

import os
from typing import Any, Mapping, Optional, Sequence, Tuple, Union, cast

import cv2  # OpenCV for image processing
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import cm
import matplotlib.colors as colors

from .logger import warn, error, debug


def visualize_similarity_matrix(
    results: Mapping[str, Any],
    model_names: Sequence[str],
    metric: str = "correlation",
    stat: str = "mean",
    figsize: Tuple[int, int] = (10, 8),
    cmap: Union[str, colors.Colormap] = "coolwarm",
    annotate: bool = True,
    plot_only_lower_triangle: bool = True,
    keep_diagonal: bool = False,
    dpi: int = 600,
    font_scale: float = 1.2,
    add_title: bool = True,
) -> Figure:
    """
    Visualize similarity matrix.
    """
    if metric not in results["summary"]:
        raise ValueError(
            f"Metric '{metric}' not found in results. Available metrics: {list(results['summary'].keys())}"
        )

    n_models = len(model_names)
    similarity_matrix = np.zeros((n_models, n_models))
    annot_matrix = np.empty((n_models, n_models), dtype=object)

    # Initialize diagonal
    for i in range(n_models):
        if metric in ["mse", "emd"]:
            similarity_matrix[i, i] = 0.0
            annot_matrix[i, i] = "0.00"
        else:
            similarity_matrix[i, i] = 1.0
            annot_matrix[i, i] = "1.00"

    # Fill Matrix
    for pair, metrics_data in results["summary"][metric].items():
        try:
            model_indices = pair.split("_vs_")
            if len(model_indices) == 2:
                i, j = map(int, model_indices)
                if i < n_models and j < n_models:
                    val = metrics_data[stat]
                    similarity_matrix[i, j] = val
                    similarity_matrix[j, i] = val

                    label = f"{val:.2f}"
                    # Check for standard deviation if available
                    std_val = metrics_data.get("std", metrics_data.get("stdev"))
                    if std_val is not None:
                        label += f"\n±{std_val:.2f}"

                    annot_matrix[i, j] = label
                    annot_matrix[j, i] = label
                else:
                    warn(f"Indices {i}, {j} out of bounds. Skipping.")
        except ValueError:
            warn(f"Could not parse indices from pair '{pair}'. Skipping.")

    # --- Plotting Configuration ---

    with sns.plotting_context("paper", font_scale=font_scale):

        fig = plt.figure(figsize=figsize, dpi=dpi)

        current_cmap = cmap
        custom_vmin, custom_vmax = None, None
        cbar_label = ""

        if metric in ["cosine", "correlation", "ssim"]:
            custom_vmin, custom_vmax = -1.0, 1.0
            cbar_label = f"{metric.capitalize()} Similarity"
            if cmap == "coolwarm_r":
                current_cmap = "coolwarm"
        elif metric == "mse":
            custom_vmin = 0.0
            if cmap == "coolwarm":
                current_cmap = "coolwarm_r"
            cbar_label = "Mean Squared Error"
        elif metric == "emd":
            custom_vmin = 0.0
            if cmap == "coolwarm":
                current_cmap = "coolwarm_r"
            cbar_label = "Earth Mover's Distance"
        elif metric.startswith("top_percent_iou"):
            custom_vmin, custom_vmax = 0.0, 1.0
            cbar_label = "Intersection over Union (IoU)"
        else:
            cbar_label = f"{metric.capitalize()} {stat.capitalize()}"

        mask: Optional[np.ndarray] = None
        if plot_only_lower_triangle:
            mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
        if not keep_diagonal:
            np.fill_diagonal(mask, True)

        # Initialize labels
        xticklabels = list(model_names)
        yticklabels = list(model_names)

        matrix_to_plot = similarity_matrix
        annot_to_plot = annot_matrix

        if plot_only_lower_triangle and not keep_diagonal:
            # Slice matrix: remove Top Row (1:) and Right Column (:-1)
            matrix_to_plot = similarity_matrix[1:, :-1]
            annot_to_plot = annot_matrix[1:, :-1]

            # Slice mask to match dimensions
            if mask is not None:
                mask = mask[1:, :-1]

            # Adjust labels:
            yticklabels = model_names[1:]
            xticklabels = model_names[:-1]

        ax = sns.heatmap(
            matrix_to_plot,
            annot=annot_to_plot if annotate else False,
            fmt="",
            cmap=current_cmap,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            cbar_kws={
                "label": cbar_label,
                "shrink": 0.8,  # Shrink colorbar slightly
                "fraction": 0.05,  # Adjust width relative to plot
                "pad": 0.02,  # Distance from plot
            },
            mask=mask,
            vmin=custom_vmin,
            vmax=custom_vmax,
            linecolor="white",
            linewidths=1.0,  # Thicker lines for clearer separation in print
        )

        # --- Cosmetic Fixes for Printing ---

        # Rotate x-axis labels to prevent overlap
        plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
        plt.yticks(rotation=0)

        # Remove individually black lined cells if mask is applied (Existing Logic)
        mesh = ax.collections[0]
        n_rows, n_cols = matrix_to_plot.shape
        edgecolors = mesh.get_edgecolors()

        # Ensure we have an array of colors to modify
        if len(edgecolors) == 1:
            edgecolors = np.repeat(edgecolors, n_rows * n_cols, axis=0)

        # Handle edge colors safely
        try:
            if len(edgecolors) == n_rows * n_cols:
                for i in range(n_rows * n_cols):
                    r = i // n_cols
                    c = i % n_cols

                    # If the cell is masked (value is masked in numpy array),
                    # Seaborn usually handles it, but if you need manual transparency:
                    is_masked = False
                    if mask is not None:
                        is_masked = mask[r, c]

                    if not is_masked:
                        edgecolors[i] = (1, 1, 1, 1)  # White borders for visible cells
                    else:
                        edgecolors[i] = (0, 0, 0, 0)  # Transparent for masked cells

                mesh.set_edgecolors(edgecolors)
        except Exception as e:
            warn(f"Could not apply custom edge colors: {e}")

        title_metric_name = (
            metric.upper() if metric in ["mse", "emd"] else metric.capitalize()
        )

        if add_title:
            plt.title(
                f"{title_metric_name} ({stat}) Comparison", pad=20, fontweight="bold"
            )

        plt.tight_layout()

        return fig


def visualize_similarity_distribution(
    results: Mapping[str, Any],
    metric: str = "correlation",
    figsize: Tuple[float, float] = (10, 6),
    bins: Union[int, str] = "auto",
    color: str = "steelblue",
    add_title: bool = True,
) -> Figure:
    """
    Plot histogram of inter-variant agreement distribution.
    """
    # Validate and explain bins
    if isinstance(bins, int):
        if bins <= 0:
            raise ValueError("bins must be a positive integer or 'auto'.")
        bins_explanation = f"Using {bins} bins: finer granularity, more detail."
    elif isinstance(bins, str):
        if bins != "auto":
            raise ValueError("bins must be a positive integer or 'auto'.")
        bins_explanation = (
            "Using 'auto' bins: numpy will choose bin count based on data."
        )
    else:
        raise TypeError("bins must be int or 'auto' string.")

    if "per_image" not in results:
        raise KeyError(
            "results must contain a 'per_image' key produced by compare_heatmaps"
        )

    per_image_section = cast(
        Mapping[str, Mapping[str, np.ndarray]], results["per_image"]
    )
    if metric not in per_image_section:
        raise ValueError(
            f"Metric '{metric}' not found in results. Available metrics: {list(per_image_section.keys())}"
        )

    metric_data = per_image_section[metric]

    # Stack comparisons into (n_pairs, n_images)
    stacked_comparisons = np.stack(list(metric_data.values()), axis=0)

    # Calculate statistics across pairs for each image
    agreement_means = np.mean(stacked_comparisons, axis=0)

    # Analyze the distribution to set appropriate scales
    data_min = np.min(agreement_means)
    data_max = np.max(agreement_means)
    data_mean = np.mean(agreement_means)
    data_std = np.std(agreement_means)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram
    counts, bins_edges, patches = ax.hist(
        agreement_means, bins=bins, alpha=0.7, edgecolor="black", color=color
    )

    # Add mean line
    ax.axvline(
        data_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {data_mean:.3f} (±{data_std:.3f})",
    )

    # Set X-axis limits based on metric type and data range
    if metric in ["correlation", "cosine"]:
        # For correlation/cosine, use full range but consider data
        x_margin = 0.05
        xlim_min = max(-1.0, data_min - x_margin)
        xlim_max = min(1.0, data_max + x_margin)
        ax.set_xlim(xlim_min, xlim_max)
        xlabel_text = f"Average {metric.capitalize()} Score"
    elif metric in ["mse", "emd"]:
        # For error metrics, start from 0 and add margin
        x_margin = (data_max - data_min) * 0.1
        ax.set_xlim(0, data_max + x_margin)
        metric_display = metric.upper()
        xlabel_text = f"Average {metric_display} Score"
    elif metric == "ssim":
        # SSIM ranges from -1 to 1
        x_margin = 0.05
        xlim_min = max(-1.0, data_min - x_margin)
        xlim_max = min(1.0, data_max + x_margin)
        ax.set_xlim(xlim_min, xlim_max)
        xlabel_text = f"Average {metric.upper()} Score"
    elif metric.startswith("top_percent_iou"):
        ax.set_xlim(0.0, 1.0)
        xlabel_text = "Average IoU Score"
    else:
        # For other metrics, use data-driven limits
        x_margin = (data_max - data_min) * 0.1
        ax.set_xlim(data_min - x_margin, data_max + x_margin)
        xlabel_text = f"Average {metric.capitalize()} Score"

    # Set Y-axis limits based on histogram counts
    y_max = np.max(counts)
    y_margin = y_max * 0.1
    ax.set_ylim(0, y_max + y_margin)

    # Labels and styling
    ax.set_xlabel(xlabel_text, fontsize=11)
    ax.set_ylabel("Number of Images", fontsize=11)

    if add_title:
        ax.set_title(
            f"Distribution of Variant Agreement ({metric.capitalize()})", fontsize=12
        )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    # Add bins explanation as a note below the plot
    ax.text(
        0.99,
        -0.18,
        bins_explanation,
        fontsize=9,
        color="gray",
        ha="right",
        va="top",
        transform=ax.transAxes,
    )

    fig.tight_layout()
    return fig


def visualize_violin_distribution(
    data: Mapping[str, Sequence[float]],
    metric: str,
    figsize: Tuple[float, float] = (16, 5),  # Wide for side-by-side
    palette: str = "muted",
    custom_model_order: Optional[Sequence[str]] = None,
    dpi: int = 600,
    font_scale: float = 1.2,
    add_title: bool = True,
) -> Figure:
    """
    Creates faceted violin plots with independent scales, containing
    a narrow boxplot inside each violin.

    Parameters:
    - data: mapping from model name to sequence of metric values.
    - metric: name of the metric (used for labels/titles).
    - figsize: overall figure size in inches (width, height).
    - palette: seaborn palette name.
    - custom_model_order: optional explicit ordering of models.
    - dpi: figure DPI for high-resolution output.
    - font_scale: seaborn font scale applied within plotting context.
    """
    # 1. Flatten Data
    records = []
    for model_name, values in data.items():
        for v in values:
            records.append({"Model": model_name, "Value": v})

    if not records:
        warn(f"No data available for violin plot of {metric}.")
        # Ensure returned figure respects figsize and dpi
        empty_fig = plt.figure(figsize=figsize, dpi=dpi)
        return empty_fig

    df = pd.DataFrame(records)

    # 2. Sort Order (Median) or Custom
    if custom_model_order:
        order = custom_model_order
    else:
        order = df.groupby("Model")["Value"].median().sort_values().index

    # Use seaborn plotting context to control fonts
    with sns.plotting_context("paper", font_scale=font_scale):
        # 3. Create FacetGrid (Independent Y-Axes)
        # We create the FacetGrid and then set the overall figure size & DPI
        g = sns.FacetGrid(
            df,
            col="Model",
            col_order=order,
            sharey=False,  # Independent scales
            height=5,
            aspect=0.5,
            hue="Model",
            palette=palette,
        )

        # --- LAYER 1: The Violin (Shape only) ---
        g.map_dataframe(
            sns.violinplot,
            y="Value",
            inner=None,  # Turn off default inner lines
            density_norm="width",
            cut=0,
            alpha=0.7,
            linewidth=0,  # Remove outline for a cleaner look behind the box
        )

        # --- LAYER 2: The Boxplot (The "Small Box" inside) ---
        g.map_dataframe(
            sns.boxplot,
            y="Value",
            width=0.15,  # Make it narrow so it sits "inside"
            boxprops={
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": "black",
            },  # White box pops
            whiskerprops={"color": "black"},
            capprops={"color": "black"},
            medianprops={"color": "black", "linewidth": 1.5},
            showfliers=False,  # Don't show outlier dots (stripplot does this)
            zorder=2,  # Ensure it draws on top of violin
        )

        # --- LAYER 3: The Strip Plot (Raw Data Dots) ---
        g.map_dataframe(
            sns.stripplot,
            y="Value",
            color="black",
            alpha=0.3,
            size=2,
            jitter=True,
            zorder=3,
        )

        # 4. Aesthetics
        g.set_titles(col_template="{col_name}")
        g.set_axis_labels("", f"{metric.capitalize()}")

        # Global Title (scale title fontsize with font_scale)
        if add_title:
            g.fig.suptitle(
                f"Distribution of {metric.capitalize()} Scores Across Architectures",
                y=1.05,
                fontsize=int(14 * font_scale),
            )

        # Add gridlines to every subplot
        for ax in g.axes.flat:
            ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Apply requested figure size and DPI
        try:
            g.fig.set_size_inches(figsize)
            g.fig.set_dpi(dpi)
        except Exception:
            # Fallback: create a new figure wrapper if setting fails
            warn(
                "Could not apply figsize/dpi to FacetGrid figure. Continuing with defaults."
            )

        plt.tight_layout()

    return g.fig


def denormalize_image(
    tensor: torch.Tensor,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Denormalizes an image tensor."""
    if tensor.dim() != 3:
        raise ValueError(
            f"Input tensor must have 3 dimensions (C, H, W), but got {tensor.dim()}"
        )

    mean_seq = list(mean)
    std_seq = list(std)
    mean_used: Sequence[float] = mean_seq
    std_used: Sequence[float] = std_seq
    if tensor.shape[0] != len(mean_seq) or tensor.shape[0] != len(std_seq):
        if tensor.shape[0] == 1:  # Grayscale
            warn(
                "Denormalizing grayscale with potentially RGB stats. Using first value of mean/std."
            )
            mean_used = [float(mean_seq[0])] if mean_seq else [0.0]
            std_used = [float(std_seq[0])] if std_seq else [1.0]
        else:
            raise ValueError(
                f"Channel mismatch: Tensor has {tensor.shape[0]} channels, mean has {len(mean_seq)}, std has {len(std_seq)}"
            )

    mean_t = torch.as_tensor(mean_used, dtype=tensor.dtype, device=tensor.device).view(
        -1, 1, 1
    )
    std_t = torch.as_tensor(std_used, dtype=tensor.dtype, device=tensor.device).view(
        -1, 1, 1
    )

    denormalized_tensor = tensor * std_t + mean_t
    return torch.clamp(denormalized_tensor, 0.0, 1.0)


def visualize_and_save_saliency(
    image_tensor: torch.Tensor,
    saliency_map: np.ndarray,
    output_dir: str,
    filename_prefix: str,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    overlay_alpha: float = 0.5,
    cmap_name: str = "bwr",
) -> None:
    """
    Visualizes saliency map, creates a colored overlay on a grayscale original, and saves plain images.
    """
    if image_tensor.is_cuda:
        debug(
            "image_tensor provided to visualize_and_save_saliency is on CUDA, moving to CPU."
        )
        image_tensor = image_tensor.cpu()

    # Create a sub-folder for each image's visualizations
    image_specific_output_dir = os.path.join(output_dir, filename_prefix)
    os.makedirs(image_specific_output_dir, exist_ok=True)

    if saliency_map.ndim != 2:
        error(
            "Saliency map has unexpected dimensions %s. Expected (H, W). Skipping visualization.",
            saliency_map.shape,
        )
        return

    saliency_map = np.clip(saliency_map, 0.0, 1.0)

    NUMPY_DIR = os.path.join(
        output_dir, "numpy_saliency_maps"
    )  # Centralized numpy maps
    os.makedirs(NUMPY_DIR, exist_ok=True)
    np.save(
        os.path.join(NUMPY_DIR, f"{filename_prefix}_saliency_map.npy"), saliency_map
    )

    try:
        img_denorm_tensor = denormalize_image(image_tensor, mean, std)
    except ValueError as e:
        error(
            f"Error during denormalization for {filename_prefix}: {e}. Skipping visualization."
        )
        return

    # Convert tensor to uint8 image array
    img_np = img_denorm_tensor.numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0.0, 1.0)
    img_uint8 = (img_np * 255).astype(np.uint8)

    # Ensure base image is in BGR format for OpenCV saving
    if img_uint8.shape[2] == 1:
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    elif img_uint8.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    else:
        error(
            "Unexpected number of channels (%s) for %s. Skipping visualization.",
            img_uint8.shape[2],
            filename_prefix,
        )
        return

    # 1. Save Original Image (Plain, without text/borders)
    orig_save_path = os.path.join(
        image_specific_output_dir, f"{filename_prefix}_original.png"
    )
    cv2.imwrite(orig_save_path, img_bgr)

    # 2. Generate and Save Heatmap Image (Plain)
    try:
        cmap = cm.get_cmap(cmap_name)
    except ValueError:
        warn(f"Colormap '{cmap_name}' not found. Using default 'viridis'.")
        cmap = cm.get_cmap("viridis")

    # Apply colormap directly to the normalized array (returns RGBA floats [0, 1])
    heatmap_rgba = cmap(saliency_map)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)

    # Resize heatmap to match original image dimensions if necessary
    if heatmap_bgr.shape[:2] != img_bgr.shape[:2]:
        debug(
            "Resizing heatmap from %s to %s for %s",
            heatmap_bgr.shape[:2],
            img_bgr.shape[:2],
            filename_prefix,
        )
        heatmap_bgr = cv2.resize(
            heatmap_bgr,
            (img_bgr.shape[1], img_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    heatmap_save_path = os.path.join(
        image_specific_output_dir, f"{filename_prefix}_heatmap_{cmap_name}.png"
    )
    cv2.imwrite(heatmap_save_path, heatmap_bgr)

    # 3. Save Grayscale Original + Colored Overlay
    # Convert the original color image to grayscale, then back to BGR so it has 3 channels for blending
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_base_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(
        gray_base_bgr, 1.0 - overlay_alpha, heatmap_bgr, overlay_alpha, 0.0
    )
    overlay_save_path = os.path.join(
        image_specific_output_dir, f"{filename_prefix}_overlay_{cmap_name}.png"
    )
    cv2.imwrite(overlay_save_path, overlay)
