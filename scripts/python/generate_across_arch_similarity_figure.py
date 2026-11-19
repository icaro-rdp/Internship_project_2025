"""Generate across-architecture explanation similarity and agreement multi-panel figures.

Renders 3-panel publication-ready comparison figures (Panel A: Grad-CAM,
Panel B: MPM, Panel C: LIME) of across-architecture prototype attribution
matrices (Correlation/Similarity, IoU at various thresholds, SSIM) with a unified
right-side scale legend and lower-triangular format.
Exports to both vector PDF and high-resolution raster PNG.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# Default model identifiers in canonical display order
DEFAULT_MODELS: list[str] = [
    "barlowtwins",
    "resnet152",
    "densenet161",
    "efficientnetb3",
    "vgg16",
    "vgg19",
    "clip_vit_b16",
]

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "barlowtwins": "BarlowTwins",
    "resnet152": "ResNet152",
    "densenet161": "DenseNet161",
    "efficientnetb3": "EfficientNetB3",
    "vgg16": "VGG16",
    "vgg19": "VGG19",
    "clip_vit_b16": "CLIPViT",
    "clip": "CLIPViT",
}

METHODS: list[tuple[str, str, str]] = [
    ("gradcam", "Grad-CAM", "A"),
    ("mpm", "MPM", "B"),
    ("lime", "LIME", "C"),
]

DEFAULT_IOU_VMAX_OFFSET: float = 0.02

METRIC_CONFIGS: dict[str, dict[str, Any]] = {
    "correlation": {
        "vmin": -1.0,
        "vmax": 1.0,
        "cap_vmax": False,
        "cbar_label": "Similarity",
        "cmap": "coolwarm",
        "file_stem": "across_arch_explanation_similarity",
    },
    "top_percent_iou_15": {
        "vmin": 0.0,
        "vmax": 1.0,
        "cap_vmax": True,
        "cbar_label": "IoU @ 15%",
        "cmap": "coolwarm",
        "file_stem": "across_arch_explanation_iou_15",
    },
    "top_percent_iou_5": {
        "vmin": 0.0,
        "vmax": 1.0,
        "cap_vmax": True,
        "cbar_label": "IoU @ 5%",
        "cmap": "coolwarm",
        "file_stem": "across_arch_explanation_iou_5",
    },
    "top_percent_iou_25": {
        "vmin": 0.0,
        "vmax": 1.0,
        "cap_vmax": True,
        "cbar_label": "IoU @ 25%",
        "cmap": "coolwarm",
        "file_stem": "across_arch_explanation_iou_25",
    },
    "ssim": {
        "vmin": -1.0,
        "vmax": 1.0,
        "cap_vmax": False,
        "cbar_label": "SSIM",
        "cmap": "coolwarm",
        "file_stem": "across_arch_explanation_ssim",
    },
}


def load_metric_matrices(
    json_path: Path,
    metric: str = "correlation",
    models: Sequence[str] = DEFAULT_MODELS,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load pairwise metric matrices and standard deviations from comparison JSON.

    Args:
        json_path: Path to experiment_2b_comparison.json.
        metric: Metric key to load (e.g. 'correlation', 'top_percent_iou_15', 'ssim').
        models: List of model identifiers to include.

    Returns:
        Tuple of (matrices_dict, std_matrices_dict) where keys are method names
        and values are (N, N) NumPy arrays.

    Raises:
        FileNotFoundError: If json_path does not exist.
        KeyError: If expected metric keys are missing.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Comparison JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        data: dict[str, Any] = json.load(file)

    matrices: dict[str, np.ndarray] = {}
    std_matrices: dict[str, np.ndarray] = {}
    n_models = len(models)

    for method_key, _, _ in METHODS:
        full_key = f"{method_key}_between_model_architectures"
        if full_key not in data:
            logger.warning("Key %s not found in JSON, skipping", full_key)
            continue

        raw_models: list[str] = data[full_key]["models"]
        summary_dict = data[full_key].get("summary", {})
        if metric not in summary_dict:
            logger.warning(
                "Metric %s not found in %s summary, skipping", metric, full_key
            )
            continue

        metric_dict = summary_dict[metric]

        mat = np.full((n_models, n_models), np.nan)
        std_mat = np.full((n_models, n_models), np.nan)

        for pair, stats in metric_dict.items():
            if "_vs_" not in pair:
                continue
            idx_i, idx_j = map(int, pair.split("_vs_"))
            model_i = raw_models[idx_i]
            model_j = raw_models[idx_j]

            if model_i in models and model_j in models:
                pos_i = models.index(model_i)
                pos_j = models.index(model_j)
                mean_val = float(stats["mean"])
                std_val = float(stats.get("std", 0.0))
                mat[pos_i, pos_j] = mean_val
                mat[pos_j, pos_i] = mean_val
                std_mat[pos_i, pos_j] = std_val
                std_mat[pos_j, pos_i] = std_val

        matrices[method_key] = mat
        std_matrices[method_key] = std_mat

    return matrices, std_matrices


def load_correlation_matrices(
    json_path: Path,
    models: Sequence[str] = DEFAULT_MODELS,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load correlation matrices (wrapper for backward compatibility).

    Args:
        json_path: Path to experiment_2b_comparison.json.
        models: List of model identifiers.

    Returns:
        Tuple of (matrices, std_matrices).
    """
    return load_metric_matrices(json_path, metric="correlation", models=models)


def compute_capped_vmax(
    matrices: dict[str, np.ndarray],
    offset: float = DEFAULT_IOU_VMAX_OFFSET,
    upper_bound: float = 1.0,
) -> float:
    """Compute shared color scale cap as max displayed value across panels plus offset.

    Args:
        matrices: Mapping of method key to (N, N) metric matrix.
        offset: Small headroom added above the maximum value so the highest cell
            does not sit at the extreme saturated end of the colormap.
        upper_bound: Maximum allowed upper limit for the scale.

    Returns:
        Capped vmax value bounded above by `upper_bound`.
    """
    panel_maxima: list[float] = []
    for m_key, _, _ in METHODS:
        if m_key not in matrices:
            continue
        mat_slice = matrices[m_key][1:, :-1]
        mask = np.triu(np.ones_like(mat_slice, dtype=bool), k=1) | np.isnan(mat_slice)
        valid = mat_slice[~mask]
        if valid.size > 0:
            panel_maxima.append(float(np.max(valid)))

    if not panel_maxima:
        return upper_bound

    return min(upper_bound, max(panel_maxima) + offset)


def render_across_arch_similarity_figure(
    matrices: dict[str, np.ndarray],
    std_matrices: dict[str, np.ndarray] | None = None,
    models: Sequence[str] = DEFAULT_MODELS,
    output_pdf: Path | None = None,
    output_png: Path | None = None,
    include_std: bool = False,
    cbar_label: str = "Similarity",
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "coolwarm",
    dpi: int = 300,
) -> plt.Figure:
    """Render 3-panel across-architecture similarity figure with single colorbar.

    Args:
        matrices: Mapping of method key to (N, N) correlation matrix.
        std_matrices: Optional mapping of method key to (N, N) standard deviation matrix.
        models: List of model identifiers.
        output_pdf: Optional path to save vector PDF.
        output_png: Optional path to save raster PNG.
        include_std: Whether to display +/- std underneath the mean in each cell.
        cbar_label: Label for the shared colorbar.
        vmin: Minimum value for color scale.
        vmax: Maximum value for color scale.
        cmap: Colormap name.
        dpi: Dots per inch for raster export.

    Returns:
        Matplotlib Figure object.
    """
    display_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]
    row_labels = display_names[1:]
    col_labels = display_names[:-1]

    fig = plt.figure(figsize=(14.0, 4.4), dpi=dpi)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.045], wspace=0.20)

    for idx, (m_key, m_title, panel_letter) in enumerate(METHODS):
        if m_key not in matrices:
            continue

        ax = fig.add_subplot(gs[0, idx])
        mat_slice = matrices[m_key][1:, :-1]
        # Only display the lower triangle part: mask upper triangle (k=1) and NaNs
        mask = np.triu(np.ones_like(mat_slice, dtype=bool), k=1) | np.isnan(mat_slice)

        std_slice = None
        if include_std and std_matrices and m_key in std_matrices:
            std_slice = std_matrices[m_key][1:, :-1]

        annot_grid = np.empty(mat_slice.shape, dtype=object)
        for r in range(mat_slice.shape[0]):
            for c in range(mat_slice.shape[1]):
                if mask[r, c]:
                    annot_grid[r, c] = ""
                else:
                    val = mat_slice[r, c]
                    text_val = f"{val:.2f}".replace("-", "−")
                    if std_slice is not None:
                        s_val = std_slice[r, c]
                        if not np.isnan(s_val):
                            text_val += f"\n±{s_val:.2f}"
                    annot_grid[r, c] = text_val

        is_last = idx == len(METHODS) - 1
        cbar_ax = fig.add_subplot(gs[0, 3]) if is_last else None

        cell_font_size = 7 if include_std else 8.5
        sns.heatmap(
            mat_slice,
            mask=mask,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=0.6,
            linecolor="white",
            annot=annot_grid,
            fmt="",
            annot_kws={"size": cell_font_size, "weight": "medium"},
            cbar=is_last,
            cbar_ax=cbar_ax,
            xticklabels=col_labels,
            yticklabels=row_labels if idx == 0 else False,
        )

        ax.set_title(
            m_title,
            fontsize=11.0,
            pad=10,
            weight="bold",
        )
        ax.tick_params(axis="both", labelsize=8.5)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

        # Panel letter label
        ax.text(
            -0.10 if idx == 0 else -0.02,
            1.05,
            panel_letter,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )

        if is_last and cbar_ax:
            cbar_ax.set_ylabel(cbar_label, fontsize=9.5, labelpad=8)
            cbar_ax.tick_params(labelsize=8.5)

    if output_pdf:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
        logger.info("Saved vector PDF figure to %s", output_pdf)

    if output_png:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, format="png", dpi=dpi, bbox_inches="tight")
        logger.info("Saved raster PNG figure to %s", output_png)

    return fig


def generate_all_metric_panels(
    json_path: Path,
    output_dir: Path,
    metrics: Sequence[str] | None = None,
    include_std: bool = False,
    dpi: int = 300,
    iou_vmax_offset: float = DEFAULT_IOU_VMAX_OFFSET,
) -> list[Path]:
    """Generate across-architecture panel figures for specified or all metrics.

    Args:
        json_path: Path to experiment_2b_comparison.json.
        output_dir: Directory where figures will be saved.
        metrics: Sequence of metric keys to generate (defaults to all METRIC_CONFIGS).
        include_std: Whether to include standard deviations in cells.
        dpi: DPI for PNG output.
        iou_vmax_offset: Headroom added to the max value across panels when capping
            IoU color scales.

    Returns:
        List of generated PDF file paths.
    """
    target_metrics = list(metrics) if metrics else list(METRIC_CONFIGS.keys())
    generated: list[Path] = []

    for metric_name in target_metrics:
        config = METRIC_CONFIGS.get(
            metric_name,
            {
                "vmin": 0.0,
                "vmax": 1.0,
                "cap_vmax": "iou" in metric_name.lower(),
                "cbar_label": metric_name,
                "cmap": "coolwarm",
                "file_stem": f"across_arch_explanation_{metric_name}",
            },
        )

        matrices, std_matrices = load_metric_matrices(
            json_path=json_path, metric=metric_name
        )
        if not matrices:
            continue

        vmax = float(config["vmax"])
        if config.get("cap_vmax", False):
            vmax = compute_capped_vmax(
                matrices=matrices,
                offset=iou_vmax_offset,
                upper_bound=vmax,
            )

        file_stem = config["file_stem"]
        pdf_path = output_dir / f"{file_stem}.pdf"
        png_path = output_dir / f"{file_stem}.png"

        fig = render_across_arch_similarity_figure(
            matrices=matrices,
            std_matrices=std_matrices,
            output_pdf=pdf_path,
            output_png=png_path,
            include_std=include_std,
            cbar_label=config["cbar_label"],
            vmin=config["vmin"],
            vmax=vmax,
            cmap=config["cmap"],
            dpi=dpi,
        )
        plt.close(fig)
        generated.append(pdf_path)

        # Also generate with_std version
        fig_std = render_across_arch_similarity_figure(
            matrices=matrices,
            std_matrices=std_matrices,
            output_pdf=output_dir / f"{file_stem}_with_std.pdf",
            output_png=output_dir / f"{file_stem}_with_std.png",
            include_std=True,
            cbar_label=config["cbar_label"],
            vmin=config["vmin"],
            vmax=vmax,
            cmap=config["cmap"],
            dpi=dpi,
        )
        plt.close(fig_std)

    # If iou_15 was generated, create canonical alias across_arch_explanation_iou.{pdf,png}
    iou_15_pdf = output_dir / "across_arch_explanation_iou_15.pdf"
    if iou_15_pdf.exists():
        alias_pdf = output_dir / "across_arch_explanation_iou.pdf"
        alias_png = output_dir / "across_arch_explanation_iou.png"
        alias_pdf.write_bytes(iou_15_pdf.read_bytes())
        iou_15_png = output_dir / "across_arch_explanation_iou_15.png"
        if iou_15_png.exists():
            alias_png.write_bytes(iou_15_png.read_bytes())
        generated.append(alias_pdf)

    return generated


def main() -> None:
    """CLI entrypoint for generating across-architecture similarity and metric panels."""
    parser = argparse.ArgumentParser(
        description="Generate across-architecture explanation similarity and metric A/B/C panels.",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=Path(
            "/mnt/storage/tier2/urihas/projects/HASURI002X49/icaro/Outputs/Experiment_2_variants/experiment_2b_comparison.json",
        ),
        help="Path to experiment_2b_comparison.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Outputs/Experiment_2_variants/Plots"),
        help="Output directory for generated plots",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["all"],
        help="Metrics to plot ('all' or specific: correlation, top_percent_iou_15, top_percent_iou_5, top_percent_iou_25, ssim)",
    )
    parser.add_argument(
        "--include-std",
        action="store_true",
        help="Include standard deviation in each cell",
    )
    parser.add_argument(
        "--cbar-label",
        type=str,
        default="Similarity",
        help="Label for shared colorbar (only used when running single metric)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster PNG output",
    )
    parser.add_argument(
        "--iou-vmax-offset",
        type=float,
        default=DEFAULT_IOU_VMAX_OFFSET,
        help="Headroom added above max panel value for IoU color scales",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    metrics = None if "all" in args.metrics else args.metrics
    generate_all_metric_panels(
        json_path=args.json_path,
        output_dir=args.output_dir,
        metrics=metrics,
        include_std=args.include_std,
        dpi=args.dpi,
        iou_vmax_offset=args.iou_vmax_offset,
    )


if __name__ == "__main__":
    main()
