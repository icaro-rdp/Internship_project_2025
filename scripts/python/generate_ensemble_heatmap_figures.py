"""Generate multi-panel ensemble vs. base-model MPM heatmap figures.

Loads 10-variant MPM prototype heatmaps for the Bagging and Stacking ensembles
(Experiment 3) alongside all 7 base model prototypes (Experiment 2), renders
overlays using the shared visualization utilities, and outputs publication-ready
figures (PNG at 600 DPI and vector PDF) plus a structured JSON manifest tracking
every sampled stimulus and its annotations.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Image_Authenticity_prediction.main.Utils.comparisons import uniform_heatmaps
from Image_Authenticity_prediction.main.Utils.logger import info
from Image_Authenticity_prediction.main.Utils.visualization import render_overlay_rgb
from scripts.python.reproduce_agreement_figures import (
    ANNOTATIONS_FILE,
    MANIFEST_FILE,
    MAPS_ROOT,
    MODEL_DISPLAY_NAMES,
    MODELS,
    load_prototypes,
    load_stimulus_image,
)

ENSEMBLE_MAPS_DIR = (
    _PROJECT_ROOT / "Outputs/Experiment_3_ensemble/XAI_Maps/Multiscale_Pixel_Masking"
)
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "Outputs/paper_figures/ensemble_heatmaps"
ENSEMBLE_MODES: tuple[str, str] = ("bagging", "stacking")
ENSEMBLE_HEADERS: dict[str, str] = {
    "bagging": "Bagging\nEnsemble",
    "stacking": "Stacking\nEnsemble",
}
TIER_ROW_LABELS: dict[str, str] = {
    "high": "High\nAuthenticity",
    "medium": "Mid\nAuthenticity",
    "low": "Low\nAuthenticity",
}


def load_ensemble_prototypes(
    ensemble_maps_dir: Path = ENSEMBLE_MAPS_DIR,
    target_res: tuple[int, int] = (224, 224),
    max_samples: int | None = None,
) -> dict[str, np.ndarray]:
    """Load 10-variant Bagging and Stacking MPM prototype heatmaps in [-1, 1].

    Args:
        ensemble_maps_dir: Directory containing ``*_exp3_prototype_maps.npy``.
        target_res: Spatial resolution ``(H, W)`` to resize maps into if needed.
        max_samples: Optional cap on number of test stimuli loaded.

    Returns:
        Mapping ``{"bagging": array, "stacking": array}`` of float32 arrays
        with shape ``(N, H, W)`` clipped to ``[-1.0, 1.0]``.

    Raises:
        FileNotFoundError: If a required prototype ``.npy`` file is missing.
    """
    prototypes: dict[str, np.ndarray] = {}
    for mode in ENSEMBLE_MODES:
        proto_path = ensemble_maps_dir / f"{mode}_exp3_prototype_maps.npy"
        if not proto_path.is_file():
            raise FileNotFoundError(
                f"Ensemble prototype heatmap not found: {proto_path}"
            )
        arr = np.load(proto_path)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        if max_samples is not None and max_samples > 0:
            arr = arr[:max_samples]
        if arr.shape[1:] != target_res:
            arr = uniform_heatmaps(arr, target_res[0], target_res[1])
        prototypes[mode] = np.clip(arr.astype(np.float32), -1.0, 1.0)
        info(
            "Loaded ensemble '%s' prototype (range [%.2f, %.2f]) -> %s",
            mode,
            float(prototypes[mode].min()),
            float(prototypes[mode].max()),
            prototypes[mode].shape,
        )
    return prototypes


def load_manifest_and_annotations(
    manifest_path: Path = MANIFEST_FILE,
    annotations_path: Path = ANNOTATIONS_FILE,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Load test stimulus manifest and full AIGCIQA2023 annotations per global index.

    Args:
        manifest_path: Path to ``test_samples_manifest.json``.
        annotations_path: Path to ``real_images_annotations.csv``.

    Returns:
        Tuple of ``(manifest_list, annotations_by_global_idx)``.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Test samples manifest not found: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as manifest_file:
        manifest: list[dict[str, Any]] = json.load(manifest_file)

    annotations_by_idx: dict[int, dict[str, Any]] = {}
    if annotations_path.is_file():
        annotations_df = pd.read_csv(annotations_path)
        for idx, row in annotations_df.iterrows():
            annotations_by_idx[int(idx)] = {
                "authenticity": _safe_float(row.get("Authenticity")),
                "quality": _safe_float(row.get("Quality")),
                "correspondence": _safe_float(row.get("Correspondence")),
                "description": _safe_str(row.get("Prompt")),
                "generator_model": _safe_str(row.get("model")),
                "category": _safe_str(row.get("Category")),
                "challenge": _safe_str(row.get("Challenge")),
                "image_path": _safe_str(row.get("image_path")),
            }
    return manifest, annotations_by_idx


def _safe_float(value: Any, decimals: int = 4) -> float | None:
    """Convert value to rounded Python float, or None if missing/non-finite."""
    if value is None or pd.isna(value):
        return None
    f_val = float(value)
    if not np.isfinite(f_val):
        return None
    return round(f_val, decimals)


def _safe_str(value: Any) -> str | None:
    """Convert value to string, or None if missing."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def _pearson_2d(map_a: np.ndarray, map_b: np.ndarray) -> float:
    """Compute spatial Pearson correlation between two 2D heatmaps."""
    flat_a = map_a.astype(np.float64).ravel()
    flat_b = map_b.astype(np.float64).ravel()
    diff_a = flat_a - float(np.mean(flat_a))
    diff_b = flat_b - float(np.mean(flat_b))
    denom = float(np.linalg.norm(diff_a) * np.linalg.norm(diff_b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(diff_a, diff_b) / denom)


def _mean_pairwise_pearson(maps: Sequence[np.ndarray]) -> float:
    """Compute mean pairwise spatial Pearson correlation across 2D heatmaps."""
    if len(maps) < 2:
        return 0.0
    pair_corrs = [
        _pearson_2d(maps[i], maps[j])
        for i in range(len(maps))
        for j in range(i + 1, len(maps))
    ]
    return float(np.mean(pair_corrs))


def _compute_tertile_pools(
    manifest: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split test indices into low, medium, and high authenticity tertiles."""
    scored_indices = sorted(
        (
            (float(item.get("authenticity_score", 0.0)), int(item["test_idx"]))
            for item in manifest
        ),
        key=lambda pair: (pair[0], pair[1]),
    )
    ordered_test_indices = np.array([idx for _, idx in scored_indices], dtype=int)
    low_pool, medium_pool, high_pool = np.array_split(ordered_test_indices, 3)
    return low_pool, medium_pool, high_pool


def sample_stratified_figure_rows(
    manifest: Sequence[dict[str, Any]],
    num_figures: int = 10,
    seed: int = 42,
) -> list[list[tuple[int, str]]]:
    """Sample unique test stimuli stratified across authenticity tertiles.

    Splits the test set into three equal-sized authenticity tertiles
    (low, medium, high) and samples ``num_figures`` unique stimuli without
    replacement from each tertile so every figure contains one High, one
    Medium, and one Low authenticity row (ordered High -> Medium -> Low).

    Args:
        manifest: Sequence of test-sample metadata dicts containing
            ``test_idx`` and ``authenticity_score``.
        num_figures: Number of 3-row candidate figures to compose.
        seed: Random seed for reproducible sampling.

    Returns:
        List of length ``num_figures``, where each item is a 3-element list of
        ``(test_idx, authenticity_tier)`` tuples ordered ``high``, ``medium``,
        ``low``.

    Raises:
        ValueError: If ``num_figures`` is not positive or exceeds tertile size.
    """
    if num_figures <= 0:
        raise ValueError(f"num_figures must be positive, got {num_figures}")

    low_pool, medium_pool, high_pool = _compute_tertile_pools(manifest)

    min_pool_size = min(len(low_pool), len(medium_pool), len(high_pool))
    if num_figures > min_pool_size:
        raise ValueError(
            f"Requested {num_figures} figures, but smallest authenticity tertile "
            f"has only {min_pool_size} stimuli."
        )

    rng = np.random.default_rng(seed)
    high_picks = rng.choice(high_pool, size=num_figures, replace=False)
    medium_picks = rng.choice(medium_pool, size=num_figures, replace=False)
    low_picks = rng.choice(low_pool, size=num_figures, replace=False)

    figures: list[list[tuple[int, str]]] = []
    for fig_idx in range(num_figures):
        figures.append(
            [
                (int(high_picks[fig_idx]), "high"),
                (int(medium_picks[fig_idx]), "medium"),
                (int(low_picks[fig_idx]), "low"),
            ]
        )
    return figures


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert float [0, 1] or uint8 image array to uint8 in [0, 255]."""
    if image.dtype == np.uint8:
        return image
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def render_ensemble_heatmap_figure(
    rows: Sequence[dict[str, Any]],
    ordered_models: Sequence[str] = MODELS,
    output_png: Path = DEFAULT_OUTPUT_DIR / "ensemble_figure_01.png",
    output_pdf: Path | None = DEFAULT_OUTPUT_DIR / "ensemble_figure_01.pdf",
    dpi: int = 600,
) -> None:
    """Render multi-row ensemble vs. base-model heatmap comparison figure.

    Layout matches the paper template:
    - Left block (larger squares, bold headers): Original, Bagging Ensemble,
      Stacking Ensemble.
    - Horizontal visual gap.
    - Right block (smaller squares, vertically centered per row, regular
      smaller headers): the 7 individual base models.

    Args:
        rows: Sequence of per-row dictionaries, each containing
            ``"input_rgb"`` (H, W, 3), ``"bagging_overlay"`` (H, W, 3),
            ``"stacking_overlay"`` (H, W, 3), and ``"model_overlays"``
            mapping model name to RGB overlay array.
        ordered_models: Ordered sequence of base model keys to plot.
        output_png: Output path for high-resolution PNG figure.
        output_pdf: Optional output path for vector PDF figure.
        dpi: Resolution in dots per inch for raster output.

    Raises:
        ValueError: If ``rows`` or ``ordered_models`` is empty.
    """
    if not rows:
        raise ValueError("At least one stimulus row is required to render a figure.")
    models_list = list(ordered_models)
    if not models_list:
        raise ValueError("At least one base model is required to render a figure.")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Exact panel geometry in inches for true 1:1 square panels
    n_rows = len(rows)
    n_models = len(models_list)
    large_size = 1.65
    small_size = 1.18
    large_gap = 0.14
    group_gap = 0.36
    small_gap = 0.12
    row_gap = 0.18
    margin_x = 0.12
    margin_bottom = 0.12
    header_height = 0.56

    left_block_w = 3 * large_size + 2 * large_gap
    right_block_w = n_models * small_size + (n_models - 1) * small_gap
    has_row_labels = any(row.get("row_label") for row in rows)
    left_label_w = 0.98 if has_row_labels else 0.0
    left_label_gap = 0.22 if has_row_labels else 0.0
    fig_w = (
        2 * margin_x
        + left_label_w
        + left_label_gap
        + left_block_w
        + group_gap
        + right_block_w
    )
    fig_h = margin_bottom + n_rows * large_size + (n_rows - 1) * row_gap + header_height

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Precompute column x positions in inches
    left_label_center_x = (margin_x + left_label_w / 2.0) / fig_w
    large_start_x = margin_x + left_label_w + left_label_gap
    large_x = [large_start_x + i * (large_size + large_gap) for i in range(3)]
    small_start_x = large_start_x + left_block_w + group_gap
    small_x = [small_start_x + i * (small_size + small_gap) for i in range(n_models)]

    # Precompute row y positions (bottom edge of large panel) from top to bottom
    top_large_y = fig_h - header_height - large_size
    row_large_y = [top_large_y - r * (large_size + row_gap) for r in range(n_rows)]
    small_y_offset = (large_size - small_size) / 2.0

    # Column headers above row 0
    large_headers = (
        "Original",
        ENSEMBLE_HEADERS["bagging"],
        ENSEMBLE_HEADERS["stacking"],
    )
    large_header_y = (row_large_y[0] + large_size + 0.09) / fig_h
    for col_x, title in zip(large_x, large_headers, strict=True):
        fig.text(
            (col_x + large_size / 2.0) / fig_w,
            large_header_y,
            title,
            fontsize=11.5,
            fontweight="bold",
            ha="center",
            va="bottom",
            linespacing=1.1,
        )

    small_header_y = (row_large_y[0] + small_y_offset + small_size + 0.09) / fig_h
    for col_x, model_key in zip(small_x, models_list, strict=True):
        fig.text(
            (col_x + small_size / 2.0) / fig_w,
            small_header_y,
            MODEL_DISPLAY_NAMES.get(model_key, model_key),
            fontsize=9.5,
            fontweight="normal",
            ha="center",
            va="bottom",
        )

    # Render image panels row by row
    for row_data, y_large in zip(rows, row_large_y, strict=True):
        large_images = (
            _to_uint8(row_data["input_rgb"]),
            _to_uint8(row_data["bagging_overlay"]),
            _to_uint8(row_data["stacking_overlay"]),
        )
        for col_x, img in zip(large_x, large_images, strict=True):
            ax = fig.add_axes(
                [col_x / fig_w, y_large / fig_h, large_size / fig_w, large_size / fig_h]
            )
            ax.imshow(img, aspect="equal")
            ax.axis("off")

        y_small = y_large + small_y_offset
        model_overlays: dict[str, np.ndarray] = row_data["model_overlays"]
        for col_x, model_key in zip(small_x, models_list, strict=True):
            ax = fig.add_axes(
                [col_x / fig_w, y_small / fig_h, small_size / fig_w, small_size / fig_h]
            )
            overlay = model_overlays.get(
                model_key, np.full((224, 224, 3), 128, dtype=np.uint8)
            )
            ax.imshow(_to_uint8(overlay), aspect="equal")
            ax.axis("off")

        row_label = row_data.get("row_label")
        if row_label:
            fig.text(
                left_label_center_x,
                (y_large + large_size / 2.0) / fig_h,
                str(row_label),
                fontsize=10.5,
                fontweight="bold",
                ha="center",
                va="center",
                linespacing=1.15,
            )

    fig.savefig(output_png, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    if output_pdf is not None:
        fig.savefig(output_pdf, format="pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    info(
        "Rendered ensemble heatmap figure -> PNG: %s%s",
        output_png,
        f", PDF: {output_pdf}" if output_pdf is not None else "",
    )


def _build_row_entry(
    row_idx: int,
    test_idx: int,
    tier: str,
    manifest: list[dict[str, Any]],
    annotations_by_idx: dict[int, dict[str, Any]],
    base_prototypes: dict[str, np.ndarray],
    ensemble_prototypes: dict[str, np.ndarray],
    active_models: Sequence[str],
    cmap_name: str,
    alpha: float,
    allow_missing_images: bool,
    images_dir: Path | None,
    show_row_labels: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build renderable image overlays and JSON sub-image record for one stimulus."""
    meta = manifest[test_idx] if test_idx < len(manifest) else {}
    global_idx = int(meta.get("global_csv_idx", test_idx))
    image_name = str(meta.get("image_name", f"{test_idx}.png"))
    ann = annotations_by_idx.get(global_idx, {})

    auth_score = ann.get("authenticity")
    if auth_score is None:
        auth_score = _safe_float(meta.get("authenticity_score"))

    input_rgb = load_stimulus_image(
        test_idx=test_idx,
        manifest=manifest,
        allow_missing=allow_missing_images,
        images_dir=images_dir,
    )

    bagging_map = ensemble_prototypes["bagging"][test_idx]
    stacking_map = ensemble_prototypes["stacking"][test_idx]
    base_maps = [base_prototypes[model_key][test_idx] for model_key in active_models]

    bagging_overlay = render_overlay_rgb(
        input_rgb,
        bagging_map,
        cmap_name=cmap_name,
        alpha=alpha,
        symmetric=True,
    )
    stacking_overlay = render_overlay_rgb(
        input_rgb,
        stacking_map,
        cmap_name=cmap_name,
        alpha=alpha,
        symmetric=True,
    )
    model_overlays = {
        model_key: render_overlay_rgb(
            input_rgb,
            base_prototypes[model_key][test_idx],
            cmap_name=cmap_name,
            alpha=alpha,
            symmetric=True,
        )
        for model_key in active_models
    }

    row_render = {
        "input_rgb": input_rgb,
        "bagging_overlay": bagging_overlay,
        "stacking_overlay": stacking_overlay,
        "model_overlays": model_overlays,
        "row_label": TIER_ROW_LABELS.get(tier) if show_row_labels else None,
    }
    sub_image_meta: dict[str, Any] = {
        "row_idx": row_idx,
        "authenticity_tier": tier,
        "test_idx": test_idx,
        "global_csv_idx": global_idx,
        "image_name": image_name,
        "image_path": ann.get("image_path") or meta.get("image_path"),
        "description": ann.get("description") or _safe_str(meta.get("prompt")),
        "generator_model": ann.get("generator_model"),
        "category": ann.get("category"),
        "challenge": ann.get("challenge"),
        "human_scores": {
            "authenticity": auth_score,
            "quality": ann.get("quality"),
            "correspondence": ann.get("correspondence"),
        },
        "xai_agreement": {
            "mean_base_models_pearson_r": round(_mean_pairwise_pearson(base_maps), 4),
            "bagging_vs_stacking_pearson_r": round(
                _pearson_2d(bagging_map, stacking_map), 4
            ),
        },
    }
    return row_render, sub_image_meta


def run_ensemble_figure_pipeline(
    num_figures: int = 10,
    seed: int = 42,
    indices: Sequence[int] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    base_maps_dir: Path = MAPS_ROOT,
    ensemble_maps_dir: Path = ENSEMBLE_MAPS_DIR,
    manifest_path: Path = MANIFEST_FILE,
    annotations_path: Path = ANNOTATIONS_FILE,
    models: Sequence[str] = MODELS,
    cmap_name: str = "coolwarm",
    alpha: float = 0.5,
    dpi: int = 600,
    allow_missing_images: bool = False,
    images_dir: Path | None = None,
    custom_name: str = "ensemble_figure_custom",
    show_row_labels: bool = False,
) -> tuple[dict[str, Any], list[Path]]:
    """Run full pipeline to generate ensemble heatmap figures and JSON manifest.

    Args:
        num_figures: Number of 3-row stratified candidate figures to render
            (ignored when ``indices`` is provided).
        seed: Random seed for stratified sampling.
        indices: Optional explicit sequence of ``test_idx`` values to render
            into a single custom figure.
        output_dir: Output directory for PNGs, PDFs, and JSON manifest.
        base_maps_dir: Root XAI_Maps directory for Experiment 2 base models.
        ensemble_maps_dir: Directory containing Experiment 3 ensemble prototypes.
        manifest_path: Path to ``test_samples_manifest.json``.
        annotations_path: Path to ``real_images_annotations.csv``.
        models: Ordered list of base model architectures.
        cmap_name: Matplotlib diverging colormap name.
        alpha: Overlay opacity in ``[0.0, 1.0]``.
        dpi: PNG output resolution in DPI.
        allow_missing_images: Whether to use placeholder tiles if images are missing.
        images_dir: Optional custom directory containing raw stimulus PNGs.
        custom_name: Filename stem used when ``indices`` is provided.

    Returns:
        Tuple of ``(manifest_payload, generated_png_paths)``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, annotations_by_idx = load_manifest_and_annotations(
        manifest_path=manifest_path,
        annotations_path=annotations_path,
    )

    base_prototypes = load_prototypes(
        method="mpm",
        maps_dir=base_maps_dir,
        models=models,
    )
    active_models = [m for m in models if m in base_prototypes]
    ensemble_prototypes = load_ensemble_prototypes(
        ensemble_maps_dir=ensemble_maps_dir,
    )

    if indices is not None:
        n_total = len(manifest)
        for idx in indices:
            if not (0 <= idx < n_total):
                raise ValueError(
                    f"Stimulus test_idx {idx} out of range [0, {n_total - 1}]."
                )
        low_pool, medium_pool, high_pool = _compute_tertile_pools(manifest)
        tier_by_idx = (
            {int(i): "low" for i in low_pool}
            | {int(i): "medium" for i in medium_pool}
            | {int(i): "high" for i in high_pool}
        )
        figure_specs: list[tuple[int, str, list[tuple[int, str]]]] = [
            (
                1,
                custom_name,
                [(int(idx), tier_by_idx.get(int(idx), "custom")) for idx in indices],
            )
        ]
        sampling_strategy = "explicit_indices"
        json_path = output_dir / f"{custom_name}_manifest.json"
    else:
        sampled_groups = sample_stratified_figure_rows(
            manifest=manifest,
            num_figures=num_figures,
            seed=seed,
        )
        figure_specs = [
            (fig_id, f"ensemble_figure_{fig_id:02d}", group)
            for fig_id, group in enumerate(sampled_groups, start=1)
        ]
        sampling_strategy = "authenticity_tertiles_without_replacement"
        json_path = output_dir / "ensemble_figures_manifest.json"

    figures_records: list[dict[str, Any]] = []
    png_paths: list[Path] = []

    for fig_id, stem, row_specs in figure_specs:
        png_path = output_dir / f"{stem}.png"
        pdf_path = output_dir / f"{stem}.pdf"
        row_renders: list[dict[str, Any]] = []
        sub_images: list[dict[str, Any]] = []

        for row_idx, (test_idx, tier) in enumerate(row_specs, start=1):
            row_render, sub_image_meta = _build_row_entry(
                row_idx=row_idx,
                test_idx=test_idx,
                tier=tier,
                manifest=manifest,
                annotations_by_idx=annotations_by_idx,
                base_prototypes=base_prototypes,
                ensemble_prototypes=ensemble_prototypes,
                active_models=active_models,
                cmap_name=cmap_name,
                alpha=alpha,
                allow_missing_images=allow_missing_images,
                images_dir=images_dir,
                show_row_labels=show_row_labels,
            )
            row_renders.append(row_render)
            sub_images.append(sub_image_meta)

        render_ensemble_heatmap_figure(
            rows=row_renders,
            ordered_models=active_models,
            output_png=png_path,
            output_pdf=pdf_path,
            dpi=dpi,
        )
        png_paths.append(png_path)
        figures_records.append(
            {
                "figure_id": fig_id,
                "files": {
                    "png": png_path.name,
                    "pdf": pdf_path.name,
                },
                "sub_images": sub_images,
            }
        )

    columns = [
        "Original",
        "Bagging Ensemble",
        "Stacking Ensemble",
        *[MODEL_DISPLAY_NAMES.get(m, m) for m in active_models],
    ]
    manifest_payload: dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "num_figures": len(figures_records),
            "rows_per_figure": len(figure_specs[0][2]) if figure_specs else 0,
            "sampling_strategy": sampling_strategy,
            "seed": None if indices is not None else seed,
            "xai_method": "Multiscale_Pixel_Masking",
            "heatmap_source": "10_variant_prototypes",
            "overlay_style": {
                "cmap": cmap_name,
                "alpha": alpha,
                "symmetric_range": [-1.0, 1.0],
                "dpi": dpi,
            },
            "columns": columns,
        },
        "figures": figures_records,
    }

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(manifest_payload, json_file, indent=2, ensure_ascii=False)
    info(
        "Saved figure manifest JSON to %s (%d figures)",
        json_path,
        len(figures_records),
    )
    return manifest_payload, png_paths


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate multi-row ensemble (Bagging & Stacking) vs. 7 base model "
            "MPM heatmap comparison figures and a structured JSON manifest."
        )
    )
    parser.add_argument(
        "--num-figures",
        type=int,
        default=10,
        help="Number of 3-row authenticity-stratified figures to generate (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified stimulus sampling (default: 42).",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help=(
            "Comma-separated test_idx values (e.g. '101,42,233') to render a single "
            "custom figure instead of random stratified sampling."
        ),
    )
    parser.add_argument(
        "--custom-name",
        type=str,
        default="ensemble_figure_custom",
        help="Output filename stem when --indices is provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write generated PNG, PDF, and JSON files.",
    )
    parser.add_argument(
        "--base-maps-dir",
        type=Path,
        default=MAPS_ROOT,
        help="Root directory containing Experiment 2 XAI_Maps.",
    )
    parser.add_argument(
        "--ensemble-maps-dir",
        type=Path,
        default=ENSEMBLE_MAPS_DIR,
        help="Directory containing Experiment 3 ensemble prototype .npy files.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="coolwarm",
        help="Diverging colormap name (default: 'coolwarm').",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Overlay blending opacity in [0.0, 1.0] (default: 0.5).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Raster output resolution in DPI (default: 600).",
    )
    parser.add_argument(
        "--show-row-labels",
        action="store_true",
        help="Show left-side High/Mid/Low Authenticity row labels.",
    )
    return parser


def main() -> int:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    parsed_indices: list[int] | None = None
    if args.indices:
        parsed_indices = [
            int(tok.strip()) for tok in args.indices.split(",") if tok.strip()
        ]

    run_ensemble_figure_pipeline(
        num_figures=args.num_figures,
        seed=args.seed,
        indices=parsed_indices,
        output_dir=args.output_dir,
        base_maps_dir=args.base_maps_dir,
        ensemble_maps_dir=args.ensemble_maps_dir,
        cmap_name=args.cmap,
        alpha=args.alpha,
        dpi=args.dpi,
        custom_name=args.custom_name,
        show_row_labels=args.show_row_labels,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
