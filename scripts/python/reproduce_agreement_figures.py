"""Reproduce across-architecture attribution agreement figures and ranking tables.

Computes mean pairwise across-architecture spatial Pearson correlations per stimulus
for Grad-CAM, MPM, and LIME, ranks stimuli by across-architecture agreement, and
renders publication-ready multi-panel figures (Figure 4, Figure A.5, and LIME extension).
Supports multi-candidate exploration with top-K and bottom-K overview panels.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# Ensure repository root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Image_Authenticity_prediction.main.Utils.comparisons import uniform_heatmaps
from Image_Authenticity_prediction.main.Utils.config import get_model_order
from Image_Authenticity_prediction.main.Utils.logger import error, info, warn
from Image_Authenticity_prediction.main.Utils.visualization import render_overlay_rgb

# Default 7 architectures: CLIP is mandatory per research specification
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

METHOD_DISPLAY_NAMES: dict[str, str] = {
    "gradcam": "Grad-CAM",
    "mpm": "Multiscale Pixel Masking",
    "lime": "LIME",
}

METHOD_DIRS: dict[str, str] = {
    "gradcam": "GradCAM",
    "mpm": "Multiscale_Pixel_Masking",
    "lime": "LIME",
}


def normalize_model_name(name: str) -> str:
    """Normalize model identifier to canonical lowercase registry name."""
    clean = name.strip().lower()
    if clean in ("clip", "clipvit", "clip_vit"):
        return "clip_vit_b16"
    return clean


def resolve_maps_dir(maps_dir: Path) -> Path:
    """Resolve directory containing GradCAM, Multiscale_Pixel_Masking, or LIME subdirectories."""
    candidates = [
        maps_dir,
        maps_dir / "XAI_Maps",
        maps_dir / "Experiment_2_variants" / "XAI_Maps",
        maps_dir / "XAI_Maps" / "Test_Set",
    ]
    for cand in candidates:
        if any(
            (cand / sub).is_dir()
            for sub in ("GradCAM", "Multiscale_Pixel_Masking", "LIME")
        ):
            return cand
    return maps_dir


def load_manifest(
    maps_dir: Path, custom_path: Path | None = None
) -> list[dict[str, Any]]:
    """Load test samples manifest linking test split indices to metadata."""
    resolved = resolve_maps_dir(maps_dir)
    candidates = [
        custom_path,
        resolved / "test_samples_manifest.json",
        maps_dir / "test_samples_manifest.json",
        _PROJECT_ROOT
        / "Outputs/Experiment_2_variants/XAI_Maps/test_samples_manifest.json",
    ]

    for path in filter(None, candidates):
        if path.is_file():
            try:
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    info(
                        "Loaded test samples manifest from %s (%d items)",
                        path,
                        len(data),
                    )
                    return data
            except (OSError, json.JSONDecodeError) as e:
                warn("Failed parsing manifest at %s: %s", path, e)

    raise FileNotFoundError(
        f"Test samples manifest not found in {[str(p) for p in candidates if p]}. "
        "Please provide --manifest-path or ensure Outputs/Experiment_2_variants/XAI_Maps/"
        "test_samples_manifest.json exists."
    )


def load_annotations_df(custom_path: Path | None = None) -> pd.DataFrame | None:
    """Load annotations CSV to fetch stimulus prompts and metadata."""
    candidates = [
        custom_path,
        _PROJECT_ROOT
        / "Image_Authenticity_prediction/Dataset/AIGCIQA2023/real_images_annotations.csv",
        _PROJECT_ROOT / "Dataset/AIGCIQA2023/real_images_annotations.csv",
    ]
    for cand in filter(None, candidates):
        if Path(cand).is_file():
            try:
                df = pd.read_csv(cand)
                info("Loaded annotations from %s (%d rows)", cand, len(df))
                return df
            except (OSError, pd.errors.EmptyDataError) as e:
                warn("Failed reading annotations CSV at %s: %s", cand, e)
    return None


def _parse_model_and_variant(
    stem: str, known_models: Sequence[str]
) -> tuple[str | None, str | None]:
    """Parse model identifier and variant tag from a file stem."""
    clean_stem = stem.replace("_maps", "")
    for km in known_models:
        prefix = f"{km}_"
        if clean_stem.startswith(prefix):
            return km, clean_stem[len(prefix) :]
        if clean_stem == km:
            return km, "default"

    parts = clean_stem.split("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (None, None)


def _select_best_variant(variant_names: list[str]) -> str:
    """Select the primary / representative variant name from variant tags."""

    def _priority(name: str) -> int:
        lower = name.lower()
        if "exp1a_variant1" in lower or "best" in lower or "orig" in lower:
            return 0
        if "exp1a" in lower:
            return 1
        if "greedy" in lower and ("variant1_" in lower or lower.endswith("variant1")):
            return 2
        if "greedy" in lower:
            return 3
        return 4

    return min(variant_names, key=_priority)


def discover_and_load_attribution_maps(
    maps_dir: Path,
    method: str,
    models: Sequence[str],
    variants_strategy: str = "prototype",
    target_res: tuple[int, int] = (224, 224),
) -> dict[str, np.ndarray]:
    """Discover, load, and aggregate attribution maps for the specified method and models."""
    resolved_dir = resolve_maps_dir(maps_dir)
    dir_name = METHOD_DIRS.get(method.lower(), method)

    method_dir: Path | None = None
    for cand in (
        resolved_dir / dir_name,
        resolved_dir / method.lower(),
        resolved_dir / method.upper(),
    ):
        if cand.is_dir():
            method_dir = cand
            break

    if method_dir is None:
        raise FileNotFoundError(
            f"Attribution maps directory for method '{method}' not found under '{resolved_dir}'.\n"
            f"Expected subdirectory: '{dir_name}' (e.g. {resolved_dir / dir_name})."
        )

    normalized_models = [normalize_model_name(m) for m in models]
    known_models = sorted(
        set(normalized_models + DEFAULT_MODELS), key=len, reverse=True
    )

    grouped_files: dict[str, dict[str, Path]] = defaultdict(dict)
    for npy_file in sorted(method_dir.glob("*_maps.npy")):
        m_name, var_tag = _parse_model_and_variant(npy_file.stem, known_models)
        if m_name and var_tag:
            canonical_m = normalize_model_name(m_name)
            if canonical_m in normalized_models:
                grouped_files[canonical_m][var_tag] = npy_file

    if not grouped_files:
        raise FileNotFoundError(
            f"No '*_maps.npy' files found for models {normalized_models} in '{method_dir}'."
        )

    prototypes: dict[str, np.ndarray] = {}
    for m_name in normalized_models:
        var_dict = grouped_files.get(m_name)
        if not var_dict:
            warn("Model '%s' has no attribution maps in %s", m_name, method_dir)
            continue

        var_tags = list(var_dict.keys())
        if variants_strategy == "best":
            selected_paths = [var_dict[_select_best_variant(var_tags)]]
        elif variants_strategy == "unpruned":
            unpruned_tags = [v for v in var_tags if "exp1a" in v or "unpruned" in v]
            selected_paths = [
                var_dict[t] for t in (unpruned_tags or [_select_best_variant(var_tags)])
            ]
        else:
            selected_paths = list(var_dict.values())

        loaded_arrays: list[np.ndarray] = []
        for path in selected_paths:
            try:
                raw_arr = np.load(path)
                if raw_arr.ndim == 2:
                    raw_arr = raw_arr[np.newaxis, ...]
                if raw_arr.ndim == 3:
                    loaded_arrays.append(
                        uniform_heatmaps(raw_arr, target_res[0], target_res[1]).astype(
                            np.float32
                        )
                    )
            except (OSError, ValueError) as e:
                warn("Failed loading map file %s: %s", path, e)

        if loaded_arrays:
            prototypes[m_name] = (
                loaded_arrays[0]
                if len(loaded_arrays) == 1
                else np.mean(np.stack(loaded_arrays, axis=0), axis=0)
            )
            info(
                "Loaded '%s' (%d variants) -> shape %s",
                m_name,
                len(loaded_arrays),
                prototypes[m_name].shape,
            )

    if len(prototypes) < 2:
        raise ValueError(
            f"Only {len(prototypes)} models loaded ({list(prototypes.keys())}). At least 2 models required."
        )

    return prototypes


def _compute_vectorized_pearson(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """Compute row-wise Pearson correlation between two 2D arrays of shape (N, D)."""
    a_diff = mat_a - np.mean(mat_a, axis=1, keepdims=True)
    b_diff = mat_b - np.mean(mat_b, axis=1, keepdims=True)
    denom = np.linalg.norm(a_diff, axis=1) * np.linalg.norm(b_diff, axis=1)
    return np.where(denom > 1e-12, np.sum(a_diff * b_diff, axis=1) / denom, 0.0)


def compute_across_architecture_agreement(
    prototypes: dict[str, np.ndarray],
    ordered_models: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[str, str], np.ndarray], list[str]]:
    """Compute pairwise across-architecture spatial Pearson correlation per stimulus."""
    if ordered_models:
        active_models = [m for m in ordered_models if m in prototypes]
    else:
        pref_order = [normalize_model_name(m) for m in get_model_order()]
        active_models = sorted(
            prototypes.keys(),
            key=lambda x: pref_order.index(x) if x in pref_order else 999,
        )

    if len(active_models) < 2:
        raise ValueError(f"At least 2 models required; got {active_models}")

    num_samples = prototypes[active_models[0]].shape[0]
    for m in active_models:
        if prototypes[m].shape[0] != num_samples:
            raise ValueError(
                f"Sample count mismatch: {active_models[0]} has {num_samples}, but {m} has {prototypes[m].shape[0]}"
            )

    flat_maps = {m: prototypes[m].reshape(num_samples, -1) for m in active_models}
    pairwise_corrs: dict[tuple[str, str], np.ndarray] = {}
    pair_matrices: list[np.ndarray] = []

    m_count = len(active_models)
    for i in range(m_count):
        for j in range(i + 1, m_count):
            m_i, m_j = active_models[i], active_models[j]
            corrs = _compute_vectorized_pearson(flat_maps[m_i], flat_maps[m_j])
            pairwise_corrs[(m_i, m_j)] = corrs
            pair_matrices.append(corrs)

    stacked_pairs = np.stack(pair_matrices, axis=0)
    return (
        np.mean(stacked_pairs, axis=0),
        np.std(stacked_pairs, axis=0),
        pairwise_corrs,
        active_models,
    )


def build_agreement_ranking_table(
    mean_agreement: np.ndarray,
    std_agreement: np.ndarray,
    manifest: list[dict[str, Any]],
    annotations_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build and export stimulus ranking table ordered by mean across-architecture agreement."""
    rows: list[dict[str, Any]] = []
    for test_idx in range(len(mean_agreement)):
        meta = manifest[test_idx] if test_idx < len(manifest) else {}
        global_idx = meta.get("global_csv_idx", test_idx)
        img_name = meta.get("image_name", f"sample_{test_idx}.png")
        auth_score = meta.get("authenticity_score", np.nan)

        prompt_str = ""
        if (
            annotations_df is not None
            and global_idx is not None
            and "Prompt" in annotations_df.columns
        ):
            try:
                g_int = int(global_idx)
                if g_int in annotations_df.index:
                    prompt_str = str(annotations_df.loc[g_int, "Prompt"])
                elif 0 <= g_int < len(annotations_df):
                    prompt_str = str(annotations_df.iloc[g_int]["Prompt"])
            except (ValueError, KeyError, IndexError):
                pass
        if not prompt_str:
            prompt_str = str(meta.get("prompt", ""))

        rows.append(
            {
                "test_idx": test_idx,
                "global_csv_idx": global_idx,
                "image_name": img_name,
                "prompt": prompt_str,
                "authenticity_score": auth_score,
                "mean_agreement": float(mean_agreement[test_idx]),
                "std_agreement": float(std_agreement[test_idx]),
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values(by="mean_agreement", ascending=False)
        .reset_index(drop=True)
    )
    df.insert(0, "rank", df.index + 1)
    return df


def find_stimulus_index(
    identifier: str,
    manifest: list[dict[str, Any]],
    ranking_df: pd.DataFrame | None = None,
) -> int:
    """Resolve an identifier (filename, stem, or prompt substring) to a test index."""
    query = identifier.strip().lower()

    if query.isdigit():
        val = int(query)
        if 0 <= val < len(manifest):
            return val

    for meta in manifest:
        t_idx = int(meta["test_idx"])
        name = str(meta.get("image_name", "")).lower()
        if query in (name, Path(name).stem.lower()) or str(
            meta.get("image_path", "")
        ).lower().endswith(f"/{query}"):
            return t_idx

    if ranking_df is not None and "prompt" in ranking_df.columns:
        for _, row in ranking_df.iterrows():
            if query in str(row["prompt"]).lower():
                return int(row["test_idx"])

    for meta in manifest:
        if query in str(meta.get("image_name", "")).lower():
            return int(meta["test_idx"])

    raise ValueError(
        f"Could not resolve stimulus identifier '{identifier}' to a test index."
    )


def load_stimulus_image(
    test_idx: int,
    manifest: list[dict[str, Any]],
    images_dir: Path | None = None,
    allow_missing: bool = False,
) -> np.ndarray:
    """Load the original RGB image array in range [0, 1] for a test stimulus."""
    meta = manifest[test_idx] if test_idx < len(manifest) else {}
    img_name = str(meta.get("image_name", f"{test_idx}.png"))
    raw_path_str = str(meta.get("image_path", ""))

    candidate_paths: list[Path] = []
    if images_dir:
        candidate_paths.extend([images_dir / img_name, images_dir / raw_path_str])
    if raw_path_str:
        p = Path(raw_path_str)
        candidate_paths.extend(
            [p, _PROJECT_ROOT / p, _PROJECT_ROOT / "Image_Authenticity_prediction" / p]
        )
    candidate_paths.append(
        _PROJECT_ROOT
        / "Image_Authenticity_prediction/Dataset/AIGCIQA2023/Image/allimg"
        / img_name
    )

    for cand in candidate_paths:
        if cand.is_file():
            bgr = cv2.imread(str(cand))
            if bgr is not None:
                return (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).astype(np.float32) / 255.0

    if allow_missing:
        warn(
            "Original image '%s' (index %d) not found locally; generating neutral placeholder.",
            img_name,
            test_idx,
        )
        placeholder = np.full((224, 224, 3), 0.7, dtype=np.float32)
        placeholder[::16, :, :] = 0.6
        placeholder[:, ::16, :] = 0.6
        return placeholder

    raise FileNotFoundError(
        f"Original image file '{img_name}' for test index {test_idx} not found. "
        "Pass --images-dir or --allow-missing-images."
    )


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Ensure array is uint8 image data in range [0, 255]."""
    return (
        arr if arr.dtype == np.uint8 else np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    )


def render_agreement_figure(
    high_input_rgb: np.ndarray,
    high_overlays: dict[str, np.ndarray],
    low_input_rgb: np.ndarray,
    low_overlays: dict[str, np.ndarray],
    ordered_models: Sequence[str],
    output_pdf: Path,
    output_png: Path,
    method_name: str = "Grad-CAM",
    title_style: str = "auto",
    dpi: int = 300,
    high_label: str | None = None,
    low_label: str | None = None,
) -> None:
    """Render and export publication-ready 2-row multi-panel agreement figure."""
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    models_list = list(ordered_models)
    widths = [1.2] + [1.0] * len(models_list)
    fig = plt.figure(figsize=(1.15 * (1.2 + len(models_list)), 3.8))
    gs = GridSpec(
        2,
        len(widths),
        width_ratios=widths,
        height_ratios=[1, 1],
        left=0.04,
        right=0.98,
        top=0.82,
        bottom=0.12,
        wspace=0.28,
        hspace=0.45,
    )

    method_clean = METHOD_DISPLAY_NAMES.get(method_name.lower(), method_name)
    if high_label and low_label:
        panel_a, panel_b = high_label, low_label
    elif title_style == "figure4" or (
        title_style == "auto" and "grad" in method_name.lower()
    ):
        panel_a, panel_b = "A  High agreement", "B  Low agreement"
    else:
        panel_a = f"{method_clean}: High agreement between architectures"
        panel_b = f"{method_clean}: Low agreement between architectures"

    fig.text(
        0.04, 0.91, panel_a, fontsize=11, fontweight="bold", ha="left", va="center"
    )
    fig.text(
        0.04, 0.49, panel_b, fontsize=11, fontweight="bold", ha="left", va="center"
    )

    headers = ["Input"] + [MODEL_DISPLAY_NAMES.get(m, m) for m in models_list]
    for c in range(len(widths)):
        bbox = gs[0, c].get_position(fig)
        fig.text(
            (bbox.x0 + bbox.x1) / 2.0,
            0.83,
            headers[c],
            fontsize=9,
            ha="center",
            va="bottom",
        )

    rows_data = [
        (0, _to_uint8(high_input_rgb), high_overlays),
        (1, _to_uint8(low_input_rgb), low_overlays),
    ]
    for row_idx, input_uint8, overlays in rows_data:
        ax_in = fig.add_subplot(gs[row_idx, 0])
        ax_in.imshow(input_uint8)
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        ax_in.set_frame_on(False)

        for c, m_name in enumerate(models_list):
            ax_m = fig.add_subplot(gs[row_idx, 1 + c])
            ax_m.imshow(
                overlays.get(m_name, np.full((224, 224, 3), 128, dtype=np.uint8))
            )
            ax_m.set_xticks([])
            ax_m.set_yticks([])
            ax_m.set_frame_on(False)

    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(output_png, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    info("Rendered figure -> PDF: %s, PNG: %s", output_pdf, output_png)


def render_multi_row_agreement_figure(
    stimuli_rows: list[dict[str, Any]],
    ordered_models: Sequence[str],
    output_pdf: Path,
    output_png: Path,
    section_title: str,
    method_name: str = "Grad-CAM",
    dpi: int = 300,
) -> None:
    """Render a multi-row grid of stimuli overlays (e.g. top-10 or bottom-10).

    Args:
        stimuli_rows: List of dicts with 'rank', 'score', 'prompt', 'input_rgb', 'overlays'.
        ordered_models: List of model identifiers.
        output_pdf: Output vector PDF path.
        output_png: Output raster PNG path.
        section_title: Title heading for the figure.
        method_name: Method name string.
        dpi: Output resolution.
    """
    if not stimuli_rows:
        return

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    k_rows = len(stimuli_rows)
    models_list = list(ordered_models)
    widths = [1.2] + [1.0] * len(models_list)

    fig_width = 1.15 * (1.2 + len(models_list))
    fig_height = 0.8 + 1.55 * k_rows
    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = GridSpec(
        k_rows,
        len(widths),
        width_ratios=widths,
        height_ratios=[1] * k_rows,
        left=0.04,
        right=0.98,
        top=1.0 - (0.55 / fig_height),
        bottom=0.03,
        wspace=0.25,
        hspace=0.45,
    )

    method_clean = METHOD_DISPLAY_NAMES.get(method_name.lower(), method_name)
    fig.text(
        0.04,
        1.0 - (0.28 / fig_height),
        f"{method_clean}: {section_title}",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="center",
    )

    headers = ["Input"] + [MODEL_DISPLAY_NAMES.get(m, m) for m in models_list]
    for c in range(len(widths)):
        bbox = gs[0, c].get_position(fig)
        fig.text(
            (bbox.x0 + bbox.x1) / 2.0,
            1.0 - (0.45 / fig_height),
            headers[c],
            fontsize=9,
            ha="center",
            va="bottom",
        )

    for r_idx, s_data in enumerate(stimuli_rows):
        rank = s_data["rank"]
        score = s_data["score"]
        raw_prompt = s_data.get("prompt", "")
        prompt_trunc = (raw_prompt[:42] + "...") if len(raw_prompt) > 45 else raw_prompt

        row_bbox = gs[r_idx, 0].get_position(fig)
        row_label = (
            f'Rank {rank} (r = {score:+.2f})  "{prompt_trunc}"'
            if prompt_trunc
            else f"Rank {rank} (r = {score:+.2f})"
        )
        fig.text(
            0.04,
            row_bbox.y1 + 0.008,
            row_label,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

        ax_in = fig.add_subplot(gs[r_idx, 0])
        ax_in.imshow(_to_uint8(s_data["input_rgb"]))
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        ax_in.set_frame_on(False)

        overlays = s_data["overlays"]
        for c, m_name in enumerate(models_list):
            ax_m = fig.add_subplot(gs[r_idx, 1 + c])
            ax_m.imshow(
                overlays.get(m_name, np.full((224, 224, 3), 128, dtype=np.uint8))
            )
            ax_m.set_xticks([])
            ax_m.set_yticks([])
            ax_m.set_frame_on(False)

    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(output_png, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    info("Rendered multi-row figure -> PDF: %s, PNG: %s", output_pdf, output_png)


def _find_image_in_dir(directory: Path, base_name: str) -> Path | None:
    """Find image file matching base_name case-insensitively."""
    target_lower = base_name.lower()
    target_stem = Path(base_name).stem.lower()
    for child in directory.iterdir():
        if child.is_file() and child.name.lower() in (
            target_lower,
            f"{target_stem}.png",
        ):
            return child
    return None


def render_from_directories(
    high_dir: Path,
    low_dir: Path,
    models: Sequence[str],
    output_pdf: Path,
    output_png: Path,
    method_name: str = "Grad-CAM",
    title_style: str = "auto",
    dpi: int = 300,
) -> None:
    """Render agreement figure directly from pre-rendered image directories."""
    if not high_dir.is_dir() or not low_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {high_dir} or {low_dir}")

    high_in = _find_image_in_dir(high_dir, "original.png") or _find_image_in_dir(
        high_dir, "input.png"
    )
    low_in = _find_image_in_dir(low_dir, "original.png") or _find_image_in_dir(
        low_dir, "input.png"
    )
    if not high_in or not low_in:
        raise FileNotFoundError(
            "Missing 'original.png' or 'input.png' in input folders."
        )

    high_overlays: dict[str, np.ndarray] = {}
    low_overlays: dict[str, np.ndarray] = {}
    available_models: list[str] = []

    for m in models:
        norm_m = normalize_model_name(m)
        disp_m = MODEL_DISPLAY_NAMES.get(norm_m, norm_m)
        p_high = _find_image_in_dir(high_dir, f"{norm_m}.png") or _find_image_in_dir(
            high_dir, f"{disp_m}.png"
        )
        p_low = _find_image_in_dir(low_dir, f"{norm_m}.png") or _find_image_in_dir(
            low_dir, f"{disp_m}.png"
        )

        if p_high and p_low:
            high_overlays[norm_m] = plt.imread(p_high)
            low_overlays[norm_m] = plt.imread(p_low)
            available_models.append(norm_m)

    if not available_models:
        raise FileNotFoundError(
            f"No matching model overlay images found in {high_dir} and {low_dir} for {models}"
        )

    render_agreement_figure(
        high_input_rgb=plt.imread(high_in),
        high_overlays=high_overlays,
        low_input_rgb=plt.imread(low_in),
        low_overlays=low_overlays,
        ordered_models=available_models,
        output_pdf=output_pdf,
        output_png=output_png,
        method_name=method_name,
        title_style=title_style,
        dpi=dpi,
    )


def _resolve_stimulus_index(
    idx_override: int | None,
    id_override: str | None,
    default_idx: int,
    manifest: list[dict[str, Any]],
    ranking_df: pd.DataFrame,
) -> int:
    """Resolve a stimulus index from manual override or ranking default."""
    if idx_override is not None:
        return int(idx_override)
    if id_override is not None:
        return find_stimulus_index(id_override, manifest, ranking_df)
    return int(default_idx)


def run_agreement_pipeline_for_method(
    method: str,
    maps_dir: Path,
    output_dir: Path,
    models: Sequence[str] = DEFAULT_MODELS,
    variants_strategy: str = "prototype",
    target_res: tuple[int, int] = (224, 224),
    top_k: int = 10,
    high_idx: int | None = None,
    low_idx: int | None = None,
    high_id: str | None = None,
    low_id: str | None = None,
    images_dir: Path | None = None,
    allow_missing_images: bool = False,
    title_style: str = "auto",
    cmap_name: str = "coolwarm",
    alpha: float = 0.5,
    dpi: int = 300,
    manifest_path: Path | None = None,
    annotations_path: Path | None = None,
) -> tuple[pd.DataFrame, Path, Path]:
    """Execute complete analysis and figure generation pipeline for one XAI method."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_method = method.lower()
    method_display = METHOD_DISPLAY_NAMES.get(clean_method, method)

    # 1. Discover and load maps
    prototypes = discover_and_load_attribution_maps(
        maps_dir=maps_dir,
        method=clean_method,
        models=models,
        variants_strategy=variants_strategy,
        target_res=target_res,
    )
    mean_agreement, std_agreement, _, active_models = (
        compute_across_architecture_agreement(
            prototypes=prototypes, ordered_models=models
        )
    )

    # 2. Metadata and ranking table
    manifest = load_manifest(maps_dir=maps_dir, custom_path=manifest_path)
    annotations_df = load_annotations_df(custom_path=annotations_path)

    ranking_df = build_agreement_ranking_table(
        mean_agreement=mean_agreement,
        std_agreement=std_agreement,
        manifest=manifest,
        annotations_df=annotations_df,
    )
    ranking_csv = (
        output_dir / f"{clean_method}_across_architecture_agreement_ranking.csv"
    )
    ranking_df.to_csv(ranking_csv, index=False)
    info(
        "Saved %s ranking table to %s (%d rows)",
        method_display,
        ranking_csv,
        len(ranking_df),
    )

    symmetric = clean_method == "lime"

    # 3. Determine number of candidates to process
    k_eff = min(top_k, max(1, len(ranking_df) // 2))

    # Helper to build row data dict
    def _build_stimulus_entry(row: pd.Series) -> dict[str, Any]:
        t_idx = int(row["test_idx"])
        img_rgb = load_stimulus_image(t_idx, manifest, images_dir, allow_missing_images)
        overlays = {
            m: render_overlay_rgb(
                img_rgb, prototypes[m][t_idx], cmap_name, alpha, symmetric
            )
            for m in active_models
        }
        return {
            "rank": int(row["rank"]),
            "test_idx": t_idx,
            "score": float(row["mean_agreement"]),
            "prompt": str(row.get("prompt", "")),
            "input_rgb": img_rgb,
            "overlays": overlays,
        }

    # 4. Generate Top-K and Bottom-K overview panels
    top_entries = [_build_stimulus_entry(ranking_df.iloc[i]) for i in range(k_eff)]
    bottom_entries = [
        _build_stimulus_entry(ranking_df.iloc[-(i + 1)]) for i in range(k_eff)
    ]

    top_pdf = output_dir / f"{clean_method}_top_{k_eff}_high_agreement.pdf"
    top_png = output_dir / f"{clean_method}_top_{k_eff}_high_agreement.png"
    render_multi_row_agreement_figure(
        stimuli_rows=top_entries,
        ordered_models=active_models,
        output_pdf=top_pdf,
        output_png=top_png,
        section_title=f"Top {k_eff} High Agreement Stimuli",
        method_name=method_display,
        dpi=dpi,
    )

    bottom_pdf = output_dir / f"{clean_method}_bottom_{k_eff}_low_agreement.pdf"
    bottom_png = output_dir / f"{clean_method}_bottom_{k_eff}_low_agreement.png"
    render_multi_row_agreement_figure(
        stimuli_rows=bottom_entries,
        ordered_models=active_models,
        output_pdf=bottom_pdf,
        output_png=bottom_png,
        section_title=f"Bottom {k_eff} Low Agreement Stimuli",
        method_name=method_display,
        dpi=dpi,
    )

    # 5. Generate individual candidate 2-row figures (Top i vs Bottom i)
    candidates_dir = output_dir / f"candidates_{clean_method}"
    for i in range(k_eff):
        high_item = top_entries[i]
        low_item = bottom_entries[i]
        h_rank = high_item["rank"]
        l_rank = low_item["rank"]

        cand_stem = f"candidate_pair_{i + 1:02d}_rank_{h_rank:03d}_vs_{l_rank:03d}"
        c_pdf = candidates_dir / f"{cand_stem}.pdf"
        c_png = candidates_dir / f"{cand_stem}.png"

        high_lbl = f"A  High agreement (Rank {h_rank}, r = {high_item['score']:+.2f})"
        low_lbl = f"B  Low agreement (Rank {l_rank}, r = {low_item['score']:+.2f})"

        render_agreement_figure(
            high_input_rgb=high_item["input_rgb"],
            high_overlays=high_item["overlays"],
            low_input_rgb=low_item["input_rgb"],
            low_overlays=low_item["overlays"],
            ordered_models=active_models,
            output_pdf=c_pdf,
            output_png=c_png,
            method_name=method_display,
            title_style=title_style,
            dpi=dpi,
            high_label=high_lbl,
            low_label=low_lbl,
        )

    # 6. Primary paper figure (Rank 1 vs Rank N, or manual overrides)
    sel_high_idx = _resolve_stimulus_index(
        high_idx, high_id, ranking_df.iloc[0]["test_idx"], manifest, ranking_df
    )
    sel_low_idx = _resolve_stimulus_index(
        low_idx, low_id, ranking_df.iloc[-1]["test_idx"], manifest, ranking_df
    )

    # If already computed in top/bottom entries, reuse; otherwise load
    high_match = next((e for e in top_entries if e["test_idx"] == sel_high_idx), None)
    if high_match:
        high_input, high_overlays = high_match["input_rgb"], high_match["overlays"]
    else:
        high_input = load_stimulus_image(
            sel_high_idx, manifest, images_dir, allow_missing_images
        )
        high_overlays = {
            m: render_overlay_rgb(
                high_input, prototypes[m][sel_high_idx], cmap_name, alpha, symmetric
            )
            for m in active_models
        }

    low_match = next((e for e in bottom_entries if e["test_idx"] == sel_low_idx), None)
    if low_match:
        low_input, low_overlays = low_match["input_rgb"], low_match["overlays"]
    else:
        low_input = load_stimulus_image(
            sel_low_idx, manifest, images_dir, allow_missing_images
        )
        low_overlays = {
            m: render_overlay_rgb(
                low_input, prototypes[m][sel_low_idx], cmap_name, alpha, symmetric
            )
            for m in active_models
        }

    fig_stems = {
        "gradcam": "figure4_gradcam_agreement",
        "mpm": "figure_a5_mpm_agreement",
    }
    stem = fig_stems.get(clean_method, f"figure_{clean_method}_agreement")
    out_pdf = output_dir / f"{stem}.pdf"
    out_png = output_dir / f"{stem}.png"

    render_agreement_figure(
        high_input_rgb=high_input,
        high_overlays=high_overlays,
        low_input_rgb=low_input,
        low_overlays=low_overlays,
        ordered_models=active_models,
        output_pdf=out_pdf,
        output_png=out_png,
        method_name=method_display,
        title_style=title_style,
        dpi=dpi,
    )

    info(
        "Generated top-%d and bottom-%d candidates in %s", k_eff, k_eff, candidates_dir
    )
    return ranking_df, out_pdf, out_png


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Reproduce across-architecture attribution agreement figures and ranking tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--maps-dir",
        type=Path,
        default=_PROJECT_ROOT / "Outputs/Experiment_2_variants/XAI_Maps",
        help="Directory containing XAI attribution maps (GradCAM, MPM, LIME).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "Outputs/Experiment_2_variants/Plots/agreement_figures",
        help="Directory to save generated figures (.pdf, .png) and ranking tables (.csv).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory of raw stimulus images.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Path to test_samples_manifest.json.",
    )
    parser.add_argument(
        "--annotations-path",
        type=Path,
        default=None,
        help="Path to real_images_annotations.csv.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["gradcam", "mpm", "lime"],
        help="Attribution methods to process ('gradcam', 'mpm', 'lime', or 'all').",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models to include in comparison (default includes CLIP).",
    )
    parser.add_argument(
        "--variants",
        choices=["prototype", "best", "unpruned"],
        default="prototype",
        help="Variant aggregation strategy.",
    )
    parser.add_argument(
        "--target-res",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Spatial resolution (H W) for map comparison.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top and bottom agreement candidate stimuli to render.",
    )
    parser.add_argument(
        "--high-idx",
        type=int,
        default=None,
        help="Test split index for high agreement.",
    )
    parser.add_argument(
        "--low-idx", type=int, default=None, help="Test split index for low agreement."
    )
    parser.add_argument(
        "--high-id",
        type=str,
        default=None,
        help="Identifier/prompt for high agreement.",
    )
    parser.add_argument(
        "--low-id", type=str, default=None, help="Identifier/prompt for low agreement."
    )
    parser.add_argument(
        "--high-dir",
        type=Path,
        default=None,
        help="Pre-rendered folder for high agreement.",
    )
    parser.add_argument(
        "--low-dir",
        type=Path,
        default=None,
        help="Pre-rendered folder for low agreement.",
    )
    parser.add_argument(
        "--title-style",
        choices=["auto", "figure4", "figure_a5"],
        default="auto",
        help="Panel header labeling style.",
    )
    parser.add_argument(
        "--cmap", type=str, default="coolwarm", help="Diverging colormap name."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Overlay opacity [0.0 - 1.0]."
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI.")
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Allow placeholder images when raw dataset images are absent.",
    )
    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.high_dir and args.low_dir:
        try:
            render_from_directories(
                high_dir=args.high_dir,
                low_dir=args.low_dir,
                models=args.models,
                output_pdf=args.output_dir / "custom_directory_agreement_figure.pdf",
                output_png=args.output_dir / "custom_directory_agreement_figure.png",
                method_name="Grad-CAM",
                title_style=args.title_style,
                dpi=args.dpi,
            )
            return 0
        except (OSError, ValueError, RuntimeError, KeyError) as e:
            error("Direct directory rendering failed: %s", e)
            return 1

    methods_requested = (
        ["gradcam", "mpm", "lime"]
        if "all" in [m.lower() for m in args.methods]
        else [m.lower() for m in args.methods]
    )
    methods = list(dict.fromkeys(methods_requested))
    target_res = (int(args.target_res[0]), int(args.target_res[1]))
    success_count = 0

    for method in methods:
        info("Processing Across-Architecture Agreement for %s", method.upper())
        try:
            run_agreement_pipeline_for_method(
                method=method,
                maps_dir=args.maps_dir,
                output_dir=args.output_dir,
                models=args.models,
                variants_strategy=args.variants,
                target_res=target_res,
                top_k=args.top_k,
                high_idx=args.high_idx,
                low_idx=args.low_idx,
                high_id=args.high_id,
                low_id=args.low_id,
                images_dir=args.images_dir,
                allow_missing_images=args.allow_missing_images,
                title_style=args.title_style,
                cmap_name=args.cmap,
                alpha=args.alpha,
                dpi=args.dpi,
                manifest_path=args.manifest_path,
                annotations_path=args.annotations_path,
            )
            success_count += 1
        except FileNotFoundError as e:
            error("[%s] Missing required files: %s", method, e)
        except (OSError, ValueError, RuntimeError, KeyError) as e:
            error("[%s] Unexpected error in pipeline: %s", method, e)

    info(
        "Pipeline finished: %d/%d methods completed successfully.",
        success_count,
        len(methods),
    )
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
