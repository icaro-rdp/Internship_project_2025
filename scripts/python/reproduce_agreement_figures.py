"""Reproduce across-architecture attribution agreement figures and ranking tables.

Computes mean pairwise across-architecture spatial Pearson correlations per stimulus
for Grad-CAM, MPM, and LIME in range [-1, 1], ranks stimuli by across-architecture agreement,
and renders publication-ready multi-panel figures (Figure 4, Figure A.5, and LIME extension).
"""

from __future__ import annotations

import argparse
import json
import sys
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
from Image_Authenticity_prediction.main.Utils.logger import error, info, warn
from Image_Authenticity_prediction.main.Utils.visualization import (
    denormalize_image,
    render_overlay_rgb,
)

# Fixed paths
MAPS_ROOT = _PROJECT_ROOT / "Outputs/Experiment_2_variants/XAI_Maps"
PLOTS_DIR = _PROJECT_ROOT / "Outputs/Experiment_2_variants/Plots/agreement_figures"
MANIFEST_FILE = MAPS_ROOT / "test_samples_manifest.json"
ANNOTATIONS_FILE = (
    _PROJECT_ROOT
    / "Image_Authenticity_prediction/Dataset/AIGCIQA2023/real_images_annotations.csv"
)

# Fixed 7 architectures per research specification
MODELS: list[str] = [
    "barlowtwins",
    "resnet152",
    "densenet161",
    "efficientnetb3",
    "vgg16",
    "vgg19",
    "clip_vit_b16",
]

DEFAULT_MODELS: list[str] = MODELS

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

METHOD_DIRS: dict[str, str] = {
    "gradcam": "GradCAM",
    "mpm": "Multiscale_Pixel_Masking",
    "lime": "LIME",
}

METHOD_SUBDIRS: dict[str, str] = {
    "gradcam": "GradCAM",
    "mpm": "MPM",
    "lime": "LIME",
}

METHOD_DISPLAY_NAMES: dict[str, str] = {
    "gradcam": "Grad-CAM",
    "mpm": "Multiscale Pixel Masking",
    "lime": "LIME",
}


def normalize_model_name(name: str) -> str:
    """Normalize model identifier to canonical lowercase registry name."""
    clean = name.strip().lower()
    return "clip_vit_b16" if clean in ("clip", "clipvit", "clip_vit") else clean


def load_prototypes(
    method: str,
    target_res: tuple[int, int] = (224, 224),
    max_samples: int | None = None,
    maps_dir: Path = MAPS_ROOT,
    models: Sequence[str] = MODELS,
    variants_strategy: str = "prototype",
) -> dict[str, np.ndarray]:
    """Load attribution maps for fixed architectures, ensuring values in [-1, 1].

    Averages all available variant maps per architecture into a prototype array.

    Args:
        method: Attribution method ('gradcam', 'mpm', or 'lime').
        target_res: Spatial resolution (H, W) for maps comparison.
        max_samples: Optional limit on number of stimuli to evaluate.
        maps_dir: Directory containing XAI maps subdirectories.
        models: Sequence of model names to load.
        variants_strategy: Strategy name (kept for interface compatibility).

    Returns:
        Mapping of model identifier to prototype heatmap array of shape (N, H, W).
    """
    clean_method = method.lower()
    dir_name = METHOD_DIRS.get(clean_method, method)
    if (maps_dir / dir_name).is_dir():
        method_dir = maps_dir / dir_name
    elif any(maps_dir.glob("*_maps.npy")):
        method_dir = maps_dir
    else:
        raise FileNotFoundError(
            f"Attribution maps directory '{maps_dir / dir_name}' not found."
        )

    prototypes: dict[str, np.ndarray] = {}
    for m in models:
        norm_m = normalize_model_name(m)
        variant_files = sorted(method_dir.glob(f"{norm_m}_exp1a_variant*_maps.npy"))
        if not variant_files:
            variant_files = sorted(method_dir.glob(f"{norm_m}*_maps.npy"))
        if not variant_files:
            warn("No attribution map files found for %s in %s", norm_m, method_dir)
            continue

        if variants_strategy == "best":
            variant_files = variant_files[:1]

        loaded: list[np.ndarray] = []
        for f in variant_files:
            arr = np.load(f)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            if max_samples is not None and max_samples > 0:
                arr = arr[:max_samples]
            if arr.shape[1:] != target_res:
                arr = uniform_heatmaps(arr, target_res[0], target_res[1])
            # Ensure maps are strictly in [-1.0, 1.0] range
            loaded.append(np.clip(arr.astype(np.float32), -1.0, 1.0))

        if loaded:
            prototypes[norm_m] = (
                loaded[0]
                if len(loaded) == 1
                else np.mean(np.stack(loaded, axis=0), axis=0)
            )
            info(
                "Loaded '%s' (%d variants, range [%.2f, %.2f]) -> %s",
                norm_m,
                len(loaded),
                float(prototypes[norm_m].min()),
                float(prototypes[norm_m].max()),
                prototypes[norm_m].shape,
            )

    if len(prototypes) < 2:
        raise ValueError(
            f"Expected at least 2 models, got {len(prototypes)} in {method_dir}."
        )

    return prototypes


# Compatibility alias
discover_and_load_attribution_maps = load_prototypes


def compute_across_architecture_agreement(
    prototypes: dict[str, np.ndarray],
    ordered_models: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[str, str], np.ndarray], list[str]]:
    """Compute across-architecture mean pairwise Pearson correlation per stimulus.

    Args:
        prototypes: Mapping of model name to prototype heatmap array (N, H, W).
        ordered_models: Optional subset and ordering of models.

    Returns:
        Tuple of (mean agreement array, std agreement array, pairwise corrs dict, active models list).
    """
    active = [m for m in (ordered_models or MODELS) if m in prototypes]
    if len(active) < 2:
        raise ValueError(f"At least 2 models required; got {active}")

    num_samples = prototypes[active[0]].shape[0]
    flat = {m: prototypes[m].reshape(num_samples, -1) for m in active}

    pair_corrs: dict[tuple[str, str], np.ndarray] = {}
    pair_matrices: list[np.ndarray] = []

    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            m_i, m_j = active[i], active[j]
            ma, mb = flat[m_i], flat[m_j]
            a_diff = ma - np.mean(ma, axis=1, keepdims=True)
            b_diff = mb - np.mean(mb, axis=1, keepdims=True)
            denom = np.linalg.norm(a_diff, axis=1) * np.linalg.norm(b_diff, axis=1)
            r = np.where(denom > 1e-12, np.sum(a_diff * b_diff, axis=1) / denom, 0.0)
            pair_corrs[(m_i, m_j)] = r
            pair_matrices.append(r)

    stacked = np.stack(pair_matrices, axis=0)
    return np.mean(stacked, axis=0), np.std(stacked, axis=0), pair_corrs, active


def build_agreement_ranking_table(
    mean_agreement: np.ndarray,
    std_agreement: np.ndarray,
    manifest: list[dict[str, Any]] | None = None,
    annotations_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build stimulus ranking table ordered by mean across-architecture agreement.

    Args:
        mean_agreement: Array of mean agreement scores per test stimulus.
        std_agreement: Array of agreement standard deviations per test stimulus.
        manifest: Optional preloaded manifest list.
        annotations_df: Optional annotations DataFrame with prompts.

    Returns:
        DataFrame sorted descending by mean_agreement.
    """
    if manifest is None and MANIFEST_FILE.is_file():
        try:
            with open(MANIFEST_FILE) as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            manifest = None

    prompts: dict[int, str] = {}
    if annotations_df is not None and "Prompt" in annotations_df.columns:
        prompts = annotations_df["Prompt"].to_dict()
    elif ANNOTATIONS_FILE.is_file():
        try:
            adf = pd.read_csv(ANNOTATIONS_FILE)
            if "Prompt" in adf.columns:
                prompts = adf["Prompt"].to_dict()
        except (OSError, pd.errors.EmptyDataError):
            pass

    rows: list[dict[str, Any]] = []
    for test_idx in range(len(mean_agreement)):
        meta = manifest[test_idx] if (manifest and test_idx < len(manifest)) else {}
        global_idx = meta.get("global_csv_idx", test_idx)
        img_name = meta.get("image_name", f"{test_idx}.png")
        prompt_str = prompts.get(global_idx, meta.get("prompt", ""))

        rows.append(
            {
                "rank": 0,
                "test_idx": test_idx,
                "global_csv_idx": global_idx,
                "image_name": img_name,
                "prompt": str(prompt_str),
                "authenticity_score": meta.get("authenticity_score", np.nan),
                "mean_agreement": float(mean_agreement[test_idx]),
                "std_agreement": float(std_agreement[test_idx]),
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values(by="mean_agreement", ascending=False)
        .reset_index(drop=True)
    )
    df["rank"] = df.index + 1
    return df


def load_stimulus_image(
    test_idx: int,
    manifest: list[dict[str, Any]] | None = None,
    allow_missing: bool = False,
    images_dir: Path | None = None,
) -> np.ndarray:
    """Load denormalized [0, 1] RGB image (224x224 crop) matching model input.

    Args:
        test_idx: Test split index.
        manifest: Optional manifest list.
        allow_missing: Whether to return neutral placeholder if image not found.
        images_dir: Optional custom directory of images.

    Returns:
        RGB image float32 array in range [0, 1] of shape (224, 224, 3).
    """
    meta = manifest[test_idx] if (manifest and test_idx < len(manifest)) else {}
    global_idx = meta.get("global_csv_idx", test_idx)
    img_name = meta.get("image_name", f"{test_idx}.png")

    if images_dir is None:
        try:
            from Image_Authenticity_prediction.main.data import clip_dataset

            img_tensor, _ = clip_dataset[int(global_idx)]
            denorm = denormalize_image(
                img_tensor,
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
            return denorm.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        except (ImportError, IndexError, KeyError, OSError, TypeError, ValueError):
            pass

    raw_candidates = [
        (images_dir / img_name) if images_dir else None,
        _PROJECT_ROOT
        / "Image_Authenticity_prediction/Dataset/AIGCIQA2023/Image/allimg"
        / img_name,
    ]
    for cand in filter(None, raw_candidates):
        if cand.is_file():
            bgr = cv2.imread(str(cand))
            if bgr is not None:
                return (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).astype(np.float32) / 255.0

    if allow_missing:
        placeholder = np.full((224, 224, 3), 0.7, dtype=np.float32)
        placeholder[::16, :, :] = 0.6
        placeholder[:, ::16, :] = 0.6
        return placeholder

    raise FileNotFoundError(
        f"Stimulus image '{img_name}' (test_idx {test_idx}) not found."
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
    ordered_models: Sequence[str] = MODELS,
    output_pdf: Path = PLOTS_DIR / "figure4_gradcam_agreement.pdf",
    output_png: Path = PLOTS_DIR / "figure4_gradcam_agreement.png",
    method_name: str = "Grad-CAM",
    title_style: str = "auto",
    dpi: int = 300,
    high_label: str | None = None,
    low_label: str | None = None,
) -> None:
    """Render publication-ready 2-row multi-panel agreement figure matching paper layout."""
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    models_list = list(ordered_models)
    num_cols = 1 + len(models_list)
    widths = [1.20] + [1.0] * len(models_list)

    fig = plt.figure(figsize=(1.35 * (1.20 + len(models_list)), 4.8), facecolor="white")
    gs = GridSpec(
        2,
        num_cols,
        width_ratios=widths,
        height_ratios=[1.0, 1.0],
        left=0.04,
        right=0.98,
        top=0.76,
        bottom=0.06,
        wspace=0.25,
        hspace=0.68,
    )

    panel_a = high_label if high_label else "A  High agreement"
    panel_b = low_label if low_label else "B  Low agreement"

    # Precise alignment with column 0 image
    fig.canvas.draw()
    bbox_0_0 = gs[0, 0].get_position(fig)
    bbox_1_0 = gs[1, 0].get_position(fig)

    fig.text(
        bbox_0_0.x0,
        bbox_0_0.y1 + 0.10,
        panel_a,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    fig.text(
        bbox_1_0.x0,
        bbox_1_0.y1 + 0.04,
        panel_b,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    headers = ["Input"] + [MODEL_DISPLAY_NAMES.get(m, m) for m in models_list]
    for c in range(num_cols):
        bbox = gs[0, c].get_position(fig)
        fig.text(
            (bbox.x0 + bbox.x1) / 2.0,
            bbox.y1 + 0.015,
            headers[c],
            fontsize=9.5,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    rows_data = [
        (0, _to_uint8(high_input_rgb), high_overlays),
        (1, _to_uint8(low_input_rgb), low_overlays),
    ]
    for row_idx, input_uint8, overlays in rows_data:
        ax_in = fig.add_subplot(gs[row_idx, 0])
        ax_in.imshow(input_uint8, aspect="equal")
        ax_in.axis("off")

        for c, m_name in enumerate(models_list):
            ax_m = fig.add_subplot(gs[row_idx, 1 + c])
            ax_m.imshow(
                overlays.get(m_name, np.full((224, 224, 3), 128, dtype=np.uint8)),
                aspect="equal",
            )
            ax_m.axis("off")

    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(output_png, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    info("Rendered figure -> PDF: %s, PNG: %s", output_pdf, output_png)


def run_agreement_pipeline(
    method: str = "gradcam",
    top_k: int = 5,
    max_samples: int | None = None,
    output_dir: Path = PLOTS_DIR,
    cmap_name: str = "coolwarm",
    alpha: float = 0.5,
    dpi: int = 300,
    allow_missing_images: bool = False,
    maps_dir: Path = MAPS_ROOT,
    models: Sequence[str] = MODELS,
    high_idx: int | None = None,
    low_idx: int | None = None,
    variants_strategy: str = "prototype",
    target_res: tuple[int, int] = (224, 224),
    title_style: str = "auto",
    manifest_path: Path | None = None,
    annotations_path: Path | None = None,
    high_id: str | None = None,
    low_id: str | None = None,
    images_dir: Path | None = None,
) -> tuple[pd.DataFrame, Path, Path]:
    """Execute agreement pipeline: load maps in [-1, 1], rank stimuli, render K agreement plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_method = method.lower()
    method_display = METHOD_DISPLAY_NAMES.get(clean_method, method)
    subfolder_name = METHOD_SUBDIRS.get(clean_method, clean_method.upper())
    target_dir = (
        output_dir
        if output_dir.name.lower() in [clean_method, subfolder_name.lower()]
        else (output_dir / subfolder_name)
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load prototypes in [-1, 1]
    prototypes = load_prototypes(
        method=clean_method,
        target_res=target_res,
        max_samples=max_samples,
        maps_dir=maps_dir,
        models=models,
        variants_strategy=variants_strategy,
    )
    active_models = [m for m in models if m in prototypes]
    mean_agg, std_agg, _, _ = compute_across_architecture_agreement(
        prototypes, ordered_models=active_models
    )

    # 2. Manifest and ranking table
    manifest = None
    m_path = manifest_path or MANIFEST_FILE
    if m_path.is_file():
        try:
            with open(m_path) as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass

    ann_df = None
    a_path = annotations_path or ANNOTATIONS_FILE
    if a_path.is_file():
        try:
            ann_df = pd.read_csv(a_path)
        except (OSError, pd.errors.EmptyDataError):
            pass

    ranking_df = build_agreement_ranking_table(
        mean_agg, std_agg, manifest=manifest, annotations_df=ann_df
    )
    csv_path = target_dir / f"{clean_method}_across_architecture_agreement_ranking.csv"
    ranking_df.to_csv(csv_path, index=False)
    info(
        "Saved %s ranking table to %s (%d rows)",
        method_display,
        csv_path,
        len(ranking_df),
    )

    # 3. Build top and bottom candidate stimuli
    k_eff = min(top_k, max(1, len(ranking_df) // 2))

    def _build_entry(row: pd.Series) -> dict[str, Any]:
        t_idx = int(row["test_idx"])
        img_rgb = load_stimulus_image(
            t_idx,
            manifest=manifest,
            allow_missing=allow_missing_images,
            images_dir=images_dir,
        )
        overlays = {
            m: render_overlay_rgb(
                img_rgb, prototypes[m][t_idx], cmap_name, alpha, symmetric=True
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

    top_entries = [_build_entry(ranking_df.iloc[i]) for i in range(k_eff)]
    bottom_entries = [_build_entry(ranking_df.iloc[-(i + 1)]) for i in range(k_eff)]

    # Handle manual index overrides for the primary pair (pair 1)
    if high_idx is not None or low_idx is not None:
        sel_h = high_idx if high_idx is not None else ranking_df.iloc[0]["test_idx"]
        sel_l = low_idx if low_idx is not None else ranking_df.iloc[-1]["test_idx"]
        h_row = ranking_df.loc[ranking_df["test_idx"] == sel_h].iloc[0]
        l_row = ranking_df.loc[ranking_df["test_idx"] == sel_l].iloc[0]
        top_entries[0] = _build_entry(h_row)
        bottom_entries[0] = _build_entry(l_row)

    # 4. Render K plots (Top i vs Bottom i) in dedicated method subfolder
    primary_pdf = target_dir / "pair_01.pdf"
    primary_png = target_dir / "pair_01.png"

    for i in range(k_eff):
        p_idx = i + 1
        hi, lo = top_entries[i], bottom_entries[i]
        pair_pdf = target_dir / f"pair_{p_idx:02d}.pdf"
        pair_png = target_dir / f"pair_{p_idx:02d}.png"
        render_agreement_figure(
            high_input_rgb=hi["input_rgb"],
            high_overlays=hi["overlays"],
            low_input_rgb=lo["input_rgb"],
            low_overlays=lo["overlays"],
            ordered_models=active_models,
            output_pdf=pair_pdf,
            output_png=pair_png,
            method_name=method_display,
            title_style=title_style,
            dpi=dpi,
        )

    return ranking_df, primary_pdf, primary_png


# Compatibility aliases
run_agreement_pipeline_for_method = run_agreement_pipeline


def find_stimulus_index(
    identifier: str,
    manifest: list[dict[str, Any]],
    ranking_df: pd.DataFrame | None = None,
) -> int:
    """Resolve an identifier (digit, image filename, or prompt keyword) to a test index."""
    query = identifier.strip().lower()
    if query.isdigit():
        val = int(query)
        if 0 <= val < len(manifest):
            return val
    for meta in manifest:
        if query in str(meta.get("image_name", "")).lower():
            return int(meta["test_idx"])
    if ranking_df is not None and "prompt" in ranking_df.columns:
        for _, row in ranking_df.iterrows():
            if query in str(row["prompt"]).lower():
                return int(row["test_idx"])
    raise ValueError(f"Stimulus identifier '{identifier}' not found.")


def render_from_directories(
    high_dir: Path,
    low_dir: Path,
    models: Sequence[str] = MODELS,
    output_pdf: Path = PLOTS_DIR / "custom_dir.pdf",
    output_png: Path = PLOTS_DIR / "custom_dir.png",
    method_name: str = "Grad-CAM",
    title_style: str = "auto",
    dpi: int = 300,
) -> None:
    """Render figure from directory of pre-rendered PNGs."""
    high_in = high_dir / "original.png"
    low_in = low_dir / "original.png"
    high_overlays = {
        m: plt.imread(high_dir / f"{m}.png")
        for m in models
        if (high_dir / f"{m}.png").is_file()
    }
    low_overlays = {
        m: plt.imread(low_dir / f"{m}.png")
        for m in models
        if (low_dir / f"{m}.png").is_file()
    }
    render_agreement_figure(
        high_input_rgb=plt.imread(high_in),
        high_overlays=high_overlays,
        low_input_rgb=plt.imread(low_in),
        low_overlays=low_overlays,
        ordered_models=list(high_overlays.keys()),
        output_pdf=output_pdf,
        output_png=output_png,
        method_name=method_name,
        title_style=title_style,
        dpi=dpi,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Reproduce across-architecture attribution agreement figures and ranking tables.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["gradcam", "mpm", "lime"],
        help="Attribution methods to process ('gradcam', 'mpm', 'lime', or 'all').",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of high/low agreement pairs to render (combines top K and bottom K into K plots).",
    )
    parser.add_argument(
        "--num-samples",
        "--max-samples",
        dest="max_samples",
        type=int,
        default=None,
        help="Optional limit on number of stimuli to evaluate (e.g. 10 for quick testing).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="List of model architectures to evaluate and plot.",
    )
    parser.add_argument(
        "--cmap", type=str, default="coolwarm", help="Diverging colormap name."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Overlay opacity [0.0 - 1.0]."
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLOTS_DIR,
        help="Directory to save generated figures and tables.",
    )
    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    methods_req = (
        ["gradcam", "mpm", "lime"]
        if "all" in [m.lower() for m in args.methods]
        else [m.lower() for m in args.methods]
    )
    methods = list(dict.fromkeys(methods_req))

    for method in methods:
        info("Processing Across-Architecture Agreement for %s", method.upper())
        try:
            run_agreement_pipeline(
                method=method,
                top_k=args.top_k,
                max_samples=args.max_samples,
                output_dir=args.output_dir,
                models=args.models,
                cmap_name=args.cmap,
                alpha=args.alpha,
                dpi=args.dpi,
            )
        except (FileNotFoundError, ValueError, OSError) as e:
            error("[%s] Pipeline failed: %s", method, e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
