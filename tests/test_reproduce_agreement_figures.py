"""Unit tests for across-architecture attribution agreement reproduction pipeline.

Tests mathematical consistency, ranking table generation, stimulus identification,
multi-panel figure layout, LIME support, and cluster/local file resolution.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scripts.python.reproduce_agreement_figures import (
    DEFAULT_MODELS,
    MODEL_DISPLAY_NAMES,
    build_agreement_ranking_table,
    compute_across_architecture_agreement,
    discover_and_load_attribution_maps,
    find_stimulus_index,
    normalize_model_name,
    render_agreement_figure,
    render_from_directories,
    run_agreement_pipeline_for_method,
)


@pytest.fixture
def mock_xai_maps_dir(tmp_path: Path) -> Path:
    """Create a temporary directory structure mimicking cluster XAI_Maps with 7 models."""
    gc_dir = tmp_path / "GradCAM"
    mpm_dir = tmp_path / "Multiscale_Pixel_Masking"
    lime_dir = tmp_path / "LIME"

    gc_dir.mkdir(parents=True)
    mpm_dir.mkdir(parents=True)
    lime_dir.mkdir(parents=True)

    num_samples = 5
    # Create distinct synthetic maps for each model so correlations are well-defined
    for i, model in enumerate(DEFAULT_MODELS):
        # Variant 1
        rng1 = np.random.default_rng(seed=100 + i)
        map_v1 = rng1.standard_normal((num_samples, 14, 14)).astype(np.float32)

        # Variant 2
        rng2 = np.random.default_rng(seed=200 + i)
        map_v2 = rng2.standard_normal((num_samples, 14, 14)).astype(np.float32)

        for d in (gc_dir, mpm_dir, lime_dir):
            np.save(d / f"{model}_exp1a_variant1_best_maps.npy", map_v1)
            np.save(d / f"{model}_exp1a_variant2_maps.npy", map_v2)

    # Manifest
    manifest = [
        {
            "test_idx": k,
            "global_csv_idx": 100 + k,
            "image_path": f"./Dataset/AIGCIQA2023/Image/allimg/{k}.png",
            "image_name": f"{k}.png",
            "authenticity_score": 50.0 + k * 5.0,
        }
        for k in range(num_samples)
    ]
    with open(tmp_path / "test_samples_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return tmp_path


@pytest.fixture
def mock_annotations_df() -> pd.DataFrame:
    """Create mock annotations DataFrame with prompts."""
    return pd.DataFrame(
        {
            "Quality": [60.0, 55.0, 50.0, 65.0, 70.0],
            "Authenticity": [50.0, 55.0, 60.0, 65.0, 70.0],
            "Correspondence": [60.0, 60.0, 60.0, 60.0, 60.0],
            "image_path": [
                f"./Dataset/AIGCIQA2023/Image/allimg/{i}.png" for i in range(5)
            ],
            "model": ["test_gen"] * 5,
            "Challenge": ["Basic"] * 5,
            "Category": ["Artifacts"] * 5,
            "Prompt": [
                "view of a clock tower from above",
                "a green pepper",
                "a boat in the canals of venice",
                "sailboat on a calm lake",
                "lemon plate on a wooden table",
            ],
        },
        index=[100, 101, 102, 103, 104],
    )


def test_default_models_includes_clip() -> None:
    """Verify CLIP (clip_vit_b16) is a mandatory core model in default models."""
    assert "clip_vit_b16" in DEFAULT_MODELS
    assert len(DEFAULT_MODELS) == 7
    assert normalize_model_name("CLIP") == "clip_vit_b16"
    assert normalize_model_name("clipvit") == "clip_vit_b16"
    assert normalize_model_name("ResNet152") == "resnet152"
    assert "clip_vit_b16" in MODEL_DISPLAY_NAMES


def test_compute_across_architecture_agreement_math() -> None:
    """Verify pairwise across-architecture Pearson correlation and mean agreement math."""
    num_samples = 3
    # Construct synthetic prototype heatmaps where correlations can be verified
    # Model A: constant gradient
    # Model B: identical to Model A -> r(A, B) = 1.0
    # Model C: inverted gradient -> r(A, C) = -1.0, r(B, C) = -1.0
    base_pattern = np.linspace(-1, 1, 16).reshape(4, 4).astype(np.float32)

    proto_a = np.stack([base_pattern * (k + 1) for k in range(num_samples)])
    proto_b = np.stack([base_pattern * (k + 1) for k in range(num_samples)])
    proto_c = np.stack([-base_pattern * (k + 1) for k in range(num_samples)])

    prototypes = {
        "model_a": proto_a,
        "model_b": proto_b,
        "model_c": proto_c,
    }

    mean_agg, std_agg, pair_corrs, active = compute_across_architecture_agreement(
        prototypes, ordered_models=["model_a", "model_b", "model_c"]
    )

    assert len(active) == 3
    assert len(mean_agg) == num_samples
    assert len(std_agg) == num_samples

    # Pairs are (A, B) with r=1.0, (A, C) with r=-1.0, (B, C) with r=-1.0
    # Mean agreement = (1.0 - 1.0 - 1.0) / 3 = -1/3 ≈ -0.3333
    for k in range(num_samples):
        np.testing.assert_allclose(
            pair_corrs[("model_a", "model_b")][k], 1.0, atol=1e-5
        )
        np.testing.assert_allclose(
            pair_corrs[("model_a", "model_c")][k], -1.0, atol=1e-5
        )
        np.testing.assert_allclose(
            pair_corrs[("model_b", "model_c")][k], -1.0, atol=1e-5
        )
        np.testing.assert_allclose(mean_agg[k], -1.0 / 3.0, atol=1e-5)


def test_build_agreement_ranking_table(mock_annotations_df: pd.DataFrame) -> None:
    """Verify ranking table ordering and required schema columns."""
    mean_agreement = np.array([0.15, 0.85, -0.20, 0.40, 0.95])
    std_agreement = np.array([0.05, 0.02, 0.08, 0.04, 0.01])

    manifest = [
        {
            "test_idx": k,
            "global_csv_idx": 100 + k,
            "image_name": f"{k}.png",
            "authenticity_score": 50.0 + k,
        }
        for k in range(5)
    ]

    df = build_agreement_ranking_table(
        mean_agreement=mean_agreement,
        std_agreement=std_agreement,
        manifest=manifest,
        annotations_df=mock_annotations_df,
    )

    expected_cols = [
        "rank",
        "test_idx",
        "global_csv_idx",
        "image_name",
        "prompt",
        "authenticity_score",
        "mean_agreement",
        "std_agreement",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 5

    # Highest agreement should be at rank 1 (sample 4 with agreement 0.95)
    assert df.iloc[0]["rank"] == 1
    assert df.iloc[0]["test_idx"] == 4
    assert df.iloc[0]["mean_agreement"] == pytest.approx(0.95)
    assert df.iloc[0]["prompt"] == "lemon plate on a wooden table"

    # Lowest agreement should be at rank 5 (sample 2 with agreement -0.20)
    assert df.iloc[-1]["rank"] == 5
    assert df.iloc[-1]["test_idx"] == 2
    assert df.iloc[-1]["mean_agreement"] == pytest.approx(-0.20)


def test_find_stimulus_index_resolution(mock_annotations_df: pd.DataFrame) -> None:
    """Verify identifier resolution via integer, image filename, stem, and prompt keyword."""
    manifest = [
        {
            "test_idx": 0,
            "global_csv_idx": 100,
            "image_name": "1554.png",
            "image_path": "path/1554.png",
        },
        {
            "test_idx": 1,
            "global_csv_idx": 101,
            "image_name": "332.png",
            "image_path": "path/332.png",
        },
        {
            "test_idx": 2,
            "global_csv_idx": 102,
            "image_name": "20.png",
            "image_path": "path/20.png",
        },
    ]
    ranking_df = pd.DataFrame(
        [
            {
                "test_idx": 0,
                "image_name": "1554.png",
                "prompt": "view of a clock tower from above",
            },
            {"test_idx": 1, "image_name": "332.png", "prompt": "a green pepper"},
            {"test_idx": 2, "image_name": "20.png", "prompt": "sailboat on water"},
        ]
    )

    # 1. By integer
    assert find_stimulus_index("1", manifest, ranking_df) == 1
    # 2. By image filename
    assert find_stimulus_index("1554.png", manifest, ranking_df) == 0
    # 3. By stem
    assert find_stimulus_index("332", manifest, ranking_df) == 1
    # 4. By prompt keyword
    assert find_stimulus_index("clock tower", manifest, ranking_df) == 0
    assert find_stimulus_index("green pepper", manifest, ranking_df) == 1
    assert find_stimulus_index("sailboat", manifest, ranking_df) == 2


def test_discover_and_load_attribution_maps_prototype(mock_xai_maps_dir: Path) -> None:
    """Verify map discovery and variant averaging into architectural prototypes."""
    prototypes = discover_and_load_attribution_maps(
        maps_dir=mock_xai_maps_dir,
        method="gradcam",
        models=DEFAULT_MODELS,
        variants_strategy="prototype",
        target_res=(224, 224),
    )

    assert len(prototypes) == 7
    assert "clip_vit_b16" in prototypes
    for m in DEFAULT_MODELS:
        assert prototypes[m].shape == (5, 224, 224)


def test_render_agreement_figure_layout(tmp_path: Path) -> None:
    """Verify multi-panel agreement figure generates valid PDF and PNG artifacts."""
    high_img = np.ones((224, 224, 3), dtype=np.float32)
    low_img = np.zeros((224, 224, 3), dtype=np.float32)

    models = ["barlowtwins", "resnet152", "clip_vit_b16"]
    high_overlays = {m: np.full((224, 224, 3), 200, dtype=np.uint8) for m in models}
    low_overlays = {m: np.full((224, 224, 3), 50, dtype=np.uint8) for m in models}

    pdf_path = tmp_path / "test_fig.pdf"
    png_path = tmp_path / "test_fig.png"

    render_agreement_figure(
        high_input_rgb=high_img,
        high_overlays=high_overlays,
        low_input_rgb=low_img,
        low_overlays=low_overlays,
        ordered_models=models,
        output_pdf=pdf_path,
        output_png=png_path,
        method_name="Grad-CAM",
        title_style="figure4",
        dpi=150,
    )

    assert pdf_path.is_file()
    assert png_path.is_file()
    assert pdf_path.stat().st_size > 1000
    assert png_path.stat().st_size > 1000


def test_render_from_directories(tmp_path: Path) -> None:
    """Verify direct directory rendering mode from pre-rendered image folders."""
    high_dir = tmp_path / "high"
    low_dir = tmp_path / "low"
    high_dir.mkdir()
    low_dir.mkdir()

    dummy_rgb = np.full((64, 64, 3), 180, dtype=np.uint8)

    plt.imsave(high_dir / "original.png", dummy_rgb)
    plt.imsave(low_dir / "original.png", dummy_rgb)

    models = ["barlowtwins", "resnet152", "clip_vit_b16"]
    for m in models:
        plt.imsave(high_dir / f"{m}.png", dummy_rgb)
        plt.imsave(low_dir / f"{m}.png", dummy_rgb)

    pdf_out = tmp_path / "dir_fig.pdf"
    png_out = tmp_path / "dir_fig.png"

    render_from_directories(
        high_dir=high_dir,
        low_dir=low_dir,
        models=models,
        output_pdf=pdf_out,
        output_png=png_out,
        method_name="Grad-CAM",
        title_style="figure_a5",
        dpi=150,
    )

    assert pdf_out.is_file()
    assert png_out.is_file()


def test_pipeline_end_to_end_all_methods(
    mock_xai_maps_dir: Path, tmp_path: Path
) -> None:
    """Verify end-to-end execution across Grad-CAM, MPM, and LIME with overrides."""
    out_dir = tmp_path / "output_plots"

    for method in ("gradcam", "mpm", "lime"):
        ranking_df, pdf_path, png_path = run_agreement_pipeline_for_method(
            method=method,
            maps_dir=mock_xai_maps_dir,
            output_dir=out_dir,
            models=DEFAULT_MODELS,
            variants_strategy="prototype",
            target_res=(32, 32),
            allow_missing_images=True,
            high_idx=1,
            low_idx=3,
            dpi=100,
        )

        assert isinstance(ranking_df, pd.DataFrame)
        assert len(ranking_df) == 5
        assert pdf_path.is_file()
        assert png_path.is_file()

        csv_file = out_dir / f"{method}_across_architecture_agreement_ranking.csv"
        assert csv_file.is_file()

        # Verify top-K and bottom-K overview panels and candidate pairs
        top_pdf = out_dir / f"{method}_top_2_high_agreement.pdf"
        bottom_pdf = out_dir / f"{method}_bottom_2_low_agreement.pdf"
        assert top_pdf.is_file()
        assert bottom_pdf.is_file()

        cand_dir = out_dir / f"candidates_{method}"
        cand_files = list(cand_dir.glob("candidate_pair_*.pdf"))
        assert len(cand_files) == 2


def test_missing_maps_dir_informative_error(tmp_path: Path) -> None:
    """Verify informative error message when maps directory is empty or missing."""
    empty_dir = tmp_path / "empty_outputs"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Attribution maps directory"):
        discover_and_load_attribution_maps(
            maps_dir=empty_dir,
            method="gradcam",
            models=DEFAULT_MODELS,
        )
