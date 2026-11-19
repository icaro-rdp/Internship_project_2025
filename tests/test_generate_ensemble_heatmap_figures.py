"""Unit tests for ensemble vs. base-model MPM heatmap figure generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.python.generate_ensemble_heatmap_figures import (
    load_ensemble_prototypes,
    run_ensemble_figure_pipeline,
    sample_stratified_figure_rows,
)
from scripts.python.reproduce_agreement_figures import MODELS


def _create_synthetic_dataset(tmp_path: Path, num_samples: int = 15) -> dict[str, Path]:
    """Create synthetic Experiment 2 & 3 MPM prototypes, manifest, and annotations."""
    base_maps_root = tmp_path / "exp2_maps"
    mpm_base_dir = base_maps_root / "Multiscale_Pixel_Masking"
    mpm_base_dir.mkdir(parents=True)

    ens_dir = tmp_path / "exp3_maps"
    ens_dir.mkdir(parents=True)

    rng = np.random.default_rng(123)
    for model in MODELS:
        v1 = rng.uniform(-1.0, 1.0, size=(num_samples, 32, 32)).astype(np.float32)
        v2 = rng.uniform(-1.0, 1.0, size=(num_samples, 32, 32)).astype(np.float32)
        np.save(mpm_base_dir / f"{model}_exp1a_variant1_maps.npy", v1)
        np.save(mpm_base_dir / f"{model}_exp1a_variant2_maps.npy", v2)

    for mode in ("bagging", "stacking"):
        arr = rng.uniform(-1.0, 1.0, size=(num_samples, 32, 32)).astype(np.float32)
        np.save(ens_dir / f"{mode}_exp3_prototype_maps.npy", arr)

    manifest = [
        {
            "test_idx": i,
            "global_csv_idx": i,
            "image_path": f"./Dataset/AIGCIQA2023/Image/allimg/{i}.png",
            "image_name": f"{i}.png",
            "authenticity_score": float(20.0 + i * 3.0),
        }
        for i in range(num_samples)
    ]
    manifest_path = base_maps_root / "test_samples_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    annotations_path = tmp_path / "annotations.csv"
    pd.DataFrame(
        {
            "Quality": [50.0 + i for i in range(num_samples)],
            "Authenticity": [20.0 + i * 3.0 for i in range(num_samples)],
            "Correspondence": [60.0 + i * 0.5 for i in range(num_samples)],
            "image_path": [
                f"./Dataset/AIGCIQA2023/Image/allimg/{i}.png"
                for i in range(num_samples)
            ],
            "model": ["controlNet" for _ in range(num_samples)],
            "Challenge": ["Basic" for _ in range(num_samples)],
            "Category": ["Objects" for _ in range(num_samples)],
            "Prompt": [f"Synthetic prompt {i}" for i in range(num_samples)],
        }
    ).to_csv(annotations_path, index=False)

    return {
        "base_maps_root": base_maps_root,
        "ens_dir": ens_dir,
        "manifest_path": manifest_path,
        "annotations_path": annotations_path,
    }


def test_sample_stratified_figure_rows_unique_and_ordered() -> None:
    """Verify unique non-overlapping samples strictly stratified across tertiles."""
    manifest = [{"test_idx": i, "authenticity_score": float(i)} for i in range(30)]
    groups_a = sample_stratified_figure_rows(manifest, num_figures=10, seed=42)
    groups_b = sample_stratified_figure_rows(manifest, num_figures=10, seed=42)

    assert groups_a == groups_b
    assert len(groups_a) == 10

    seen_indices: set[int] = set()
    for group in groups_a:
        assert [tier for _, tier in group] == ["high", "medium", "low"]
        high_idx, med_idx, low_idx = (idx for idx, _ in group)
        assert high_idx >= 20
        assert 10 <= med_idx < 20
        assert low_idx < 10
        seen_indices.update([high_idx, med_idx, low_idx])

    assert len(seen_indices) == 30


def test_sample_stratified_figure_rows_validation() -> None:
    """Verify invalid num_figures raises ValueError."""
    manifest = [{"test_idx": i, "authenticity_score": float(i)} for i in range(9)]
    with pytest.raises(ValueError, match="positive"):
        sample_stratified_figure_rows(manifest, num_figures=0)
    with pytest.raises(ValueError, match="tertile"):
        sample_stratified_figure_rows(manifest, num_figures=4)


def test_load_ensemble_prototypes_missing_raises(tmp_path: Path) -> None:
    """Verify FileNotFoundError when ensemble prototype files are missing."""
    with pytest.raises(FileNotFoundError, match="bagging_exp3_prototype_maps.npy"):
        load_ensemble_prototypes(ensemble_maps_dir=tmp_path)


def test_run_ensemble_figure_pipeline_stratified_and_custom(tmp_path: Path) -> None:
    """Verify end-to-end figure rendering and JSON manifest in both modes."""
    paths = _create_synthetic_dataset(tmp_path, num_samples=15)
    out_dir = tmp_path / "outputs"

    payload, pngs = run_ensemble_figure_pipeline(
        num_figures=3,
        seed=7,
        output_dir=out_dir,
        base_maps_dir=paths["base_maps_root"],
        ensemble_maps_dir=paths["ens_dir"],
        manifest_path=paths["manifest_path"],
        annotations_path=paths["annotations_path"],
        dpi=100,
        allow_missing_images=True,
    )

    assert len(pngs) == 3
    for png_path in pngs:
        assert png_path.is_file()
        assert png_path.with_suffix(".pdf").is_file()

    json_path = out_dir / "ensemble_figures_manifest.json"
    assert json_path.is_file()
    disk_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert disk_payload == payload

    meta = payload["metadata"]
    assert meta["num_figures"] == 3
    assert meta["rows_per_figure"] == 3
    assert meta["sampling_strategy"] == "authenticity_tertiles_without_replacement"
    assert meta["seed"] == 7
    assert meta["overlay_style"]["dpi"] == 100
    assert len(meta["columns"]) == 10
    assert meta["columns"][:3] == [
        "Original",
        "Bagging Ensemble",
        "Stacking Ensemble",
    ]

    assert len(payload["figures"]) == 3
    seen_test_ids: set[int] = set()
    for fig_entry in payload["figures"]:
        assert len(fig_entry["sub_images"]) == 3
        for sub in fig_entry["sub_images"]:
            seen_test_ids.add(sub["test_idx"])
            assert sub["description"].startswith("Synthetic prompt ")
            assert sub["generator_model"] == "controlNet"
            assert sub["category"] == "Objects"
            assert sub["challenge"] == "Basic"
            assert set(sub["human_scores"].keys()) == {
                "authenticity",
                "quality",
                "correspondence",
            }
            assert set(sub["xai_agreement"].keys()) == {
                "mean_base_models_pearson_r",
                "bagging_vs_stacking_pearson_r",
            }
            assert -1.0 <= sub["xai_agreement"]["mean_base_models_pearson_r"] <= 1.0
            assert -1.0 <= sub["xai_agreement"]["bagging_vs_stacking_pearson_r"] <= 1.0
    assert len(seen_test_ids) == 9

    # Custom explicit indices mode
    custom_payload, custom_pngs = run_ensemble_figure_pipeline(
        indices=[1, 7, 13],
        output_dir=out_dir,
        base_maps_dir=paths["base_maps_root"],
        ensemble_maps_dir=paths["ens_dir"],
        manifest_path=paths["manifest_path"],
        annotations_path=paths["annotations_path"],
        dpi=100,
        allow_missing_images=True,
        custom_name="paper_pick",
    )

    assert len(custom_pngs) == 1
    assert (out_dir / "paper_pick.png").is_file()
    assert (out_dir / "paper_pick.pdf").is_file()
    assert (out_dir / "paper_pick_manifest.json").is_file()
    assert custom_payload["metadata"]["sampling_strategy"] == "explicit_indices"
    assert custom_payload["metadata"]["seed"] is None
    sub_imgs = custom_payload["figures"][0]["sub_images"]
    assert [s["test_idx"] for s in sub_imgs] == [1, 7, 13]
    assert [s["authenticity_tier"] for s in sub_imgs] == [
        "low",
        "medium",
        "high",
    ]
