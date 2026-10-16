"""Unit tests for scripts/visualize_saliency_overlays.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from scripts.visualize_saliency_overlays import (
    discover_heatmap_files,
    generate_comparison_grid,
    render_overlay_rgb,
    save_individual_overlays,
    select_sample_indices,
)


class DummyDataset:
    """Mock PyTorch dataset returning synthetic images and target scores."""

    def __init__(self, size: int = 10) -> None:
        self.size = size
        self.images = [torch.randn(3, 32, 32) for _ in range(size)]
        self.scores = [float(i) / float(size) for i in range(size)]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        return self.images[idx], self.scores[idx]


@pytest.fixture
def mock_maps_dir(tmp_path: Path) -> Path:
    """Create a temporary directory structure mimicking XAI_Maps."""
    gc_dir = tmp_path / "GradCAM"
    mpm_dir = tmp_path / "Multiscale_Pixel_Masking"
    gc_dir.mkdir(parents=True)
    mpm_dir.mkdir(parents=True)

    dummy_maps = np.zeros((5, 14, 14), dtype=np.float32)

    np.save(gc_dir / "clip_vit_b16_exp1a_variant1_best_maps.npy", dummy_maps)
    np.save(gc_dir / "clip_vit_b16_exp1b_variant1_greedy_pruned_maps.npy", dummy_maps)
    np.save(gc_dir / "resnet152_exp1a_variant1_best_maps.npy", dummy_maps)
    np.save(mpm_dir / "clip_vit_b16_exp1a_variant1_best_maps.npy", dummy_maps)

    return tmp_path


def test_discover_heatmap_files_best_variant(mock_maps_dir: Path) -> None:
    """Verify discovery filters to best/primary variant by default."""
    discovered = discover_heatmap_files(
        maps_dir=mock_maps_dir,
        methods=("gradcam", "mpm"),
        variants_filter="best",
    )
    assert "gradcam" in discovered
    assert "clip_vit_b16" in discovered["gradcam"]
    # Only the primary variant is kept under 'best'
    assert len(discovered["gradcam"]["clip_vit_b16"]) == 1
    assert "exp1a_variant1_best" in discovered["gradcam"]["clip_vit_b16"]


def test_discover_heatmap_files_filter_models(mock_maps_dir: Path) -> None:
    """Verify filtering by specific model names."""
    discovered = discover_heatmap_files(
        maps_dir=mock_maps_dir,
        methods=("gradcam",),
        models_filter=["resnet152"],
        variants_filter="all",
    )
    assert "resnet152" in discovered["gradcam"]
    assert "clip_vit_b16" not in discovered["gradcam"]


def test_select_sample_indices_quantiles() -> None:
    """Verify quantile selection returns distinct indices across the score range."""
    dataset = DummyDataset(size=20)
    indices = select_sample_indices(dataset, num_samples=3)
    assert len(indices) == 3
    assert len(set(indices)) == 3
    assert all(0 <= idx < 20 for idx in indices)


def test_select_sample_indices_explicit() -> None:
    """Verify explicit comma-separated indices are parsed and bounded."""
    dataset = DummyDataset(size=10)
    indices = select_sample_indices(dataset, indices_arg="0, 4, 9, 999")
    assert indices == [0, 4, 9]


def test_render_overlay_rgb_dimensions_and_type() -> None:
    """Verify render_overlay_rgb outputs uint8 RGB with matching shape."""
    img_rgb = np.zeros((64, 64, 3), dtype=np.float32)
    saliency_2d = np.random.uniform(-1.0, 1.0, (14, 14)).astype(np.float32)

    overlay = render_overlay_rgb(img_rgb, saliency_2d, cmap_name="bwr", alpha=0.5)
    assert overlay.shape == (64, 64, 3)
    assert overlay.dtype == np.uint8
    assert overlay.min() >= 0
    assert overlay.max() <= 255


def test_generate_comparison_grid_clean_strip_and_annotated(tmp_path: Path) -> None:
    """Verify clean borderless strip and annotated figure are written to disk."""
    orig_rgb = np.ones((32, 32, 3), dtype=np.float32) * 0.5
    models_data: list[dict[str, Any]] = [
        {
            "model_name": "clip_vit_b16",
            "method": "gradcam",
            "variant": "exp1a_variant1",
            "saliency": np.zeros((32, 32), dtype=np.float32),
        }
    ]

    # 1. Clean borderless strip
    clean_strip_path = tmp_path / "grids" / "strip_sample_0001.png"
    generate_comparison_grid(
        sample_idx=1,
        orig_rgb=orig_rgb,
        auth_score=0.75,
        models_data=models_data,
        output_path=clean_strip_path,
        annotate=False,
    )
    assert clean_strip_path.exists()
    assert clean_strip_path.stat().st_size > 0

    # 2. Labeled annotated figure
    labeled_grid_path = tmp_path / "grids" / "grid_sample_0001.png"
    generate_comparison_grid(
        sample_idx=1,
        orig_rgb=orig_rgb,
        auth_score=0.75,
        models_data=models_data,
        output_path=labeled_grid_path,
        annotate=True,
        dpi=100,
    )
    assert labeled_grid_path.exists()
    assert labeled_grid_path.stat().st_size > 0

    # 3. Individual plain images
    sample_dir = tmp_path / "individual" / "sample_0001"
    save_individual_overlays(
        sample_idx=1,
        orig_rgb=orig_rgb,
        models_data=models_data,
        sample_dir=sample_dir,
    )
    assert (sample_dir / "original.png").exists()
    assert (sample_dir / "clip_vit_b16_exp1a_variant1_gradcam_overlay.png").exists()
    assert (sample_dir / "clip_vit_b16_exp1a_variant1_gradcam_heatmap.png").exists()
    assert (sample_dir / "clip_vit_b16_exp1a_variant1_gradcam_heatmap.png").exists()
