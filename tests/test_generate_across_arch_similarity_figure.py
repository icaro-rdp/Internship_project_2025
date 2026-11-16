"""Tests for across-architecture explanation similarity figure generator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.python.generate_across_arch_similarity_figure import (
    DEFAULT_MODELS,
    generate_all_metric_panels,
    load_correlation_matrices,
    load_metric_matrices,
    render_across_arch_similarity_figure,
)


def _create_mock_json(tmp_path: Path) -> Path:
    mock_data = {}
    for m in ["gradcam", "mpm", "lime"]:
        key = f"{m}_between_model_architectures"
        mock_data[key] = {
            "models": DEFAULT_MODELS,
            "summary": {
                "correlation": {
                    "0_vs_1": {"mean": 0.55, "std": 0.20},
                    "1_vs_2": {"mean": -0.15, "std": 0.10},
                    "4_vs_5": {"mean": 0.60, "std": 0.05},
                },
                "top_percent_iou_15": {
                    "0_vs_1": {"mean": 0.28, "std": 0.05},
                    "1_vs_2": {"mean": 0.12, "std": 0.03},
                },
            },
        }

    json_file = tmp_path / "mock_comparison.json"
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(mock_data, f)
    return json_file


def test_load_correlation_matrices_and_render(tmp_path: Path) -> None:
    """Test loading comparison JSON and rendering PDF/PNG panel figures."""
    json_file = _create_mock_json(tmp_path)

    matrices, std_matrices = load_correlation_matrices(json_file, DEFAULT_MODELS)
    assert "gradcam" in matrices
    assert "mpm" in matrices
    assert "lime" in matrices
    assert matrices["gradcam"].shape == (7, 7)
    assert np.isclose(matrices["gradcam"][0, 1], 0.55)
    assert np.isclose(matrices["gradcam"][1, 0], 0.55)

    pdf_path = tmp_path / "across_arch_explanation_similarity.pdf"
    png_path = tmp_path / "across_arch_explanation_similarity.png"

    fig = render_across_arch_similarity_figure(
        matrices=matrices,
        std_matrices=std_matrices,
        models=DEFAULT_MODELS,
        output_pdf=pdf_path,
        output_png=png_path,
        include_std=False,
    )
    assert fig is not None
    assert pdf_path.exists()
    assert png_path.exists()
    assert pdf_path.stat().st_size > 0
    assert png_path.stat().st_size > 0


def test_render_with_std(tmp_path: Path) -> None:
    """Test rendering figure with standard deviation text."""
    n = len(DEFAULT_MODELS)
    mat = np.zeros((n, n))
    mat[0, 1] = 0.42
    mat[1, 0] = 0.42
    smat = np.zeros((n, n))
    smat[0, 1] = 0.12
    smat[1, 0] = 0.12

    matrices = {"gradcam": mat, "mpm": mat, "lime": mat}
    std_matrices = {"gradcam": smat, "mpm": smat, "lime": smat}

    pdf_path = tmp_path / "with_std.pdf"
    fig = render_across_arch_similarity_figure(
        matrices=matrices,
        std_matrices=std_matrices,
        models=DEFAULT_MODELS,
        output_pdf=pdf_path,
        include_std=True,
    )
    assert fig is not None
    assert pdf_path.exists()


def test_load_metric_matrices_and_generate_all(tmp_path: Path) -> None:
    """Test loading multiple metrics and generating all panels."""
    json_file = _create_mock_json(tmp_path)

    iou_mats, _ = load_metric_matrices(json_file, metric="top_percent_iou_15")
    assert "gradcam" in iou_mats
    assert np.isclose(iou_mats["gradcam"][0, 1], 0.28)

    out_dir = tmp_path / "panels"
    generated = generate_all_metric_panels(
        json_path=json_file,
        output_dir=out_dir,
        metrics=["correlation", "top_percent_iou_15"],
    )
    assert len(generated) >= 2
    assert (out_dir / "across_arch_explanation_similarity.pdf").exists()
    assert (out_dir / "across_arch_explanation_iou_15.pdf").exists()
    assert (out_dir / "across_arch_explanation_iou.pdf").exists()
