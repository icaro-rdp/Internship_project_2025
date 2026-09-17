"""Unit and integration tests for Experiment 2 (Explainability Map Generation & Comparison)."""

import json
from pathlib import Path

import numpy as np
import pytest

from Image_Authenticity_prediction.main.Experiments import experiment_two
from Image_Authenticity_prediction.main.Experiments.experiment_two import (
    NpEncoder,
    get_weight_files,
    load_and_resize_map,
    run_comparisons,
    run_experiment_2,
    save_plots_for_result,
    setup_directories,
)


class TestExperimentTwoDirectoryAndWeightDiscovery:
    """Tests for directory hierarchy initialization and weight checkpoint discovery."""

    def test_setup_directories_creates_hierarchy(
        self,
        mock_experiment_dirs: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify setup_directories creates all required XAI map and plot output folders."""
        exp2_base = mock_experiment_dirs["root"] / "FreshExp2"
        custom_dirs = {
            "output": exp2_base,
            "weights": exp2_base / "Weights",
            "maps": exp2_base / "XAI_Maps",
            "gradcam": exp2_base / "XAI_Maps" / "GradCAM",
            "mpm": exp2_base / "XAI_Maps" / "Multiscale_Pixel_Masking",
            "plots": exp2_base / "Plots",
        }
        monkeypatch.setattr(experiment_two, "DIRS", custom_dirs)

        setup_directories()

        for folder in custom_dirs.values():
            assert folder.exists(), f"Expected directory {folder} to exist"

    def test_get_weight_files_returns_empty_when_dir_missing(
        self,
        mock_experiment_dirs: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify get_weight_files returns an empty dictionary when weights path does not exist."""
        missing_dir = mock_experiment_dirs["root"] / "NonexistentWeights"
        monkeypatch.setitem(experiment_two.DIRS, "weights", missing_dir)

        weights = get_weight_files(models_filter=None, variants_filter="all")
        assert weights == {}

    def test_get_weight_files_filtering_by_model(
        self,
        mock_experiment_dirs: dict[str, Path],
        mock_exp1_weight_files: list[Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify get_weight_files filters weight checkpoints by specific model architecture."""
        monkeypatch.setitem(
            experiment_two.DIRS, "weights", mock_experiment_dirs["exp1_weights"]
        )

        vgg_only = get_weight_files(models_filter=["vgg16"], variants_filter="all")
        assert "vgg16" in vgg_only
        assert "resnet152" not in vgg_only
        assert len(vgg_only["vgg16"]) > 0

    def test_get_weight_files_filtering_by_variant_type(
        self,
        mock_experiment_dirs: dict[str, Path],
        mock_exp1_weight_files: list[Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify get_weight_files filters checkpoints by greedy, negative, or base variants."""
        monkeypatch.setitem(
            experiment_two.DIRS, "weights", mock_experiment_dirs["exp1_weights"]
        )

        greedy_weights = get_weight_files(models_filter=None, variants_filter="greedy")
        for model_weights in greedy_weights.values():
            for wpath in model_weights:
                assert "greedy" in str(wpath)

        base_weights = get_weight_files(models_filter=None, variants_filter="base")
        for model_weights in base_weights.values():
            for wpath in model_weights:
                assert "greedy" not in str(wpath) and "negative" not in str(wpath)


class TestExperimentTwoMapProcessing:
    """Tests for heatmap deserialization, resizing, and serialization encoders."""

    def test_np_encoder_serializes_numpy_structures(self) -> None:
        """Verify NpEncoder correctly encodes numpy integers, floats, and ndarrays into JSON."""
        data = {
            "int_val": np.int64(42),
            "float_val": np.float32(3.1415),
            "array_val": np.array([1.0, 2.0, 3.0]),
        }
        encoded_json = json.dumps(data, cls=NpEncoder)
        decoded = json.loads(encoded_json)

        assert decoded["int_val"] == 42
        assert abs(decoded["float_val"] - 3.1415) < 1e-4
        assert decoded["array_val"] == [1.0, 2.0, 3.0]

    def test_load_and_resize_map_success(
        self,
        mock_experiment_dirs: dict[str, Path],
    ) -> None:
        """Verify load_and_resize_map loads array from disk and resizes to target spatial resolution."""
        arr = np.random.rand(4, 64, 64).astype(np.float32)
        target_path = mock_experiment_dirs["exp2_gradcam"] / "sample_map.npy"
        np.save(target_path, arr)

        resized = load_and_resize_map(target_path, target_res=(224, 224))
        assert resized is not None
        assert resized.shape == (4, 224, 224)

    def test_load_and_resize_map_handles_invalid_file(self) -> None:
        """Verify load_and_resize_map handles non-existent or corrupted files gracefully."""
        bad_path = Path("/invalid/path/that/does/not/exist.npy")
        resized = load_and_resize_map(bad_path, target_res=(224, 224))
        assert resized is None


class TestExperimentTwoComparisons:
    """Tests for quantitative saliency map comparisons and visualization saving."""

    def test_run_comparisons_between_model_architectures(
        self,
        mock_experiment_dirs: dict[str, Path],
        mock_saliency_arrays: dict[str, list[Path]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify run_comparisons computes prototype comparisons across distinct architectures."""
        monkeypatch.setitem(experiment_two.DIRS, "output", mock_experiment_dirs["root"])
        monkeypatch.setitem(
            experiment_two.DIRS, "gradcam", mock_experiment_dirs["exp2_gradcam"]
        )
        monkeypatch.setitem(
            experiment_two.DIRS, "mpm", mock_experiment_dirs["exp2_mpm"]
        )
        monkeypatch.setitem(
            experiment_two.DIRS, "plots", mock_experiment_dirs["exp2_plots"]
        )

        results = run_comparisons(
            methods=["gradcam"],
            kinds=("between_model_architectures",),
            metrics=("correlation",),
            target_res=(224, 224),
            models_filter=["vgg16", "resnet152"],
            save_json=True,
        )

        assert "gradcam_between_model_architectures" in results
        comp = results["gradcam_between_model_architectures"]
        assert "models" in comp
        assert "summary" in comp
        assert "correlation" in comp["summary"]

        json_out = mock_experiment_dirs["root"] / "experiment_2b_comparison.json"
        assert json_out.exists(), "Comparison results JSON file should be saved"

    def test_run_comparisons_within_model_variants(
        self,
        mock_experiment_dirs: dict[str, Path],
        mock_saliency_arrays: dict[str, list[Path]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify run_comparisons computes intra-model variant comparisons for each architecture."""
        monkeypatch.setitem(experiment_two.DIRS, "output", mock_experiment_dirs["root"])
        monkeypatch.setitem(
            experiment_two.DIRS, "gradcam", mock_experiment_dirs["exp2_gradcam"]
        )
        monkeypatch.setitem(
            experiment_two.DIRS, "mpm", mock_experiment_dirs["exp2_mpm"]
        )
        monkeypatch.setitem(
            experiment_two.DIRS, "plots", mock_experiment_dirs["exp2_plots"]
        )

        results = run_comparisons(
            methods=["gradcam"],
            kinds=("within_model_variants",),
            metrics=("correlation",),
            target_res=(224, 224),
            models_filter=["vgg16"],
            save_json=False,
        )

        assert "gradcam_within_model_variants" in results
        assert "vgg16" in results["gradcam_within_model_variants"]

    def test_save_plots_for_result_skips_absent_metrics(
        self,
        mock_experiment_dirs: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify save_plots_for_result skips metrics not found in summary without raising."""
        monkeypatch.setitem(
            experiment_two.DIRS, "plots", mock_experiment_dirs["exp2_plots"]
        )

        comp_res = {
            "per_image": {},
            "summary": {
                "correlation": {
                    "pair1": {
                        "mean": 0.8,
                        "std": 0.1,
                        "min": 0.6,
                        "max": 0.9,
                        "median": 0.82,
                    }
                }
            },
        }
        save_plots_for_result(
            comp_res=comp_res,
            labels=["model_a", "model_b"],
            method="gradcam",
            scope_name="test_scope",
            metrics=("nonexistent_metric",),
        )


class TestExperimentTwoExecution:
    """Tests for complete Experiment 2 entry point orchestration."""

    def test_run_experiment_2_comparison_only_mode(
        self,
        mock_experiment_dirs: dict[str, Path],
        mock_saliency_arrays: dict[str, list[Path]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify run_experiment_2 executes comparison mode without attempting weight generation."""
        monkeypatch.setitem(experiment_two.DIRS, "output", mock_experiment_dirs["root"])
        monkeypatch.setitem(
            experiment_two.DIRS, "gradcam", mock_experiment_dirs["exp2_gradcam"]
        )
        monkeypatch.setitem(
            experiment_two.DIRS, "mpm", mock_experiment_dirs["exp2_mpm"]
        )
        monkeypatch.setitem(
            experiment_two.DIRS, "plots", mock_experiment_dirs["exp2_plots"]
        )

        run_experiment_2(
            models=["vgg16", "resnet152"],
            comparison_only=True,
            xai_methods="gradcam",
            comparison_kinds=("between_model_architectures",),
            comparison_metrics=("correlation",),
            save_comparison_json=True,
        )

        json_out = mock_experiment_dirs["root"] / "experiment_2b_comparison.json"
        assert json_out.exists()

    def test_run_experiment_2_empty_weights_completes_safely(
        self,
        mock_experiment_dirs: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify run_experiment_2 completes safely when no weights are found for generation."""
        empty_dir = mock_experiment_dirs["root"] / "EmptyWeightsExp2"
        empty_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setitem(experiment_two.DIRS, "weights", empty_dir)
        monkeypatch.setitem(experiment_two.DIRS, "output", mock_experiment_dirs["root"])

        run_experiment_2(
            models=["vgg16"],
            comparison_only=False,
            run_comparison=False,
            save_maps=False,
        )
