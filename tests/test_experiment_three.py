"""Unit and integration tests for Experiment 3 (Variant-Aligned Ensembles & XAI)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from Image_Authenticity_prediction.main.Experiments import experiment_three
from Image_Authenticity_prediction.main.Experiments.experiment_three import (
    CANONICAL_MODEL_ORDER,
    MODEL_REGISTRY,
    StackingMetaLearner,
    compute_prediction_reliability,
    create_global_test_indices,
    create_variant_split,
    experiment_3c_evaluate_ensemble,
    experiment_3d_plot_ensemble_heatmaps,
    fit_ridge_stacking_meta,
    get_labels,
    get_predictions,
    parse_variant_indices,
    reset_regression_head,
    run_experiment_3,
    setup_directories,
)
from Image_Authenticity_prediction.main.Utils.pruning import FeatureMapsPruner


class TestExperimentThreeDataSplitsAndIsolation:
    """Tests verifying data partition reproducibility and strict validation/test isolation."""

    def test_create_global_test_indices_fraction_and_seed(self) -> None:
        """Verify global test indices match specified fraction and are reproducible with seed."""
        dataset_size = 120
        test_fraction = 0.2
        expected_test_size = 24

        indices_run1 = create_global_test_indices(
            dataset_size=dataset_size, test_fraction=test_fraction, seed=42
        )
        indices_run2 = create_global_test_indices(
            dataset_size=dataset_size, test_fraction=test_fraction, seed=42
        )

        assert indices_run1 == indices_run2
        assert len(indices_run1) == expected_test_size
        assert len(set(indices_run1)) == expected_test_size
        assert all(0 <= idx < dataset_size for idx in indices_run1)

    def test_create_variant_split_disjoint_partitions(
        self, synthetic_dataset: Dataset
    ) -> None:
        """Verify train, validation, and test subsets share zero common image indices."""
        dataset_size = len(synthetic_dataset)
        global_test_indices = create_global_test_indices(
            dataset_size=dataset_size, test_fraction=0.2, seed=42
        )

        train_ds, val_ds, test_ds, _val_indices = create_variant_split(
            backbone_dataset=synthetic_dataset,
            global_test_indices=global_test_indices,
            variant_idx=1,
            val_fraction=0.125,
        )

        train_set = set(train_ds.indices)
        val_set = set(val_ds.indices)
        test_set = set(test_ds.indices)

        assert not (train_set & val_set)
        assert not (train_set & test_set)
        assert not (val_set & test_set)
        assert len(train_set) + len(val_set) + len(test_set) == dataset_size

    def test_create_variant_split_preserves_val_indices_for_pruning(
        self, synthetic_dataset: Dataset
    ) -> None:
        """Verify returned validation indices match validation subset for pruning phase isolation."""
        global_test_indices = create_global_test_indices(
            dataset_size=len(synthetic_dataset), test_fraction=0.2, seed=42
        )

        _, val_ds, _, val_indices = create_variant_split(
            backbone_dataset=synthetic_dataset,
            global_test_indices=global_test_indices,
            variant_idx=1,
        )

        assert val_indices == list(val_ds.indices)

    def test_variant_train_val_splits_differ_across_variants(
        self, synthetic_dataset: Dataset
    ) -> None:
        """Verify different variant indices produce diverse training and validation splits."""
        global_test_indices = create_global_test_indices(
            dataset_size=len(synthetic_dataset), test_fraction=0.2, seed=42
        )

        train_ds_1, val_ds_1, test_ds_1, _ = create_variant_split(
            backbone_dataset=synthetic_dataset,
            global_test_indices=global_test_indices,
            variant_idx=1,
        )

        train_ds_2, val_ds_2, test_ds_2, _ = create_variant_split(
            backbone_dataset=synthetic_dataset,
            global_test_indices=global_test_indices,
            variant_idx=2,
        )

        assert set(test_ds_1.indices) == set(test_ds_2.indices)
        assert set(train_ds_1.indices) != set(train_ds_2.indices)
        assert set(val_ds_1.indices) != set(val_ds_2.indices)


class TestExperimentThreeModelOperations:
    """Tests for regression head reinitialization and prediction extraction utilities."""

    def test_reset_regression_head_resets_linear_parameters(
        self, tiny_model: nn.Module
    ) -> None:
        """Verify reset_regression_head resets linear layer weights in the model regression head."""
        initial_params = [
            param.clone().detach() for param in tiny_model.regression_head.parameters()
        ]

        reset_regression_head(tiny_model)

        reset_params = [
            param.clone().detach() for param in tiny_model.regression_head.parameters()
        ]

        changed = any(
            not torch.equal(init, rst)
            for init, rst in zip(initial_params, reset_params, strict=True)
        )
        assert changed, "Expected at least one parameter in regression head to change"

    def test_get_predictions_and_get_labels(
        self, tiny_model: nn.Module, synthetic_dataset: Dataset
    ) -> None:
        """Verify get_predictions and get_labels correctly aggregate batch outputs across loader."""
        loader = DataLoader(synthetic_dataset, batch_size=10, shuffle=False)

        preds = get_predictions(tiny_model, loader, device="cpu")
        labels = get_labels(loader)

        assert preds.shape[0] == len(synthetic_dataset)
        assert labels.shape[0] == len(synthetic_dataset)
        assert isinstance(preds, torch.Tensor)
        assert isinstance(labels, torch.Tensor)


class TestExperimentThreeMetaLearner:
    """Tests for StackingMetaLearner architecture and gradient optimization."""

    def test_stacking_meta_learner_forward_pass(self) -> None:
        """Verify StackingMetaLearner produces expected output shape given base model predictions."""
        num_base_models = 6
        batch_size = 16
        meta_learner = StackingMetaLearner(num_base_models=num_base_models)

        x_input = torch.randn(batch_size, num_base_models)
        y_out = meta_learner(x_input)

        assert y_out.shape == (batch_size, 1)

    def test_stacking_meta_learner_gradient_step(self) -> None:
        """Verify StackingMetaLearner parameters receive gradients and update under MSE loss."""
        num_base_models = 4
        meta_learner = StackingMetaLearner(num_base_models=num_base_models)
        optimizer = torch.optim.Adam(meta_learner.parameters(), lr=1e-2)
        criterion = nn.MSELoss()

        x = torch.randn(8, num_base_models)
        target = torch.randn(8, 1)

        optimizer.zero_grad()
        loss = criterion(meta_learner(x), target)
        loss.backward()

        assert meta_learner.fc.weight.grad is not None
        assert meta_learner.fc.bias.grad is not None
        optimizer.step()

    def test_fit_ridge_stacking_meta_closed_form(self) -> None:
        """Verify fit_ridge_stacking_meta solves closed-form Ridge regression and populates StackingMetaLearner."""
        rng = np.random.default_rng(42)
        x_val = rng.normal(50.0, 10.0, size=(60, 7)).astype(np.float64)
        true_weights = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.1], dtype=np.float64)
        true_bias = 2.5
        y_val = x_val @ true_weights + true_bias

        meta_learner, weights, bias, val_mse, val_r2 = fit_ridge_stacking_meta(
            x_val, y_val, alpha=1e-4
        )

        assert weights.shape == (7,)
        np.testing.assert_allclose(weights, true_weights, atol=1e-3)
        assert abs(bias - true_bias) < 1e-2
        assert val_mse < 1e-4
        assert val_r2 > 0.999

        x_tensor = torch.tensor(x_val, dtype=torch.float32)
        with torch.no_grad():
            preds = meta_learner(x_tensor).squeeze(-1).numpy()
        np.testing.assert_allclose(preds, y_val, atol=1e-2)

    def test_model_registry_includes_all_seven_architectures(self) -> None:
        """Verify MODEL_REGISTRY and CANONICAL_MODEL_ORDER include all 7 architectures including clip_vit_b16."""
        expected_order = (
            "vgg16",
            "vgg19",
            "resnet152",
            "densenet161",
            "efficientnetb3",
            "barlowtwins",
            "clip_vit_b16",
        )
        assert CANONICAL_MODEL_ORDER == expected_order
        for model_name in expected_order:
            assert model_name in MODEL_REGISTRY
        assert MODEL_REGISTRY["clip_vit_b16"]["target_layer"] == "visual_projection"

    def test_feature_maps_pruner_supports_nn_linear(
        self, synthetic_dataset: Dataset, tmp_path: Path
    ) -> None:
        """Verify FeatureMapsPruner handles nn.Linear layers such as CLIP visual_projection."""

        class TinyLinearProjectionModel(nn.Module):
            """Tiny model exposing an nn.Linear projection layer before the regression head."""

            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(3, 8, kernel_size=3, padding=1),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                self.visual_projection = nn.Linear(8, 6, bias=False)
                self.regression_head = nn.Linear(6, 1)

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                feat = self.visual_projection(self.conv(x))
                return self.regression_head(feat), feat

        model = TinyLinearProjectionModel()
        loader = DataLoader(synthetic_dataset, batch_size=10, shuffle=False)
        criterion = nn.MSELoss()

        def dummy_eval(
            mdl: nn.Module, dl: DataLoader, crit: nn.Module, dev: str
        ) -> float:
            del dl, crit, dev
            return float(torch.sum(mdl.visual_projection.weight**2).item()) + 1.0

        pruner = FeatureMapsPruner(
            model=model,
            dataloader=loader,
            layer_name="visual_projection",
            criterion=criterion,
            eval_function=dummy_eval,
            device="cpu",
        )
        scores = pruner.compute_importance_scores()
        assert scores.shape == (6, 2)

        save_path = str(tmp_path / "pruned_linear.pth")
        history = pruner.negative_impact_pruning(
            model_save_path=save_path, threshold=0.0
        )
        assert "removed_features" in history
        assert "reduction_percentage" in history


class TestExperimentThreePipelineStages:
    """Tests for Experiment 3 execution phases (3A, 3B, 3C, 3D, and complete pipeline)."""

    def test_setup_directories_creates_all_folders(
        self,
        mock_experiment_dirs: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify setup_directories creates weights, rankings, results, and heatmap directories."""
        fresh_base = mock_experiment_dirs["root"] / "FreshExp3"
        dirs = {
            "exp1_weights": fresh_base / "Exp1_Weights",
            "exp2_mpm": fresh_base / "Exp2_MPM",
            "weights": fresh_base / "Weights",
            "rankings": fresh_base / "Ranking_arrays",
            "ranking_plots": fresh_base / "Ranking_Plots",
            "training_plots": fresh_base / "Training_Plots",
            "training_history": fresh_base / "Training_History",
            "results": fresh_base / "Results",
            "heatmaps": fresh_base / "XAI_Maps" / "Multiscale_Pixel_Masking",
            "plots": fresh_base / "Plots",
        }
        monkeypatch.setattr(experiment_three, "DIRS", dirs)

        setup_directories()

        for key, path in dirs.items():
            if key not in ("exp1_weights", "exp2_mpm"):
                assert path.exists()

    def test_experiment_3c_empty_weights_returns_empty_dict(
        self,
        mock_experiment_dirs: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify experiment_3c_evaluate_ensemble gracefully handles missing weight files."""
        empty_weights = mock_experiment_dirs["root"] / "EmptyWeightsExp3"
        empty_weights.mkdir(parents=True, exist_ok=True)
        monkeypatch.setitem(experiment_three.DIRS, "exp1_weights", empty_weights)
        monkeypatch.setitem(experiment_three.DIRS, "weights", empty_weights)

        results = experiment_3c_evaluate_ensemble(models_filter=["vgg16"], device="cpu")
        assert results == {}

    def test_experiment_3c_bagging_and_stacking_evaluation(
        self,
        mock_experiment_dirs: dict[str, Path],
        mock_exp1_weight_files: list[Path],
        synthetic_dataset: Dataset,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify Stage 3C evaluates variant-aligned Bagging and Stacking and saves artifacts."""
        exp1_weights_dir = mock_experiment_dirs["exp1_weights"]
        weights_dir = mock_experiment_dirs["exp3_weights"]
        results_dir = mock_experiment_dirs["exp3_results"]
        plots_dir = mock_experiment_dirs["exp3_plots"]

        monkeypatch.setitem(experiment_three.DIRS, "exp1_weights", exp1_weights_dir)
        monkeypatch.setitem(
            experiment_three.DIRS, "weights", experiment_three.OUTPUT_DIR / "Weights"
        )
        monkeypatch.setattr(
            experiment_three,
            "OUTPUT_DIR",
            mock_experiment_dirs["root"] / "Outputs" / "Experiment_3_ensemble",
        )
        monkeypatch.setitem(experiment_three.DIRS, "weights", weights_dir)
        monkeypatch.setitem(experiment_three.DIRS, "results", results_dir)
        monkeypatch.setitem(experiment_three.DIRS, "plots", plots_dir)
        monkeypatch.setattr(experiment_three, "imageNet_dataset", synthetic_dataset)
        monkeypatch.setattr(experiment_three, "denseNet_dataset", synthetic_dataset)
        monkeypatch.setattr(experiment_three, "clip_dataset", synthetic_dataset)

        def fake_get_predictions(
            model: nn.Module, dataloader: DataLoader, device: str
        ) -> torch.Tensor:
            n_items = len(dataloader.dataset)
            return torch.linspace(40.0, 80.0, steps=n_items).unsqueeze(1)

        def fake_get_labels(dataloader: DataLoader) -> torch.Tensor:
            n_items = len(dataloader.dataset)
            return torch.linspace(42.0, 78.0, steps=n_items)

        with (
            patch(
                "Image_Authenticity_prediction.main.Experiments.experiment_three.get_predictions",
                side_effect=fake_get_predictions,
            ),
            patch(
                "Image_Authenticity_prediction.main.Experiments.experiment_three.get_labels",
                side_effect=fake_get_labels,
            ),
            patch(
                "Image_Authenticity_prediction.main.Experiments.experiment_three.load_model_with_weights",
                return_value=MagicMock(),
            ),
        ):
            results = experiment_3c_evaluate_ensemble(
                models_filter=["vgg16", "resnet152"],
                global_test_indices={
                    "imagenet": [0, 1, 2, 3],
                    "densenet": [0, 1, 2, 3],
                    "clip": [0, 1, 2, 3],
                },
                device="cpu",
                ensemble_mode=["bagging", "stacking"],
                variants=[1, 2],
            )

        assert "bagging" in results
        assert "stacking" in results
        assert "pruned_ensemble_bagging" in results
        assert "pruned_ensemble_stacking" in results

        for strategy in ("bagging", "stacking"):
            strat_data = results[strategy]
            assert "variants" in strat_data
            assert len(strat_data["variants"]) == 2
            assert "mean" in strat_data
            assert "std" in strat_data
            assert "prediction_reliability" in strat_data
            assert "combined_correlation_ceiling" in strat_data
            for metric in ("mse", "rmse", "mae", "plcc", "srcc", "krcc"):
                assert metric in strat_data["mean"]

        assert (weights_dir / "stacking_exp3_variant1_meta.pth").exists()
        assert (weights_dir / "stacking_exp3_variant2_meta.pth").exists()
        assert (weights_dir / "stacking_coefficients.json").exists()
        assert (results_dir / "experiment_3c_ensemble_results.json").exists()
        assert (results_dir / "experiment_3c_comparison.json").exists()

    def test_experiment_3d_map_reuse_and_prototype_generation(
        self,
        mock_experiment_dirs: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify Stage 3D reuses Experiment 2 MPM maps, resizes to 224x224, and saves variant/prototype .npy files."""
        exp2_mpm_dir = mock_experiment_dirs["exp2_mpm"]
        mpm_out_dir = (
            mock_experiment_dirs["root"]
            / "Outputs"
            / "Experiment_3_ensemble"
            / "XAI_Maps"
            / "Multiscale_Pixel_Masking"
        )
        mpm_out_dir.mkdir(parents=True, exist_ok=True)
        weights_dir = mock_experiment_dirs["exp3_weights"]
        results_dir = mock_experiment_dirs["exp3_results"]
        plots_dir = mock_experiment_dirs["exp3_plots"]

        monkeypatch.setitem(experiment_three.DIRS, "exp2_mpm", exp2_mpm_dir)
        monkeypatch.setitem(experiment_three.DIRS, "mpm", mpm_out_dir)
        monkeypatch.setitem(experiment_three.DIRS, "weights", weights_dir)
        monkeypatch.setitem(experiment_three.DIRS, "results", results_dir)
        monkeypatch.setitem(experiment_three.DIRS, "plots", plots_dir)

        rng = np.random.default_rng(123)
        n_test = 6
        models = ["vgg16", "densenet161"]
        for variant_idx in (1, 2):
            # vgg16 at 224x224, densenet161 at 300x300 to test bilinear resizing
            vgg_maps = rng.uniform(-1.0, 1.0, size=(n_test, 224, 224)).astype(
                np.float32
            )
            dense_maps = rng.uniform(-1.0, 1.0, size=(n_test, 300, 300)).astype(
                np.float32
            )
            np.save(
                exp2_mpm_dir / f"vgg16_exp1a_variant{variant_idx}_maps.npy", vgg_maps
            )
            np.save(
                exp2_mpm_dir / f"densenet161_exp1a_variant{variant_idx}_maps.npy",
                dense_maps,
            )

        coeff_payload = {
            "ridge_alpha": 1.0,
            "model_order": models,
            "variants": [
                {
                    "variant_idx": 1,
                    "models": models,
                    "weights": [0.6, 0.4],
                    "weights_by_model": {"vgg16": 0.6, "densenet161": 0.4},
                    "intercept": 1.0,
                },
                {
                    "variant_idx": 2,
                    "models": models,
                    "weights": [0.7, -0.2],
                    "weights_by_model": {"vgg16": 0.7, "densenet161": -0.2},
                    "intercept": 0.5,
                },
            ],
        }
        with open(
            weights_dir / "stacking_coefficients.json", "w", encoding="utf-8"
        ) as f:
            json.dump(coeff_payload, f)

        summary = experiment_3d_plot_ensemble_heatmaps(
            models_filter=models,
            device="cpu",
            save_plots=False,
            modes=["bagging", "stacking"],
            variants=[1, 2],
            reuse_exp2_maps=True,
            live_mpm=False,
        )

        assert summary["status"] == "success"
        assert summary["execution_mode"] == "reuse_exp2_maps"

        for mode in ("bagging", "stacking"):
            for k in (1, 2):
                var_file = mpm_out_dir / f"{mode}_exp3_variant{k}_maps.npy"
                assert var_file.exists()
                loaded = np.load(var_file)
                assert loaded.shape == (n_test, 224, 224)
                assert loaded.dtype == np.float32
                assert float(np.min(loaded)) >= -1.0 - 1e-5
                assert float(np.max(loaded)) <= 1.0 + 1e-5

            proto_file = mpm_out_dir / f"{mode}_exp3_prototype_maps.npy"
            assert proto_file.exists()
            proto_loaded = np.load(proto_file)
            assert proto_loaded.shape == (n_test, 224, 224)
            assert mode in summary["within_ensemble_consistency"]
        assert "between_prototypes_agreement" in summary

    def test_experiment_3d_live_mpm_mode(
        self,
        mock_experiment_dirs: dict[str, Path],
        mock_exp1_weight_files: list[Path],
        synthetic_dataset: Dataset,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify Stage 3D live MPM mode computes and saves ensemble maps when live_mpm=True."""
        exp1_weights_dir = mock_experiment_dirs["exp1_weights"]
        mpm_out_dir = (
            mock_experiment_dirs["root"]
            / "Outputs"
            / "Experiment_3_ensemble"
            / "XAI_Maps"
            / "Multiscale_Pixel_Masking"
        )
        mpm_out_dir.mkdir(parents=True, exist_ok=True)
        weights_dir = mock_experiment_dirs["exp3_weights"]
        results_dir = mock_experiment_dirs["exp3_results"]
        plots_dir = mock_experiment_dirs["exp3_plots"]

        monkeypatch.setitem(experiment_three.DIRS, "exp1_weights", exp1_weights_dir)
        monkeypatch.setitem(experiment_three.DIRS, "mpm", mpm_out_dir)
        monkeypatch.setitem(experiment_three.DIRS, "weights", weights_dir)
        monkeypatch.setitem(experiment_three.DIRS, "results", results_dir)
        monkeypatch.setitem(experiment_three.DIRS, "plots", plots_dir)
        monkeypatch.setattr(experiment_three, "OUTPUT_DIR", weights_dir.parent)
        monkeypatch.setattr(experiment_three, "imageNet_dataset", synthetic_dataset)
        monkeypatch.setattr(experiment_three, "denseNet_dataset", synthetic_dataset)
        monkeypatch.setattr(experiment_three, "clip_dataset", synthetic_dataset)

        fake_mpm_instance = MagicMock()
        fake_mpm_instance.generate_map.return_value = np.linspace(
            -5.0, 5.0, num=32 * 32, dtype=np.float32
        ).reshape(32, 32)

        with (
            patch(
                "Image_Authenticity_prediction.main.Experiments.experiment_three.load_model_with_weights",
                return_value=MagicMock(),
            ),
            patch(
                "Image_Authenticity_prediction.main.Experiments.experiment_three.MultiscalePixelMasking",
                return_value=fake_mpm_instance,
            ),
        ):
            summary = experiment_3d_plot_ensemble_heatmaps(
                global_test_indices={
                    "imagenet": [0, 1, 2],
                    "densenet": [0, 1, 2],
                    "clip": [0, 1, 2],
                },
                models_filter=["vgg16", "resnet152"],
                device="cpu",
                save_plots=False,
                modes=["bagging"],
                variants=[1],
                reuse_exp2_maps=False,
                live_mpm=True,
                target_resolution=(32, 32),
            )

        assert summary["status"] == "success"
        assert summary["execution_mode"] == "live_mpm"
        var_file = mpm_out_dir / "bagging_exp3_variant1_maps.npy"
        proto_file = mpm_out_dir / "bagging_exp3_prototype_maps.npy"
        assert var_file.exists()
        assert proto_file.exists()
        assert np.load(var_file).shape == (3, 32, 32)

    def test_parse_variants_and_reliability_helpers(self) -> None:
        """Verify parse_variant_indices and compute_prediction_reliability helpers."""
        assert parse_variant_indices("all", num_variants=10) == list(range(1, 11))
        assert parse_variant_indices("1,3,5") == [1, 3, 5]

        r_model, ceiling = compute_prediction_reliability(
            [
                np.array([10.0, 20.0, 30.0]),
                np.array([10.0, 20.0, 30.0]),
            ]
        )
        assert abs(r_model - 1.0) < 1e-6
        assert abs(ceiling - np.sqrt(0.67)) < 1e-6

    def test_run_experiment_3_orchestration_all_flags_disabled(
        self,
        mock_experiment_dirs: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify run_experiment_3 returns clean results dictionary when all phase flags are disabled."""
        monkeypatch.setitem(
            experiment_three.DIRS, "weights", mock_experiment_dirs["exp3_weights"]
        )
        monkeypatch.setitem(
            experiment_three.DIRS, "results", mock_experiment_dirs["exp3_results"]
        )
        monkeypatch.setitem(
            experiment_three.DIRS, "heatmaps", mock_experiment_dirs["exp3_heatmaps"]
        )
        monkeypatch.setitem(
            experiment_three.DIRS, "plots", mock_experiment_dirs["exp3_plots"]
        )

        results = run_experiment_3(
            models=["vgg16"],
            run_training=False,
            run_pruning=False,
            run_evaluation=False,
            run_heatmaps=False,
            save_results=False,
        )

        assert isinstance(results, dict)
        assert "training" not in results
        assert "pruning" not in results
        assert "evaluation" not in results
