"""Unit and integration tests for Experiment 3 (Bagging & Stacking Ensembles)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from Image_Authenticity_prediction.main.Experiments import experiment_three
from Image_Authenticity_prediction.main.Experiments.experiment_three import (
    StackingMetaLearner,
    create_global_test_indices,
    create_variant_split,
    experiment_3c_evaluate_ensemble,
    get_labels,
    get_predictions,
    reset_regression_head,
    run_experiment_3,
    setup_directories,
)


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


class TestExperimentThreePipelineStages:
    """Tests for Experiment 3 execution phases (3A, 3B, 3C, and complete pipeline)."""

    def test_setup_directories_creates_all_folders(
        self,
        mock_experiment_dirs: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify setup_directories creates weights, rankings, results, and heatmap directories."""
        fresh_base = mock_experiment_dirs["root"] / "FreshExp3"
        dirs = {
            "weights": fresh_base / "Weights",
            "rankings": fresh_base / "Ranking_arrays",
            "ranking_plots": fresh_base / "Ranking_Plots",
            "training_plots": fresh_base / "Training_Plots",
            "training_history": fresh_base / "Training_History",
            "results": fresh_base / "Results",
            "heatmaps": fresh_base / "Heatmaps",
        }
        monkeypatch.setattr(experiment_three, "DIRS", dirs)

        setup_directories()

        for path in dirs.values():
            assert path.exists()

    def test_experiment_3c_empty_weights_returns_empty_dict(
        self,
        mock_experiment_dirs: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify experiment_3c_evaluate_ensemble gracefully handles missing pruned weight files."""
        empty_weights = mock_experiment_dirs["root"] / "EmptyWeightsExp3"
        empty_weights.mkdir(parents=True, exist_ok=True)
        monkeypatch.setitem(experiment_three.DIRS, "weights", empty_weights)

        results = experiment_3c_evaluate_ensemble(models_filter=["vgg16"], device="cpu")
        assert results == {}

    def test_experiment_3c_bagging_ensemble_evaluation(
        self,
        mock_experiment_dirs: dict[str, Path],
        mock_exp3_weight_files: list[Path],
        synthetic_dataset: Dataset,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify bagging ensemble averages variant predictions and computes test performance metrics."""
        weights_dir = mock_experiment_dirs["exp3_weights"]
        results_dir = mock_experiment_dirs["exp3_results"]
        plots_dir = mock_experiment_dirs["exp3_plots"]

        monkeypatch.setitem(experiment_three.DIRS, "weights", weights_dir)
        monkeypatch.setitem(experiment_three.DIRS, "results", results_dir)
        monkeypatch.setitem(experiment_three.DIRS, "plots", plots_dir)

        dummy_preds = torch.tensor([[45.0], [55.0], [65.0], [75.0]])
        dummy_labels = torch.tensor([42.0, 50.0, 68.0, 72.0])

        with (
            patch(
                "Image_Authenticity_prediction.main.Experiments.experiment_three.get_predictions",
                return_value=dummy_preds,
            ),
            patch(
                "Image_Authenticity_prediction.main.Experiments.experiment_three.get_labels",
                return_value=dummy_labels,
            ),
            patch(
                "Image_Authenticity_prediction.main.Experiments.experiment_three.load_model_with_weights",
                return_value=MagicMock(),
            ),
        ):
            results = experiment_3c_evaluate_ensemble(
                models_filter=["vgg16"],
                global_test_indices={
                    "imagenet": [0, 1, 2, 3],
                    "densenet": [0, 1, 2, 3],
                },
                device="cpu",
                ensemble_mode=["bagging"],
            )

        assert "pruned_ensemble_bagging" in results
        bagging_metrics = results["pruned_ensemble_bagging"]
        assert "mse" in bagging_metrics
        assert "rmse" in bagging_metrics
        assert "plcc" in bagging_metrics
        assert "srcc" in bagging_metrics
        assert "krcc" in bagging_metrics
        assert isinstance(bagging_metrics["mse"], float)

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
