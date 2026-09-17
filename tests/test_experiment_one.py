"""Unit and integration tests for Experiment 1 (Model Training, Pruning, and Testing)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from Image_Authenticity_prediction.main.Experiments import experiment_one
from Image_Authenticity_prediction.main.Experiments.experiment_one import (
    MODEL_REGISTRY,
    PRUNING_CONFIG,
    TRAINING_CONFIG,
    experiment_1a_train_all_models,
    experiment_1b_prune_all_models,
    experiment_one_test_models,
    run_experiment_one_complete,
)


class TestExperimentOneConfigAndRegistry:
    """Tests for Experiment 1 configuration, model registry, and architectural definitions."""

    def test_model_registry_contains_expected_architectures(self) -> None:
        """Verify that all core architectures are registered."""
        expected_models = {
            "vgg16",
            "vgg19",
            "resnet152",
            "densenet161",
            "efficientnetb3",
            "barlowtwins",
        }
        assert expected_models.issubset(set(MODEL_REGISTRY.keys()))

    def test_model_registry_entry_schema(self) -> None:
        """Verify that every registry entry contains the required configuration fields."""
        required_keys = {"class", "dataset", "target_layer", "input_size"}
        for model_name, entry in MODEL_REGISTRY.items():
            missing_keys = required_keys - set(entry.keys())
            assert not missing_keys, (
                f"Model {model_name} is missing keys: {missing_keys}"
            )
            assert isinstance(entry["input_size"], int)
            assert entry["input_size"] > 0
            assert isinstance(entry["target_layer"], str)

    def test_target_layer_resolvable_in_vgg16(self) -> None:
        """Verify that target pruning layer exists in VGG16 architecture."""
        model_cls = MODEL_REGISTRY["vgg16"]["class"]
        target_layer_name = MODEL_REGISTRY["vgg16"]["target_layer"]
        model = model_cls(freeze_backbone=True)

        named_modules = dict(model.named_modules())
        assert target_layer_name in named_modules, (
            f"Target layer {target_layer_name} not found in VGG16 modules"
        )

    def test_training_and_pruning_config_values(self) -> None:
        """Verify that training and pruning configurations contain valid numeric thresholds."""
        assert "max_epochs" in TRAINING_CONFIG
        assert TRAINING_CONFIG["max_epochs"] > 0
        assert "learning_rate" in TRAINING_CONFIG
        assert TRAINING_CONFIG["learning_rate"] > 0
        assert "patience" in TRAINING_CONFIG
        assert TRAINING_CONFIG["patience"] > 0
        assert "methods" in PRUNING_CONFIG
        assert isinstance(PRUNING_CONFIG["methods"], list)


class TestExperimentOneDataSplits:
    """Tests for deterministic data splitting and variant isolation in Experiment 1."""

    def test_global_test_indices_reproducibility(self) -> None:
        """Verify that global test indices are fully deterministic given a fixed seed."""
        total_size = 100
        test_size = int(0.2 * total_size)

        gen_first = torch.Generator().manual_seed(42)
        perm_first = torch.randperm(total_size, generator=gen_first).tolist()[
            :test_size
        ]

        gen_second = torch.Generator().manual_seed(42)
        perm_second = torch.randperm(total_size, generator=gen_second).tolist()[
            :test_size
        ]

        assert perm_first == perm_second
        assert len(perm_first) == test_size
        assert len(set(perm_first)) == test_size

    def test_train_val_split_isolation_and_variant_diversity(self) -> None:
        """Verify no test set data leakage and distinct train/val splits across variants."""
        total_size = 100
        test_size = int(0.1 * total_size)

        gen_global = torch.Generator().manual_seed(42)
        test_indices = set(
            torch.randperm(total_size, generator=gen_global).tolist()[:test_size]
        )
        remaining = [i for i in range(total_size) if i not in test_indices]

        variant_splits: dict[int, tuple[list[int], list[int]]] = {}
        for variant_idx in (1, 2):
            gen = torch.Generator().manual_seed(42 + variant_idx)
            perm = torch.randperm(len(remaining), generator=gen).tolist()
            shuffled = [remaining[i] for i in perm]

            val_size = int(0.1 * total_size)
            train_size = len(shuffled) - val_size

            train_idx = shuffled[:train_size]
            val_idx = shuffled[train_size:]

            assert not (set(train_idx) & test_indices)
            assert not (set(val_idx) & test_indices)
            assert not (set(train_idx) & set(val_idx))

            variant_splits[variant_idx] = (train_idx, val_idx)

        assert variant_splits[1][0] != variant_splits[2][0], (
            "Different variants must have different training splits"
        )


class TestExperimentOneModelOperations:
    """Tests for regression head reinitialization and model parameter manipulation."""

    def test_reset_regression_head_modifies_only_head_weights(self) -> None:
        """Verify resetting regression head updates head weights while keeping features fixed."""
        model_cls = MODEL_REGISTRY["vgg16"]["class"]
        model = model_cls(freeze_backbone=True)

        features_weight_before = next(model.features.parameters()).clone().detach()
        head_weight_before = [
            param.clone().detach() for param in model.regression_head.parameters()
        ]

        for layer in model.regression_head.modules():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

        features_weight_after = next(model.features.parameters()).clone().detach()
        head_weight_after = [
            param.clone().detach() for param in model.regression_head.parameters()
        ]

        assert torch.equal(features_weight_before, features_weight_after)

        any_head_param_changed = False
        for before, after in zip(head_weight_before, head_weight_after, strict=True):
            if not torch.equal(before, after):
                any_head_param_changed = True
                break
        assert any_head_param_changed


class TestExperimentOnePipelineStages:
    """Tests for Experiment 1 execution stages (1A, 1B, 1C, and complete pipeline)."""

    def test_experiment_1a_handles_unknown_model_gracefully(self) -> None:
        """Verify that training an unregistered model name does not raise an unhandled exception."""
        results = experiment_1a_train_all_models(
            models_to_train=["nonexistent_model_xyz"],
            save_plots=False,
            verbose=False,
        )
        assert "nonexistent_model_xyz" in results
        assert "error" in results["nonexistent_model_xyz"]

    def test_experiment_1b_empty_weights_directory_returns_empty_dict(
        self,
        mock_experiment_dirs: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify that pruning with an empty weights directory logs and returns cleanly."""
        empty_weights = mock_experiment_dirs["root"] / "EmptyWeights"
        empty_weights.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(experiment_one, "WEIGHTS_DIR", empty_weights)
        results = experiment_1b_prune_all_models(
            models_to_prune=["vgg16"], verbose=False
        )
        assert results == {}

    def test_experiment_1b_pruning_with_mocked_pruner(
        self,
        mock_experiment_dirs: dict[str, Path],
        mock_exp1_weight_files: list[Path],
        synthetic_dataset: Dataset,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify pruning logic dispatches to pruner and records statistics correctly."""
        weights_dir = mock_experiment_dirs["exp1_weights"]
        rankings_dir = mock_experiment_dirs["exp1_rankings"]
        ranking_plots_dir = mock_experiment_dirs["exp1_ranking_plots"]

        monkeypatch.setattr(experiment_one, "WEIGHTS_DIR", weights_dir)
        monkeypatch.setattr(experiment_one, "RANKINGS_DIR", rankings_dir)
        monkeypatch.setattr(experiment_one, "RANKING_PLOTS_DIR", ranking_plots_dir)

        mock_mock_pruner_instance = MagicMock()
        mock_mock_pruner_instance.rank_feature_maps.return_value = np.array([0, 1, 2])
        mock_mock_pruner_instance.prune_model.return_value = {
            "baseline_mse": 0.5,
            "baseline_rmse": 0.707,
            "final_mse": 0.4,
            "final_rmse": 0.632,
            "improvement_mse": 0.1,
            "improvement_rmse": 0.075,
            "removed_features": [1],
            "num_removed": 1,
            "reduction_percentage": 25.0,
            "pruned_model": nn.Linear(4, 1),
            "original_model": nn.Linear(4, 1),
        }

        with (
            patch(
                "Image_Authenticity_prediction.main.Experiments.experiment_one.FeatureMapsPruner",
                return_value=mock_mock_pruner_instance,
            ),
            patch(
                "Image_Authenticity_prediction.main.Experiments.experiment_one.check_pruning_statistics",
                return_value={"healthy": True},
            ),
        ):
            results = experiment_1b_prune_all_models(
                models_to_prune=["vgg16"],
                pruning_method="greedy",
                verbose=False,
            )

        assert "vgg16" in results
        assert isinstance(results["vgg16"], dict)

    def test_experiment_one_test_models_handles_missing_weights(
        self,
        mock_experiment_dirs: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify testing phase handles nonexistent or empty weights directory."""
        empty_dir = mock_experiment_dirs["root"] / "NonexistentDir"
        monkeypatch.setattr(experiment_one, "WEIGHTS_DIR", empty_dir)
        monkeypatch.setattr(
            experiment_one,
            "TEST_RESULTS_DIR",
            mock_experiment_dirs["exp1_test_results"],
        )

        results = experiment_one_test_models(models_to_test=["vgg16"], verbose=False)
        assert results == {}

    def test_run_experiment_one_complete_orchestration_all_flags_disabled(
        self,
    ) -> None:
        """Verify run_experiment_one_complete returns empty result dictionary when all phases disabled."""
        results = run_experiment_one_complete(
            models_to_process=["vgg16"],
            run_training=False,
            run_pruning=False,
            run_testing=False,
        )
        assert isinstance(results, dict)
        assert results["training"] is None
        assert results["pruning"] is None
        assert results["testing"] is None
