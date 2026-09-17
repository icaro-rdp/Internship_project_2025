"""Unit tests for experiment CLI parsing and subcommand dispatching."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from Image_Authenticity_prediction import __main__ as cli_entry


class TestExperimentsCLI:
    """Tests for CLI arguments parsing and experiment dispatch handlers."""

    @pytest.fixture
    def cli_parser(self) -> argparse.ArgumentParser:
        """Create and return the top-level argument parser from __main__.

        Returns:
            Configured ArgumentParser with all subcommands.
        """
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        # Re-register subparsers identically to __main__.main()
        # Experiment One
        exp1_parser = subparsers.add_parser("experiment-one")
        exp1_parser.add_argument("--models", type=str, nargs="+")
        exp1_parser.add_argument("--train", action="store_true")
        exp1_parser.add_argument("--prune", action="store_true")
        exp1_parser.add_argument("--test", action="store_true")
        exp1_parser.add_argument(
            "--pruning-method",
            type=str,
            default="both",
            choices=["greedy", "negative_impact", "both"],
        )
        exp1_parser.add_argument("--threshold", type=float, default=0.0)

        # Experiment Two
        exp2_parser = subparsers.add_parser("experiment-two")
        exp2_parser.add_argument("--models", type=str, nargs="+")
        exp2_parser.add_argument(
            "--xai-methods",
            type=str,
            default="both",
            choices=["gradcam", "mpm", "both"],
        )
        exp2_parser.add_argument("--variants", type=str, default="all")
        exp2_parser.add_argument("--save-maps", action="store_true", default=True)
        exp2_parser.add_argument("--comparison-only", action="store_true")
        exp2_parser.add_argument("--run-comparison", action="store_true")
        exp2_parser.add_argument(
            "--comparison-kinds",
            type=str,
            nargs="+",
            default=["between_model_architectures"],
        )
        exp2_parser.add_argument(
            "--comparison-metrics",
            type=str,
            nargs="+",
            default=["correlation"],
        )
        exp2_parser.add_argument("--target-resolution", type=str, default="224,224")

        # Experiment Three
        exp3_parser = subparsers.add_parser("experiment-three")
        exp3_parser.add_argument("--models", type=str, nargs="+")
        exp3_parser.add_argument(
            "--strategy",
            type=str,
            default="both",
            choices=["bagging", "stacking", "both"],
        )
        exp3_parser.add_argument("--train", action="store_true", default=True)
        exp3_parser.add_argument("--no-train", dest="train", action="store_false")
        exp3_parser.add_argument("--evaluate", action="store_true", default=True)
        exp3_parser.add_argument("--no-evaluate", dest="evaluate", action="store_false")
        exp3_parser.add_argument("--save-results", action="store_true", default=True)
        exp3_parser.add_argument(
            "--no-save-results", dest="save_results", action="store_false"
        )
        exp3_parser.add_argument("--heatmaps", action="store_true")

        return parser

    def test_parse_experiment_one_flags(
        self, cli_parser: argparse.ArgumentParser
    ) -> None:
        """Verify experiment-one command parses models, pruning methods, and thresholds."""
        args = cli_parser.parse_args(
            [
                "experiment-one",
                "--models",
                "vgg16",
                "resnet152",
                "--train",
                "--prune",
                "--pruning-method",
                "greedy",
                "--threshold",
                "0.05",
            ]
        )

        assert args.command == "experiment-one"
        assert args.models == ["vgg16", "resnet152"]
        assert args.train is True
        assert args.prune is True
        assert args.test is False
        assert args.pruning_method == "greedy"
        assert args.threshold == 0.05

    def test_parse_experiment_two_flags(
        self, cli_parser: argparse.ArgumentParser
    ) -> None:
        """Verify experiment-two command parses XAI methods, variants, and comparison options."""
        args = cli_parser.parse_args(
            [
                "experiment-two",
                "--models",
                "vgg16",
                "--xai-methods",
                "gradcam",
                "--variants",
                "greedy",
                "--comparison-only",
                "--comparison-kinds",
                "between_model_architectures",
                "within_model_variants",
                "--comparison-metrics",
                "correlation",
                "rmse",
            ]
        )

        assert args.command == "experiment-two"
        assert args.models == ["vgg16"]
        assert args.xai_methods == "gradcam"
        assert args.variants == "greedy"
        assert args.comparison_only is True
        assert "between_model_architectures" in args.comparison_kinds
        assert "within_model_variants" in args.comparison_kinds
        assert args.comparison_metrics == ["correlation", "rmse"]

    def test_parse_experiment_three_flags(
        self, cli_parser: argparse.ArgumentParser
    ) -> None:
        """Verify experiment-three command parses strategy, execution flags, and heatmap options."""
        args = cli_parser.parse_args(
            [
                "experiment-three",
                "--models",
                "vgg16",
                "--strategy",
                "bagging",
                "--no-train",
                "--evaluate",
                "--heatmaps",
            ]
        )

        assert args.command == "experiment-three"
        assert args.models == ["vgg16"]
        assert args.strategy == "bagging"
        assert args.train is False
        assert args.evaluate is True
        assert args.heatmaps is True

    def test_dispatch_experiment_one_command(self) -> None:
        """Verify experiment_one_command dispatches parsed args to run_experiment_one_complete."""
        mock_args = MagicMock(
            models=["vgg16"],
            train=True,
            prune=False,
            test=True,
            pruning_method="greedy",
            threshold=0.0,
        )

        with (
            patch(
                "main.Experiments.experiment_one.run_experiment_one_complete"
            ) as mock_run,
            patch(
                "Image_Authenticity_prediction.main.Experiments.experiment_one.run_experiment_one_complete",
                new=mock_run,
            ),
        ):
            cli_entry.experiment_one_command(mock_args)
            mock_run.assert_called_once_with(
                models_to_process=["vgg16"],
                run_training=True,
                run_pruning=False,
                run_testing=True,
                pruning_method="greedy",
                threshold=0.0,
            )

    def test_dispatch_experiment_two_command(self) -> None:
        """Verify experiment_two_command dispatches parsed args to run_experiment_2."""
        mock_args = MagicMock(
            models=["vgg16"],
            save_maps=True,
            variants="all",
            xai_methods="both",
            comparison_only=False,
            run_comparison=True,
            comparison_kinds=["between_model_architectures"],
            comparison_metrics=["correlation"],
            target_resolution="224,224",
        )

        with (
            patch("main.Experiments.experiment_two.run_experiment_2") as mock_run,
            patch(
                "Image_Authenticity_prediction.main.Experiments.experiment_two.run_experiment_2",
                new=mock_run,
            ),
        ):
            cli_entry.experiment_two_command(mock_args)
            mock_run.assert_called_once_with(
                models=["vgg16"],
                save_maps=True,
                variants="all",
                xai_methods="both",
                comparison_only=False,
                run_comparison=True,
                comparison_kinds=("between_model_architectures",),
                comparison_metrics=("correlation",),
                comparison_target_resolution=(224, 224),
                save_comparison_json=True,
            )

    def test_dispatch_experiment_three_command(self) -> None:
        """Verify experiment_three_command dispatches parsed args to run_experiment_3."""
        mock_args = MagicMock(
            models=["vgg16"],
            strategy="both",
            train=True,
            evaluate=True,
            heatmaps=False,
            save_results=True,
        )

        with (
            patch("main.Experiments.experiment_three.run_experiment_3") as mock_run,
            patch(
                "Image_Authenticity_prediction.main.Experiments.experiment_three.run_experiment_3",
                new=mock_run,
            ),
        ):
            cli_entry.experiment_three_command(mock_args)
            mock_run.assert_called_once_with(
                models=["vgg16"],
                run_training=True,
                run_pruning=True,
                run_evaluation=True,
                run_heatmaps=False,
                save_results=True,
                ensemble_mode=["bagging", "stacking"],
            )
