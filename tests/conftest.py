"""Shared pytest fixtures and test doubles for the experiments test suite."""

import collections.abc
import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SyntheticAuthenticityDataset(Dataset):
    """Synthetic dataset producing dummy image tensors and authenticity scores."""

    def __init__(
        self,
        num_samples: int = 40,
        image_shape: tuple[int, int, int] = (3, 32, 32),
    ) -> None:
        """Initialize the synthetic dataset.

        Args:
            num_samples: Number of synthetic image-label pairs to generate.
            image_shape: Spatial dimension of generated images as (C, H, W).
        """
        self.num_samples = num_samples
        self.image_shape = image_shape
        generator = torch.Generator().manual_seed(42)
        self.data = torch.randn(
            (num_samples, *image_shape),
            generator=generator,
            dtype=torch.float32,
        )
        self.labels = (
            torch.rand(
                (num_samples, 1),
                generator=generator,
                dtype=torch.float32,
            )
            * 100.0
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Number of dataset items.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch a single sample by its integer index.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            Tuple containing the image tensor and scalar score tensor.
        """
        return self.data[idx], self.labels[idx]


class TinyPredictor(nn.Module):
    """Lightweight CNN model adhering to the authenticity predictor interface."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 8) -> None:
        """Initialize tiny predictor layers.

        Args:
            in_channels: Number of input channels.
            feature_dim: Number of intermediate channels in convolutional feature extractor.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the model forward pass.

        Args:
            x: Input batch tensor with shape (B, C, H, W).

        Returns:
            Tuple containing prediction tensor (B, 1) and feature tensor.
        """
        feat = self.features(x)
        return self.regression_head(feat), feat


@pytest.fixture
def synthetic_dataset() -> SyntheticAuthenticityDataset:
    """Fixture providing a deterministic in-memory synthetic dataset.

    Returns:
        Instance of SyntheticAuthenticityDataset.
    """
    return SyntheticAuthenticityDataset(num_samples=40)


@pytest.fixture
def tiny_model() -> TinyPredictor:
    """Fixture providing an instantiated lightweight CNN predictor.

    Returns:
        Instance of TinyPredictor.
    """
    return TinyPredictor()


@pytest.fixture
def mock_experiment_dirs(
    tmp_path: Path,
) -> collections.abc.Generator[dict[str, Path], None, None]:
    """Fixture providing mock output directory hierarchy for experiments.

    Args:
        tmp_path: Temporary directory managed by pytest.

    Yields:
        Mapping of directory purpose names to their temporary filesystem paths.
    """
    exp1_dir = tmp_path / "Outputs" / "Experiment_1_variants"
    weights_dir = exp1_dir / "Weights"
    rankings_dir = exp1_dir / "Ranking_arrays"
    ranking_plots_dir = exp1_dir / "Ranking_Plots"
    training_plots_dir = exp1_dir / "Training_Plots"
    training_history_dir = exp1_dir / "Training_History"
    test_results_dir = exp1_dir / "Test_Results"

    exp2_dir = tmp_path / "Outputs" / "Experiment_2_variants"
    gradcam_dir = exp2_dir / "XAI_Maps" / "GradCAM"
    mpm_dir = exp2_dir / "XAI_Maps" / "Multiscale_Pixel_Masking"
    exp2_plots_dir = exp2_dir / "Plots"

    exp3_dir = tmp_path / "Outputs" / "Experiment_3_ensemble"
    exp3_weights_dir = exp3_dir / "Weights"
    exp3_results_dir = exp3_dir / "Results"
    exp3_plots_dir = exp3_dir / "Plots"
    exp3_heatmaps_dir = exp3_dir / "Heatmaps"

    all_dirs = [
        weights_dir,
        rankings_dir,
        ranking_plots_dir,
        training_plots_dir,
        training_history_dir,
        test_results_dir,
        gradcam_dir,
        mpm_dir,
        exp2_plots_dir,
        exp3_weights_dir,
        exp3_results_dir,
        exp3_plots_dir,
        exp3_heatmaps_dir,
    ]

    for folder in all_dirs:
        folder.mkdir(parents=True, exist_ok=True)

    yield {
        "root": tmp_path,
        "exp1_weights": weights_dir,
        "exp1_rankings": rankings_dir,
        "exp1_ranking_plots": ranking_plots_dir,
        "exp1_training_plots": training_plots_dir,
        "exp1_training_history": training_history_dir,
        "exp1_test_results": test_results_dir,
        "exp2_gradcam": gradcam_dir,
        "exp2_mpm": mpm_dir,
        "exp2_plots": exp2_plots_dir,
        "exp3_weights": exp3_weights_dir,
        "exp3_results": exp3_results_dir,
        "exp3_plots": exp3_plots_dir,
        "exp3_heatmaps": exp3_heatmaps_dir,
    }


@pytest.fixture
def mock_exp1_weight_files(
    mock_experiment_dirs: dict[str, Path],
) -> list[Path]:
    """Fixture creating realistic dummy weight checkpoints for Experiment 1 and 2.

    Args:
        mock_experiment_dirs: Mapping of directory purpose to temporary paths.

    Returns:
        List of created weight checkpoint paths.
    """
    weights_dir = mock_experiment_dirs["exp1_weights"]
    model = TinyPredictor()
    state_dict = model.state_dict()

    filenames = [
        "vgg16_exp1a_variant1_best.pth",
        "vgg16_exp1a_variant2_best.pth",
        "vgg16_exp1b_variant1_greedy_pruned_best.pth",
        "vgg16_exp1b_variant1_negative_pruned_best.pth",
        "resnet152_exp1a_variant1_best.pth",
        "resnet152_exp1a_variant2_best.pth",
        "resnet152_exp1b_variant1_greedy_pruned_best.pth",
    ]

    created_paths: list[Path] = []
    for fname in filenames:
        path = weights_dir / fname
        torch.save(state_dict, path)
        created_paths.append(path)

    return created_paths


@pytest.fixture
def mock_exp3_weight_files(
    mock_experiment_dirs: dict[str, Path],
) -> list[Path]:
    """Fixture creating realistic dummy weight checkpoints for Experiment 3.

    Args:
        mock_experiment_dirs: Mapping of directory purpose to temporary paths.

    Returns:
        List of created weight checkpoint paths.
    """
    weights_dir = mock_experiment_dirs["exp3_weights"]
    model = TinyPredictor()
    state_dict = model.state_dict()

    filenames = [
        "vgg16_exp3a_variant1_best.pth",
        "vgg16_exp3a_variant2_best.pth",
        "vgg16_exp3b_variant1_greedy_pruned.pth",
        "vgg16_exp3b_variant2_greedy_pruned.pth",
        "resnet152_exp3a_variant1_best.pth",
        "resnet152_exp3a_variant2_best.pth",
        "resnet152_exp3b_variant1_greedy_pruned.pth",
        "resnet152_exp3b_variant2_greedy_pruned.pth",
    ]

    created_paths: list[Path] = []
    for fname in filenames:
        path = weights_dir / fname
        torch.save(state_dict, path)
        created_paths.append(path)

    return created_paths


@pytest.fixture
def mock_saliency_arrays(
    mock_experiment_dirs: dict[str, Path],
) -> dict[str, list[Path]]:
    """Fixture creating synthetic numpy saliency maps for explainability comparisons.

    Args:
        mock_experiment_dirs: Mapping of directory purpose to temporary paths.

    Returns:
        Mapping containing lists of created saliency map paths for each method.
    """
    gradcam_dir = mock_experiment_dirs["exp2_gradcam"]
    mpm_dir = mock_experiment_dirs["exp2_mpm"]

    np.random.seed(42)
    created: dict[str, list[Path]] = {"gradcam": [], "mpm": []}

    models = ["vgg16", "resnet152"]
    variants = ["exp1a_variant1", "exp1b_variant1_greedy_pruned"]

    for model_name in models:
        for variant in variants:
            gradcam_arr = np.random.rand(10, 224, 224).astype(np.float32)
            mpm_arr = np.random.rand(10, 224, 224).astype(np.float32)

            g_path = gradcam_dir / f"{model_name}_{variant}_maps.npy"
            m_path = mpm_dir / f"{model_name}_{variant}_maps.npy"

            np.save(g_path, gradcam_arr)
            np.save(m_path, mpm_arr)

            created["gradcam"].append(g_path)
            created["mpm"].append(m_path)

    return created
