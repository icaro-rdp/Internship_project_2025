"""Tests for the Multiscale Pixel Masking explainer."""

import numpy as np
import pytest
import torch
from torch import nn

from Image_Authenticity_prediction.main.Utils.explainability import (
    MultiscalePixelMasking,
)

IMAGE_SHAPE = (3, 9, 11)  # Non-square to catch row/column swaps.
SIGMAS = [1, 2, 3, 6]  # Even sigmas exercise the asymmetric occluder.
MASK_VALUE = 0.5


class TinyRegressor(nn.Module):
    """Small CNN returning ``(predictions, features)`` like the project models."""

    def __init__(self) -> None:
        """Build a fixed-seed conv -> pool -> linear network."""
        super().__init__()
        torch.manual_seed(0)
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
        )
        self.head = nn.Linear(16, 2)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return regression outputs and the pooled features."""
        features = self.features(images)
        return self.head(features), features


def reference_saliency(
    model: nn.Module, image: torch.Tensor, target_index: int
) -> np.ndarray:
    """Compute raw MPM saliency one occluded image at a time (the specification)."""
    _, height, width = image.shape
    saliency = np.zeros((height, width), dtype=np.float64)
    with torch.no_grad():
        original_score = model(image.unsqueeze(0))[0][0, target_index].item()
        for sigma in SIGMAS:
            for row in range(height):
                for col in range(width):
                    masked = image.clone()
                    top, left = max(0, row - sigma // 2), max(0, col - sigma // 2)
                    masked[
                        :, top : row + (sigma + 1) // 2, left : col + (sigma + 1) // 2
                    ] = MASK_VALUE
                    score = model(masked.unsqueeze(0))[0][0, target_index].item()
                    saliency[row, col] += original_score - score
    return saliency


@pytest.fixture
def model() -> TinyRegressor:
    """Provide the deterministic test network."""
    return TinyRegressor().eval()


@pytest.fixture
def image() -> torch.Tensor:
    """Provide a deterministic random image."""
    return torch.randn(IMAGE_SHAPE, generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("target_index", [0, 1])
def test_matches_per_pixel_reference(
    model: TinyRegressor, image: torch.Tensor, target_index: int
) -> None:
    """Vectorised batches reproduce the one-image-at-a-time definition."""
    explainer = MultiscalePixelMasking(
        model, SIGMAS, pixel_batch_size=16, mask_value=MASK_VALUE, use_tqdm=False
    )
    saliency = explainer.generate_map(image, target_index, normalize=False)
    expected = reference_saliency(model, image, target_index)
    np.testing.assert_allclose(saliency, expected, rtol=1e-5, atol=1e-6)


def test_batch_size_does_not_change_map(
    model: TinyRegressor, image: torch.Tensor
) -> None:
    """Batch size, including ragged final batches, only affects throughput."""
    maps = [
        MultiscalePixelMasking(
            model, SIGMAS, batch_size, mask_value=MASK_VALUE, use_tqdm=False
        ).generate_map(image.unsqueeze(0), normalize=False)
        for batch_size in (1, 7, 1000)
    ]
    for saliency in maps[1:]:
        np.testing.assert_allclose(saliency, maps[0], rtol=1e-5, atol=1e-6)


def test_occlusion_masks_clip_at_image_border() -> None:
    """Occluders centred near a corner are clipped rather than wrapped."""
    corner_and_last = torch.tensor([0, 9 * 11 - 1])
    masks = MultiscalePixelMasking._occlusion_masks((9, 11), corner_and_last, 3)
    assert masks.shape == (2, 1, 9, 11)
    assert masks[0, 0, :2, :2].all() and masks[0].sum() == 4
    assert masks[1, 0, -2:, -2:].all() and masks[1].sum() == 4


def test_normalized_map_spans_unit_range(
    model: TinyRegressor, image: torch.Tensor
) -> None:
    """Normalised output is rescaled to [-1, 1]."""
    explainer = MultiscalePixelMasking(model, SIGMAS, 32, MASK_VALUE, use_tqdm=False)
    saliency = explainer.generate_map(image)
    assert saliency.shape == IMAGE_SHAPE[1:]
    assert saliency.min() == pytest.approx(-1.0)
    assert saliency.max() == pytest.approx(1.0)
