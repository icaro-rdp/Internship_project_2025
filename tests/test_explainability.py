"""Tests for the Multiscale Pixel Masking explainer."""

import numpy as np
import pytest
import torch
from torch import nn

from Image_Authenticity_prediction.main.Utils.explainability import (
    LIME,
    GradCAM,
    MultiscalePixelMasking,
    compute_slic_segments,
    weighted_r2,
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


def test_stability_defaults_and_cleanup(
    model: TinyRegressor, image: torch.Tensor
) -> None:
    """MPM uses safe fp32 defaults, batch_size=64, and clean cleanup."""
    explainer = MultiscalePixelMasking(model, SIGMAS, mask_value=MASK_VALUE)
    assert explainer.pixel_batch_size == 64
    assert explainer.use_amp is False
    explainer.generate_map(image)
    explainer.cleanup(failed=True)
    explainer.cleanup(failed=False)


def test_gradcam_preserves_parameter_requires_grad(
    model: TinyRegressor, image: torch.Tensor
) -> None:
    """GradCAM computes heatmaps via autograd.grad without mutating parameter flags."""
    grad_cam = GradCAM(model, "features.0", relu=False)
    heatmap = grad_cam.generate_map(image.unsqueeze(0), 0)
    assert heatmap.shape == IMAGE_SHAPE[1:]
    assert all(p.requires_grad for p in model.parameters())
    grad_cam.cleanup()
    assert grad_cam.hooks == []


def test_per_scale_stride_upsamples_to_full_resolution(
    model: TinyRegressor, image: torch.Tensor
) -> None:
    """Per-scale strides evaluate a coarse grid and upsample back to (H, W)."""
    explainer = MultiscalePixelMasking(
        model, SIGMAS, pixel_batch_size=16, mask_value=MASK_VALUE, stride=[1, 2, 3, 4]
    )
    assert explainer.strides == [1, 2, 3, 4]
    saliency = explainer.generate_map(image)
    assert saliency.shape == IMAGE_SHAPE[1:]
    assert saliency.min() == pytest.approx(-1.0)
    assert saliency.max() == pytest.approx(1.0)


def test_mpm_input_validation(model: TinyRegressor) -> None:
    """Validate that invalid parameters raise ValueError with informative messages."""
    with pytest.raises(ValueError, match="pixel_batch_size must be positive"):
        MultiscalePixelMasking(model, SIGMAS, pixel_batch_size=0)

    with pytest.raises(ValueError, match="sigma_list cannot be empty"):
        MultiscalePixelMasking(model, [])

    with pytest.raises(ValueError, match="All occluder sizes.*must be positive"):
        MultiscalePixelMasking(model, [1, -2, 3])

    with pytest.raises(ValueError, match="stride must be positive"):
        MultiscalePixelMasking(model, SIGMAS, stride=-1)

    with pytest.raises(ValueError, match="All stride values must be positive"):
        MultiscalePixelMasking(model, SIGMAS, stride=[1, 0, 1, 1])


def test_mpm_log_debug_stats(model: TinyRegressor, image: torch.Tensor) -> None:
    """MPM generates map and logs debug stats without error."""
    explainer = MultiscalePixelMasking(
        model,
        SIGMAS,
        pixel_batch_size=16,
        mask_value=MASK_VALUE,
    )
    saliency = explainer.generate_map(image, log_debug_stats=True)
    assert saliency.shape == IMAGE_SHAPE[1:]
    explainer.cleanup()


def test_gradcam_log_debug_stats(model: TinyRegressor, image: torch.Tensor) -> None:
    """GradCAM runs and logs debug statistics when enabled."""
    grad_cam = GradCAM(model, "features.0", relu=False)
    heatmap = grad_cam.generate_map(image.unsqueeze(0), 0, log_debug_stats=True)
    assert heatmap.shape == IMAGE_SHAPE[1:]
    grad_cam.cleanup(failed=False)


def test_compute_slic_segments_properties(image: torch.Tensor) -> None:
    """SLIC segmentation returns contiguous int16 labels starting at 0."""
    img_rgb = torch.clamp(image.permute(1, 2, 0), 0.0, 1.0).numpy().astype(np.float32)
    segments = compute_slic_segments(img_rgb, n_segments=6, compactness=10.0)
    assert segments.shape == IMAGE_SHAPE[1:]
    assert segments.dtype == np.int16
    assert segments.min() == 0
    num_unique = len(np.unique(segments))
    assert num_unique == int(segments.max()) + 1


def test_weighted_r2_computation() -> None:
    """Weighted R^2 returns 1 for exact fit and 0 for mean predictor."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([0.5, 1.0, 1.5, 2.0])
    assert weighted_r2(y_true, y_true, weights) == pytest.approx(1.0)
    w_mean = np.sum(weights * y_true) / np.sum(weights)
    assert weighted_r2(y_true, np.full_like(y_true, w_mean), weights) == pytest.approx(
        0.0
    )
    assert weighted_r2(np.ones(4), np.ones(4), weights) == pytest.approx(1.0)


def test_lime_generates_symmetric_normalized_map_and_metadata(
    model: TinyRegressor, image: torch.Tensor
) -> None:
    """LIME produces a zero-centered map in [-1, 1] and records beta, R^2, and segments."""
    explainer = LIME(
        model,
        n_segments=6,
        compactness=10.0,
        num_perturbations=24,
        keep_prob=0.7,
        kernel_width=0.25,
        ridge_alpha=1.0,
        seed=42,
        batch_size=8,
        use_tqdm=False,
    )
    raw = explainer.generate_map(image, target_index=0, normalize=False)
    saliency = explainer.generate_map(image, target_index=0, log_debug_stats=True)
    assert saliency.shape == IMAGE_SHAPE[1:]
    assert saliency.dtype == np.float32
    expected_max = float(np.max(np.abs(raw))) / (float(np.max(np.abs(raw))) + 1e-8)
    assert np.max(np.abs(saliency)) == pytest.approx(expected_max, rel=1e-4)
    assert explainer.last_beta is not None
    assert explainer.last_segments is not None
    assert explainer.last_r2 is not None
    assert explainer.last_yhat0 is not None
    assert len(explainer.last_beta) == int(explainer.last_segments.max()) + 1

    # Passing precomputed segments yields identical output
    saliency_pre = explainer.generate_map(
        image,
        target_index=0,
        segments=explainer.last_segments,
    )
    np.testing.assert_allclose(saliency, saliency_pre, rtol=1e-6, atol=1e-6)
    explainer.cleanup()


def test_lime_post_masking_resize(model: TinyRegressor, image: torch.Tensor) -> None:
    """LIME supports post-masking spatial upsampling while returning original canvas resolution."""
    explainer = LIME(
        model,
        n_segments=5,
        num_perturbations=12,
        batch_size=6,
        model_input_size=(16, 16),
    )
    saliency = explainer.generate_map(image, normalize=False)
    assert saliency.shape == IMAGE_SHAPE[1:]


def test_lime_input_validation(model: TinyRegressor) -> None:
    """LIME raises ValueError when initialized with out-of-range hyperparameters."""
    with pytest.raises(ValueError, match="n_segments must be positive"):
        LIME(model, n_segments=0)
    with pytest.raises(ValueError, match="compactness must be positive"):
        LIME(model, compactness=0.0)
    with pytest.raises(ValueError, match="num_perturbations must be > 1"):
        LIME(model, num_perturbations=0)
    with pytest.raises(ValueError, match="keep_prob must be in"):
        LIME(model, keep_prob=1.0)
    with pytest.raises(ValueError, match="kernel_width must be positive"):
        LIME(model, kernel_width=-0.1)
    with pytest.raises(ValueError, match="ridge_alpha must be positive"):
        LIME(model, ridge_alpha=-1.0)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        LIME(model, batch_size=0)
