#!/usr/bin/env python3
"""Benchmark Multiscale Pixel Masking (MPM) optimizations on Blackwell GPUs.

Tests varying pixel batch sizes (64, 128, 256, 512), mixed precision (bfloat16),
cuDNN autotuning, and measure speedup, peak VRAM, and numerical fidelity against baseline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Image_Authenticity_prediction.main.Models.models import (
    DenseNet161AuthenticityPredictor,
)
from Image_Authenticity_prediction.main.Utils.explainability import (
    MultiscalePixelMasking,
)


def load_test_image(dataset_dir: Path, img_size: int = 224) -> torch.Tensor:
    """Find and load one real test image as normalized (1, 3, H, W) tensor."""
    image_dir = dataset_dir / "Image"
    candidates = list(image_dir.glob("*/*.jpg")) + list(image_dir.glob("*/*.png"))
    if not candidates:
        candidates = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    if candidates:
        img_path = candidates[0]
        print(f"Loaded test image from: {img_path}")
        img = Image.open(img_path).convert("RGB")
        return transform(img).unsqueeze(0)
    else:
        print("Warning: No image file found, generating synthetic RGB tensor.")
        return torch.randn(1, 3, img_size, img_size)


class AutocastMPM(MultiscalePixelMasking):
    """MPM wrapper enabling torch.autocast for forward evaluations."""

    def __init__(self, *args, autocast_dtype=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.autocast_dtype = autocast_dtype

    def generate_map(self, image: torch.Tensor, target_index: int = 0, **kwargs):
        if self.autocast_dtype is not None:
            with torch.amp.autocast(device_type="cuda", dtype=self.autocast_dtype):
                return super().generate_map(image, target_index=target_index, **kwargs)
        return super().generate_map(image, target_index=target_index, **kwargs)


def run_benchmark():
    parser = argparse.ArgumentParser(description="Benchmark MPM on Blackwell GPU")
    parser.add_argument(
        "--weights",
        type=str,
        default="Outputs/Experiment_1_variants/Weights/densenet161_exp1a_variant1_best.pth",
        help="Path to DenseNet-161 weights",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="Image_Authenticity_prediction/Dataset/AIGCIQA2023",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--sigmas",
        nargs="+",
        type=int,
        default=[4, 8, 16],
        help="Occluder scale sizes",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Running on Device: {device} ===")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Model: {gpu_name} ({total_vram:.1f} GiB VRAM)")
    else:
        print("ERROR: CUDA not available!")
        sys.exit(1)

    # 1. Instantiate Model
    print("Initializing DenseNet-161 model...")
    model = DenseNet161AuthenticityPredictor(freeze_backbone=True)
    weights_path = Path(args.weights)
    if weights_path.exists():
        print(f"Loading weights from {weights_path}...")
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
    else:
        print(f"Weights not found at {weights_path}; benchmarking with base initialized architecture.")
    model.eval()
    model.to(device)

    # 2. Load test image
    img_tensor = load_test_image(Path(args.dataset_dir)).to(device)

    # 3. Configurations to benchmark
    # Baseline: Batch 64, FP32 (matching GPU 1 setup)
    configs = [
        {"name": "Baseline (GPU1 match)", "batch": 64, "dtype": None, "cudnn": False},
        {"name": "Batch 128 (FP32)", "batch": 128, "dtype": None, "cudnn": False},
        {"name": "Batch 256 (FP32)", "batch": 256, "dtype": None, "cudnn": True},
        {"name": "Batch 512 (FP32)", "batch": 512, "dtype": None, "cudnn": True},
        {"name": "Batch 256 (bfloat16 + cuDNN)", "batch": 256, "dtype": torch.bfloat16, "cudnn": True},
        {"name": "Batch 512 (bfloat16 + cuDNN)", "batch": 512, "dtype": torch.bfloat16, "cudnn": True},
    ]

    results = []
    baseline_map = None

    print(f"\nBenchmarking MPM across {len(configs)} configurations (Scales: {args.sigmas})...\n")

    for cfg in configs:
        b_size = cfg["batch"]
        dtype = cfg["dtype"]
        cudnn_flag = cfg["cudnn"]
        name = cfg["name"]

        torch.backends.cudnn.benchmark = cudnn_flag
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        explainer = AutocastMPM(
            model=model,
            sigma_list=args.sigmas,
            pixel_batch_size=b_size,
            mask_value=0.0,
            use_tqdm=False,
            stride=1,
            autocast_dtype=dtype,
        )

        # Synchronize and run timed execution
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        saliency_map = explainer.generate_map(img_tensor, target_index=0, normalize=True)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3)

        # Numerical comparison against baseline
        if baseline_map is None:
            baseline_map = saliency_map
            corr = 1.0000
            mae = 0.0000
        else:
            # Flatten and compute Pearson correlation and MAE
            flat_base = baseline_map.flatten()
            flat_curr = saliency_map.flatten()
            mae = float(np.mean(np.abs(flat_base - flat_curr)))
            if np.std(flat_base) > 1e-6 and np.std(flat_curr) > 1e-6:
                corr = float(np.corrcoef(flat_base, flat_curr)[0, 1])
            else:
                corr = 1.0

        # Calculate time for full variant (273 images)
        variant_mins = (elapsed * 273) / 60.0

        results.append(
            {
                "name": name,
                "batch": b_size,
                "precision": "BF16" if dtype == torch.bfloat16 else "FP32",
                "time_sec": elapsed,
                "peak_vram_gb": peak_vram,
                "variant_mins": variant_mins,
                "corr": corr,
                "mae": mae,
            }
        )

        print(
            f"  -> [{name:^30}] Time: {elapsed:5.2f}s | Peak VRAM: {peak_vram:4.1f} GiB | "
            f"Est. Variant (273 imgs): {variant_mins:5.1f}m | r={corr:.4f}, MAE={mae:.4f}"
        )

    # Print Summary Table
    base_time = results[0]["time_sec"]
    print("\n" + "=" * 95)
    print(f"{'Configuration':<32} | {'Batch':<5} | {'Prec':<5} | {'Sec/Img':<7} | {'Speedup':<7} | {'Variant (273)':<14} | {'VRAM':<7} | {'Corr (r)':<8}")
    print("-" * 95)
    for r in results:
        speedup = base_time / r["time_sec"]
        print(
            f"{r['name']:<32} | {r['batch']:<5} | {r['precision']:<5} | "
            f"{r['time_sec']:>6.2f}s | {speedup:>6.2f}x | {r['variant_mins']:>10.1f} min | "
            f"{r['peak_vram_gb']:>5.1f} GB | {r['corr']:>8.4f}"
        )
    print("=" * 95 + "\n")


if __name__ == "__main__":
    run_benchmark()
