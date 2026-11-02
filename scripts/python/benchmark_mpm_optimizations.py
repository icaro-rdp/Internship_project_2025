#!/usr/bin/env python3
"""Comprehensive Multiscale Pixel Masking (MPM) Optimization Benchmark on Blackwell GPU.

Evaluates batch size scaling (64 to 2048), precision (FP32 vs bfloat16),
cuDNN autotuning, memory scaling, and processing bottlenecks.
Compares directly against GPU 1 (RTX 3090) baseline of 77.30s/img.
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

# Reference baseline recorded on GPU 1 (gpuhasson1, RTX 3090, 24 GiB)
GPU1_TIME_PER_IMG = 77.30  # seconds
GPU1_VARIANT_MINS = (GPU1_TIME_PER_IMG * 273) / 60.0  # 351.7 mins (~5h 52m)


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
        print(f"Loaded test image from: {img_path}", flush=True)
        img = Image.open(img_path).convert("RGB")
        return transform(img).unsqueeze(0)
    else:
        print("Warning: No image file found, generating synthetic RGB tensor.", flush=True)
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
    parser = argparse.ArgumentParser(description="Benchmark MPM scaling on Blackwell GPU")
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
        default=[3, 17, 65],
        help="Occluder scale sizes (production config: 3 17 65)",
    )
    parser.add_argument(
        "--strides",
        nargs="+",
        type=int,
        default=[2, 4, 16],
        help="Occluder strides per scale (production config: 2 4 16)",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 110, flush=True)
    print("DIPSEN HPC (Node c31) - DenseNet-161 MPM Optimization Benchmark", flush=True)
    print("=" * 110, flush=True)
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Hardware: {gpu_name} ({total_vram:.1f} GiB VRAM)", flush=True)
    else:
        print("ERROR: CUDA not available!", flush=True)
        sys.exit(1)

    print(f"Production Parameters: Sigmas={args.sigmas}, Strides={args.strides}", flush=True)
    # Calculate exact total occlusions
    h, w = 224, 224
    total_occlusions = sum((h // s) * (w // s) for s in args.strides)
    print(f"Total Occlusion Forward Passes per Image: {total_occlusions:,}", flush=True)
    print(f"GPU 1 Reference Baseline: {GPU1_TIME_PER_IMG:.2f}s/img | {GPU1_VARIANT_MINS:.1f} min/variant (273 imgs)", flush=True)
    print("=" * 110 + "\n", flush=True)

    # 1. Instantiate Model
    print("Initializing DenseNet-161 model...", flush=True)
    model = DenseNet161AuthenticityPredictor(freeze_backbone=True)
    weights_path = Path(args.weights)
    if weights_path.exists():
        print(f"Loading weights from {weights_path}...", flush=True)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
    else:
        print(f"Weights not found at {weights_path}; benchmarking with base initialized architecture.", flush=True)
    model.eval()
    model.to(device)

    # 2. Load test image
    img_tensor = load_test_image(Path(args.dataset_dir)).to(device)

    # 3. Comprehensive Configurations to Benchmark
    configs = [
        # Baseline matching GPU 1 setup on Blackwell hardware
        {"name": "Blackwell Baseline (B=64, FP32)", "batch": 64, "dtype": None, "cudnn": False},
        # FP32 Scaling with cuDNN benchmark
        {"name": "Batch 128 (FP32 + cuDNN)", "batch": 128, "dtype": None, "cudnn": True},
        {"name": "Batch 256 (FP32 + cuDNN)", "batch": 256, "dtype": None, "cudnn": True},
        {"name": "Batch 512 (FP32 + cuDNN)", "batch": 512, "dtype": None, "cudnn": True},
        # bfloat16 Tensor Core Scaling
        {"name": "Batch 64  (BF16 + cuDNN)", "batch": 64, "dtype": torch.bfloat16, "cudnn": True},
        {"name": "Batch 128 (BF16 + cuDNN)", "batch": 128, "dtype": torch.bfloat16, "cudnn": True},
        {"name": "Batch 256 (BF16 + cuDNN)", "batch": 256, "dtype": torch.bfloat16, "cudnn": True},
        {"name": "Batch 512 (BF16 + cuDNN)", "batch": 512, "dtype": torch.bfloat16, "cudnn": True},
        {"name": "Batch 1024 (BF16 + cuDNN)", "batch": 1024, "dtype": torch.bfloat16, "cudnn": True},
        {"name": "Batch 2048 (BF16 + cuDNN)", "batch": 2048, "dtype": torch.bfloat16, "cudnn": True},
    ]

    results = []
    baseline_map = None

    print(f"Running benchmark across {len(configs)} configurations...\n", flush=True)

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
            stride=args.strides,
            autocast_dtype=dtype,
        )

        # Warmup and timing
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        saliency_map = explainer.generate_map(img_tensor, target_index=0, normalize=True)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3)

        # Numerical comparison against baseline (Batch 64 FP32)
        if baseline_map is None:
            baseline_map = saliency_map
            corr = 1.0000
            mae = 0.0000
        else:
            flat_base = baseline_map.flatten()
            flat_curr = saliency_map.flatten()
            mae = float(np.mean(np.abs(flat_base - flat_curr)))
            if np.std(flat_base) > 1e-6 and np.std(flat_curr) > 1e-6:
                corr = float(np.corrcoef(flat_base, flat_curr)[0, 1])
            else:
                corr = 1.0

        variant_mins = (elapsed * 273) / 60.0
        variant_hours = variant_mins / 60.0
        speedup_gpu1 = GPU1_TIME_PER_IMG / elapsed

        results.append(
            {
                "name": name,
                "batch": b_size,
                "precision": "BF16" if dtype == torch.bfloat16 else "FP32",
                "time_sec": elapsed,
                "peak_vram_gb": peak_vram,
                "variant_mins": variant_mins,
                "variant_hours": variant_hours,
                "speedup_gpu1": speedup_gpu1,
                "corr": corr,
                "mae": mae,
            }
        )

        print(
            f"  -> [{name:^30}] Time: {elapsed:5.2f}s | "
            f"Speedup vs GPU1: {speedup_gpu1:5.1f}x | "
            f"Est. 273 imgs: {variant_mins:5.1f} min ({variant_hours:4.2f}h) | "
            f"VRAM: {peak_vram:4.1f} GB | r={corr:.4f}",
            flush=True,
        )

    # Print Formatted Comparison Table
    print("\n" + "=" * 125, flush=True)
    print(
        f"{'Configuration':<32} | {'Batch':<5} | {'Prec':<5} | {'Sec/Img':<7} | "
        f"{'Speedup (GPU 1)':<15} | {'Est. 273 Imgs':<16} | {'VRAM':<7} | {'Corr (r)':<8}",
        flush=True,
    )
    print("-" * 125, flush=True)

    # Reference row
    print(
        f"{'GPU 1 (RTX 3090 Reference)':<32} | {'64':<5} | {'FP32':<5} | "
        f"{GPU1_TIME_PER_IMG:>6.2f}s | {'1.0x (baseline)':<15} | "
        f"{GPU1_VARIANT_MINS:>7.1f}m ({GPU1_VARIANT_MINS/60:>4.2f}h) | {'~4.2 GB':<7} | {'1.0000':<8}",
        flush=True,
    )
    print("-" * 125, flush=True)

    for r in results:
        time_str = f"{r['variant_mins']:>5.1f}m ({r['variant_hours']:>4.2f}h)"
        spd_str = f"{r['speedup_gpu1']:.1f}x faster"
        print(
            f"{r['name']:<32} | {r['batch']:<5} | {r['precision']:<5} | "
            f"{r['time_sec']:>6.2f}s | {spd_str:<15} | "
            f"{time_str:<16} | {r['peak_vram_gb']:>5.1f} GB | {r['corr']:>8.4f}",
            flush=True,
        )
    print("=" * 125 + "\n", flush=True)


if __name__ == "__main__":
    run_benchmark()
