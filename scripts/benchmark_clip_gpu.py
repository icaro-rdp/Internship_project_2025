"""Benchmarking script for CLIP ViT-B/16 training and Grad-CAM throughput.

Measures hardware specifications, model loading latency, peak VRAM allocation,
training step throughput (forward + backward on head + optimizer step), and
Grad-CAM attribution latency across batch sizes.
"""

import argparse
import json
import platform
import socket
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPVisionModelWithProjection


def get_system_and_gpu_info() -> dict[str, str | int | float | bool]:
    """Collect host hardware and PyTorch CUDA environment metadata.

    Returns:
        Dictionary containing platform, CUDA, and device specifications.
    """
    info: dict[str, str | int | float | bool] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_idx)
        total_vram_gb = props.total_memory / (1024**3)
        info["device_name"] = props.name
        info["compute_capability"] = f"{props.major}.{props.minor}"
        info["total_vram_gb"] = round(total_vram_gb, 2)
        info["multi_processor_count"] = props.multi_processor_count
        info["cuda_runtime_version"] = torch.version.cuda or "unknown"
        info["cudnn_version"] = (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 0
        )
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    else:
        info["device_name"] = "CPU"
        info["compute_capability"] = "N/A"
        info["total_vram_gb"] = 0.0

    return info


def build_clip_regressor(
    device: torch.device,
) -> tuple[CLIPVisionModelWithProjection, nn.Sequential, float]:
    """Load CLIP Vision Model and attach the project regression head.

    Args:
        device: Torch execution device.

    Returns:
        Tuple of (frozen_clip_backbone, trainable_regression_head, load_time_sec).
    """
    t_start = time.perf_counter()
    clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")

    # Freeze backbone as specified in ADR 0001 Step 1
    for param in clip.parameters():
        param.requires_grad = False
    clip.eval().to(device)

    # Attach ADR 0001 Step 1 regression head (512 -> 128 -> 1)
    head = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1),
    ).to(device)
    head.train()

    if device.type == "cuda":
        torch.cuda.synchronize()
    load_time_sec = time.perf_counter() - t_start

    return clip, head, load_time_sec


def benchmark_training_step(
    clip: CLIPVisionModelWithProjection,
    head: nn.Sequential,
    device: torch.device,
    batch_size: int,
    warmup_iters: int = 5,
    bench_iters: int = 25,
) -> dict[str, float]:
    """Benchmark forward pass + backward pass on head + optimizer step.

    Args:
        clip: Frozen CLIP vision backbone.
        head: Downstream regression head.
        device: Execution device.
        batch_size: Number of images per batch.
        warmup_iters: Iterations to execute before measuring.
        bench_iters: Measured iterations.

    Returns:
        Dictionary with timing, throughput, and memory metrics.
    """
    optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    dummy_images = torch.randn(batch_size, 3, 224, 224, device=device)
    dummy_targets = torch.randn(batch_size, 1, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    # Warmup
    for _ in range(warmup_iters):
        optimizer.zero_grad()
        with torch.no_grad():
            outputs = clip(pixel_values=dummy_images)
            features = outputs.image_embeds
        preds = head(features)
        loss = criterion(preds, dummy_targets)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs using CUDA events or high-precision timer
    latencies_ms: list[float] = []

    for _ in range(bench_iters):
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            optimizer.zero_grad()
            with torch.no_grad():
                outputs = clip(pixel_values=dummy_images)
                features = outputs.image_embeds
            preds = head(features)
            loss = criterion(preds, dummy_targets)
            loss.backward()
            optimizer.step()

            end_event.record()
            torch.cuda.synchronize()
            latencies_ms.append(start_event.elapsed_time(end_event))
        else:
            t0 = time.perf_counter()
            optimizer.zero_grad()
            with torch.no_grad():
                outputs = clip(pixel_values=dummy_images)
                features = outputs.image_embeds
            preds = head(features)
            loss = criterion(preds, dummy_targets)
            loss.backward()
            optimizer.step()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    avg_ms = float(sum(latencies_ms) / len(latencies_ms))
    throughput_samples_per_sec = float((batch_size / avg_ms) * 1000.0)

    peak_vram_gb = 0.0
    if device.type == "cuda":
        peak_vram_gb = round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)

    return {
        "batch_size": float(batch_size),
        "mean_latency_ms": round(avg_ms, 2),
        "throughput_images_per_sec": round(throughput_samples_per_sec, 2),
        "peak_vram_gb": peak_vram_gb,
    }


def benchmark_gradcam_step(
    clip: CLIPVisionModelWithProjection,
    head: nn.Sequential,
    device: torch.device,
    warmup_iters: int = 3,
    bench_iters: int = 15,
) -> dict[str, float]:
    """Benchmark ViT Grad-CAM attribution pass (ADR 0001 Step 2).

    Simulates the forward pass, backward hook on layer_norm1, score backward,
    token reshaping (1, 197, 768) -> (1, 768, 14, 14), and bilinear resize.

    Args:
        clip: CLIP vision backbone.
        head: Regression head.
        device: Execution device.
        warmup_iters: Warmup iteration count.
        bench_iters: Measured iteration count.

    Returns:
        Dictionary with single-sample attribution latency and memory.
    """
    target_layer = clip.vision_model.encoder.layers[11].layer_norm1
    captured: dict[str, torch.Tensor] = {}

    def fwd_hook(
        module: nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor
    ) -> None:
        captured["acts"] = out

    def bwd_hook(
        module: nn.Module,
        grad_in: tuple[torch.Tensor | None, ...],
        grad_out: tuple[torch.Tensor | None, ...],
    ) -> None:
        if grad_out[0] is not None:
            captured["grads"] = grad_out[0]

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    dummy_image = torch.randn(1, 3, 224, 224, device=device)

    def run_one() -> torch.Tensor:
        img = dummy_image.clone().detach().requires_grad_(True)
        clip.zero_grad()
        head.zero_grad()
        outputs = clip(pixel_values=img)
        preds = head(outputs.image_embeds)
        score = preds[0, 0]
        score.backward()

        # Token reshape: (1, 197, 768) -> (1, 768, 14, 14)
        acts = captured["acts"][:, 1:, :].reshape(1, 14, 14, 768).permute(0, 3, 1, 2)
        grads = captured["grads"][:, 1:, :].reshape(1, 14, 14, 768).permute(0, 3, 1, 2)
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = F.relu(torch.sum(weights * acts, dim=1, keepdim=True))
        return F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)

    # Warmup
    for _ in range(warmup_iters):
        _ = run_one()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    latencies_ms: list[float] = []
    for _ in range(bench_iters):
        if device.type == "cuda":
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()
            _ = run_one()
            ev_end.record()
            torch.cuda.synchronize()
            latencies_ms.append(ev_start.elapsed_time(ev_end))
        else:
            t0 = time.perf_counter()
            _ = run_one()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    h_fwd.remove()
    h_bwd.remove()

    avg_ms = float(sum(latencies_ms) / len(latencies_ms))
    peak_vram_gb = 0.0
    if device.type == "cuda":
        peak_vram_gb = round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)

    return {
        "gradcam_latency_ms": round(avg_ms, 2),
        "gradcam_throughput_images_per_sec": round(1000.0 / avg_ms, 2),
        "gradcam_peak_vram_gb": peak_vram_gb,
    }


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark GPU environment for CLIP ViT training and Grad-CAM."
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="Batch sizes to evaluate (default: 16 32 64)",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1368,
        help="Sample count in dataset to estimate wall times (default: 1368)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs per variant (default: 50)",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=10,
        help="Number of Monte Carlo variants for Exp 1A (default: 10)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to save JSON benchmark summary",
    )
    return parser.parse_args()


def main() -> None:
    """Run environment detection and benchmark suite."""
    args = parse_arguments()

    print("=" * 72)
    print(" CLIP ViT-B/16 HARDWARE & TRAINING BENCHMARK")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys_info = get_system_and_gpu_info()

    print(f"Hostname       : {sys_info['hostname']}")
    print(f"Device         : {sys_info['device_name']}")
    if sys_info["cuda_available"]:
        print(f"Total VRAM     : {sys_info['total_vram_gb']} GB")
        print(f"Compute Cap    : {sys_info['compute_capability']}")
        print(f"CUDA Driver/RT : {sys_info['cuda_runtime_version']}")
        print(f"PyTorch Vers   : {sys_info['torch_version']}")
        print(f"BF16 Supported : {sys_info['bf16_supported']}")
    else:
        print("CUDA WARNING   : GPU is NOT available. Running on CPU.")
    print("-" * 72)

    print("Loading CLIP Vision Model ('openai/clip-vit-base-patch16')...")
    clip, head, load_time = build_clip_regressor(device)
    print(f"Model loaded in: {load_time:.2f}s")
    print("-" * 72)

    # 1. Training step benchmark
    print("Running Training Throughput Benchmark (Exp 1A: Frozen Backbone)...")
    batch_results: list[dict[str, float]] = []

    for bs in args.batch_sizes:
        try:
            res = benchmark_training_step(clip, head, device, batch_size=bs)
            batch_results.append(res)
            print(
                f"  Batch Size {bs:2d} | "
                f"Latency: {res['mean_latency_ms']:6.2f} ms/batch | "
                f"Throughput: {res['throughput_images_per_sec']:7.2f} img/s | "
                f"Peak VRAM: {res['peak_vram_gb']:5.2f} GB"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  Batch Size {bs:2d} | OUT OF MEMORY (OOM)")
            torch.cuda.empty_cache()

    print("-" * 72)

    # 2. Grad-CAM attribution benchmark
    print("Running Grad-CAM Attribution Benchmark (Exp 2: Token Reshape)...")
    gcam_res = benchmark_gradcam_step(clip, head, device)
    print(
        f"  Single image  | "
        f"Latency: {gcam_res['gradcam_latency_ms']:6.2f} ms/img | "
        f"Throughput: {gcam_res['gradcam_throughput_images_per_sec']:7.2f} img/s | "
        f"Peak VRAM: {gcam_res['gradcam_peak_vram_gb']:5.2f} GB"
    )
    print("-" * 72)

    # 3. Wall clock time projections
    train_samples = int(args.dataset_size * 0.8)  # 80% train split
    best_train_res = (
        max(batch_results, key=lambda x: x["throughput_images_per_sec"])
        if batch_results
        else None
    )

    print("Projected Wall Clock Times (Exp 1A & Exp 2):")
    if best_train_res:
        img_per_sec = best_train_res["throughput_images_per_sec"]
        epoch_sec = train_samples / img_per_sec
        variant_min = (epoch_sec * args.epochs) / 60.0
        ten_variants_min = variant_min * args.variants

        print(
            f"  Optimal Batch Size: {int(best_train_res['batch_size'])} "
            f"({img_per_sec:.1f} img/s)"
        )
        print(
            f"  1 Variant ({args.epochs} epochs, {train_samples} train imgs) : {variant_min:.1f} minutes"
        )
        print(
            f"  {args.variants} Variants (Exp 1A full ensemble)         : {ten_variants_min:.1f} minutes"
        )

    gcam_total_sec = (args.dataset_size * gcam_res["gradcam_latency_ms"]) / 1000.0
    print(
        f"  Grad-CAM full dataset ({args.dataset_size} images)     : "
        f"{gcam_total_sec:.1f} seconds ({gcam_total_sec / 60.0:.2f} min)"
    )
    print("=" * 72)

    full_report = {
        "system_info": sys_info,
        "model_load_time_sec": round(load_time, 2),
        "training_benchmarks": batch_results,
        "gradcam_benchmark": gcam_res,
        "projections": {
            "dataset_size": args.dataset_size,
            "train_samples": train_samples,
            "epochs": args.epochs,
            "variants": args.variants,
            "estimated_variant_minutes": round(variant_min, 2)
            if best_train_res
            else None,
            "estimated_ten_variants_minutes": round(ten_variants_min, 2)
            if best_train_res
            else None,
            "estimated_gradcam_dataset_seconds": round(gcam_total_sec, 2),
        },
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)
        print(f"Benchmark JSON saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
