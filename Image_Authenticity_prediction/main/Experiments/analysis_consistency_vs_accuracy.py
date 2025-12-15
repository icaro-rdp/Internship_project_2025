import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
import sys

# ============================================================================
# 1. SETUP & CONFIGURATION
# ============================================================================
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from main.train import test_model
from main.data import IMAGENET_DATASET, NUM_WORKERS, DENSENET_DATASET

# IMPORT YOUR MODEL CLASSES HERE
# (Adjust these names to match your actual class definitions in main/Models)
from main.Models import (
    VGG16AuthenticityPredictor,
    VGG19AuthenticityPredictor,
    ResNet152AuthenticityPredictor,
    DenseNet161AuthenticityPredictor,
    EfficientNetB3AuthenticityPredictor,
    BarlowTwinsAuthenticityPredictor,
)

from main.Utils.logger import set_level, info, warn, error, debug

# --- Functions ---


def get_consistency_scores(
    json_path, model_name, method="gradcam", metric_key="top_percent_iou_15"
):
    info(f"--- [Step 1] Processing Consistency Scores for: {model_name} ---")
    with open(json_path, "r") as f:
        data = json.load(f)

    key = f"{method}_within_model_variants"
    debug(f"JSON Key accessed: '{key}'")

    if key not in data or model_name not in data[key]:
        raise ValueError(f"Data for {model_name} not found in JSON")

    correlations_map = data[key][model_name]["per_image"][metric_key]
    debug(
        f"Found {len(correlations_map)} pairwise comparisons (e.g., '0_vs_1', '0_vs_2')."
    )

    # Identify variants involved
    variants_involved = set()
    for pair_key in correlations_map.keys():
        # Assuming keys are "0_vs_1", "1_vs_2", etc.
        parts = pair_key.split("_vs_")
        if len(parts) == 2:
            variants_involved.add(int(parts[0]))
            variants_involved.add(int(parts[1]))

    debug(f"Variants identified from JSON: {sorted(list(variants_involved))}")

    # Initialize arrays based on the first pair found
    first_pair_key = next(iter(correlations_map.keys()))
    first_pair_data = correlations_map[first_pair_key]
    n_images = len(first_pair_data)

    debug(
        f"Detected dataset size: {n_images} images (based on pair '{first_pair_key}')."
    )

    sum_scores = np.zeros(n_images)
    count_pairs = 0

    for pair_key, scores in correlations_map.items():
        current_scores = np.array(scores)
        if len(current_scores) != n_images:
            warn(
                f"Pair {pair_key} has length {len(current_scores)}, expected {n_images}"
            )
            continue

        sum_scores += current_scores
        count_pairs += 1

    avg_scores = sum_scores / count_pairs

    debug(f"Aggregation complete.")
    debug(f"Output Shape: {avg_scores.shape} (1 value per image)")
    info(f"Consistency scores computed for {count_pairs} pairs.")

    return avg_scores, sorted(list(variants_involved))


def get_prediction_errors(
    model_cls, weights_dir, model_name, variant_ids, device="cuda"
):
    # Look for variant weights (only greedy_pruned ones)
    # We filter specifically for the variants found in the JSON

    weight_files = []

    # Strategy: Try to find exp1b (pruned) for these variants.
    # If NONE found, try exp1a (best).
    # Note: JSON uses 0-indexed variants (0-9), but weight files use 1-indexed (1-10)

    # 1. Try exp1b
    for vid in variant_ids:
        file_vid = vid + 1  # Convert 0-indexed to 1-indexed for file names
        found = list(
            weights_dir.glob(f"{model_name}_exp1b_variant{file_vid}_greedy_pruned.pth")
        )
        if found:
            weight_files.append(found[0])

    # 2. If no exp1b files found, try exp1a
    if not weight_files:
        info(f"No exp1b weights found for {model_name}. Falling back to exp1a...")
        for vid in variant_ids:
            file_vid = vid + 1  # Convert 0-indexed to 1-indexed for file names
            found = list(
                weights_dir.glob(f"{model_name}_exp1a_variant{file_vid}_best.pth")
            )
            if found:
                weight_files.append(found[0])

    info(f"Found {len(weight_files)} weight files matching variants {variant_ids}.")

    if len(weight_files) != len(variant_ids):
        warn(
            f"Mismatch! Expected {len(variant_ids)} weights, found {len(weight_files)}. Missing variants?"
        )

    if model_name == "densenet161":
        test_loader = DataLoader(
            DENSENET_DATASET["train"],
            batch_size=32,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
    else:
        test_loader = DataLoader(
            IMAGENET_DATASET["train"],
            batch_size=32,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

    all_squared_errors = []
    all_labels = []

    for w_path in weight_files:
        debug(f"Evaluating {w_path.name}...")
        model = model_cls(freeze_backbone=False)
        model.load_state_dict(
            torch.load(w_path, map_location=device, weights_only=True)
        )
        model.to(device)
        model.eval()

        metrics = test_model(
            model,
            test_loader,
            torch.nn.MSELoss(),
            device=device,
            return_additional_metrics=True,
        )

        # Absolute deviation |pred - true|
        if isinstance(metrics["preds"], torch.Tensor):
            preds = metrics["preds"].detach().cpu().numpy()
            labels = metrics["labels"].detach().cpu().numpy()
        else:
            preds = metrics["preds"]
            labels = metrics["labels"]

        abs_dev = np.abs(preds - labels)
        all_squared_errors.append(abs_dev)
        all_labels.append(labels)

    if not all_squared_errors:
        raise ValueError(f"No errors calculated for {model_name}. Check weight files.")

    # Average errors across all variants, labels should be same for all
    return np.mean(np.stack(all_squared_errors), axis=0), all_labels[0]


# --- Main Execution ---


def main():
    set_level("DEBUG")
    # Configuration Paths
    JSON_PATH = (
        project_root
        / "main/Experiments/Outputs/Experiment_2_variants/XAI_Maps/Train_Set/experiment_2b_comparison.json"
    )
    WEIGHTS_DIR = (
        project_root / "main/Experiments/Outputs/Experiment_1_variants/Weights"
    )

    # Metric Key Configuration
    METRIC_KEYS = [
        "top_percent_iou_15",
        "correlation",
        "top_percent_iou_5",
        "top_percent_iou_25",
    ]

    # Map JSON keys to Python Classes
    MODEL_MAP = {
        "vgg16": VGG16AuthenticityPredictor,
        "vgg19": VGG19AuthenticityPredictor,
        "resnet152": ResNet152AuthenticityPredictor,
        "densenet161": DenseNet161AuthenticityPredictor,
        "efficientnetb3": EfficientNetB3AuthenticityPredictor,
        "barlowtwins": BarlowTwinsAuthenticityPredictor,
    }

    OUTPUT_DIR = (
        project_root / "main/Experiments/Outputs/Experiment_Cons_vs_Accuracy/Train_Set"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load JSON to find available models
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    available_models = list(data["gradcam_within_model_variants"].keys())
    info(f"Models found in JSON: {available_models}")

    for metric_key in METRIC_KEYS:
        info(f"{'#'*60}")
        info(f"Processing Metric: {metric_key}")
        info(f"{'#'*60}")

        results_summary = {}
        csv_data = []

        for model_name in available_models:
            if model_name not in MODEL_MAP:
                warn(f"Skipping {model_name}: No class mapping defined in MODEL_MAP.")
                continue

            info(f"{'='*40}")
            info(f"Processing: {model_name}")
            info(f"{'='*40}")

            try:
                # 1. Get Consistency (X)
                info("Extracting consistency scores...")
                consistency, variant_ids = get_consistency_scores(
                    JSON_PATH, model_name, metric_key=metric_key
                )

                # 2. Get Prediction Error (Y)
                info(f"Calculating prediction errors for variants {variant_ids}...")
                ModelClass = MODEL_MAP[model_name]
                errors, true_labels = get_prediction_errors(
                    ModelClass, WEIGHTS_DIR, model_name, variant_ids
                )

                # 3. Correlate
                corr_err, p_val_err = pearsonr(consistency, errors)
                corr_true, p_val_true = pearsonr(consistency, true_labels)

                debug(f"Correlation (Consistency vs MAE): r={corr_err}, p={p_val_err}")
                debug(
                    f"Correlation (Consistency vs True Labels): r={corr_true}, p={p_val_true}"
                )

                csv_data.append(
                    {
                        "model": model_name,
                        "metric_key": metric_key,
                        "correlation_consistency_vs_mae": corr_err,
                        "p_value_consistency_vs_mae": p_val_err,
                        "correlation_consistency_vs_true_labels": corr_true,
                        "p_value_consistency_vs_true_labels": p_val_true,
                    }
                )

                results_summary[model_name] = {"r": corr_err, "p": p_val_err}

                info(f"-> Correlation (Consistency vs MAE): {corr_err:.4f}")
                info(f"-> P-value: {p_val_err:.4e}")
                info(f"-> Correlation (Consistency vs True Labels): {corr_true:.4f}")
                info(f"-> P-value: {p_val_true:.4e}")

                info(f"{'-'*40}")

            except Exception as e:
                error(f"Failed to process {model_name}: {e}")

        info("=" * 40)
        info(f"FINAL SUMMARY (Consistency vs. Accuracy) - {metric_key}")
        info("=" * 40)
        info(f"{'Model':<15} | {'Correlation':<12}")
        info("-" * 40)
        for model_name, stats in results_summary.items():
            info(f"{model_name:<15} | {stats['r']:<12.4f}")

        # Save to CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = OUTPUT_DIR / f"consistency_vs_accuracy_summary_{metric_key}.csv"
            df.to_csv(csv_path, index=False)
            info(f"Saved summary CSV to {csv_path}")
        else:
            warn(f"No data collected for metric '{metric_key}'. CSV not saved.")


if __name__ == "__main__":
    main()
