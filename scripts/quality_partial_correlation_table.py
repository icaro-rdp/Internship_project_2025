"""Reproduce the paper table relating authenticity predictions to quality ratings.

For every Experiment 1 model variant, the test-split authenticity predictions
(Â) are correlated with the true authenticity (A) and quality (Q) ratings, and
the partial correlation r(A, Â | Q) is computed. Per architecture, the table
reports mean ± SD across variants and marks partial correlations whose mean
differs significantly from zero (one-sample t-test).

The script runs in two stages:

1. Inference: loads each ``<model>_exp1a_variant<k>_best.pth`` checkpoint and
   predicts the test split. Predictions are cached to ``predictions.csv`` and
   reused on later runs (pass ``--recompute`` to force new inference).
2. Statistics: computes correlations from the cached predictions and writes
   per-variant results, the summary table (CSV) and a LaTeX table.

Example:
    uv run scripts/quality_partial_correlation_table.py --device cuda
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from Image_Authenticity_prediction.main.data import (
    BATCH_SIZE,
    DENSENET_DATASET,
    IMAGENET_DATASET,
)
from Image_Authenticity_prediction.main.Models import (
    BarlowTwinsAuthenticityPredictor,
    DenseNet161AuthenticityPredictor,
    EfficientNetB3AuthenticityPredictor,
    ResNet152AuthenticityPredictor,
    VGG16AuthenticityPredictor,
    VGG19AuthenticityPredictor,
)
from Image_Authenticity_prediction.main.train import test_model
from Image_Authenticity_prediction.main.Utils.cleanup import clear_gpu_memory

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS_DIR = REPO_ROOT / "Outputs" / "Experiment_1_variants" / "Weights"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Outputs" / "Quality_partial_correlation"

# Table row order: weight-file prefix -> (display name, model class, dataset splits)
ARCHITECTURES: dict[str, tuple[str, type[nn.Module], dict[str, Dataset]]] = {
    "barlowtwins": (
        "Barlow Twins",
        BarlowTwinsAuthenticityPredictor,
        IMAGENET_DATASET,
    ),
    "resnet152": ("ResNet152", ResNet152AuthenticityPredictor, IMAGENET_DATASET),
    "densenet161": ("DenseNet161", DenseNet161AuthenticityPredictor, DENSENET_DATASET),
    "efficientnetb3": (
        "EfficientNetB3",
        EfficientNetB3AuthenticityPredictor,
        IMAGENET_DATASET,
    ),
    "vgg19": ("VGG19", VGG19AuthenticityPredictor, IMAGENET_DATASET),
    "vgg16": ("VGG16", VGG16AuthenticityPredictor, IMAGENET_DATASET),
}

VARIANT_PATTERN = re.compile(r"_variant(\d+)_best\.pth$")


# ==============================================================================
# STAGE 1: INFERENCE
# ==============================================================================


def find_variant_weights(weights_dir: Path, model_key: str) -> list[tuple[int, Path]]:
    """List the Experiment 1A checkpoints of one architecture, sorted by variant.

    Args:
        weights_dir: Directory holding the ``.pth`` checkpoints.
        model_key: Weight-file prefix of the architecture (e.g. ``vgg16``).

    Returns:
        Pairs of (variant number, checkpoint path) in ascending variant order.

    Raises:
        FileNotFoundError: If no checkpoint exists for the architecture.
    """
    variant_weights = [
        (int(match.group(1)), path)
        for path in weights_dir.glob(f"{model_key}_exp1a_variant*_best.pth")
        if (match := VARIANT_PATTERN.search(path.name))
    ]
    if not variant_weights:
        raise FileNotFoundError(
            f"No exp1a checkpoints for {model_key} in {weights_dir}"
        )
    return sorted(variant_weights)


def predict_architecture(
    model_key: str,
    weights_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    """Predict test-split authenticity with every variant of one architecture.

    Args:
        model_key: Key of the architecture in ``ARCHITECTURES``.
        weights_dir: Directory holding the ``.pth`` checkpoints.
        device: Device on which inference runs.
        batch_size: Number of images per inference batch.
        num_workers: Number of data-loading worker processes.

    Returns:
        Long-format frame with one row per (variant, test image) holding the
        true authenticity, true quality, and predicted authenticity.

    Raises:
        ValueError: If the labels returned by the model loop do not match the
            annotation rows of the test split.
    """
    display_name, model_cls, dataset_splits = ARCHITECTURES[model_key]
    test_split: Subset = dataset_splits["test"]
    annotations = test_split.dataset.data.iloc[test_split.indices]
    loader = DataLoader(
        test_split, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    variant_frames = []
    for variant, weights_path in find_variant_weights(weights_dir, model_key):
        logger.info("Predicting %s variant %d (%s)", model_key, variant, weights_path)
        model = model_cls(freeze_backbone=False)
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
        results = test_model(
            model, loader, nn.MSELoss(), device=device, return_additional_metrics=True
        )
        del model
        if device.type == "cuda":
            clear_gpu_memory()

        if not np.allclose(results["labels"], annotations["Authenticity"]):
            raise ValueError(
                f"Label order mismatch for {model_key} variant {variant}; "
                "predictions cannot be aligned with quality ratings."
            )
        variant_frames.append(
            pd.DataFrame(
                {
                    "architecture": display_name,
                    "variant": variant,
                    "image_index": test_split.indices,
                    "authenticity": annotations["Authenticity"].to_numpy(),
                    "quality": annotations["Quality"].to_numpy(),
                    "predicted_authenticity": results["preds"],
                }
            )
        )
    return pd.concat(variant_frames, ignore_index=True)


def collect_predictions(
    weights_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    """Predict the test split with every variant of every table architecture.

    Args:
        weights_dir: Directory holding the ``.pth`` checkpoints.
        device: Device on which inference runs.
        batch_size: Number of images per inference batch.
        num_workers: Number of data-loading worker processes.

    Returns:
        Concatenated long-format predictions of all architectures.
    """
    return pd.concat(
        [
            predict_architecture(
                model_key, weights_dir, device, batch_size, num_workers
            )
            for model_key in ARCHITECTURES
        ],
        ignore_index=True,
    )


# ==============================================================================
# STAGE 2: STATISTICS
# ==============================================================================


def partial_correlation(r_xy: float, r_xz: float, r_yz: float) -> float:
    """Compute the first-order partial correlation r(x, y | z).

    Args:
        r_xy: Pearson correlation between x and y.
        r_xz: Pearson correlation between x and the control variable z.
        r_yz: Pearson correlation between y and the control variable z.

    Returns:
        Correlation between x and y after linearly partialling z out of both.
    """
    return (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))


def compute_variant_correlations(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute r(A, Â), r(Q, Â), and r(A, Â | Q) for every model variant.

    Args:
        predictions: Long-format predictions from ``collect_predictions``.

    Returns:
        One row per (architecture, variant) with the three correlations.
    """
    rows = []
    for (architecture, variant), group in predictions.groupby(
        ["architecture", "variant"], sort=False
    ):
        r_a_pred = stats.pearsonr(
            group["authenticity"], group["predicted_authenticity"]
        )
        r_q_pred = stats.pearsonr(group["quality"], group["predicted_authenticity"])
        r_a_q = stats.pearsonr(group["authenticity"], group["quality"])
        rows.append(
            {
                "architecture": architecture,
                "variant": variant,
                "r_A_pred": r_a_pred.statistic,
                "r_Q_pred": r_q_pred.statistic,
                "partial_r_A_pred_given_Q": partial_correlation(
                    r_a_pred.statistic, r_a_q.statistic, r_q_pred.statistic
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_architectures(
    variant_correlations: pd.DataFrame, alpha: float
) -> pd.DataFrame:
    """Aggregate variant correlations into mean ± SD per architecture.

    Args:
        variant_correlations: Output of ``compute_variant_correlations``.
        alpha: Significance level of the one-sample t-test of the partial
            correlations against zero.

    Returns:
        One row per architecture (in table order) with means, sample SDs
        (ddof=1), the t-test result, and a significance flag.
    """
    rows = []
    for architecture, group in variant_correlations.groupby("architecture", sort=False):
        partial = group["partial_r_A_pred_given_Q"]
        t_test = stats.ttest_1samp(partial, popmean=0.0)
        rows.append(
            {
                "architecture": architecture,
                "n_variants": len(group),
                "r_A_pred_mean": group["r_A_pred"].mean(),
                "r_A_pred_sd": group["r_A_pred"].std(),
                "r_Q_pred_mean": group["r_Q_pred"].mean(),
                "r_Q_pred_sd": group["r_Q_pred"].std(),
                "partial_mean": partial.mean(),
                "partial_sd": partial.std(),
                "partial_t": t_test.statistic,
                "partial_p": t_test.pvalue,
                "partial_significant": t_test.pvalue < alpha,
            }
        )
    return pd.DataFrame(rows)


def format_latex_table(summary: pd.DataFrame) -> str:
    """Render the architecture summary as a booktabs LaTeX table.

    Args:
        summary: Output of ``summarize_architectures``.

    Returns:
        LaTeX source of the table, including caption and label.
    """
    body_rows = []
    for row in summary.itertuples(index=False):
        star = "^{*}" if row.partial_significant else ""
        body_rows.append(
            f"{row.architecture} & "
            f"${row.r_A_pred_mean:.2f} \\pm {row.r_A_pred_sd:.2f}$ & "
            f"${row.r_Q_pred_mean:.2f} \\pm {row.r_Q_pred_sd:.2f}$ & "
            f"${row.partial_mean:.2f}{star} \\pm {row.partial_sd:.2f}$ \\\\"
        )
    n_variants = int(summary["n_variants"].max())
    header = (
        r"\textbf{Architecture} & $r(A, \hat{A})$ (PLCC) & $r(Q, \hat{A})$ & "
        r"\textbf{Partial} $r(A, \hat{A} \mid Q)$ \\"
    )
    caption = (
        r"\caption{\textbf{The relation between authenticity predictions and "
        r"quality ratings}. Correlations between true \textsc{Authenticity} "
        r"scores ($A$), predicted \textsc{Authenticity} scores ($\hat{A}$), and "
        r"true \textsc{Quality} scores ($Q$) across model variants. Entries are "
        rf"mean $\pm$ SD across {n_variants} variants per architecture. The "
        r"partial correlation indicates the residual correlation between true "
        r"and predicted \textsc{Authenticity} after partialling out $Q$ from "
        r"both. Stars (*) indicate that the mean partial correlation differs "
        r"significantly from zero (one-sample $t$-test against zero, "
        r"$p < .05$).}"
    )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            header,
            r"\midrule",
            *body_rows,
            r"\bottomrule",
            r"\end{tabular}",
            caption,
            r"\label{tab:quality_partial_correlation}",
            r"\end{table}",
            "",
        ]
    )


# ==============================================================================
# CLI
# ==============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device", default="cuda", help="Inference device, e.g. 'cuda', 'cuda:1'."
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Run inference even if cached predictions exist.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    """Run inference (unless cached) and write the correlation table outputs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.csv"

    if predictions_path.exists() and not args.recompute:
        logger.info("Loading cached predictions from %s", predictions_path)
        predictions = pd.read_csv(predictions_path)
    else:
        predictions = collect_predictions(
            args.weights_dir,
            torch.device(args.device),
            args.batch_size,
            args.num_workers,
        )
        predictions.to_csv(predictions_path, index=False)
        logger.info("Saved predictions to %s", predictions_path)

    variant_correlations = compute_variant_correlations(predictions)
    summary = summarize_architectures(variant_correlations, args.alpha)

    variant_correlations.to_csv(
        args.output_dir / "per_variant_correlations.csv", index=False
    )
    summary.to_csv(args.output_dir / "table_summary.csv", index=False)
    latex_path = args.output_dir / "table_quality_partial_correlation.tex"
    latex_path.write_text(format_latex_table(summary))

    logger.info("Summary:\n%s", summary.round(3).to_string(index=False))
    logger.info("LaTeX table written to %s", latex_path)


if __name__ == "__main__":
    main()
