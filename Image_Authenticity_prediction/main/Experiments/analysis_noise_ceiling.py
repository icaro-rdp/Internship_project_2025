"""Noise ceiling and split-half reliability analysis for perceptual ratings.

This module evaluates the inter-rater reliability and upper bound on model
performance (noise ceiling) for human perceptual evaluations of Quality,
Authenticity, and Match (Correspondence) on the AIGCIQA2023 photographic subset.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DATASET_DIR = PACKAGE_ROOT / "Dataset" / "AIGCIQA2023"
CANONICAL_SCORES_DIR = PACKAGE_ROOT / "Dataset" / "Single_scores"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Outputs" / "Noise_ceiling_analysis"

METRIC_COLUMNS = {
    "Quality": 4,
    "Authenticity": 5,
    "Match": 6,
}


def load_removed_ids(removed_images_file: Path) -> set[int]:
    """Load image unique IDs that must be excluded from analysis.

    Args:
        removed_images_file: Path to text file containing paths of removed images.

    Returns:
        Set of integer image IDs to exclude.
    """
    removed_ids: set[int] = set()
    if not removed_images_file.exists():
        logger.warning(
            "Removed images file not found at %s. No images excluded.",
            removed_images_file,
        )
        return removed_ids

    logger.info("Loading removed IDs from %s", removed_images_file)
    with open(removed_images_file, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            line_str = line.strip()
            if not line_str:
                continue
            stem = Path(line_str).stem
            try:
                removed_ids.add(int(stem))
            except ValueError:
                logger.warning(
                    "Could not parse integer from stem '%s' (line: %s)",
                    stem,
                    line_str,
                )

    logger.info("Loaded %d excluded image IDs.", len(removed_ids))
    return removed_ids


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Safely coerce a pandas Series to numeric values.

    Args:
        series: Pandas Series to convert.

    Returns:
        Series coerced to float numeric types with NaNs for invalid entries.
    """
    return pd.to_numeric(series, errors="coerce")


def read_raw_table(file_path: Path) -> pd.DataFrame:
    """Read CSV or Excel without header, auto-stripping header row if present.

    Args:
        file_path: Path to participant ratings file.

    Returns:
        DataFrame with 0-based integer column indices.

    Raises:
        ValueError: If file suffix is not supported.
    """
    suffix = file_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        raw_data = pd.read_excel(file_path, usecols="B:G", header=None)
        id_col = 0
    elif suffix == ".csv":
        raw_data = pd.read_csv(file_path, header=None)
        id_col = 1
    else:
        raise ValueError(f"Unsupported file type: {file_path.name}")

    if raw_data.shape[0] > 0 and raw_data.shape[1] > id_col:
        # If the first entry in uniqueID column is not a number, it's a header line
        first_id_val = pd.to_numeric(raw_data.iloc[0, id_col], errors="coerce")
        if pd.isna(first_id_val):
            raw_data = raw_data.iloc[1:].reset_index(drop=True)

    return raw_data


def load_participant_file(file_path: Path) -> pd.DataFrame:
    """Standardize single-participant file into canonical columns.

    Extracts column B (pos 1 in CSV, pos 0 in Excel B:G) as uniqueID, and
    Quality, Authenticity, Match.

    Args:
        file_path: Path to CSV or Excel file.

    Returns:
        Cleaned DataFrame with columns: uniqueID, Quality, Authenticity, Match.

    Raises:
        ValueError: If file type is unsupported or column count is insufficient.
    """
    suffix = file_path.suffix.lower()
    raw_data = read_raw_table(file_path)

    if suffix in {".xlsx", ".xls"}:
        if raw_data.shape[1] < 6:
            raise ValueError(
                f"Expected at least 6 columns in Excel file {file_path.name}, got {raw_data.shape[1]}"
            )
        unique_id_series = coerce_numeric(raw_data.iloc[:, 0]).astype("Int64")
        quality_series = coerce_numeric(raw_data.iloc[:, 3])
        authenticity_series = coerce_numeric(raw_data.iloc[:, 4])
        match_series = coerce_numeric(raw_data.iloc[:, 5])
    else:
        if raw_data.shape[1] < 7:
            raise ValueError(
                f"Expected at least 7 columns in CSV file {file_path.name}, got {raw_data.shape[1]}"
            )
        unique_id_series = coerce_numeric(raw_data.iloc[:, 1]).astype("Int64")
        quality_series = coerce_numeric(raw_data.iloc[:, 4])
        authenticity_series = coerce_numeric(raw_data.iloc[:, 5])
        match_series = coerce_numeric(raw_data.iloc[:, 6])

    standardized_df = pd.DataFrame(
        {
            "uniqueID": unique_id_series,
            "Quality": quality_series,
            "Authenticity": authenticity_series,
            "Match": match_series,
        }
    )

    standardized_df = standardized_df.dropna(subset=["uniqueID"])
    standardized_df = standardized_df.dropna(
        subset=["Quality", "Authenticity", "Match"], how="all"
    )
    standardized_df["uniqueID"] = standardized_df["uniqueID"].astype("Int64")
    return standardized_df


def read_metric_series_from_file(
    file_path: Path, metric_col_idx: int, removed_ids: set[int]
) -> pd.Series:
    """Extract a single metric series from a participant file.

    Args:
        file_path: Path to participant ratings file.
        metric_col_idx: 0-based column index of the metric.
        removed_ids: Set of image IDs to exclude.

    Returns:
        Series with image unique IDs as index and metric ratings as values.

    Raises:
        ValueError: If file lacks sufficient columns.
    """
    raw_data = read_raw_table(file_path)

    if raw_data.shape[1] <= max(1, metric_col_idx):
        raise ValueError(
            f"{file_path.name}: expected > {metric_col_idx} columns, got {raw_data.shape[1]}"
        )

    unique_ids = coerce_numeric(raw_data.iloc[:, 1]).astype("Int64")
    metric_values = coerce_numeric(raw_data.iloc[:, metric_col_idx])

    metric_series = pd.Series(
        metric_values.values, index=unique_ids.values, name=file_path.stem
    )
    metric_series = metric_series.dropna()
    metric_series = metric_series[~metric_series.index.isna()]
    metric_series.index = metric_series.index.astype(int)
    metric_series = metric_series[~metric_series.index.isin(removed_ids)]
    return metric_series.groupby(level=0).mean()


def build_metric_matrix(
    files: list[Path], metric_col_idx: int, removed_ids: set[int]
) -> pd.DataFrame:
    """Outer-join all participant series into an images x participants matrix.

    Args:
        files: List of participant score files.
        metric_col_idx: 0-based column index of target metric.
        removed_ids: Set of excluded image IDs.

    Returns:
        DataFrame of shape (n_images, n_participants).

    Raises:
        RuntimeError: If no participant series could be loaded.
    """
    series_list: list[pd.Series] = []
    column_names: list[str] = []

    for file_path in files:
        try:
            series = read_metric_series_from_file(
                file_path, metric_col_idx, removed_ids
            )
            series_list.append(series)
            column_names.append(file_path.stem)
        except (
            ValueError,
            OSError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
        ) as error_msg:
            logger.warning(
                "Skipping %s for matrix build: %s", file_path.name, error_msg
            )

    if not series_list:
        raise RuntimeError("No participant series loaded successfully.")

    matrix = pd.concat(series_list, axis=1)
    matrix.columns = column_names
    return matrix


def calculate_pearson_correlation(series_a: pd.Series, series_b: pd.Series) -> float:
    """Calculate Pearson correlation across shared non-null indices.

    Args:
        series_a: First series.
        series_b: Second series.

    Returns:
        Pearson correlation coefficient, or NaN if fewer than 3 paired samples.
    """
    paired = pd.concat([series_a, series_b], axis=1).dropna()
    if paired.shape[0] < 3:
        return float("nan")
    corr_val = float(paired.iloc[:, 0].corr(paired.iloc[:, 1], method="pearson"))
    return corr_val


def fisher_z_transform(correlation: float) -> float:
    """Apply Fisher z-transformation to correlation coefficient.

    Args:
        correlation: Correlation coefficient in [-1, 1].

    Returns:
        Fisher z-transformed value, or NaN if input is not finite.
    """
    if not np.isfinite(correlation):
        return float("nan")
    clipped = float(np.clip(correlation, -0.999999, 0.999999))
    return float(np.arctanh(clipped))


def inverse_fisher_z(z_value: float) -> float:
    """Convert Fisher z-score back to Pearson correlation coefficient.

    Args:
        z_value: Fisher z-value.

    Returns:
        Pearson correlation coefficient.
    """
    if not np.isfinite(z_value):
        return float("nan")
    return float(np.tanh(z_value))


def spearman_brown_prophecy(correlation: float, multiplier: float = 2.0) -> float:
    """Predict full test reliability from split-half reliability.

    Formula: r_full = (m * r) / (1 + (m - 1) * r)

    Args:
        correlation: Split-half correlation coefficient.
        multiplier: Test length multiplier (default 2.0 for equal halves).

    Returns:
        Estimated full reliability, or NaN if input is invalid.
    """
    if not np.isfinite(correlation):
        return float("nan")
    denominator = 1.0 + (multiplier - 1.0) * correlation
    if abs(denominator) < 1e-12:
        return float("nan")
    return float((multiplier * correlation) / denominator)


def compute_split_half_by_subjects(
    matrix: pd.DataFrame,
    use_strict_complete_case: bool = True,
    subj_cov_thresh: float = 0.90,
    img_cov_thresh: float = 0.90,
    min_half_fraction: float = 0.70,
    n_splits: int = 20,
    seed: int = 12345,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute split-half reliability and noise ceiling across subjects.

    Args:
        matrix: Images x participants rating matrix.
        use_strict_complete_case: Whether to drop any image missing even one rater.
        subj_cov_thresh: Minimum fraction of images a subject must rate if thresholded.
        img_cov_thresh: Minimum fraction of participants an image must have if thresholded.
        min_half_fraction: Fraction of raters in each half required per image.
        n_splits: Number of random split iterations.
        seed: Random seed for split sampling.

    Returns:
        Tuple of (splits_df, summary_dict).

    Raises:
        RuntimeError: If matrix has insufficient observations after filtering.
    """
    coverage = matrix.notna()
    n_images, _ = coverage.shape
    per_subj_counts = coverage.sum(axis=0)

    if use_strict_complete_case:
        filtered_matrix = matrix.dropna(axis=1, how="all")
        filtered_matrix = filtered_matrix.dropna(axis=0, how="any")
        policy = "strict_complete_case"
    else:
        min_images_per_subject = int(np.ceil(subj_cov_thresh * n_images))
        keep_subj = per_subj_counts >= min_images_per_subject
        filtered_matrix = matrix.loc[:, keep_subj]
        n_participants_f = filtered_matrix.shape[1]
        min_subjects_per_image = int(np.ceil(img_cov_thresh * n_participants_f))
        keep_imgs = filtered_matrix.notna().sum(axis=1) >= min_subjects_per_image
        filtered_matrix = filtered_matrix.loc[keep_imgs]
        policy = f"thresholded(subj>={subj_cov_thresh:.0%}, img>={img_cov_thresh:.0%})"

    if filtered_matrix.shape[0] < 3 or filtered_matrix.shape[1] < 2:
        raise RuntimeError(
            f"Not enough data after filtering ({filtered_matrix.shape}). Adjust coverage thresholds."
        )

    rng = default_rng(seed)
    participants = np.array(filtered_matrix.columns)
    n_kept_participants = len(participants)
    half_size_1 = n_kept_participants // 2
    half_size_2 = n_kept_participants - half_size_1
    min_raters_per_half = max(2, int(np.floor(min_half_fraction * half_size_1)))

    records: list[dict[str, Any]] = []
    for split_idx in range(n_splits):
        shuffled_indices = rng.permutation(n_kept_participants)
        group1 = participants[shuffled_indices[:half_size_1]]
        group2 = participants[shuffled_indices[half_size_1:]]

        mean_group1 = filtered_matrix[group1].mean(axis=1, skipna=True)
        mean_group2 = filtered_matrix[group2].mean(axis=1, skipna=True)

        counts1 = filtered_matrix[group1].count(axis=1)
        counts2 = filtered_matrix[group2].count(axis=1)
        valid_mask = (counts1 >= min_raters_per_half) & (counts2 >= min_raters_per_half)

        r_split = calculate_pearson_correlation(
            mean_group1[valid_mask], mean_group2[valid_mask]
        )
        z_split = fisher_z_transform(r_split)
        r_sb = spearman_brown_prophecy(r_split, multiplier=2.0)
        z_sb = fisher_z_transform(r_sb)

        records.append(
            {
                "split_idx": split_idx + 1,
                "n_in_g1": len(group1),
                "n_in_g2": len(group2),
                "min_raters_per_half": int(min_raters_per_half),
                "n_images_used": int(valid_mask.sum()),
                "r_split": r_split,
                "z_split": z_split,
                "r_sb_full": r_sb,
                "z_sb_full": z_sb,
            }
        )

    splits_df = pd.DataFrame(records)
    mean_z_split = float(np.nanmean(splits_df["z_split"].values))
    mean_r_split = inverse_fisher_z(mean_z_split)
    mean_z_sb = float(np.nanmean(splits_df["z_sb_full"].values))
    mean_r_sb = inverse_fisher_z(mean_z_sb)

    noise_ceiling_r_max = (
        float(np.sqrt(mean_r_sb))
        if np.isfinite(mean_r_sb) and mean_r_sb >= 0
        else float("nan")
    )

    summary: dict[str, Any] = {
        "policy": policy,
        "matrix_shape_after_filter": list(filtered_matrix.shape),
        "participants_after_filter": n_kept_participants,
        "half_sizes": [int(half_size_1), int(half_size_2)],
        "min_raters_per_half": int(min_raters_per_half),
        "mean_r_split": mean_r_split,
        "mean_r_sb_full": mean_r_sb,
        "noise_ceiling_r_max": noise_ceiling_r_max,
        "mean_images_used": float(splits_df["n_images_used"].mean()),
        "std_images_used": (
            float(splits_df["n_images_used"].std(ddof=1)) if len(splits_df) > 1 else 0.0
        ),
    }

    return splits_df, summary


def load_and_aggregate_scores(
    files: list[Path],
    removed_ids: set[int],
    bin_edges: np.ndarray,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float],
    list[np.ndarray],
    list[np.ndarray],
]:
    """Load all participant rating files and compute aggregated summary statistics.

    Args:
        files: List of participant CSV/Excel files.
        removed_ids: Set of image IDs to exclude.
        bin_edges: 1D array of rating bin edges for concentration vectors.

    Returns:
        Tuple containing:
            - per_image_df: DataFrame of per-image means and rater counts.
            - per_subject_df: DataFrame of per-subject QA correlations.
            - raw_long_df: DataFrame containing stacked long-format raw ratings.
            - overall_means: Dict of overall mean ratings across all entries.
            - quality_histograms: List of per-subject Quality concentration arrays.
            - authenticity_histograms: List of per-subject Authenticity concentration arrays.
    """
    quality_dict: dict[int, list[float]] = defaultdict(list)
    authenticity_dict: dict[int, list[float]] = defaultdict(list)
    match_dict: dict[int, list[float]] = defaultdict(list)

    per_subject_results: list[dict[str, Any]] = []
    quality_histograms: list[np.ndarray] = []
    authenticity_histograms: list[np.ndarray] = []

    for file_path in files:
        try:
            df_file = load_participant_file(file_path)
            df_subject = df_file[~df_file["uniqueID"].isin(removed_ids)].copy()

            for _, row in df_subject.iterrows():
                uid_raw = row["uniqueID"]
                if pd.isna(uid_raw):
                    continue
                uid = int(uid_raw)

                q_val = row["Quality"]
                a_val = row["Authenticity"]
                m_val = row["Match"]

                if pd.notna(q_val):
                    quality_dict[uid].append(float(q_val))
                if pd.notna(a_val):
                    authenticity_dict[uid].append(float(a_val))
                if pd.notna(m_val):
                    match_dict[uid].append(float(m_val))

            q_subject = coerce_numeric(df_subject["Quality"]).to_numpy(dtype=float)
            a_subject = coerce_numeric(df_subject["Authenticity"]).to_numpy(dtype=float)

            valid_mask = ~np.isnan(q_subject) & ~np.isnan(a_subject)
            q_valid = q_subject[valid_mask]
            a_valid = a_subject[valid_mask]

            pearson_raw = float("nan")
            spearman_raw = float("nan")
            pearson_bin = float("nan")
            spearman_bin = float("nan")

            if q_valid.size > 1 and a_valid.size > 1:
                try:
                    pearson_raw, _ = pearsonr(q_valid, a_valid)
                except (ValueError, ZeroDivisionError, FloatingPointError):
                    pearson_raw = float("nan")
                try:
                    spearman_raw, _ = spearmanr(q_valid, a_valid)
                except (ValueError, ZeroDivisionError, FloatingPointError):
                    spearman_raw = float("nan")

            if q_valid.size > 0 and a_valid.size > 0:
                q_counts, _ = np.histogram(q_valid, bins=bin_edges)
                a_counts, _ = np.histogram(a_valid, bins=bin_edges)

                q_sum = q_counts.sum()
                a_sum = a_counts.sum()

                if q_sum > 0 and a_sum > 0:
                    q_prob = q_counts / float(q_sum)
                    a_prob = a_counts / float(a_sum)
                    quality_histograms.append(q_prob)
                    authenticity_histograms.append(a_prob)

                    try:
                        pearson_bin, _ = pearsonr(q_prob, a_prob)
                    except (ValueError, ZeroDivisionError, FloatingPointError):
                        pearson_bin = float("nan")
                    try:
                        spearman_bin, _ = spearmanr(q_prob, a_prob)
                    except (ValueError, ZeroDivisionError, FloatingPointError):
                        spearman_bin = float("nan")

            per_subject_results.append(
                {
                    "subject_file": file_path.name,
                    "n_QA": int(q_valid.size),
                    "pearson_bin_conc_QA": float(pearson_bin),
                    "spearman_bin_conc_QA": float(spearman_bin),
                    "pearson_raw_QA": float(pearson_raw),
                    "spearman_raw_QA": float(spearman_raw),
                }
            )

        except (
            ValueError,
            OSError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
        ) as error_msg:
            logger.warning(
                "Skipping %s during aggregation: %s", file_path.name, error_msg
            )

    all_ids = sorted(set(quality_dict) | set(authenticity_dict) | set(match_dict))
    image_rows: list[dict[str, Any]] = []

    def safe_mean(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else float("nan")

    for uid in all_ids:
        q_vals = quality_dict.get(uid, [])
        a_vals = authenticity_dict.get(uid, [])
        m_vals = match_dict.get(uid, [])

        image_rows.append(
            {
                "uniqueID": uid,
                "Q_mean": safe_mean(q_vals),
                "Q_n": len(q_vals),
                "A_mean": safe_mean(a_vals),
                "A_n": len(a_vals),
                "M_mean": safe_mean(m_vals),
                "M_n": len(m_vals),
            }
        )

    per_image_df = (
        pd.DataFrame(image_rows).sort_values("uniqueID").reset_index(drop=True)
    )
    per_subject_df = pd.DataFrame(per_subject_results)

    flat_q = [v for vals in quality_dict.values() for v in vals]
    flat_a = [v for vals in authenticity_dict.values() for v in vals]
    flat_m = [v for vals in match_dict.values() for v in vals]

    overall_means = {
        "Quality": safe_mean(flat_q),
        "Authenticity": safe_mean(flat_a),
        "Match": safe_mean(flat_m),
    }

    raw_records: list[dict[str, Any]] = []
    for metric_name, m_dict in [
        ("Quality", quality_dict),
        ("Authenticity", authenticity_dict),
        ("Match", match_dict),
    ]:
        for uid, vals in m_dict.items():
            for val in vals:
                raw_records.append(
                    {"uniqueID": uid, "metric": metric_name, "value": float(val)}
                )
    raw_long_df = pd.DataFrame(raw_records)

    return (
        per_image_df,
        per_subject_df,
        raw_long_df,
        overall_means,
        quality_histograms,
        authenticity_histograms,
    )


def check_out_of_range_ratings(
    raw_long_df: pd.DataFrame,
    per_image_df: pd.DataFrame,
    max_rating: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify ratings or per-image means that exceed the expected rating scale.

    Args:
        raw_long_df: Long-format DataFrame containing single ratings.
        per_image_df: Aggregated DataFrame of per-image means.
        max_rating: Upper limit of valid rating scale (default 5.0).

    Returns:
        Tuple of (raw_flagged_df, means_flagged_df).
    """
    raw_flagged = raw_long_df[raw_long_df["value"] > max_rating].sort_values(
        ["metric", "value"], ascending=[True, False]
    )

    means_mask = (
        (per_image_df["Q_mean"] > max_rating)
        | (per_image_df["A_mean"] > max_rating)
        | (per_image_df["M_mean"] > max_rating)
    )
    means_flagged = per_image_df[means_mask].copy()

    return raw_flagged, means_flagged


def plot_per_image_histograms(
    per_image_df: pd.DataFrame,
    plot_dir: Path,
    bins: int = 20,
) -> list[Path]:
    """Plot distribution histograms of per-image mean ratings.

    Args:
        per_image_df: DataFrame with Q_mean, A_mean, M_mean columns.
        plot_dir: Destination folder for output figures.
        bins: Number of histogram bins.

    Returns:
        List of generated image file paths.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    generated_plots: list[Path] = []

    metrics_to_plot = [
        ("Quality", per_image_df["Q_mean"]),
        ("Authenticity", per_image_df["A_mean"]),
        ("Match", per_image_df["M_mean"]),
    ]

    for metric_name, series in metrics_to_plot:
        valid_series = series.dropna().astype(float)
        if valid_series.empty:
            logger.warning("No data for %s; skipping histogram plot.", metric_name)
            continue

        data_vals = valid_series.values
        as_percent = float(np.nanmax(data_vals)) <= 1.05
        if as_percent:
            vals_to_plot = data_vals * 100.0
            x_label = f"{metric_name} (%)"
            hist_range = (0.0, 100.0)
        else:
            vals_to_plot = data_vals
            x_label = metric_name
            xmin, xmax = float(np.nanmin(vals_to_plot)), float(np.nanmax(vals_to_plot))
            span = xmax - xmin
            hist_range = (
                (xmin - 0.02 * span, xmax + 0.02 * span)
                if span > 0
                else (xmin - 1.0, xmax + 1.0)
            )

        fig, axis = plt.subplots(figsize=(8, 5))
        axis.hist(
            vals_to_plot,
            bins=bins,
            range=hist_range,
            color="#3498db",
            edgecolor="#21618c",
        )
        axis.set_title(f"Histogram of per-image mean {metric_name}")
        axis.set_xlabel(x_label)
        axis.set_ylabel("Count of images")
        axis.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()

        output_file = plot_dir / f"hist_per_image_mean_{metric_name}.png"
        fig.savefig(output_file, dpi=600, bbox_inches="tight")
        plt.close(fig)
        generated_plots.append(output_file)
        logger.info("Saved histogram: %s", output_file)

    return generated_plots


def plot_mean_histograms_across_subjects(
    quality_histograms: list[np.ndarray],
    authenticity_histograms: list[np.ndarray],
    bin_edges: np.ndarray,
    plot_dir: Path,
) -> Path | None:
    """Plot mean rating concentration distributions across subjects.

    Args:
        quality_histograms: List of per-subject Quality concentration vectors.
        authenticity_histograms: List of per-subject Authenticity concentration vectors.
        bin_edges: Rating bin boundary array.
        plot_dir: Destination folder.

    Returns:
        Path of saved figure, or None if no histograms available.
    """
    if not quality_histograms or not authenticity_histograms:
        logger.info(
            "No per-subject histograms available; skipping mean histogram plot."
        )
        return None

    plot_dir.mkdir(parents=True, exist_ok=True)
    q_arr = np.vstack(quality_histograms)
    a_arr = np.vstack(authenticity_histograms)

    q_mean = q_arr.mean(axis=0)
    a_mean = a_arr.mean(axis=0)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(bin_centers, q_mean, marker="o", color="#3498db", label="Quality")
    axis.plot(bin_centers, a_mean, marker="s", color="#e67e22", label="Authenticity")
    axis.set_xlabel("Rating (0-5)")
    axis.set_ylabel("Mean probability per bin")
    axis.set_title(
        f"Mean Rating Concentration Across Subjects ({len(bin_centers)} bins)"
    )
    axis.legend(frameon=True)
    axis.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    output_path = plot_dir / "mean_histograms_quality_authenticity.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved mean concentration plot: %s", output_path)
    return output_path


def run_noise_ceiling_analysis(
    scores_dir: Path | None = None,
    dataset_dir: Path | None = None,
    output_dir: Path | None = None,
    n_splits: int = 20,
    bins: int = 20,
    seed: int = 12345,
    use_strict_complete_case: bool = False,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """Execute complete noise ceiling and split-half reliability pipeline.

    Args:
        scores_dir: Directory containing per-participant score files.
        dataset_dir: Directory containing AIGCIQA2023 annotations.
        output_dir: Destination directory for results and plots.
        n_splits: Number of split-half iterations.
        bins: Number of equal-width bins for rating histograms.
        seed: Random seed for split sampling.
        use_strict_complete_case: Whether to drop images missing any raters.
        generate_plots: Whether to save distribution and summary plots.

    Returns:
        Dictionary containing summary results and file locations.

    Raises:
        FileNotFoundError: If scores directory contains no valid data files.
    """
    resolved_scores_dir = scores_dir or CANONICAL_SCORES_DIR
    resolved_dataset_dir = dataset_dir or CANONICAL_DATASET_DIR
    resolved_out_dir = output_dir or DEFAULT_OUTPUT_DIR
    plot_dir = resolved_out_dir / "Plots"

    resolved_out_dir.mkdir(parents=True, exist_ok=True)
    if generate_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(resolved_scores_dir.glob("*_scores.csv"))
    excel_files = sorted(
        p
        for p in resolved_scores_dir.iterdir()
        if p.suffix.lower() in {".xlsx", ".xls"}
    )
    candidate_files = csv_files if csv_files else excel_files

    if not candidate_files:
        raise FileNotFoundError(
            f"No source files found in {resolved_scores_dir}. Expected '*_scores.csv' or Excel files."
        )

    logger.info(
        "Found %d participant score files in %s",
        len(candidate_files),
        resolved_scores_dir,
    )

    removed_ids_file = resolved_dataset_dir / "removed_images.txt"
    removed_ids = load_removed_ids(removed_ids_file)

    bin_edges = np.linspace(0.0, 5.0, bins + 1)

    (
        per_image_df,
        per_subject_df,
        raw_long_df,
        overall_means,
        q_hists,
        a_hists,
    ) = load_and_aggregate_scores(candidate_files, removed_ids, bin_edges)

    per_image_csv = resolved_out_dir / "per_image_means_QAM.csv"
    per_image_df.to_csv(per_image_csv, index=False)
    logger.info("Saved per-image summary to %s", per_image_csv)

    per_subject_csv = resolved_out_dir / "per_subject_QA_bin_corr.csv"
    per_subject_df.to_csv(per_subject_csv, index=False)
    logger.info("Saved per-subject QA correlations to %s", per_subject_csv)

    raw_gt5, means_gt5 = check_out_of_range_ratings(
        raw_long_df, per_image_df, max_rating=5.0
    )
    if not raw_gt5.empty:
        raw_gt5_csv = resolved_out_dir / "images_values_gt5_raw.csv"
        raw_gt5.to_csv(raw_gt5_csv, index=False)
        logger.warning(
            "Found %d individual ratings > 5.0. Saved to %s", len(raw_gt5), raw_gt5_csv
        )

    if not means_gt5.empty:
        means_gt5_csv = resolved_out_dir / "images_means_gt5.csv"
        means_gt5.to_csv(means_gt5_csv, index=False)
        logger.warning(
            "Found %d per-image means > 5.0. Saved to %s", len(means_gt5), means_gt5_csv
        )

    split_half_results: dict[str, Any] = {}
    for metric_name, col_idx in METRIC_COLUMNS.items():
        logger.info("Computing split-half reliability for %s...", metric_name)
        matrix = build_metric_matrix(candidate_files, col_idx, removed_ids)

        splits_df, summary = compute_split_half_by_subjects(
            matrix,
            use_strict_complete_case=use_strict_complete_case,
            subj_cov_thresh=0.90,
            img_cov_thresh=0.90,
            min_half_fraction=0.70,
            n_splits=n_splits,
            seed=seed,
        )

        metric_csv = (
            resolved_out_dir / f"split_half_reliability_{metric_name.lower()}.csv"
        )
        splits_df.to_csv(metric_csv, index=False)
        logger.info("Saved %s split-half details to %s", metric_name, metric_csv)

        logger.info(
            "%s Reliability: Mean r_split = %.4f | Spearman-Brown r = %.4f | Noise ceiling r_max = %.4f",
            metric_name,
            summary["mean_r_split"],
            summary["mean_r_sb_full"],
            summary["noise_ceiling_r_max"],
        )
        split_half_results[metric_name] = summary

    if generate_plots:
        plot_per_image_histograms(per_image_df, plot_dir, bins=bins)
        plot_mean_histograms_across_subjects(q_hists, a_hists, bin_edges, plot_dir)

    overall_summary: dict[str, Any] = {
        "dataset_statistics": {
            "participant_files_count": len(candidate_files),
            "excluded_ids_count": len(removed_ids),
            "retained_unique_images": len(per_image_df),
        },
        "overall_means": overall_means,
        "split_half_reliability": split_half_results,
        "output_directory": str(resolved_out_dir),
    }

    summary_json_path = resolved_out_dir / "noise_ceiling_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as json_file:
        json.dump(overall_summary, json_file, indent=2)
    logger.info("Saved full summary JSON to %s", summary_json_path)

    return overall_summary


def main() -> None:
    """CLI entrypoint for noise ceiling analysis."""
    parser = argparse.ArgumentParser(
        description="Run noise ceiling and split-half reliability analysis"
    )
    parser.add_argument(
        "--scores-dir",
        type=Path,
        default=None,
        help="Path to Single_scores directory (default: canonical location)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Path to AIGCIQA2023 dataset directory (default: canonical location)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: Outputs/Noise_ceiling_analysis)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=20,
        help="Number of random split-half permutations (default: 20)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of bins for rating distribution analysis (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for split permutation sampling (default: 12345)",
    )
    parser.add_argument(
        "--strict-complete-case",
        action="store_true",
        help="Drop images not evaluated by all participants (complete cases)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_noise_ceiling_analysis(
        scores_dir=args.scores_dir,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        bins=args.bins,
        seed=args.seed,
        use_strict_complete_case=args.strict_complete_case,
        generate_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
