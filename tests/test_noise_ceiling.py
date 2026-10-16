"""Unit tests for noise ceiling and split-half reliability analysis."""

from pathlib import Path

import numpy as np
import pandas as pd

from Image_Authenticity_prediction.main.Experiments.analysis_noise_ceiling import (
    calculate_pearson_correlation,
    check_out_of_range_ratings,
    compute_split_half_by_subjects,
    fisher_z_transform,
    inverse_fisher_z,
    load_participant_file,
    load_removed_ids,
    run_noise_ceiling_analysis,
    spearman_brown_prophecy,
)


def test_load_removed_ids(tmp_path: Path) -> None:
    """Verify parsing of removed image stems and handling of non-integer lines."""
    dummy_file = tmp_path / "removed_images.txt"
    dummy_file.write_text(
        "./path/to/52.png\n./path/to/105.png\n\ninvalid_stem.png\n88\n",
        encoding="utf-8",
    )

    ids = load_removed_ids(dummy_file)
    assert ids == {52, 105, 88}


def test_fisher_z_and_inverse_roundtrip() -> None:
    """Verify Fisher z transform and inverse transform recover original correlations."""
    correlations = [0.1, 0.45, 0.78, 0.95, -0.3]
    for r in correlations:
        z = fisher_z_transform(r)
        r_rec = inverse_fisher_z(z)
        assert np.isclose(r, r_rec, atol=1e-6)

    # Edge cases
    assert np.isnan(fisher_z_transform(float("nan")))
    assert np.isnan(inverse_fisher_z(float("nan")))


def test_spearman_brown_prophecy_calculation() -> None:
    """Verify Spearman-Brown formula: r_full = (2 * r) / (1 + r)."""
    # r = 0.5 -> r_full = 1.0 / 1.5 = 2/3 ≈ 0.666667
    r_sb = spearman_brown_prophecy(0.5, multiplier=2.0)
    assert np.isclose(r_sb, 2.0 / 3.0, atol=1e-6)

    # r = 0.0 -> r_full = 0.0
    assert np.isclose(spearman_brown_prophecy(0.0), 0.0)

    # r = 1.0 -> r_full = 1.0
    assert np.isclose(spearman_brown_prophecy(1.0), 1.0)


def test_calculate_pearson_correlation() -> None:
    """Verify paired Pearson correlation computation and handling of missing pairs."""
    series_a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=[10, 20, 30, 40, 50])
    series_b = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0], index=[10, 20, 30, 40, 50])
    corr = calculate_pearson_correlation(series_a, series_b)
    assert np.isclose(corr, 1.0)

    # Less than 3 common points
    short_b = pd.Series([2.0, 4.0], index=[10, 20])
    assert np.isnan(calculate_pearson_correlation(series_a, short_b))


def test_load_participant_file(tmp_path: Path) -> None:
    """Verify participant CSV parsing with positional columns."""
    csv_file = tmp_path / "01_scores.csv"
    # Columns: 0=idx, 1=uniqueID, 2=name, 3=model, 4=Quality, 5=Authenticity, 6=Match
    csv_file.write_text(
        "1,100,img1.png,ModelA,4.5,3.5,4.0\n"
        "2,200,img2.png,ModelA,2.0,1.5,2.5\n"
        "3,,img3.png,ModelB,3.0,3.0,3.0\n",
        encoding="utf-8",
    )

    df = load_participant_file(csv_file)
    assert list(df.columns) == ["uniqueID", "Quality", "Authenticity", "Match"]
    assert len(df) == 2
    assert df.loc[0, "uniqueID"] == 100
    assert df.loc[0, "Quality"] == 4.5
    assert df.loc[0, "Authenticity"] == 3.5
    assert df.loc[0, "Match"] == 4.0


def test_check_out_of_range_ratings() -> None:
    """Verify detection of ratings greater than rating scale maximum."""
    raw_df = pd.DataFrame(
        [
            {"uniqueID": 1, "metric": "Quality", "value": 4.5},
            {"uniqueID": 2, "metric": "Authenticity", "value": 5.8},
        ]
    )
    per_image_df = pd.DataFrame(
        [
            {"uniqueID": 1, "Q_mean": 4.5, "A_mean": 4.0, "M_mean": 3.0},
            {"uniqueID": 2, "Q_mean": 3.0, "A_mean": 5.2, "M_mean": 4.0},
        ]
    )

    raw_gt5, means_gt5 = check_out_of_range_ratings(
        raw_df, per_image_df, max_rating=5.0
    )
    assert len(raw_gt5) == 1
    assert raw_gt5.iloc[0]["value"] == 5.8
    assert len(means_gt5) == 1
    assert means_gt5.iloc[0]["uniqueID"] == 2


def test_compute_split_half_by_subjects() -> None:
    """Verify split-half sampling on synthetic matrix."""
    rng = np.random.default_rng(42)
    # 20 images x 6 participants
    ground_truth = rng.uniform(1.0, 5.0, size=(20, 1))
    noise = rng.normal(0, 0.2, size=(20, 6))
    matrix_data = ground_truth + noise
    df_matrix = pd.DataFrame(
        matrix_data,
        index=range(1, 21),
        columns=[f"subj_{i:02d}" for i in range(1, 7)],
    )

    splits_df, summary = compute_split_half_by_subjects(
        df_matrix,
        use_strict_complete_case=True,
        min_half_fraction=0.6,
        n_splits=10,
        seed=123,
    )

    assert len(splits_df) == 10
    assert summary["policy"] == "strict_complete_case"
    assert summary["participants_after_filter"] == 6
    assert 0.0 < summary["mean_r_split"] <= 1.0
    assert summary["mean_r_sb_full"] >= summary["mean_r_split"]
    assert summary["noise_ceiling_r_max"] >= summary["mean_r_sb_full"]


def test_end_to_end_noise_ceiling_synthetic(tmp_path: Path) -> None:
    """Verify full noise ceiling pipeline runs and produces expected files."""
    scores_dir = tmp_path / "Single_scores"
    scores_dir.mkdir()
    dataset_dir = tmp_path / "AIGCIQA2023"
    dataset_dir.mkdir()
    out_dir = tmp_path / "Output_Noise_Ceiling"

    # Create removed_images.txt
    removed_file = dataset_dir / "removed_images.txt"
    removed_file.write_text("999.png\n", encoding="utf-8")

    # Create 4 participant CSVs
    for p_id in range(1, 5):
        csv_file = scores_dir / f"{p_id:02d}_scores.csv"
        rows = [
            "1,10,img1.png,M1,4.0,3.5,4.0",
            "2,20,img2.png,M1,2.5,2.0,3.0",
            "3,30,img3.png,M1,3.0,3.0,3.0",
            "4,40,img4.png,M1,4.5,4.0,4.5",
            "5,999,removed.png,M1,1.0,1.0,1.0",
        ]
        csv_file.write_text("\n".join(rows) + "\n", encoding="utf-8")

    summary = run_noise_ceiling_analysis(
        scores_dir=scores_dir,
        dataset_dir=dataset_dir,
        output_dir=out_dir,
        n_splits=5,
        bins=10,
        seed=42,
        use_strict_complete_case=False,
        generate_plots=True,
    )

    assert summary["dataset_statistics"]["participant_files_count"] == 4
    assert summary["dataset_statistics"]["retained_unique_images"] == 4

    # Verify generated files
    assert (out_dir / "per_image_means_QAM.csv").exists()
    assert (out_dir / "per_subject_QA_bin_corr.csv").exists()
    assert (out_dir / "split_half_reliability_quality.csv").exists()
    assert (out_dir / "split_half_reliability_authenticity.csv").exists()
    assert (out_dir / "split_half_reliability_match.csv").exists()
    assert (out_dir / "noise_ceiling_summary.json").exists()
    assert (out_dir / "Plots" / "hist_per_image_mean_Quality.png").exists()
    assert (out_dir / "Plots" / "mean_histograms_quality_authenticity.png").exists()
