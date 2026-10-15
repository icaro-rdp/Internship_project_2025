"""Unit tests for Quality vs Authenticity descriptive analysis pipeline."""

from pathlib import Path

from Image_Authenticity_prediction.main.Experiments.analysis_quality_vs_authenticity import (
    compare_cohort_discrepancies,
    compute_correlation_metrics,
    compute_decile_statistics,
    compute_deciles,
    compute_joint_decile_contingency,
    compute_standardized_discrepancy,
    extract_boundary_dissociations,
    extract_top_k_dissociations,
    load_combined_dataset,
    load_discarded_dataset,
    load_kept_dataset,
    run_descriptive_analysis,
)


def test_dataset_loaders() -> None:
    """Verify loading of Kept (1,368), Discarded (1,032), and Combined (2,400)."""
    df_kept = load_kept_dataset()
    assert len(df_kept) == 1368
    assert "cohort" in df_kept.columns
    assert (df_kept["cohort"] == "Kept").all()

    df_discarded = load_discarded_dataset()
    assert len(df_discarded) == 1032
    assert "cohort" in df_discarded.columns
    assert (df_discarded["cohort"] == "Discarded").all()

    df_combined = load_combined_dataset()
    assert len(df_combined) == 2400


def test_decile_computation_and_statistics() -> None:
    """Verify 10-bin equal-frequency deciles and summary statistics."""
    df_kept = load_kept_dataset()
    df_dec = compute_deciles(df_kept, "Authenticity", num_bins=10)
    assert "authenticity_decile" in df_dec.columns

    # Check 10 bins exist
    decile_counts = df_dec["authenticity_decile"].value_counts()
    assert len(decile_counts) == 10
    # Equal frequency: each decile should have ~136-137 images
    assert (decile_counts >= 135).all() and (decile_counts <= 138).all()

    stats_df = compute_decile_statistics(df_dec, "authenticity_decile", "Quality")
    assert len(stats_df) == 10
    for col in ["N", "mean", "std", "sem", "median", "iqr", "min", "max"]:
        assert col in stats_df.columns
        assert not stats_df[col].isna().any()


def test_correlation_metrics() -> None:
    """Verify Pearson and Spearman correlation calculation and 95% CIs."""
    df_kept = load_kept_dataset()
    metrics = compute_correlation_metrics(df_kept)

    assert metrics["n"] == 1368
    assert 0.80 <= metrics["pearson_r"] <= 0.90
    assert 0.80 <= metrics["spearman_rho"] <= 0.90
    ci_low, ci_high = metrics["pearson_ci_95"]
    assert ci_low < metrics["pearson_r"] < ci_high


def test_standardized_discrepancy_and_top_k() -> None:
    """Verify z-score calculations and quadrant-constrained top-K dissociations."""
    df_kept = load_kept_dataset()
    df_scored = compute_standardized_discrepancy(df_kept)

    assert "z_Quality" in df_scored.columns
    assert "z_Authenticity" in df_scored.columns
    assert "z_diff" in df_scored.columns
    assert abs(df_scored["z_Quality"].mean()) < 1e-5
    assert abs(df_scored["z_Authenticity"].mean()) < 1e-5

    top_q, top_a = extract_top_k_dissociations(df_kept, top_k=15)
    assert len(top_q) == 15
    assert len(top_a) == 15
    # High-Q / Low-A must have positive z_diff AND positive z_Quality AND negative z_Authenticity
    assert (top_q["z_diff"] > 0).all()
    assert (top_q["z_Quality"] > 0).all()
    assert (top_q["z_Authenticity"] < 0).all()

    # High-A / Low-Q must have negative z_diff AND positive z_Authenticity AND negative z_Quality
    assert (top_a["z_diff"] < 0).all()
    assert (top_a["z_Authenticity"] > 0).all()
    assert (top_a["z_Quality"] < 0).all()


def test_joint_decile_contingency() -> None:
    """Verify 10x10 cross-tabulation matrix structure and row/col sums."""
    df_kept = load_kept_dataset()
    ct = compute_joint_decile_contingency(df_kept)
    assert ct.shape == (10, 10)
    assert ct.values.sum() == 1368
    # A1 x Q10 should be 0
    assert ct.loc["A1", "Q10"] == 0
    # A10 x Q1 should be 0
    assert ct.loc["A10", "Q1"] == 0


def test_boundary_dissociations_and_cohort_comparison() -> None:
    """Verify extraction of boundary decile items and statistical contrast test."""
    df_kept = load_kept_dataset()
    df_disc = load_discarded_dataset()

    a1_top_q, a10_low_q = extract_boundary_dissociations(df_kept, top_n=5)
    assert len(a1_top_q) == 5
    assert len(a10_low_q) == 5
    # Max quality in A1 exceeds min quality in A10
    assert a1_top_q["Quality"].max() > a10_low_q["Quality"].min()

    comparison = compare_cohort_discrepancies(df_kept, df_disc)
    assert comparison["t_statistic"] > 0
    assert comparison["t_pvalue"] < 1e-4
    assert comparison["discarded_prop_q_gt_a"] > comparison["kept_prop_q_gt_a"]


def test_run_descriptive_analysis_pipeline(tmp_path: Path) -> None:
    """Verify complete analysis runner creates multi-panel plots and tables in tmp dir."""
    res = run_descriptive_analysis(output_dir=tmp_path, top_k=5)

    assert "kept_corr" in res
    assert "discarded_corr" in res
    assert "combined_corr" in res
    assert "cohort_comparison" in res

    # Verify tables
    tables_dir = tmp_path / "Tables"
    assert (tables_dir / "kept_quality_by_auth_deciles.csv").exists()
    assert (tables_dir / "kept_top_5_dissociations.csv").exists()
    assert (tables_dir / "kept_joint_10x10_deciles.csv").exists()
    assert (tables_dir / "discarded_joint_10x10_deciles.csv").exists()
    assert (tables_dir / "combined_joint_10x10_deciles.csv").exists()
    assert (tables_dir / "kept_boundary_dissociations.csv").exists()
    assert (tables_dir / "discarded_boundary_dissociations.csv").exists()
    assert (tables_dir / "combined_boundary_dissociations.csv").exists()
    assert (tables_dir / "cohort_discrepancy_comparison.csv").exists()

    # Verify multi-panel plots
    plots_panel_dir = tmp_path / "Plots" / "Panels"
    assert (plots_panel_dir / "figure_qa_condensed_4panel.png").exists()
    assert (plots_panel_dir / "figure_qa_bidirectional_6panel.png").exists()
    assert (plots_panel_dir / "figure_joint_10x10_heatmaps.png").exists()


def test_equal_width_binning_and_pipeline(tmp_path: Path) -> None:
    """Verify equal-width binning, pipeline generation, and 100% contingency sum."""
    df_kept = load_kept_dataset()
    df_bins = compute_deciles(
        df_kept, "Authenticity", num_bins=10, bin_strategy="equal_width"
    )
    assert "authenticity_decile" in df_bins.columns
    assert df_bins["authenticity_decile"].isna().sum() == 0

    counts = df_bins["authenticity_decile"].value_counts().sort_index()
    assert len(counts) == 10
    assert counts.sum() == 1368
    # In equal-width binning, tails have significantly fewer items than central bins
    assert counts["A1"] < 50
    assert counts["A10"] < 50
    assert counts["A6"] > 200

    res = run_descriptive_analysis(
        output_dir=tmp_path, top_k=5, bin_strategy="equal_width"
    )
    assert res["bin_strategy"] == "equal_width"

    # Verify equal-width table filenames
    tables_dir = tmp_path / "Tables"
    assert (tables_dir / "kept_quality_by_auth_bins_equal_width.csv").exists()
    assert (tables_dir / "kept_joint_10x10_bins_equal_width.csv").exists()
    assert (tables_dir / "cohort_discrepancy_comparison_equal_width.csv").exists()

    # Verify equal-width plot filenames
    plots_panel_dir = tmp_path / "Plots" / "Panels"
    assert (plots_panel_dir / "figure_qa_condensed_4panel_equal_width.png").exists()
    assert (plots_panel_dir / "figure_qa_bidirectional_6panel_equal_width.png").exists()
    assert (plots_panel_dir / "figure_joint_10x10_heatmaps_equal_width.png").exists()
