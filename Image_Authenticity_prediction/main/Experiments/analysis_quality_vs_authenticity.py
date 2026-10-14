"""Quality vs Authenticity descriptive analysis pipeline (Kept, Excluded, Combined)."""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from scipy import stats

logger = logging.getLogger(__name__)

CANONICAL_DATASET_DIR = (
    Path(__file__).resolve().parent.parent.parent / "Dataset" / "AIGCIQA2023"
)

# Vibrant pastel palettes: Blue (Kept), Orange/Amber (Excluded), Green (Combined)
COHORT_STYLES = {
    "Kept": {
        "label": "Kept Photorealistic",
        "box": "#aed6f1",  # Rich pastel sky blue
        "point": "#3498db",  # Medium-bright blue
        "trend": "#c0392b",  # Crimson coral accent
        "cmap": LinearSegmentedColormap.from_list(
            "kept_blues", ["#ffffff", "#d4e6f1", "#5dade2", "#21618c"]
        ),
    },
    "Excluded": {
        "label": "Excluded Non-Photorealistic",
        "box": "#fad7a0",  # Rich pastel amber/peach
        "point": "#e67e22",  # Medium-bright orange
        "trend": "#2874a6",  # Deep slate blue accent
        "cmap": LinearSegmentedColormap.from_list(
            "excluded_oranges", ["#ffffff", "#fdebd0", "#f5b041", "#b95814"]
        ),
    },
    "Combined": {
        "label": "Combined Full Dataset",
        "box": "#a9dfbf",  # Rich pastel sage/mint green
        "point": "#27ae60",  # Medium-bright green
        "trend": "#c0392b",  # Crimson coral accent
        "cmap": LinearSegmentedColormap.from_list(
            "combined_greens", ["#ffffff", "#d5f5e3", "#52be80", "#1e8449"]
        ),
    },
}

OUTLIER_HIGH_Q_LOW_A = "#d9534f"  # Coral red for High-Q / Low-A outliers
OUTLIER_HIGH_A_LOW_Q = "#8e44ad"  # Plum purple for High-A / Low-Q outliers
SOFT_CHARCOAL = "#333333"


# ==============================================================================
# SECTION 1: DATA & ANALYSIS
# ==============================================================================


def load_datasets(dataset_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load Kept, Excluded, and Combined datasets."""
    base_dir = dataset_dir or CANONICAL_DATASET_DIR

    df_kept = pd.read_csv(base_dir / "real_images_annotations.csv")
    df_kept["cohort"] = "Kept"

    df_excl = pd.read_csv(base_dir / "removed_images_details.csv")
    df_excl["cohort"] = "Excluded"

    df_comb = pd.concat([df_kept, df_excl], ignore_index=True)

    return {
        "Kept": df_kept,
        "Excluded": df_excl,
        "Combined": df_comb,
    }


def load_kept_dataset(dataset_dir: Path | None = None) -> pd.DataFrame:
    """Load Kept photorealistic dataset."""
    return load_datasets(dataset_dir)["Kept"]


def load_discarded_dataset(dataset_dir: Path | None = None) -> pd.DataFrame:
    """Load Excluded non-photorealistic dataset labeled as Discarded."""
    df = load_datasets(dataset_dir)["Excluded"].copy()
    df["cohort"] = "Discarded"
    return df


def load_combined_dataset(dataset_dir: Path | None = None) -> pd.DataFrame:
    """Load Combined full dataset."""
    return load_datasets(dataset_dir)["Combined"]


def compute_standardized_discrepancy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute z-scores and standardized discrepancy (z_diff = z_Quality - z_Authenticity)."""
    df_s = df.copy()
    df_s["z_Quality"] = (df_s["Quality"] - df_s["Quality"].mean()) / df_s[
        "Quality"
    ].std()
    df_s["z_Authenticity"] = (
        df_s["Authenticity"] - df_s["Authenticity"].mean()
    ) / df_s["Authenticity"].std()
    df_s["z_diff"] = df_s["z_Quality"] - df_s["z_Authenticity"]
    return df_s


def compute_joint_decile_contingency(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 10x10 joint contingency cross-tabulation matrix."""
    df_c = df.copy()
    if "authenticity_decile" not in df_c.columns:
        df_c = compute_deciles(df_c, "Authenticity", 10, prefix="A")
    if "quality_decile" not in df_c.columns:
        df_c = compute_deciles(df_c, "Quality", 10, prefix="Q")
    return pd.crosstab(
        df_c["authenticity_decile"], df_c["quality_decile"], dropna=False
    )


def compute_deciles(
    df: pd.DataFrame,
    score_col: str,
    num_bins: int = 10,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Add equal-frequency quantile decile column (e.g. A1..A10 or Q1..Q10)."""
    df_out = df.copy()
    p = prefix or score_col[0].upper()
    labels = [f"{p}{i + 1}" for i in range(num_bins)]
    df_out[f"{score_col.lower()}_decile"] = pd.qcut(
        df_out[score_col], q=num_bins, labels=labels, duplicates="drop"
    )
    return df_out


def compute_decile_statistics(
    df: pd.DataFrame,
    bin_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Compute summary statistics of target variable within each decile bin."""
    grouped = df.groupby(bin_col, observed=False)[target_col]
    q25, q75 = grouped.quantile(0.25), grouped.quantile(0.75)
    return pd.DataFrame(
        {
            "N": grouped.count(),
            "mean": grouped.mean(),
            "std": grouped.std(),
            "sem": grouped.sem(),
            "median": grouped.median(),
            "q25": q25,
            "q75": q75,
            "iqr": q75 - q25,
            "min": grouped.min(),
            "max": grouped.max(),
        }
    )


def compute_correlation_metrics(df: pd.DataFrame) -> dict:
    """Compute Pearson and Spearman correlation with 95% CI."""
    x = df["Quality"].to_numpy(dtype=float)
    y = df["Authenticity"].to_numpy(dtype=float)
    n = len(x)

    r, p_r = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)

    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(0.975)
    ci_low, ci_high = np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)

    return {
        "n": n,
        "pearson_r": float(r),
        "pearson_p": float(p_r),
        "pearson_ci_95": (float(ci_low), float(ci_high)),
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
    }


def extract_top_k_dissociations(
    df: pd.DataFrame,
    top_k: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract quadrant-constrained top-K High-Q/Low-A and High-A/Low-Q cases."""
    df_s = df.copy()
    df_s["z_Quality"] = (df_s["Quality"] - df_s["Quality"].mean()) / df_s[
        "Quality"
    ].std()
    df_s["z_Authenticity"] = (
        df_s["Authenticity"] - df_s["Authenticity"].mean()
    ) / df_s["Authenticity"].std()
    df_s["z_diff"] = df_s["z_Quality"] - df_s["z_Authenticity"]

    pool_q = df_s[(df_s["z_Quality"] > 0) & (df_s["z_Authenticity"] < 0)]
    pool_a = df_s[(df_s["z_Authenticity"] > 0) & (df_s["z_Quality"] < 0)]

    top_high_q = pool_q.sort_values("z_diff", ascending=False).head(top_k).copy()
    top_high_q["dissociation_type"] = "High Quality / Low Authenticity"
    top_high_q["rank"] = range(1, len(top_high_q) + 1)

    top_high_a = pool_a.sort_values("z_diff", ascending=True).head(top_k).copy()
    top_high_a["dissociation_type"] = "High Authenticity / Low Quality"
    top_high_a["rank"] = range(1, len(top_high_a) + 1)

    return top_high_q, top_high_a


def extract_boundary_dissociations(
    df: pd.DataFrame,
    top_n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract highest-Q items in A1 and lowest-Q items in A10."""
    df_c = df.copy()
    if "authenticity_decile" not in df_c.columns:
        df_c = compute_deciles(df_c, "Authenticity", 10, prefix="A")

    a1_top_q = (
        df_c[df_c["authenticity_decile"] == "A1"]
        .sort_values("Quality", ascending=False)
        .head(top_n)
        .copy()
    )
    a1_top_q["boundary_type"] = "Lowest Authenticity (A1) with Highest Quality"

    a10_low_q = (
        df_c[df_c["authenticity_decile"] == "A10"]
        .sort_values("Quality", ascending=True)
        .head(top_n)
        .copy()
    )
    a10_low_q["boundary_type"] = "Highest Authenticity (A10) with Lowest Quality"

    return a1_top_q, a10_low_q


def compare_cohort_discrepancies(
    df_kept: pd.DataFrame,
    df_excl: pd.DataFrame,
) -> dict:
    """Compare (Quality - Authenticity) between Kept and Excluded stimuli."""
    diff_k = df_kept["Quality"] - df_kept["Authenticity"]
    diff_e = df_excl["Quality"] - df_excl["Authenticity"]

    t_stat, t_pval = stats.ttest_ind(diff_e, diff_k, equal_var=False)
    mwu_stat, mwu_pval = stats.mannwhitneyu(diff_e, diff_k)

    n_e, n_k = len(diff_e), len(diff_k)
    pooled_sd = np.sqrt(
        ((n_e - 1) * diff_e.var(ddof=1) + (n_k - 1) * diff_k.var(ddof=1))
        / (n_e + n_k - 2)
    )
    cohens_d = (diff_e.mean() - diff_k.mean()) / pooled_sd

    return {
        "kept_diff_mean": float(diff_k.mean()),
        "kept_diff_std": float(diff_k.std()),
        "kept_diff_median": float(diff_k.median()),
        "kept_prop_q_gt_a": float((diff_k > 0).mean()),
        "discarded_diff_mean": float(diff_e.mean()),
        "discarded_diff_std": float(diff_e.std()),
        "discarded_diff_median": float(diff_e.median()),
        "discarded_prop_q_gt_a": float((diff_e > 0).mean()),
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_pval),
        "cohens_d": float(cohens_d),
        "mwu_statistic": float(mwu_stat),
        "mwu_pvalue": float(mwu_pval),
    }


# ==============================================================================
# SECTION 2: MULTI-PANEL PLOTTING
# ==============================================================================


def _setup_plot_theme() -> None:
    """Configure publication aesthetics."""
    sns.set_theme(style="ticks", font_scale=1.08)
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.edgecolor"] = "#444444"
    plt.rcParams["axes.linewidth"] = 0.9
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def _draw_boxplot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    bin_col: str,
    target_col: str,
    style: dict,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Draw pastel decile boxplot with jittered strip points and median trend."""
    sns.boxplot(
        data=df,
        x=bin_col,
        y=target_col,
        color=style["box"],
        width=0.6,
        boxprops={"edgecolor": "#444444", "linewidth": 1.0, "alpha": 0.92},
        whiskerprops={"color": "#444444", "linewidth": 1.0},
        capprops={"color": "#444444", "linewidth": 1.0},
        medianprops={"color": style["trend"], "linewidth": 2.3},
        showfliers=False,
        zorder=2,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x=bin_col,
        y=target_col,
        color=style["point"],
        alpha=0.42,
        size=3.6,
        jitter=0.22,
        zorder=3,
        ax=ax,
    )

    medians = df.groupby(bin_col, observed=False)[target_col].median()
    ax.plot(
        range(len(medians)),
        medians.to_numpy(),
        color=style["trend"],
        linestyle="--",
        linewidth=1.8,
        marker="o",
        markersize=5.0,
        label="Decile Median",
        zorder=4,
    )

    ax.set_title(title, fontweight="bold", fontsize=11, pad=8)
    ax.set_xlabel(xlabel, fontsize=10, labelpad=5)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=5)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=8.5)
    sns.despine(ax=ax, top=True, right=True)


def _draw_scatter_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    top_high_q: pd.DataFrame,
    top_high_a: pd.DataFrame,
    corr: dict,
    style: dict,
    title: str,
) -> None:
    """Draw scatter plot with quadrant mean lines, linear fit, and outliers."""
    ax.axvline(
        df["Authenticity"].mean(),
        color="#888888",
        linestyle=":",
        linewidth=1.1,
        zorder=1,
    )
    ax.axhline(
        df["Quality"].mean(),
        color="#888888",
        linestyle=":",
        linewidth=1.1,
        zorder=1,
    )

    ax.scatter(
        df["Authenticity"],
        df["Quality"],
        color=style["point"],
        alpha=0.48,
        s=28,
        edgecolors="none",
        label=f"Stimuli (N = {len(df):,})",
        zorder=2,
    )

    sns.regplot(
        data=df,
        x="Authenticity",
        y="Quality",
        scatter=False,
        color=SOFT_CHARCOAL,
        line_kws={"linewidth": 1.6, "linestyle": "--", "label": "Linear Fit (95% CI)"},
        ax=ax,
    )

    ax.scatter(
        top_high_q["Authenticity"],
        top_high_q["Quality"],
        facecolors="none",
        edgecolors=OUTLIER_HIGH_Q_LOW_A,
        s=78,
        linewidths=1.8,
        label=f"Top {len(top_high_q)} High-Q / Low-A",
        zorder=5,
    )
    ax.scatter(
        top_high_a["Authenticity"],
        top_high_a["Quality"],
        facecolors="none",
        edgecolors=OUTLIER_HIGH_A_LOW_Q,
        s=78,
        linewidths=1.8,
        label=f"Top {len(top_high_a)} High-A / Low-Q",
        zorder=5,
    )

    ci = corr["pearson_ci_95"]
    stat_text = (
        f"Pearson r = {corr['pearson_r']:.3f} (95% CI [{ci[0]:.2f}, {ci[1]:.2f}])\n"
        f"Spearman ρ = {corr['spearman_rho']:.3f}"
    )
    ax.text(
        0.04,
        0.95,
        stat_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.92,
        },
        zorder=6,
    )

    ax.set_title(title, fontweight="bold", fontsize=11, pad=8)
    ax.set_xlabel("Authenticity Rating (MOS)", fontsize=10, labelpad=5)
    ax.set_ylabel("Quality Rating (MOS)", fontsize=10, labelpad=5)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        fontsize=8,
    )
    sns.despine(ax=ax, top=True, right=True)


def plot_condensed_6panel(
    cohort_data: dict[str, dict],
    output_path: Path,
) -> None:
    """Generate 3x2 condensed figure (Rows: Kept, Excluded, Combined)."""
    _setup_plot_theme()
    fig, axes = plt.subplots(3, 2, figsize=(15, 18), dpi=300)

    panel_letters = [("A", "B"), ("C", "D"), ("E", "F")]
    for row_idx, cohort in enumerate(["Kept", "Excluded", "Combined"]):
        data = cohort_data[cohort]
        df = data["df"]
        style = COHORT_STYLES[cohort]
        let_box, let_scat = panel_letters[row_idx]

        _draw_boxplot_panel(
            ax=axes[row_idx, 0],
            df=df,
            bin_col="authenticity_decile",
            target_col="Quality",
            style=style,
            title=f"{let_box}: {style['label']} — Quality by Authenticity Decile (N = {len(df):,})",
            xlabel="Authenticity Deciles (A1 = Lowest, A10 = Highest)",
            ylabel="Quality Rating (MOS)",
        )
        _draw_scatter_panel(
            ax=axes[row_idx, 1],
            df=df,
            top_high_q=data["top_q"],
            top_high_a=data["top_a"],
            corr=data["corr"],
            style=style,
            title=f"{let_scat}: {style['label']} — Quality vs. Authenticity",
        )

    fig.suptitle(
        "Quality vs. Authenticity Distributions and Dissociations (Kept, Excluded, and Combined)",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.975])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_bidirectional_9panel(
    cohort_data: dict[str, dict],
    output_path: Path,
) -> None:
    """Generate 3x3 bidirectional figure (Rows: Kept, Excluded, Combined)."""
    _setup_plot_theme()
    fig, axes = plt.subplots(3, 3, figsize=(22, 18), dpi=300)

    panel_letters = [("A", "B", "C"), ("D", "E", "F"), ("G", "H", "I")]
    for row_idx, cohort in enumerate(["Kept", "Excluded", "Combined"]):
        data = cohort_data[cohort]
        df = data["df"]
        style = COHORT_STYLES[cohort]
        let_aq, let_qa, let_scat = panel_letters[row_idx]

        _draw_boxplot_panel(
            ax=axes[row_idx, 0],
            df=df,
            bin_col="authenticity_decile",
            target_col="Quality",
            style=style,
            title=f"{let_aq}: {cohort} — Quality by Authenticity Decile (N = {len(df):,})",
            xlabel="Authenticity Deciles (A1 = Lowest, A10 = Highest)",
            ylabel="Quality Rating (MOS)",
        )
        _draw_boxplot_panel(
            ax=axes[row_idx, 1],
            df=df,
            bin_col="quality_decile",
            target_col="Authenticity",
            style=style,
            title=f"{let_qa}: {cohort} — Authenticity by Quality Decile (N = {len(df):,})",
            xlabel="Quality Deciles (Q1 = Lowest, Q10 = Highest)",
            ylabel="Authenticity Rating (MOS)",
        )
        _draw_scatter_panel(
            ax=axes[row_idx, 2],
            df=df,
            top_high_q=data["top_q"],
            top_high_a=data["top_a"],
            corr=data["corr"],
            style=style,
            title=f"{let_scat}: {cohort} — Quality vs. Authenticity Scatter",
        )

    fig.suptitle(
        "Bidirectional Decile Analysis Across Kept, Excluded, and Combined Cohorts",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.975])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_comparative_joint_heatmaps(
    cohort_data: dict[str, dict],
    output_path: Path,
) -> None:
    """Plot 1x3 side-by-side 10x10 joint contingency heatmaps (X=Authenticity, Y=Quality)."""
    _setup_plot_theme()
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5), dpi=300)

    panel_letters = ["A", "B", "C"]
    for col_idx, cohort in enumerate(["Kept", "Excluded", "Combined"]):
        data = cohort_data[cohort]
        ct = data["joint_ct"]
        style = COHORT_STYLES[cohort]

        # Transpose so X = Authenticity (A1..A10) and Y = Quality (Q1..Q10), with Q10 at top
        grid = ct.T.iloc[::-1]
        vmax = int(grid.to_numpy().max())

        sns.heatmap(
            grid,
            annot=True,
            fmt="d",
            cmap=style["cmap"],
            norm=PowerNorm(gamma=0.65, vmin=0, vmax=vmax),
            annot_kws={"fontsize": 9.5, "fontweight": "bold"},
            cbar_kws={"label": "Number of Stimuli"},
            linewidths=0.6,
            linecolor="#dcdcdc",
            ax=axes[col_idx],
        )
        axes[col_idx].set_title(
            f"{panel_letters[col_idx]}: {style['label']} (N = {len(data['df']):,}, r = {data['corr']['pearson_r']:.3f})",
            fontsize=11.5,
            fontweight="bold",
            pad=10,
        )
        axes[col_idx].set_xlabel(
            "Authenticity Deciles (A1 = Lowest, A10 = Highest)",
            fontsize=10,
            labelpad=7,
        )
        axes[col_idx].set_ylabel(
            "Quality Deciles (Q1 = Lowest, Q10 = Highest)",
            fontsize=10,
            labelpad=7,
        )
        axes[col_idx].tick_params(axis="y", rotation=0)

    fig.suptitle(
        "10×10 Joint Decile Contingency Matrices: Quality (Y) vs. Authenticity (X)",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


# ==============================================================================
# SECTION 3: PIPELINE RUNNER & CLI
# ==============================================================================


def run_descriptive_analysis(
    dataset_dir: Path | None = None,
    output_dir: Path | None = None,
    top_k: int = 20,
) -> dict:
    """Execute Quality vs Authenticity analysis across Kept, Excluded, and Combined cohorts."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    out_base = output_dir or (repo_root / "Outputs" / "Descriptive_analysis")
    plots_panel_dir = out_base / "Plots" / "Panels"
    tables_dir = out_base / "Tables"

    plots_panel_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_datasets(dataset_dir)
    cohort_data: dict[str, dict] = {}

    for cohort, df in datasets.items():
        prefix_slug = "discarded" if cohort == "Excluded" else cohort.lower()

        df = compute_deciles(df, "Authenticity", 10, prefix="A")
        df = compute_deciles(df, "Quality", 10, prefix="Q")

        corr = compute_correlation_metrics(df)
        stats_a_q = compute_decile_statistics(df, "authenticity_decile", "Quality")
        stats_q_a = compute_decile_statistics(df, "quality_decile", "Authenticity")
        joint_ct = pd.crosstab(
            df["authenticity_decile"], df["quality_decile"], dropna=False
        )

        top_q, top_a = extract_top_k_dissociations(df, top_k=top_k)
        a1_top_q, a10_low_q = extract_boundary_dissociations(df, top_n=top_k)

        # Save CSV tables
        stats_a_q.to_csv(tables_dir / f"{prefix_slug}_quality_by_auth_deciles.csv")
        stats_q_a.to_csv(tables_dir / f"{prefix_slug}_auth_by_quality_deciles.csv")
        joint_ct.to_csv(tables_dir / f"{prefix_slug}_joint_10x10_deciles.csv")
        pd.concat([top_q, top_a], ignore_index=True).to_csv(
            tables_dir / f"{prefix_slug}_top_{top_k}_dissociations.csv", index=False
        )
        pd.concat([a1_top_q, a10_low_q], ignore_index=True).to_csv(
            tables_dir / f"{prefix_slug}_boundary_dissociations.csv", index=False
        )

        cohort_data[cohort] = {
            "df": df,
            "corr": corr,
            "joint_ct": joint_ct,
            "top_q": top_q,
            "top_a": top_a,
        }

    cohort_comp = compare_cohort_discrepancies(
        cohort_data["Kept"]["df"], cohort_data["Excluded"]["df"]
    )
    pd.DataFrame([cohort_comp]).to_csv(
        tables_dir / "cohort_discrepancy_comparison.csv", index=False
    )

    logger.info("Generating multi-panel figures...")
    plot_condensed_6panel(
        cohort_data,
        output_path=plots_panel_dir / "figure_qa_condensed_4panel.png",
    )
    plot_bidirectional_9panel(
        cohort_data,
        output_path=plots_panel_dir / "figure_qa_bidirectional_6panel.png",
    )
    plot_comparative_joint_heatmaps(
        cohort_data,
        output_path=plots_panel_dir / "figure_joint_10x10_heatmaps.png",
    )

    logger.info("Descriptive analysis complete! Outputs saved to %s", out_base)
    return {
        "kept_corr": cohort_data["Kept"]["corr"],
        "discarded_corr": cohort_data["Excluded"]["corr"],
        "combined_corr": cohort_data["Combined"]["corr"],
        "cohort_comparison": cohort_comp,
        "output_directory": str(out_base),
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run Quality vs Authenticity descriptive analysis"
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_descriptive_analysis(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
