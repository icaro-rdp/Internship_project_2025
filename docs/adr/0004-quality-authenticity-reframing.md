---
status: accepted
---

# Quality vs. Authenticity Empirical Dissociation and Reframing

To address Reviewer 1 Major Comment 2 (#6), we perform an empirical decile discretization and dissociation analysis on human Quality ($Q$) and Authenticity ($A$) Mean Opinion Scores (MOS), establishing how these constructs relate across both kept photorealistic scenes and excluded non-photorealistic stimuli.

## Context & Problem

In Reviewer 1 Major Comment 2 (#6), the reviewer questioned treating image quality and authenticity as separable psychological constructs:
> *"Quality and authenticity ratings are very strongly correlated ($r = 0.84$; Section 5.1)... Thus, the shared variance between the two measures may reflect overlapping components of human judgment rather than two psychologically separable dimensions. Importantly, partialling out quality establishes whether a model predicts variance unique to authenticity; it does not establish that variance shared with quality is irrelevant to authenticity."*

The reviewer invited either:
1. Providing empirical evidence or stronger justification for treating quality-shared variance as distinct from authenticity, or
2. Reframing the partial-correlation analysis as identifying variance *unique* to authenticity after controlling for quality, without claiming shared variance is non-authenticity proxy variance.

## Decisions

- **Empirical Dissociation via Quantile Deciles:** Discretize stimuli into 10 equal-frequency quantiles (`pd.qcut`, $N \approx 137$ per decile for Kept, $N \approx 103$ for Discarded) rather than arbitrary equal-width intervals. This ensures uniform statistical power and reliable IQR estimation across all bins.
- **Bidirectional Decile Analysis:** Evaluate both directions:
  1. Authenticity Deciles $\rightarrow$ Quality distribution (evaluating whether high authenticity guarantees high quality and the extent of within-bin quality spread).
  2. Quality Deciles $\rightarrow$ Authenticity distribution (evaluating whether high quality guarantees high authenticity).
- **10×10 Joint Decile Cross-Tabulation Grid:** Generate full 2D contingency tables and annotated heatmaps cross-tabulating Authenticity deciles ($A_1 \dots A_{10}$) against Quality deciles ($Q_1 \dots Q_{10}$) to evaluate corner cell occupancy ($A_1 \times Q_{10}$ and $A_{10} \times Q_1$) and off-diagonal dispersion.
- **Top-$K$ Dissociation Extraction ($K=20$):** Formally define perceptual dissociation using standardized discrepancy scores:
  - High-Quality / Low-Authenticity: ranked by descending $z_Q - z_A$.
  - High-Authenticity / Low-Quality: ranked by descending $z_A - z_Q$.
  Export the top 20 cases per direction with generative model and prompt metadata.
- **Partitioned Cohorts (Kept vs. Discarded):** Analyze the Kept dataset ($N=1,368$) and Discarded dataset ($N=1,032$) separately with dedicated plots, plus a combined evaluation ($N=2,400$). Excluded stimuli (cartoons, typography, imagination) serve as a validation contrast demonstrating that non-photorealistic prompts decouple rendering fidelity from perceived authenticity.
- **Hybrid Plotting Architecture:** Provide granular individual figures (Style A: boxplots with jittered raw data points) for detailed exploration and rebuttal evidence, alongside clean multi-panel figures (Panels A–D and A–F in Style C) suitable for publication in the manuscript or rebuttal supplement.
- **Code Organization:** Place executable Python analysis code in `Image_Authenticity_prediction/main/descriptive_analysis.py` (structured into analysis, plotting, and runner/CLI sections) and wire into `Image_Authenticity_prediction/__main__.py` under the CLI command `descriptive-qa`. All generated plots, tables, and reports are saved to `Outputs/Descriptive_analysis/`.

## Considered Options

- **Pure Reframing without Empirical Analysis (Rejected):** Yields ground to the reviewer without presenting the empirical reality of the dataset, where extreme range overlaps and dissociation cases exist.
- **Equal-Width Bins (Rejected):** Severely compromises statistical power at distribution tails (e.g., $N=13$ in Bin 1 vs. $N=262$ in Bin 6).
- **Single-Direction Binning ($A \rightarrow Q$ only) (Rejected):** Fails to test whether high technical quality can occur without perceived authenticity, which is central to evaluating AI image generation artifacts.

## Consequences

- Completely satisfies the supervisor meeting plan (12/10/2026) with @uhasson.
- Provides immediate quantitative tables and publication-ready figures for the Reviewer 1 rebuttal letter.
- Establishes a permanent, reproducible CLI workflow (`uv run python -m Image_Authenticity_prediction descriptive-qa`).
