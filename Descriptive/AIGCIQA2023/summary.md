
# Image Quality Assessment Dataset: Comprehensive Descriptive Analysis

## 1. Dataset Overview

The analysis examines an image quality assessment dataset containing 2,400 images with two primary metrics:
- **Quality Scores**: Numerical ratings evaluating overall image quality
- **Authenticity Scores**: Numerical ratings assessing image authenticity

Each image is evaluated along these two dimensions, creating a multivariate quality assessment framework.

## 2. Basic Statistical Measures

### 2.1 Quality Scores

| Statistic | Value |
|-----------|-------|
| Count | 2,400 |
| Mean | 49.92 |
| Median | 50.75 |
| Standard Deviation | 9.32 |
| Minimum | 26.63 |
| Maximum | 72.11 |
| 25th Percentile (Q1) | 41.63 |
| 75th Percentile (Q3) | 57.53 |
| Interquartile Range (IQR) | 15.90 |
| Skewness | -0.01 |
| Kurtosis | -1.06 |

### 2.2 Authenticity Scores

| Statistic | Value |
|-----------|-------|
| Count | 2,400 |
| Mean | 49.77 |
| Median | 49.92 |
| Standard Deviation | 8.06 |
| Minimum | 28.42 |
| Maximum | 73.72 |
| 25th Percentile (Q1) | 43.41 |
| 75th Percentile (Q3) | 55.57 |
| Interquartile Range (IQR) | 12.16 |
| Skewness | 0.07 |
| Kurtosis | -0.63 |

### 2.3 Outlier Analysis

No significant outliers were detected in either Quality or Authenticity scores using the z-score method (±3 standard deviations), suggesting a dataset without extreme values.

## 3. Distribution Analysis

### 3.1 Quality Scores Distribution

The Quality scores exhibit a **strong bimodal distribution** with the following characteristics:

- **Distribution Pattern**: Two distinct peaks with a significant valley between them
- **Confidence in Bimodality**: 80%
- **Bimodality Coefficient**: 0.514 (approaching the threshold of 0.555)
- **Kurtosis**: -1.06 (strongly negative, indicating platykurtosis)
- **Hartigan's Dip Test**: p-value of 0.0 (strong evidence against unimodality)
- **Valley Depth Ratio**: 0.374 (substantial separation between modes)
- **Optimal Components**: Gaussian Mixture Model analysis confirms 2 distinct components
- **Mode Locations**: Peaks approximately at 42 and 57

The strongly platykurtic distribution (negative kurtosis) with dual peaks indicates the presence of two distinct subpopulations in the image quality assessment.

### 3.2 Authenticity Scores Distribution

The Authenticity scores show a **primarily unimodal distribution** with slight structural complexity:

- **Distribution Pattern**: Primarily single-peaked with minor secondary structure
- **Confidence in Unimodality**: 60% (40% for bimodality)
- **Bimodality Coefficient**: 0.424 (below the threshold of 0.555)
- **Kurtosis**: -0.63 (moderately negative)
- **Hartigan's Dip Test**: p-value of 0.424 (fails to reject unimodality)
- **Valley Depth Ratio**: 0.070 (minimal separation between potential modes)
- **Optimal Components**: GMM analysis suggests 2 components, but with less distinct separation

The authenticity distribution exhibits a more continuous spread compared to the clearly divided quality scores.

### 3.3 Normality Tests

| Test | Quality Score Result | Authenticity Score Result |
|------|----------------------|---------------------------|
| Shapiro-Wilk | p < 0.0001 (Not Normal) | p < 0.0001 (Not Normal) |
| D'Agostino-Pearson | p < 0.0001 (Not Normal) | p < 0.0001 (Not Normal) |
| Kolmogorov-Smirnov | p < 0.0001 (Not Normal) | p = 0.0001 (Not Normal) |
| Anderson-Darling | Not Normal at 5% level | Not Normal at 5% level |

All normality tests definitively reject the null hypothesis of normality for both measures, confirming non-normal distributions.

## 4. Correlation Analysis

### 4.1 Correlation Metrics

| Correlation Type | Value |
|------------------|-------|
| Pearson Correlation | 0.86 |
| Spearman Rank Correlation | 0.86 |

The strong and nearly identical Pearson and Spearman correlation coefficients (0.86) indicate a **robust, linear relationship** between Quality and Authenticity scores that holds across the entire dataset regardless of parametric assumptions.

# 5. Implications for DNN Training

# Current Approach: Joint MSE Loss

You're currently using MSE loss to predict both quality and authenticity together. While this is a common approach, the bimodal distribution of quality scores introduces significant challenges:

## Problems with MSE for Your Data

### For Quality Scores (Bimodal):

1. **Regression to the Valley**:
   - MSE inherently predicts values near the distribution mean (~50)
   - This falls precisely in the valley between your two modes (~42 and ~57)
   - Result: Your model is likely being trained to predict values that rarely occur in your actual data

2. **Biased Predictions**:
   - Quality predictions will be systematically pulled toward the middle
   - High-quality images (second mode ~57) will be underpredicted
   - Low-quality images (first mode ~42) will be overpredicted

3. **Misleading Evaluation**:
   - A seemingly decent MSE might mask the fact that predictions don't match the actual bimodal distribution
   - Your model may never predict values at the actual peaks where most ground truth values exist

### For Authenticity Scores (Unimodal but Non-Normal):

1. **Better but Still Suboptimal**:
   - Less problematic than for quality scores
   - However, the platykurtic (flat) distribution still violates MSE's normality assumption

