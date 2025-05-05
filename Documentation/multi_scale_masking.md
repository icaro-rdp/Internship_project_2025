# Multi-Scale Saliency Mapping Methodologies

This document details two related methodologies for generating saliency maps based on occlusion at multiple scales. The first describes the original approach applied to EEG-image compatibility, and the second presents an adaptation for analyzing image authenticity based on a regression model's output.

## Original Method: Multi-Scale Saliency for EEG-Image Compatibility

This method aims to identify image regions most influential in determining the compatibility score between an Electroencephalography (EEG) signal and a visual stimulus (image). The core principle involves systematically occluding image patches at various scales and measuring the consequent reduction in the compatibility score.

**Procedure:**

1.  **Mask Definition:** For a given pixel coordinate $(x, y)$ and a specific scale parameter $\sigma$, a binary mask $m_{\sigma}(x, y)$ is constructed. This mask possesses a value of zero for all pixels contained within a $\sigma \times \sigma$ window centered at $(x, y)$, and a value of one for all pixels outside this window.

2.  **Mask Application:** The defined mask $m_{\sigma}(x, y)$ is applied to the original image $v$ via element-wise multiplication (Hadamard product, denoted by $\odot$). This operation effectively nullifies the pixel values within the specified $\sigma \times \sigma$ region centered at $(x, y)$, thereby occluding or suppressing that image patch. The masked image is represented as $m_{\sigma}(x, y) \odot v$.

3.  **Compatibility Score Evaluation:** Let $F(e, v)$ represent the compatibility score between an EEG signal $e$ and the original image $v$. The compatibility score is computed for both the original image pair, $F(e, v)$, and the pair involving the masked image, $F(e, m_{\sigma}(x, y) \odot v)$.

4.  **Saliency Calculation (Scale-Specific):** The saliency value for the pixel $(x, y)$ at scale $\sigma$, denoted as $S_{\sigma}(x, y)$, is determined by the difference between the original compatibility score and the score obtained with the masked image:
    $$S_{\sigma}(x, y) = F(e, v) - F(e, m_{\sigma}(x, y) \odot v)$$
    A larger positive value for $S_{\sigma}(x, y)$ indicates a greater decrease in compatibility upon masking the patch, signifying higher importance (saliency) of the occluded region at that scale for the EEG-image pairing.

5.  **Multi-Scale Aggregation:** The process outlined in steps 1-4 is reiterated for a predefined set of distinct scales $\{\sigma_1, \sigma_2, ..., \sigma_N\}$. The final saliency value for a pixel $(x, y)$, denoted as $S(x, y)$, is derived by aggregating the scale-specific saliency values. A common aggregation method is the normalized summation across all evaluated scales:
    $$S(x, y) = \text{Normalize}\left( \sum_{i=1}^{N} S_{\sigma_i}(x, y) \right)$$

## Adapted Method: Multi-Scale Saliency for Image Authenticity Regression

This section describes an adaptation of the multi-scale occlusion methodology for identifying image regions that significantly influence the output of an image authenticity regression model.

**Conceptual Framework:**

* **Core Task:** Given an image authenticity regression model, denoted as $R$, which maps an input image $v$ to a continuous authenticity score $R(v)$. The interpretation of the score (e.g., higher indicates more authentic) depends on the model's definition.
* **Objective:** To determine which spatial regions within image $v$, across multiple scales, exert the most substantial influence on the predicted authenticity score $R(v)$.
* **Saliency Measure Adaptation:** The saliency calculation is modified to reflect the change in the regression model's output score rather than the EEG-image compatibility score. The scale-specific saliency $S_{\sigma}(x, y)$ is defined as:
    $$S_{\sigma}(x, y) = R(v) - R(m_{\sigma}(x, y) \odot v)$$
    Where $R(v)$ is the authenticity score for the original image, and $R(m_{\sigma}(x, y) \odot v)$ is the score for the image with the $\sigma \times \sigma$ patch at $(x, y)$ occluded.

* **Interpretation of $S_{\sigma}(x, y)$:**
    * **High Positive Value:** Occluding this patch significantly *decreases* the predicted authenticity score. This suggests the patch contains features strongly indicative of *authenticity* according to the model $R$.
    * **Value Near Zero:** Occluding this patch yields negligible change in the score, implying the patch has minimal influence on the authenticity prediction.
    * **Negative Value:** Occluding this patch *increases* the predicted authenticity score. This implies the patch contains features interpreted by the model as evidence of *inauthenticity* (e.g., manipulation artifacts); removing them enhances the image's perceived authenticity to the model.

**Implementation Steps:**

1.  **Prerequisites:**
    * A pre-trained image authenticity regression model $R$.
    * The input image $v$ designated for analysis.
    * A predefined set of scales, `sigma_list` (e.g., `[5, 11, 21, 31, 51]`). The selection should consider image resolution and the anticipated scale of relevant features (smaller scales for subtle artifacts, larger scales for broader manipulations).

2.  **Baseline Score Calculation:**
    * Compute the authenticity score for the original, unoccluded image: $score_{original} = R(v)$.

3.  **Saliency Map Initialization:**
    * Initialize a final aggregated saliency map, $S_{final}$, with dimensions matching image $v$, filled with zeros.
    * Optionally, initialize distinct maps $S_{\sigma}$ for each scale $\sigma$ in `sigma_list` to retain scale-specific information.

4.  **Iterate Through Scales:**
    * For each scale $\sigma$ in `sigma_list`:
        a. Initialize the scale-specific saliency map $S_{\sigma}$ (same dimensions as $v$, filled with zeros).
        b. **Iterate Through Pixel Locations:** (Note: This can be computationally intensive. Patch-based iteration or approximations might be employed for efficiency).
            * For each pixel coordinate $(x, y)$:
                i.  **Create Mask:** Generate the binary mask $m_{\sigma}(x, y)$ (a matrix of ones with a $\sigma \times \sigma$ block of zeros centered at $(x, y)$). Ensure proper handling of boundary conditions (e.g., clipping the zero block at image edges).
                ii. **Apply Mask:** Create the masked image $v_{masked} = m_{\sigma}(x, y) \odot v$. Consider the occlusion strategy: setting pixels to zero is straightforward but may introduce artifacts. Alternatives include replacing with a mean color (e.g., dataset mean), local average, or noise. Zeroing aligns with the original method described.
                iii. **Calculate Masked Score:** Compute the authenticity score for the masked image: $score_{masked} = R(v_{masked})$.
                iv. **Calculate Saliency:** Determine the change in score: $saliency\_value = score_{original} - score_{masked}$.
                v.  **Store Saliency:** Assign the calculated saliency value to the corresponding location in the scale-specific map: $S_{\sigma}[y, x] = saliency\_value$. (Note: Matrix indexing often uses `[row, column]`, corresponding to `[y, x]`).

5.  **Aggregate Saliency Maps:**
    * After processing all pixel locations for a given scale $\sigma$, accumulate its contribution into the final map: $S_{final} = S_{final} + S_{\sigma}$ (element-wise addition). Normalization of $S_{\sigma}$ prior to aggregation is an alternative approach. Direct summation inherently weights scales causing larger score fluctuations more heavily.

6.  **Final Normalization (for Visualization):**
    * Upon completing iterations over all scales, normalize the aggregated map $S_{final}$ to a convenient range for visualization (e.g., $[0, 1]$ or $[-1, 1]$).
    * Example (Min-Max Scaling to $[0, 1]$): $S_{normalized} = \frac{S_{final} - \min(S_{final})}{\max(S_{final}) - \min(S_{final})}$
    * Example (Scaling to $[-1, 1]$, preserving sign): $S_{normalized} = \frac{S_{final}}{\max(|S_{final}|)}$

7.  **Visualization:**
    * Display the normalized saliency map $S_{normalized}$, typically as a heatmap. Overlaying the heatmap semi-transparently onto the original image $v$ can provide effective visual context. Employing a diverging colormap (e.g., blue-white-red) is recommended if distinguishing between regions indicative of authenticity (positive saliency) and inauthenticity (negative saliency) is important.

## Scale Selection Rationale

The provided example list of scales, $\sigma_{list} = [3, 5, 9, 17, 33, 65]$, exhibits specific characteristics:

* **Odd Sizes:** All values are odd, ensuring a unique central pixel $(x, y)$ for each occlusion window.
* **Exponential Growth Pattern:** The sequence follows the pattern $2^n + 1$ for $n = 1, 2, 3, 4, 5, 6$.
    * $2^1 + 1 = 3$
    * $2^2 + 1 = 5$
    * $2^3 + 1 = 9$
    * $2^4 + 1 = 17$
    * $2^5 + 1 = 33$
    * $2^6 + 1 = 65$

This approximately exponential scaling offers several advantages:

* **Efficient Coverage:** It allows the analysis to span a broad range of feature sizes (from very local $3 \times 3$ patches to larger $65 \times 65$ regions) using a relatively small number of scales.
* **Multi-Resolution Analysis:** It effectively probes the model's sensitivity to occlusions at different levels of locality, capturing both fine-grained details and broader contextual information.
* **Potential Motivations:** Such scaling might be motivated by factors like the hierarchical structure of convolutional neural networks, the expected distribution of feature sizes in the target domain, or empirical validation demonstrating its effectiveness.