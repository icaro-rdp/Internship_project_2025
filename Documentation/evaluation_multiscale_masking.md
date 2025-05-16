# Workflow for Comparing multiple Models' pixel-wise Saliency Maps

## General Workflow for N Models

The process involves the following key steps:

### Step 1: Organize and Prepare Saliency Map Data for Each Model

1.  **Input Data Structure**: For each of your N models, pixel wise saliency maps (typically individual image heatmaps in `.npy` format) are stored in a dedicated sub-directory of the model. For example:
    ```
    /Models/
        ├── Model_A/
        │   ├──Saliency_experiment_outputs/
        │   │   ├── Baseline_model/
        │   │   │   ├── numpy_saliency_map/
        │   │   │   │   ├── batch_maps.npy
        │   └── ...
        └── ...
    ```
    Each model's directory contains its respective saliency maps in a batch format (e.g., `batch_maps.npy`). The `batch_maps.npy` file contains all the saliency maps for that model, stacked together.

2.  **Automated Batching**: Run the initial preparation script. This script will:
    * Iterate through each model's sub-directory (Model_A to Model_N).
    * Load all individual saliency map files.
    * Consolidate these maps into a single "batch" file (e.g., `batch_maps.npy`) for each model. This batch file will contain all heatmaps for that specific model, stacked together.
    * This step is performed for all N models, resulting in N `batch_maps.npy` files, one for each model.

### Step 2: Load Data for Comparison

1.  **Load Batched Heatmaps**: For each of the N models, load its corresponding `batch_maps.npy` file. This will give you N sets of batched heatmaps.
2.  **Assign Model Identifiers**: Create a list of names or identifiers for your N models. These names will be used for labeling in the results and visualizations (e.g., `["Model_A_Name", "Model_B_Name", ..., "Model_N_Name"]`).

### Step 3: Compute Pairwise Similarities

1.  **Run Comparison**: Use the core comparison function, providing it with:
    * The list of N loaded batched heatmaps.
    * A selection of desired similarity/dissimilarity metrics (e.g., MSE, Cosine Similarity, SSIM, EMD, Pearson Correlation).
2.  **Metric Calculation**: The function will then compute these metrics for every possible unique pair of models from your set of N models (e.g., Model_A vs. Model_B, Model_A vs. Model_C, ..., Model_B vs. Model_C, ...).
    * This comparison is done image-by-image within the batches, and then summary statistics (like the mean similarity) are calculated for each model pair and each metric.

### Step 4: Visualize Comparison Results

1.  **Generate Similarity Matrices**: Use the visualization function, providing it with:
    * The results from the comparison step (Step 3).
    * The list of N model names/identifiers.
2.  **Display Results**: For each chosen metric, a similarity matrix (heatmap) will be generated. This N x N matrix visually represents the calculated similarity (e.g., mean similarity) between every pair of models. The axes of the matrix will be labeled with your provided model names, making it easy to interpret.

## Resutls 

### 1. Correlation Analysis

1.  **Impact of Pruning on Model Heatmaps:**
    * For most models (BarlowTwins, DenseNet-161, EfficientNet-B3, ResNet-152), pruning has a very minor effect on the generated heatmaps. This is evidenced by the extremely high correlation scores (approximately 0.98 to 0.99) between the baseline and pruned versions of these models. This suggests that, for these architectures, the core features influencing their heatmap outputs are largely preserved after pruning.
    * VGG-16 shows a slightly greater impact from pruning, with a correlation of 0.84 between its baseline and pruned versions. While still high, it indicates some changes in heatmap characteristics.
    * VGG-19 is most affected by pruning, with a correlation of only 0.56 between its baseline and pruned heatmaps. This suggests that pruning significantly alters the feature importance or visual explanations for VGG-19.

2.  **Correlations Within Model Architectures:**
    * As expected, the highest correlations are generally observed between the baseline and pruned versions of the same model.
    * Models of the same architecture family also tend to show some correlation, though this varies. For instance, VGG-16 models (Baseline/Pruned) correlate more strongly with each other than with VGG-19 models.

3.  **Correlations Across Different Model Architectures:**
    * **BarlowTwins:** These models show a high positive correlation with each other (0.99). They exhibit a moderate negative correlation with DenseNet models (around -0.46 to -0.56) and some slight positive correlation with VGG models (around 0.20 to 0.35). Their correlation with EfficientNet and ResNet models is generally low.
    * **DenseNet-161:** These models show strong negative correlations with VGG-16 models (ranging from -0.46 to -0.61). Their correlation with VGG-19 models is weaker and more mixed. Correlations with EfficientNet and ResNet are minimal.
    * **EfficientNet-B3 & ResNet-152:** Both EfficientNet-B3 and ResNet-152 models (baseline and pruned) show very high internal correlation but exhibit low to negligible correlation with most other model families. This suggests their learned feature representations, as visualized by heatmaps, are quite distinct from the other architectures tested. ResNet models show a slight negative correlation with VGG-19 models (around -0.21 to -0.30).
    * **VGG Models:**
        * VGG-16 models have a strong negative correlation with DenseNet models.
        * VGG models show some positive correlation with BarlowTwins models.
        * The correlation between VGG-16 and VGG-19 heatmaps is positive but not very strong (e.g., VGG-16 Pruned vs VGG-19 Pruned is 0.27; VGG-19 Baseline vs VGG-16 Pruned is 0.19).
