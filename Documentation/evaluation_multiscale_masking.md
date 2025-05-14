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
