# Neural Network Visualization Methodology

- **Importance Score Processing**

  - I inverted the z-scored importance values since my original metric assigned negative values to important channels
  - This ensures channels critical to model performance receive positive weights in my visualizations

- **Feature Map Normalization**

  - I Z-standardized each feature map independently to zero mean and unit variance

- **Weighted Aggregation Approach**

  - I multiplied each normalized map by its importance score before summing across channels
  - This highlights spatial locations where important channels show strong activations
  - Creates a visualization that reveals what the model "attends to" when making predictions

- **Visualization**

  - I rescaled the final heatmap to [0,1] for visualization across images.
  - Applied gamma correction to enhance mid-range values and make subtle patterns more apparent.
  - Overlayed the heatmap on the original image to show where the model focuses its attention.

- **Model Comparison**
  - Applied the same importance metric across all models for fair comparison.
  - Visualized the heatmaps side-by-side to compare what each model prioritizes.
