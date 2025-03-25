# Neural Network Visualization Methodology

- **Importance Score Processing**

  - I inverted the z-scored importance values since my original metric assigned negative values to important channels
  - This ensures channels critical to model performance receive positive weights in my visualizations

- **Feature Map Normalization**

  - I Z-standardized each feature map independently to zero mean and unit variance

- **Weighted Aggregation Approach**
  - I have created new 512 importance scores for each model, in which each of the zeroed out channels is assigned a score of 0. (only the scores of important channels are z-scored and keeped)
  - I multiplied each normalized map by its importance score 
  - Summed across channels to create a weighted heatmap
  - Re-scaling the heatmap to [0,1] for visualization

- **Visualization**

  - Taken the dataset image, and reversed the transformations steps (imageNet normalization) to obtain the original image
  - Applied gamma correction to enhance mid-range values and make subtle patterns more apparent.
  - Scaled the heatmap to the same size as the original image (14x14 -> 224x224)
  - Overlayed the heatmap on the original image to show where the model focuses its attention.
