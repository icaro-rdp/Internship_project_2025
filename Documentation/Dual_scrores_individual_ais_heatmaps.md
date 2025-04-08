# Image Authenticity Analysis with Feature Map Importance and Heatmaps

## 1. Feature Map Importance Score Calculation

The first step involves calculating importance scores for each feature map in the final convolutional layer by measuring how predictions change when individual feature maps are zeroed out:

1. **Setup Phase**:
   - Load a VGG-16 model fine-tuned for image authenticity prediction
   - Use a dataset of real images with authenticity scores
   - Target the final convolutional layer (features.28) with 512 channels

2. **Per-Object Importance Score Calculation**:
   - For each image in the test dataset:
     - Get baseline prediction and residual error (absolute difference between prediction and ground truth)
     - For each of the 512 channels in the final convolutional layer:
       - Zero out the channel's weights and bias
       - Get new prediction and residual error with the channel removed
       - Calculate two key metrics:
         - `delta_residual` = change in residual error (error_pruned - error_baseline)
         - `delta_prediction` = change in prediction value (prediction_pruned - prediction_baseline)
       - Restore original weights and bias
     - Store these dual scores for each channel and image

## 2. Score Transformation and Normalization

Before generating heatmaps, importance scores undergo several transformations:

1. **Score Sign Extraction**:
   - Extract the sign of the residual scores using sign function, needed to determine the direction of impact.

2. **Z-score Normalization**:
   - Normalize the absolute values of `delta_prediction` using Z-score normalization:
     - `z_score = (x - mean) / std`
     - This transforms the scores to have a mean of 0 and standard deviation of 1, making them comparable across channels and images.

3. **Combined Transformation**:
   - Use the absolute values of `delta_prediction` to refer to the magnitude of the impact, and the sign of `delta_residual` to refer to the direction of the impact.
   - This combines the normalized prediction impact with the sign of residual impact, ensuring that the final score reflects both the magnitude and direction of the impact.
   - The final importance score combines normalized prediction impact with the sign of residual impact

## 3. Heatmap Generation with AIS and GradCAM

Two visualization techniques are implemented:

### AIS (Activation-based Importance Scores) Implementation:
1. **Forward Hook Only**:
   - Register hook to capture activations from target layer

2. **Heatmap Generation**:
   - Run forward pass to get activations
   - Apply ReLU to activations to focus on positive influences
   - Weight each channel's activation map by pre-calculated importance scores (negative and positive)
   - Sum weighted activations across channels
   - Resize to input image size (224x224) using bilinear interpolation
   - Normalize by maximum absolute value to preserve sign information
   - Result: Heatmap values in range [-1, 1]

## 4. Visualization with Red-Blue Colormap

For the AIS visualization:

1. **Dual-Color Interpretation**:
   - Red-blue colormap ('bwr') is used to represent the heatmap values:
     - Red: Positive values (helpful features that increase realism)
     - Blue: Negative values (features that decrease realism)
     - Intensity: Magnitude of impact

2. **Visualization Process**:
   - Render original image
   - Create red-blue heatmap using TwoSlopeNorm for proper centering at zero
   - Overlay heatmap on image with 60% original image, 40% heatmap weighting
   - Create a only positive heatmap for positive values (red) and a only negative heatmap for negative values (blue)
   
