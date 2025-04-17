# 1) Decide which models backends to use

Looking at the AIS paper, the authors used the following models for an evaluation:
 - ResNet152
 - DenseNet161
 - VGG19
 - BarlowTwins
 

# 2) Models adaptations and Training

For all the models, the origninal backend weights are freezed, and a regression head is added on top of the model. The regression head is a simple MLP with 2 layers and ReLU activations. The output of the regression head is a single value, which is the predicted score for the input image.
The following modification and training, is performed for each model, in order to be able to use it as a feature extractor for the ais heatmaps generation.

# 3) Feature extraction

Following their suggestions, for all the models, the feature extraction are performed using the following suggestions:

Specifically:
*   **VGG19** has **512x14x14** feature maps in its deepest layer, in the layer `features.28`.
*   **ResNet-152** has **2,048x7x7** feature maps in its deepest layer, in the layer `features.7.2.conv3`'
*   **DenseNet-161** has **48x7x7** feature maps in its deepest layer, in the layer `features.denseblock4.denselayer32.norm1`.
*   **Barlow Twins**, has **2,048x7x7** feature maps in its deepest layer in the layer `features.7.2.conv3`.
*  **EfficientNetB3** has **1536x8x8** feature maps in its deepest layer, in the layer `features.8.0`.

The feature maps from these model, cotrarely to the vgg16 are less localized, since the 7x7 and 8x8 feature maps are less localized than the 14x14 feature maps from VGG16. Therefore, the scalling to the original image, should be less precise and more distributed.

# 4) InceptionV3 exclusion

I have decided to exclude from the evaluation of the ais heatmaps the incpetionv3 model, even if the prediction of the aesthetic score is not affected by the model, the feature maps are less localized than the other models. Therefore, the heatmaps are less interpretable and more distributed, sice the model uses a parallel architecture with multiple branches. 

# 5) Results 

# Neural Network Model Comparison Analysis

## Introduction

This document analyzes the relationships between six different neural network architectures (BarlowTwins, DenseNet161, EfficientNetB3, ResNet152, VGG16, and VGG19) using four different similarity/distance metrics:

1. Cosine Similarity
2. Structural Similarity (SSIM)
3. Earth Mover's Distance (EMD)
4. Mean Squared Error (MSE)

Each metric provides different insights into how the models' internal representations relate to each other, helping us understand the functional similarities and differences between these architectures.

## 1. Cosine Similarity Analysis

Cosine similarity measures the cosine of the angle between two vectors, representing how similar their orientations are in a multi-dimensional space. Values range from -1 (completely opposite) to 1 (exactly the same).

Key observations:
- All models have perfect similarity with themselves (1.0 on diagonal)
- VGG16 and VGG19 show the strongest negative correlation (-0.32), suggesting they capture substantially different features despite being from the same family
- BarlowTwins shows modest positive correlation with VGG19 (0.13) and DenseNet161 (0.10)
- EfficientNetB3 has slight negative correlation with VGG19 (-0.16)
- Most other model pairs show very weak correlations (near zero), indicating generally orthogonal feature representations

These patterns suggest that despite all being image classification models, they develop significantly different internal representations of visual features.

## 2. Structural Similarity (SSIM) Analysis

SSIM evaluates the perceptual similarity between images or, in this case, feature maps. Unlike cosine similarity, SSIM considers structural information, luminance, and contrast patterns. Values range from -1 to 1, with 1 indicating perfect similarity.

Key observations:
- The overall correlation values are generally higher and more positive than cosine similarity
- EfficientNetB3 shows modest positive correlation with DenseNet161 (0.09)
- ResNet152 and VGG19 show some similarity (0.08)
- VGG16 and VGG19 show a slight negative correlation (-0.08)
- Most models maintain weak but positive correlations with each other

This suggests that while the feature directions may differ (as seen in cosine similarity), the structural patterns captured have more commonality across different architectures.

## 3. Earth Mover's Distance (EMD) Analysis

EMD measures the minimum "cost" required to transform one distribution into another. Lower values indicate more similar distributions. In the context of neural networks, it measures how different the feature distributions are between models.

Key observations:
- The diagonal shows perfect similarity (0.0) as expected
- EfficientNetB3 consistently shows the highest distances from other models (0.44-0.49)
- BarlowTwins and VGG19 show the lowest EMD (0.18), indicating more similar feature distributions
- DenseNet161 and VGG16 are also relatively similar (0.23)

The EMD results suggest that EfficientNetB3's feature distribution is the most distinct among the models, while BarlowTwins and VGG19 share more distribution similarities despite their architectural differences.

## 4. Mean Squared Error (MSE) Analysis

MSE measures the average squared difference between corresponding elements. Lower values indicate more similarity. This metric is sensitive to both small and large differences between feature maps.

Key observations:
- All MSE values are relatively high (1.7-2.7), indicating substantial differences between all models
- VGG16 and VGG19 show the highest MSE (2.65), surprisingly suggesting they produce very different feature activations despite architectural similarity
- BarlowTwins and VGG19 have the lowest MSE (1.73)
- BarlowTwins and DenseNet161 also show relatively low MSE (1.81)

The high MSE values across all model pairs confirm that each architecture produces substantially different feature activations, even when they might share structural similarities.

