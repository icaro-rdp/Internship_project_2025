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