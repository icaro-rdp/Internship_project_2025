# 1) Decide which models backends to use

Looking at the AIS paper, the authors used the following models for an evaluation:
 - ResNet152
 - DenseNet161
 - InceptionV3
 - BarlowTwins

# 2) Models adaptations and Training

For all the models, the origninal backend weights are freezed, and a regression head is added on top of the model. The regression head is a simple MLP with 2 layers and ReLU activations. The output of the regression head is a single value, which is the predicted score for the input image.
The following modification and training, is performed for each model, in order to be able to use it as a feature extractor for the ais heatmaps generation.

# 3) Feature extraction

Following their suggestions, for all the models, the feature extraction are performed using the following suggestions:

For ResNet-152, DenseNet-161, and InceptionV3, the feature maps from the deepest convolutional layer are extracted. 

Specifically:
*   **ResNet-152** has **2,048x7x7** feature maps in its deepest layer. 
*   **DenseNet-161** has **2208x7x7** feature maps in its deepest layer. 
*   **InceptionV3** has **2,048x8x8** feature maps in its deepest layer. 
*   **Barlow Twins**, has **2,048x7x7** feature maps in its deepest layer.

The feature maps from these model, cotrarely to the vgg16 are less localized, since the 7x7 and 8x8 feature maps are less localized than the 14x14 feature maps from VGG16. Therefore, the scalling to the original image, should be less precise and more distributed.