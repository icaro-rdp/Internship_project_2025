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

For ResNet-152, DenseNet-161, and InceptionV3, the feature maps from the deepest convolutional layer are extracted, and then **global pooling is applied to these feature maps to generate object embeddings**.

Specifically:
*   **ResNet-152** has **2,048** feature maps in its deepest layer. Global pooling is then used on these maps to obtain the embeddings.
*   **DenseNet-161** has **2208** feature maps in its deepest layer. Similar to ResNet-152, **global pooling** is applied to these feature maps to generate the embeddings.
*   **InceptionV3** has **2,048** feature maps in its deepest layer. The process for obtaining embeddings involves applying **global pooling** to these feature maps.
For **Barlow Twins**, which uses a ResNet-50 architecture (related to ResNet-152), the feature maps from the **last convolutional layer** are extracted. The authors of Barlow Twins suggest using the **global average pooling** method to obtain the embeddings from these feature maps. The Barlow Twins model is trained on ImageNet-1K, and the authors of the paper suggest using the **global average pooling** method to obtain the embeddings from these feature maps.

