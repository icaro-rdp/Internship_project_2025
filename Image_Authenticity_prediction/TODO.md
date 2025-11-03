- Refactor the code to and folder structure to make it more modular and easier to read.
- Training with early stopping for the individual models.
- Pruning for the individual models.
- Go trough the GradCAM and MPM explanations for the individual models.
- Add the SD to the comparison plots.
- Add samples of non realistic images in the appendix.


| Model | Example `target_layer` Access | Notes |
| :--- | :--- | :--- |
| **ResNet** (e.g., ResNet50) | `model.layer4` | Targets the entire final convolutional block. For more precision, you can use `model.layer4[-1]` (the last `Bottleneck`). |
| **VGG16** | `model.features[28]` | This is the **last `Conv2d` layer**. Using `model.features[-1]` (which is `MaxPool2d`) is also a common choice. |
| **VGG19** | `model.features[34]` | This is the **last `Conv2d` layer**. Using `model.features[-1]` (which is `MaxPool2d`) is also common. |
| **DenseNet** (e.g., DenseNet121) | `model.features.denseblock4` | Targets the final dense block. `model.features.norm5` (the final BatchNorm) is another valid target. |
| **EfficientNet** | `model.features[-1]` | Targets the final inverted residual block (e.g., `MBConv`) in the `features` module. |
| **BarlowTwins** | `model.backbone.layer4` | **This is key:** `model` is your *final fine-tuned model*. `backbone` is the name of the BarlowTwins-trained feature extractor *inside* your model. (This example assumes a ResNet backbone). |