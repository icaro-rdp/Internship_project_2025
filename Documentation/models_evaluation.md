# Evaluation of different models on out of distribution data

## What we want to evaluate

With out of distribution data, we mean data that is not part of the training set. This is important to evaluate because it gives us an idea of how well the model generalizes to new data. In this case, we will use test data (extracted by using a reproducible seed) on the original dataset. 
The evaluation will be done using the following metrics:
- MSE (Mean Squared Error): This is a common metric for regression tasks. It measures the average squ
ared difference between the predicted and actual values. A lower MSE indicates a better model.

The evaluation will be done using the following models:
- BarlowTwins: This is a self-supervised model that learns to represent the data in a way that is invariant to certain transformations. It is based on the idea of maximizing the similarity between the representations of two augmented views of the same image while minimizing the redundancy between the representations of different images.
- DenseNet161: This is a convolutional neural network that is designed to be very deep (up to 161 layers). It uses dense connections between layers to improve the flow of information and gradients throughout the network. It is a popular model for image classification tasks.
- ResNet152: This is another convolutional neural network that is designed to be very deep (up to 152 layers). It uses residual connections between layers to improve the flow of information and gradients throughout the network. It is also a popular model for image classification tasks.
- InceptionV3 : This is a convolutional neural network that uses a combination of different types of layers (convolutional, pooling, and fully connected) to learn to represent the data. It is designed to be very deep and wide, with many parallel paths through the network. It is also a popular model for image classification tasks.
- EfficientNetB3: This is a convolutional neural network that uses a combination of different types of layers (convolutional, pooling, and fully connected) to learn to represent the data. It is designed to be very deep and wide, with many parallel paths through the network. It is also a popular model for image classification tasks.

## Types of evaluation

- Baseline model: The original fine-tuned model that was trained on the original dataset. This model will be used as a baseline to compare the performance against the pruned version of the model itself.
- Pruned model: The pruned version of the original model. This model will be used to evaluate the performance of the pruning process. The pruning process is done by removing the least important channels from a target convolutional layer. 

## Results

