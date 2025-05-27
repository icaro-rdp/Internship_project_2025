
## Stacking Ensemble Overview

The script first loads several pre-trained **base models**: BarlowTwinsAuthenticityPredictor, EfficientNetB3AuthenticityPredictor, DenseNet161AuthenticityPredictor, ResNet152AuthenticityPredictor, VGG16AuthenticityPredictor, VGG19AuthenticityPredictor, and InceptionV3AuthenticityPredictor, along with their respective weights. These base models are used to generate predictions that will serve as input features for the meta-learner.

---
## K-Fold Cross-Validation for Meta-Learner Training Data

To prevent information leakage and create a robust training dataset for the meta-learner, the script employs **K-Fold cross-validation** on the training data. Here's how it works:

1.  **Initialization**: The main training dataset (`IMAGENET_DATASET['train']`) is identified. The script initializes tensors to store **out-of-fold (OOF) predictions** (`oof_predictions_all_models`) and the corresponding true labels (`oof_labels`). A `KFold` object is configured with `N_SPLITS = N` splits, ensuring shuffling and a fixed random state for reproducibility.

2.  **Iteration through Folds**: The script iterates five times (once for each fold). In each iteration:
    * The training data is split into a training fold and a validation fold based on local indices within the main training subset.
    * The true labels for the current validation fold samples are stored in `oof_labels`.

3.  **Generating OOF Predictions**: For each base model:
    * The script determines if the model uses `DENSENET_DATASET` or `IMAGENET_DATASET` transformations, as some models require specific preprocessing.
    * A `Subset` and `DataLoader` are created for the current validation fold, using global indices to ensure correct data and transformations are applied.
    * Predictions are obtained from the current base model on this validation fold using the `get_predictions` function.
    * These predictions are then stored in the `oof_predictions_all_models` tensor at the rows corresponding to the validation fold samples and the column corresponding to the current base model.

The result of this K-Fold process is `oof_predictions_all_models`, where each row represents a sample from the original training set, and each column contains the predictions for that sample from a different base model. These OOF predictions are "unseen" by the models that generated them during their own training.

## Meta-Learner Construction and Training 

1.  **Data Preparation**:
    * The `oof_predictions_all_models` (now referred to as `combined_predictions`) and the `oof_labels` (now `labels_tensor`) are combined into a `TensorDataset`. This dataset forms the training data for the meta-learner.
    * A `DataLoader` (`train_dataloader`) is created to batch and shuffle this training data.

2.  **Meta-Learner (Stacking Model) Definition**:
    * A simple neural network, `StackingEnsemble`, is defined using `torch.nn.Module`. It consists of a single fully connected linear layer (`nn.Linear`).
    * The input size of this layer is the number of base models (i.e., the number of columns in `combined_predictions`), and it outputs a single value (`num_classes=1`) as it appears to be a regression or binary classification task with a single output neuron.

3.  **Training the Meta-Learner**:
    * The `StackingEnsemble` model is initialized.
    * An **MSELoss** function (`criterion`) is chosen, indicating a regression task or a classification task approached as regression of logits/probabilities.
    * An **Adam optimizer** is selected to update the meta-learner's weights.
    * The `train_stacking_model` function trains the meta-learner for a specified number of epochs (20 in this case).
    * In each epoch, the model iterates through the `train_dataloader`. For each batch, it calculates the output, computes the loss against the true labels, performs backpropagation, and updates the model parameters. The training loss is printed after each epoch.

## Testing the Stacked Model

After training the meta-learner, the script evaluates its performance on a separate test set:

1.  **Test Data Preparation**:
    * The script checks if `IMAGENET_DATASET['test']` exists.
    * It extracts true labels (`y_test_true_tensor`) from the test set, handling cases where it might be a `Subset` or a full `Dataset` and attempting different ways to access labels.
    * A tensor `test_predictions_all_models` is initialized to store predictions from base models on the test set.

2.  **Base Model Predictions on Test Set**:
    * Each base model makes predictions on the test data. Similar to the OOF generation, it handles which dataset (`DENSENET_DATASET['test']` or `IMAGENET_DATASET['test']`) and transformations to use for each model.
    * The `get_predictions` function is used to get these predictions, which are stored in `test_predictions_all_models`.

3.  **Meta-Learner Evaluation**:
    * The `test_predictions_all_models` (now `X_meta_test`) are fed into the trained `stacking_model` (meta-learner) to get the final ensemble predictions (`final_predictions_on_test`).
    * The performance is evaluated using **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and **R-squared (R^2)** by comparing the `final_predictions_on_test` with the `y_test_true_numpy`.

Finally, the state dictionary of the trained `stacking_model` is saved to a file named `stacking_ensemble_final.pth`.