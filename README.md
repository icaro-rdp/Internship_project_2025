# Visual Similarity Analysis Project

This repository contains tools and notebooks for analyzing visual similarity between images using various computational models and comparing them with human judgments.

## Project Structure

- **Code examples/** - Contains implementation examples and notebooks
  - ais_heatmap/ - Implementation of Alignment-Importance Heatmaps for explaining human comparisons
  - PrunedInception3.ipynb - Identifies best feature subsets for different categories
  - selectSetsFid.ipynb - Feature selection for FID score computation

- **Dataset/** - Contains various datasets used in the project
  - AIGCIQA2023/ - AI-generated image quality assessment dataset
  - AIGCQA-20K/ - Extended quality assessment dataset
  - HPDv2/ - Human Perception Dataset version 2
  - Peterson_data/ - Peterson dataset for similarity judgments

- **Dataset_exploration/** - Notebooks for exploring the datasets
  - AIGCIQA2023.ipynb
  - AIGQA-30K.ipynb
  - HPDv2.ipynb

- **Models/**, **Pruning_masks/**, **Pruning_sets/** - Directories for model artifacts and pruning data

## Core Functionality

- Finetuning of pre-trained models for regression tasks
- Evaluation of model performance using Root Mean Square Error (RMSE)
- Pruining of neural network features using various strategies
- Representation Similarity Matrix (RSM) analysis
- Feature extraction from CNNs (VGG-16, Inception)
- Computation of similarity matrices using cosine similarity

## Key Notebooks

- `VGG-16_pruning_RSA.ipynb` - Representational Similarity Analysis with VGG-16
- `VGG-16_pruning_RSME.ipynb` - Root Mean Square Error analysis with VGG-16

## Setup

This repo is just for visualization and code examples. The actual datasets and models are not included. To actually run the code, you need to download the datasets and models separately. (contact the authors for access)