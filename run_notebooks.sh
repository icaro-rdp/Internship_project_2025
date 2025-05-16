#!/bin/bash

echo "Starting notebook executions..."

# echo "Running BarlowTwins_multiscale_mask_saliency_parallel.ipynb"
# jupyter nbconvert --to notebook --execute --inplace "Models/BarlowTwins/BarlowTwins_multiscale_mask_saliency_parallel.ipynb"
# echo "Finished BarlowTwins_multiscale_mask_saliency_parallel.ipynb"

echo "Running DenseNet-161_multiscale_mask_saliency_parallel.ipynb"
jupyter nbconvert --to notebook --execute --inplace "/home/icaro.redepaolini@unitn.it/icaro_rdp_projects/Models/DenseNet-161/DenseNet-161_multiscale_mask_saliency_parallel.ipynb"
echo "Finished DenseNet-161_multiscale_mask_saliency_parallel.ipynb"

echo "Running EfficentNet-B3_multiscale_mask_saliency_parallel.ipynb"
jupyter nbconvert --to notebook --execute --inplace "/home/icaro.redepaolini@unitn.it/icaro_rdp_projects/Models/EfficientNet-B3/EfficentNet-B3_multiscale_mask_saliency_parallel.ipynb"
echo "Finished EfficentNet-B3_multiscale_mask_saliency_parallel.ipynb"

echo "Running ResNet-152_multiscale_mask_saliency_parallel.ipynb"
jupyter nbconvert --to notebook --execute --inplace "/home/icaro.redepaolini@unitn.it/icaro_rdp_projects/Models/ResNet-152/ResNet-152_multiscale_mask_saliency_parallel.ipynb"
echo "Finished ResNet-152_multiscale_mask_saliency_parallel.ipynb"

echo "Running VGG19_multiscale_mask_saliency_parallel.ipynb"
jupyter nbconvert --to notebook --execute --inplace "/home/icaro.redepaolini@unitn.it/icaro_rdp_projects/Models/VGG19/VGG19_multiscale_mask_saliency_parallel.ipynb"
echo "Finished VGG19_multiscale_mask_saliency_parallel.ipynb"

echo "Running VGG16_multiscale_mask_saliency_parallel.ipynb"
jupyter nbconvert --to notebook --execute --inplace "/home/icaro.redepaolini@unitn.it/icaro_rdp_projects/Models/VGG16/dual_scores/VGG16_multiscale_mask_saliency_parallel.ipynb"
echo "Finished VGG16_multiscale_mask_saliency_parallel.ipynb"

echo "All notebook executions finished."
