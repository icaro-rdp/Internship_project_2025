"""Model definitions for authenticity predictors using various backbones."""

import torch
import torch.hub
from torch import nn
from torchvision.models import (
    DenseNet161_Weights,
    EfficientNet_B3_Weights,
    Inception_V3_Weights,
    ResNet152_Weights,
    VGG16_Weights,
    VGG19_Weights,
    densenet161,
    efficientnet_b3,
    inception_v3,
    resnet152,
    vgg16,
    vgg19,
)
from transformers import CLIPVisionModelWithProjection


class CLIPViTAuthenticityPredictor(nn.Module):
    """Authenticity predictor using CLIP ViT-B/16 vision backbone.

    Attaches a regression head to the 512-dimensional visual projection embeddings
    of OpenAI CLIP ViT-B/16.
    """

    def __init__(self, freeze_backbone: bool = True) -> None:
        """Initialize the CLIP ViT predictor with frozen backbone and regression head.

        Args:
            freeze_backbone: Whether to freeze all backbone vision model parameters.
        """
        super().__init__()
        clip_model = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-base-patch16"
        )

        if freeze_backbone:
            for param in clip_model.parameters():
                param.requires_grad = False

        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection

        self.regression_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through vision backbone and regression head.

        Args:
            x: Batch of RGB image tensors of shape (B, 3, 224, 224).

        Returns:
            Tuple containing authenticity predictions of shape (B, 1) and
            visual projection features of shape (B, 512).
        """
        vision_outputs = self.vision_model(pixel_values=x)
        features = self.visual_projection(vision_outputs.pooler_output)
        predictions = self.regression_head(features)
        return predictions, features


class BarlowTwinsAuthenticityPredictor(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        # Load pre-trained BarlowTwins ResNet50
        barlow_twins_resnet = torch.hub.load(
            "facebookresearch/barlowtwins:main", "resnet50"
        )

        # Freeze backbone if requested
        if freeze_backbone:
            for param in barlow_twins_resnet.parameters():
                param.requires_grad = False

        self.features = nn.Sequential(*list(barlow_twins_resnet.children())[:-2])
        self.avgpool = barlow_twins_resnet.avgpool

        self.regression_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        predictions = self.regression_head(x)
        return predictions, x


class EfficientNetB3AuthenticityPredictor(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        # Load pre-trained EfficientNet-B3
        efficent_net = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in efficent_net.features.parameters():
                param.requires_grad = False

        # Extract features up to fc2
        self.features = efficent_net.features
        self.avgpool = efficent_net.avgpool

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),  # Predict authenticity
        )

    def forward(self, x):
        # Pass through the backbone features
        x = self.features(x)
        # Apply pooling
        x = self.avgpool(x)
        # Flatten the features
        features = torch.flatten(x, 1)
        # Pass through regression head
        predictions = self.regression_head(features)

        return predictions, features


class DenseNet161AuthenticityPredictor(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        # Load pre-trained DenseNet-161
        densenet = densenet161(weights=DenseNet161_Weights.DEFAULT)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in densenet.parameters():
                param.requires_grad = False

        # Store the features
        self.features = densenet.features

        # DenseNet already includes a ReLU and pooling after features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # DenseNet-161's output feature dimension is 2208 instead of 2048
        self.regression_head = nn.Sequential(
            nn.Linear(2208, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        predictions = self.regression_head(x)
        return predictions, x  # Return predictions and features


class ResNet152AuthenticityPredictor(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        # Load pre-trained ResNet-152
        resnet = resnet152(weights=ResNet152_Weights.DEFAULT)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False

        # Store the backbone (excluding the final fc layer)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = resnet.avgpool

        self.regression_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Reduced dropout ratio
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        predictions = self.regression_head(x)
        return predictions, x


class VGG16AuthenticityPredictor(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        # Load pre-trained VGG16
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)

        for module in vgg.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        # Freeze backbone if requested
        if freeze_backbone:
            for param in vgg.features.parameters():
                param.requires_grad = False

        # Extract features up to fc2
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.fc1 = vgg.classifier[:-1]  # Up to fc2 (4096 -> 128)

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.fc1(x)
        predictions = self.regression_head(features)
        return predictions, features


class VGG19AuthenticityPredictor(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        # Load pre-trained VGG19
        vgg = vgg19(weights=VGG19_Weights.DEFAULT)

        for module in vgg.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        # Freeze backbone if requested
        if freeze_backbone:
            for param in vgg.features.parameters():
                param.requires_grad = False

        # Extract features up to fc2
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.fc1 = vgg.classifier[:-1]  # Up to fc2 (4096 -> 128)

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.fc1(x)
        predictions = self.regression_head(features)
        return predictions, features


class InceptionV3AuthenticityPredictor(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        # Load pre-trained InceptionV3
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)

        # Disable auxiliary outputs for inference
        inception.aux_logits = False

        # Freeze backbone if requested
        if freeze_backbone:
            for param in inception.parameters():
                param.requires_grad = False

        # Store the backbone (excluding the final fc layer)
        # InceptionV3 structure is different from ResNet, so we need to adapt
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.regression_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        predictions = self.regression_head(x)
        return predictions, x  # Return predictions and features
