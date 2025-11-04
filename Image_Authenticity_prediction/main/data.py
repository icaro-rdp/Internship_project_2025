import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import numpy as np
from torch.utils.data import DataLoader, random_split
from typing import Optional, Tuple, Callable, Dict, Union
from torch import Tensor


class ImageAuthenticityDataset(Dataset):
    """Dataset for image quality assessment."""

    def __init__(self, csv_file_name: str, transform: Optional[Callable] = None) -> None:
        """
        Args:
            csv_file_name (str): Name of the CSV file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
    
        self.base_dir: str = 'Dataset/AIGCIQA2023'
        self.csv_file: str = os.path.join(self.base_dir, csv_file_name)
        self.data: pd.DataFrame = pd.read_csv(self.csv_file)
        self.transform: Optional[Callable] = transform

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Retrieves an image and its labels by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple (image, labels) where:
                image (torch.Tensor): The transformed image tensor.
                labels (torch.Tensor): Tensor containing authenticity score.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name: str = self.data.iloc[idx, 3]
        image: Image.Image = Image.open(img_name).convert('RGB')
        authenticity: float = self.data.iloc[idx, 1]  # Authenticity column
        labels: Tensor = torch.tensor([authenticity], dtype=torch.float)


        if self.transform:
            image = self.transform(image)

        return image, labels

IMAGENET_TRANSFORM: transforms.Compose = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DENSENET_TRANSFORM: transforms.Compose = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


ANNOTATION_FILE: str = 'real_images_annotations.csv'
BATCH_SIZE: int = 64
NUM_WORKERS: int = 10


# Create the datasets
imageNet_dataset: ImageAuthenticityDataset = ImageAuthenticityDataset(csv_file_name=ANNOTATION_FILE, transform=IMAGENET_TRANSFORM)
denseNet_dataset: ImageAuthenticityDataset = ImageAuthenticityDataset(csv_file_name=ANNOTATION_FILE, transform=DENSENET_TRANSFORM)


# Set seed reproducibility
GENERATOR: torch.Generator = torch.Generator().manual_seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42) 
np.random.seed(42)

imagenet_total_size: int = len(imageNet_dataset)
imagenet_train_size: int = int(0.8 * imagenet_total_size)
imagenet_test_size: int = imagenet_total_size - imagenet_train_size

imagenet_train_ds, imagenet_test_ds = random_split(
    imageNet_dataset,
    [imagenet_train_size, imagenet_test_size],
    generator=GENERATOR 
)


densenet_total_size: int = len(denseNet_dataset)
densenet_train_size: int = int(0.8 * densenet_total_size)
densenet_test_size: int = densenet_total_size - densenet_train_size

densenet_train_ds, densenet_test_ds = random_split(
    denseNet_dataset,
    [densenet_train_size, densenet_test_size],
    generator=GENERATOR 
)

IMAGENET_DATASET: Dict[str, Dataset] = {
    'train': imagenet_train_ds,
    'test': imagenet_test_ds
}

DENSENET_DATASET: Dict[str, Dataset] = {
    'train': densenet_train_ds,
    'test': densenet_test_ds
}

INCEPTIONV3_DATASET: Dict[str, Dataset] = {
    'train': densenet_train_ds,
    'test': densenet_test_ds
}




