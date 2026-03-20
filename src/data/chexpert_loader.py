"""
CheXpert Dataset Loader for TeleZK-FL.

Loads CheXpert-v1.0-small chest X-ray dataset with multi-label binary
classification for 5 competition pathologies. Uses CheXbert auto-labels
and the U-Ones mapping policy for uncertain labels.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# Standard ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# The 5 competition pathologies
CHEXPERT_PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


class CheXpertDataset(Dataset):
    """PyTorch Dataset for CheXpert chest X-ray classification.

    Args:
        data_dir: Root directory containing CheXpert-v1.0-small.
        df: DataFrame with image paths and labels.
        pathologies: List of pathology column names to use.
        transform: Optional torchvision transform to apply.
    """

    def __init__(
        self,
        data_dir: str,
        df: pd.DataFrame,
        pathologies: List[str] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        self.data_dir = data_dir
        self.df = df.reset_index(drop=True)
        self.pathologies = pathologies or CHEXPERT_PATHOLOGIES
        self.transform = transform or self._default_transform()

        # Pre-process labels: U-Ones policy and NaN handling
        self._process_labels()

    def _default_transform(self) -> transforms.Compose:
        """Standard transform: resize, center crop, normalize."""
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _process_labels(self):
        """Apply U-Ones policy: map -1 (uncertain) -> 1, NaN -> 0."""
        for col in self.pathologies:
            if col in self.df.columns:
                # Fill NaN/blank with 0.0
                self.df[col] = self.df[col].fillna(0.0)
                # U-Ones: map uncertain (-1.0) to positive (1.0)
                self.df[col] = self.df[col].replace(-1.0, 1.0)
                # Clamp to binary {0, 1}
                self.df[col] = self.df[col].clip(0.0, 1.0)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # Build image path — CSV paths are like "CheXpert-v1.0-small/train/..."
        # We need to join relative to the parent of CheXpert-v1.0-small
        img_rel_path = row["Path"]
        # Handle both forward and backward slashes
        img_rel_path = img_rel_path.replace("\\", "/")

        # The data_dir points to CheXpert-v1.0-small, but paths in CSV
        # start with "CheXpert-v1.0-small/..." so go up one level
        parent_dir = os.path.dirname(self.data_dir)
        img_path = os.path.join(parent_dir, img_rel_path)

        # Load image as RGB (some CheXpert images are grayscale)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback: return a black image if file is corrupted
            print(f"Warning: Could not load {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # Extract labels for the 5 pathologies
        labels = torch.tensor(
            [float(row[col]) for col in self.pathologies],
            dtype=torch.float32,
        )

        return image, labels


def get_chexpert_train_test(
    data_dir: str,
    label_csv: str,
    pathologies: List[str] = None,
) -> Tuple[CheXpertDataset, CheXpertDataset]:
    """Load CheXpert train and test (validation) datasets.

    Args:
        data_dir: Path to CheXpert-v1.0-small directory.
        label_csv: Path to train_cheXbert.csv with improved labels.
        pathologies: List of pathology names. Defaults to 5 competition ones.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    pathologies = pathologies or CHEXPERT_PATHOLOGIES

    # Load CheXbert labels for training data
    df = pd.read_csv(label_csv)

    # Split into train and valid based on path prefix
    train_mask = df["Path"].str.contains("train", case=False)
    valid_mask = df["Path"].str.contains("valid", case=False)

    train_df = df[train_mask].copy()

    # If the label CSV doesn't contain valid images, load from the
    # dataset's own valid.csv
    if valid_mask.sum() > 0:
        valid_df = df[valid_mask].copy()
    else:
        # Try loading the internal valid.csv
        valid_csv_path = os.path.join(data_dir, "valid.csv")
        if os.path.exists(valid_csv_path):
            valid_df = pd.read_csv(valid_csv_path)
        else:
            # Fallback: use last 10% of training data
            split_idx = int(len(train_df) * 0.9)
            valid_df = train_df.iloc[split_idx:].copy()
            train_df = train_df.iloc[:split_idx].copy()

    # Training transforms (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_dataset = CheXpertDataset(
        data_dir=data_dir,
        df=train_df,
        pathologies=pathologies,
        transform=train_transform,
    )

    test_dataset = CheXpertDataset(
        data_dir=data_dir,
        df=valid_df,
        pathologies=pathologies,
        transform=val_transform,
    )

    print(f"CheXpert loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    return train_dataset, test_dataset
