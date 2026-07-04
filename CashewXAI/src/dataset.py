# =============================================================================
# Dataset — CashewDataset Class & Data Augmentation Pipeline
# =============================================================================
# Multi-XAI Hybrid CNN-Transformer Framework for Cashew Disease Detection
# =============================================================================

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs.config import (
    IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE,
    BATCH_SIZE, NUM_WORKERS, TRAIN_PATH,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data Augmentation Transforms
# ─────────────────────────────────────────────────────────────────────────────

train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=40, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.7),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

test_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


# ─────────────────────────────────────────────────────────────────────────────
# Custom Dataset Class
# ─────────────────────────────────────────────────────────────────────────────

class CashewDataset(Dataset):
    """
    Custom PyTorch Dataset for cashew leaf disease images.

    Reads images organized in class-wise subdirectories:
        root_dir/
            anthracnose/
            gumosis/
            healthy/
            leaf miner/
            red rust/

    Supports both .jpg and .png image formats.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        if not self.root_dir.exists():
            print(f"⚠ Warning: Path {root_dir} not found. Please check your data directory.")
            return

        # Discover classes from subdirectory names (sorted for reproducibility)
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Collect all image paths with their corresponding labels
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            for img_path in list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")):
                self.samples.append((img_path, self.class_to_idx[cls_name]))

        print(f"✅ Loaded {len(self.samples)} images from {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, label


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_train_loader(data_path=None, batch_size=None, num_workers=None):
    """
    Create training DataLoader with augmentation pipeline.

    Args:
        data_path:    Path to training data directory (defaults to config)
        batch_size:   Batch size (defaults to config)
        num_workers:  Number of data loading workers (defaults to config)

    Returns:
        tuple: (DataLoader, CashewDataset) or (None, dataset) if empty
    """
    data_path = data_path or TRAIN_PATH
    batch_size = batch_size or BATCH_SIZE
    num_workers = num_workers or NUM_WORKERS

    dataset = CashewDataset(data_path, transform=train_transform)

    if len(dataset) > 0:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        return loader, dataset
    else:
        print("⚠ Dataset is empty. Cannot create DataLoader.")
        return None, dataset


def get_test_loader(data_path=None, batch_size=None, num_workers=None):
    """
    Create test/validation DataLoader without augmentation.

    Args:
        data_path:    Path to test data directory (defaults to config)
        batch_size:   Batch size (defaults to config)
        num_workers:  Number of data loading workers (defaults to config)

    Returns:
        tuple: (DataLoader, CashewDataset) or (None, dataset) if empty
    """
    data_path = data_path or TRAIN_PATH
    batch_size = batch_size or BATCH_SIZE
    num_workers = num_workers or NUM_WORKERS

    dataset = CashewDataset(data_path, transform=test_transform)

    if len(dataset) > 0:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return loader, dataset
    else:
        return None, dataset


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    loader, dataset = get_train_loader()
    if loader:
        batch_imgs, batch_labels = next(iter(loader))
        print(f"Batch shape : {batch_imgs.shape}")
        print(f"Labels      : {batch_labels[:5]}")
        print(f"Classes     : {dataset.classes}")
