import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

# পাথ ভেরিফিকেশন
PROJECT_ROOT = '/content/drive/MyDrive/Cashew_XAI_Project'
TRAIN_PATH = os.path.join(PROJECT_ROOT, 'datasets/train')

# ImageNet stats
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=40, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.7),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

class CashewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        if not self.root_dir.exists():
            print(f"Warning: Path {root_dir} not found. Please refresh Drive.")
            return
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for img_path in list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")):
                self.samples.append((img_path, self.class_to_idx[cls]))
        print(f"Success: Loaded {len(self.samples)} images from {len(self.classes)} classes.")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform: img = self.transform(image=img)["image"]
        return img, label

dataset = CashewDataset(TRAIN_PATH, transform=train_transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2) if len(dataset) > 0 else None