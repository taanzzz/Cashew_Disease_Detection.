import os
import cv2
import torch
import pandas as pd
import timm
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ১. পাথ ও ট্রান্সফর্ম পুনরায় নিশ্চিত করা
PROJECT_ROOT = '/content/drive/MyDrive/Cashew_XAI_Project'
TRAIN_PATH = os.path.join(PROJECT_ROOT, 'datasets/train')
CKPT_PATH = os.path.join(PROJECT_ROOT, 'checkpoints/hybrid_cnn_transformer_best.pth')
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

test_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

# ২. হাইব্রিড মডেল ক্লাস ডিফাইন করা (ইনিশিয়ালাইজেশনের জন্য)
class HybridCashewModel(nn.Module):
    def __init__(self, cnn_name="efficientnet_b0", trans_name="cait_xxs24_224", num_classes=5):
        super(HybridCashewModel, self).__init__()
        self.cnn = timm.create_model(cnn_name, pretrained=True, features_only=True)
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)
        cnn_out_dim = self.cnn.feature_info[-1]['num_chs']
        self.transformer = timm.create_model(trans_name, pretrained=True)
        trans_out_dim = self.transformer.num_features
        self.transformer.head = nn.Identity()
        self.fusion_dim = cnn_out_dim + trans_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        cnn_feats = self.cnn(x)[-1]
        cnn_feats = self.cnn_pool(cnn_feats).view(cnn_feats.size(0), -1)
        trans_feats = self.transformer(x)
        combined = torch.cat((cnn_feats, trans_feats), dim=1)
        return self.classifier(combined)

# ৩. ডাটাসেট ক্লাস
class CashewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        if not self.root_dir.exists(): return
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for img_path in list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")):
                self.samples.append((img_path, self.class_to_idx[cls]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform: img = self.transform(image=img)["image"]
        return img, label

# ৪. প্রেডিকশন ফাংশন
def predict_external_images(model, image_dir, transform, class_names, device):
    model.eval()
    results = []
    img_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    if not img_paths: return None

    with torch.no_grad():
        for img_path in img_paths[:10]:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (224, 224))
            tensor = transform(image=img_resized)["image"].unsqueeze(0).to(device)
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            results.append({
                "Image": img_path.name,
                "Predicted": class_names[pred_idx.item()],
                "Confidence": f"{conf.item()*100:.2f}%"
            })
    return pd.DataFrame(results)

# ৫. রান করার লজিক
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hybrid_model = HybridCashewModel().to(device)

if os.path.exists(CKPT_PATH):
    hybrid_model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    print("Model loaded successfully! ✅")

    dataset = CashewDataset(TRAIN_PATH, transform=test_transform)
    if len(dataset.samples) > 0:
        EXTERNAL_DATA_PATH = os.path.join(TRAIN_PATH, dataset.classes[0])
        df_results = predict_external_images(hybrid_model, EXTERNAL_DATA_PATH, test_transform, dataset.classes, device)
        if df_results is not None:
            print("--- Real World Prediction Results ---")
            display(df_results)
    else:
        print("No data found in Drive!")
else:
    print("Error: Checkpoint file not found at", CKPT_PATH)