import shap
import torch
import torch.nn as nn
import timm
import numpy as np
import os
import cv2
import gc
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A

# ১. মেমোরি পরিষ্কার
gc.collect()

# ২. মডেল ক্লাস (Matched architecture)
class HybridCashewModel(nn.Module):
    def __init__(self, cnn_name='efficientnet_b0', trans_name='cait_xxs24_224', num_classes=5):
        super(HybridCashewModel, self).__init__()
        self.cnn = timm.create_model(cnn_name, pretrained=True, features_only=True)
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)
        cnn_out_dim = self.cnn.feature_info[-1]['num_chs']
        self.transformer = timm.create_model(trans_name, pretrained=True)
        self.transformer.head = nn.Identity()
        self.fusion_dim = cnn_out_dim + self.transformer.num_features
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        cnn_feats = self.cnn(x)[-1]
        cnn_feats = self.cnn_pool(cnn_feats).view(cnn_feats.size(0), -1)
        trans_feats = self.transformer(x)
        combined = torch.cat((cnn_feats, trans_feats), dim=1)
        return self.classifier(combined)

# ৩. কনফিগারেশন
PROJECT_ROOT = '/content/drive/MyDrive/Cashew_XAI_Project'
CKPT_PATH = os.path.join(PROJECT_ROOT, 'checkpoints/hybrid_cnn_transformer_best.pth')
specific_img_path = '/content/drive/MyDrive/Cashew_XAI_Project/datasets/train/anthracnose353_.jpg'
class_names = ['anthracnose', 'gumosis', 'healthy', 'leaf miner', 'red rust']

if os.path.exists(CKPT_PATH) and os.path.exists(specific_img_path):
    device = torch.device('cpu')
    model = HybridCashewModel().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    # SHAP requirements
    for module in model.modules():
        if hasattr(module, 'inplace'): module.inplace = False

    # মডেলের ইনপুট সাইজ ২২৪ এ ফিক্সড রাখতে হবে CaiT এর জন্য
    SIZE = 224
    transform = A.Compose([
        A.Resize(SIZE, SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    img = cv2.imread(specific_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_img = transform(image=img)["image"].unsqueeze(0).to(device)
    test_img.requires_grad = True

    # ৪. এক্সট্রিমলি লাইট ব্যাকগ্রাউন্ড (RAM বাঁচাতে)
    background = torch.zeros((1, 3, SIZE, SIZE)).to(device)

    print('Calculating SHAP (Final Stability Mode)...')
    gc.collect()

    # GradientExplainer uses autograd to find feature importance
    explainer = shap.GradientExplainer(model, background)
    # nsamples কমানো হয়েছে যাতে সেশন ক্র্যাশ না করে
    shap_values = explainer.shap_values(test_img, nsamples=20, ranked_outputs=1)

    # ৫. প্লটিং
    shap_numpy = [np.transpose(s, (0, 2, 3, 1)) for s in shap_values[0]]
    test_numpy = np.transpose(test_img.detach().numpy(), (0, 2, 3, 1))

    plt.close('all')
    shap.image_plot(shap_numpy, test_numpy, show=True)

    with torch.no_grad():
        output = model(test_img)
        pred_idx = torch.argmax(output).item()
    print(f'Predicted: {class_names[pred_idx]}')
else:
    print('Error: Path not valid or files missing.')