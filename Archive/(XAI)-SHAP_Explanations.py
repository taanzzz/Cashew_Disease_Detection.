import torch
import torch.nn as nn
import timm
import os

# ১. হাইব্রিড মডেল ক্লাস পুনরায় ডিফাইন করা
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

# ২. মডেল ইনিশিয়ালাইজ এবং চেকপয়েন্ট লোড করা
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hybrid_model = HybridCashewModel()

# ড্রাইভ থেকে বেস্ট মডেল লোড করা
PROJECT_ROOT = '/content/drive/MyDrive/Cashew_XAI_Project'
CKPT_PATH = os.path.join(PROJECT_ROOT, 'checkpoints/hybrid_cnn_transformer_best.pth')

if os.path.exists(CKPT_PATH):
    hybrid_model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    hybrid_model.to(device)
    print("Model successfully loaded from Drive! ✅")
else:
    print("Checkpoint not found. Please ensure the model was trained and saved correctly.")