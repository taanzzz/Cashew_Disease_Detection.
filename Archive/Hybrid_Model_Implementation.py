import torch
import torch.nn as nn
import timm

class HybridCashewModel(nn.Module):
    def __init__(self, cnn_name="efficientnet_b0", trans_name="cait_xxs24_224", num_classes=5):
        super(HybridCashewModel, self).__init__()

        # 1. CNN Branch
        self.cnn = timm.create_model(cnn_name, pretrained=True, features_only=True)
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)
        cnn_out_dim = self.cnn.feature_info[-1]['num_chs']

        # 2. Transformer Branch
        self.transformer = timm.create_model(trans_name, pretrained=True)
        trans_out_dim = self.transformer.num_features
        self.transformer.head = nn.Identity()

        # 3. Fusion Layer
        self.fusion_dim = cnn_out_dim + trans_out_dim
        print(f"Fusion Dimension calculated: {self.fusion_dim} ({cnn_out_dim} + {trans_out_dim})")

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # CNN features
        cnn_feats = self.cnn(x)[-1]
        cnn_feats = self.cnn_pool(cnn_feats).view(cnn_feats.size(0), -1)

        # Transformer features
        trans_feats = self.transformer(x)

        # Concatenate & Classify
        combined = torch.cat((cnn_feats, trans_feats), dim=1)
        out = self.classifier(combined)
        return out

# Re-initialize the model
hybrid_model = HybridCashewModel()
print("Fixed Hybrid Model initialized with correct dimensions! ✅")

# হাইব্রিড মডেল ট্রেনিং শুরু (সংশোধিত ডাইমেনশন সহ)
best_hybrid_acc = train_model(
    hybrid_model,
    train_loader,
    epochs=10,
    lr=5e-5,
    model_name="hybrid_cnn_transformer"
)

print(f"\nHybrid Model Training Complete! Best Accuracy: {best_hybrid_acc:.4f}")