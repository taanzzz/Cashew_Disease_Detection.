# =============================================================================
# Model — Hybrid CNN-Transformer Architecture
# =============================================================================
# Multi-XAI Hybrid CNN-Transformer Framework for Cashew Disease Detection
#
# Architecture:
#   CNN Branch   : EfficientNet-B0 (feature extractor) → AdaptiveAvgPool2d
#   Trans Branch : CaiT-XXS24-224 (feature extractor)  → Identity head
#   Fusion       : Concatenation → FC(512) → ReLU → Dropout → FC(num_classes)
# =============================================================================

import torch
import torch.nn as nn
import timm

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs.config import (
    CNN_BACKBONE, TRANSFORMER_BACKBONE,
    NUM_CLASSES, FUSION_HIDDEN_DIM, DROPOUT_RATE,
)


class HybridCashewModel(nn.Module):
    """
    Hybrid CNN-Transformer model for cashew disease classification.

    Fuses local spatial features from a CNN backbone with global contextual
    representations from a Vision Transformer backbone through late fusion
    (feature concatenation) followed by a fully-connected classifier head.

    Args:
        cnn_name:    Name of the CNN backbone from timm (default: efficientnet_b0)
        trans_name:  Name of the Transformer backbone from timm (default: cait_xxs24_224)
        num_classes: Number of output disease categories (default: 5)
    """

    def __init__(
        self,
        cnn_name=CNN_BACKBONE,
        trans_name=TRANSFORMER_BACKBONE,
        num_classes=NUM_CLASSES,
    ):
        super(HybridCashewModel, self).__init__()

        # ── CNN Branch ──────────────────────────────────────────────────
        # Extract multi-scale feature maps (features_only=True)
        self.cnn = timm.create_model(cnn_name, pretrained=True, features_only=True)
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)
        cnn_out_dim = self.cnn.feature_info[-1]['num_chs']

        # ── Transformer Branch ──────────────────────────────────────────
        # Use full model but replace classification head with identity
        self.transformer = timm.create_model(trans_name, pretrained=True)
        trans_out_dim = self.transformer.num_features
        self.transformer.head = nn.Identity()

        # ── Fusion & Classifier ─────────────────────────────────────────
        self.fusion_dim = cnn_out_dim + trans_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(FUSION_HIDDEN_DIM, num_classes),
        )

        print(f"✅ Hybrid Model initialized")
        print(f"   CNN    : {cnn_name} → {cnn_out_dim}-dim")
        print(f"   Trans  : {trans_name} → {trans_out_dim}-dim")
        print(f"   Fusion : {self.fusion_dim}-dim → {num_classes} classes")

    def forward(self, x):
        # CNN feature extraction (use last feature map)
        cnn_feats = self.cnn(x)[-1]
        cnn_feats = self.cnn_pool(cnn_feats).view(cnn_feats.size(0), -1)

        # Transformer feature extraction
        trans_feats = self.transformer(x)

        # Late fusion via concatenation
        combined = torch.cat((cnn_feats, trans_feats), dim=1)

        # Classification
        out = self.classifier(combined)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model = HybridCashewModel()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Expected: [1, 5]
