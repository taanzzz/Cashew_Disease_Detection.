# =============================================================================
# XAI — SHAP Feature Attribution Analysis
# =============================================================================
# Multi-XAI Hybrid CNN-Transformer Framework for Cashew Disease Detection
#
# Uses GradientExplainer (gradient-based Shapley value approximation) to
# compute per-pixel feature importance for model predictions.
#
# Note: SHAP is computationally expensive. nsamples is kept low (20) to
#       prevent OOM errors in Colab free tier.
# =============================================================================

import shap
import torch
import torch.nn as nn
import timm
import numpy as np
import os
import gc
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs.config import (
    IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE,
    CHECKPOINT_DIR, TRAIN_PATH, CLASS_NAMES,
    SHAP_NUM_SAMPLES, XAI_OUTPUT_DIR,
)
from src.model import HybridCashewModel


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Explainer
# ─────────────────────────────────────────────────────────────────────────────

def run_shap_explanation(img_path, model=None, save_fig=True):
    """
    Generate SHAP explanation for a single image using GradientExplainer.

    Args:
        img_path:  Absolute path to the image file
        model:     Trained HybridCashewModel (loads from checkpoint if None)
        save_fig:  Whether to save the SHAP plot

    Returns:
        int: Predicted class index
    """
    gc.collect()

    # Use CPU to avoid CUDA OOM with SHAP
    device = torch.device('cpu')

    # Load model
    if model is None:
        model = HybridCashewModel().to(device)
        ckpt_path = os.path.join(CHECKPOINT_DIR, 'hybrid_cnn_transformer_best.pth')
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print("✅ Model loaded for SHAP analysis.")
        else:
            print(f"❌ Checkpoint not found: {ckpt_path}")
            return None
    else:
        model = model.to(device)

    model.eval()

    # Disable inplace operations (required by SHAP)
    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False

    # Prepare image
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_img = transform(image=img)["image"].unsqueeze(0).to(device)
    test_img.requires_grad = True

    # Background data (minimal to save RAM)
    background = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE)).to(device)

    # Compute SHAP values
    print("🔍 Computing SHAP values (this may take a moment)...")
    gc.collect()

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(test_img, nsamples=SHAP_NUM_SAMPLES, ranked_outputs=1)

    # Visualize
    shap_numpy = [np.transpose(s, (0, 2, 3, 1)) for s in shap_values[0]]
    test_numpy = np.transpose(test_img.detach().numpy(), (0, 2, 3, 1))

    plt.close('all')
    shap.image_plot(shap_numpy, test_numpy, show=False)

    if save_fig:
        os.makedirs(XAI_OUTPUT_DIR, exist_ok=True)
        fig_path = os.path.join(XAI_OUTPUT_DIR, 'shap_explanation.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"✅ SHAP plot saved to: {fig_path}")

    plt.show()

    # Print prediction
    with torch.no_grad():
        output = model(test_img)
        pred_idx = torch.argmax(output).item()

    print(f"🏷️  Predicted class: {CLASS_NAMES[pred_idx]}")
    return pred_idx


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Auto-select a sample image
    sample_class = CLASS_NAMES[0]
    class_dir = os.path.join(TRAIN_PATH, sample_class)

    if os.path.exists(class_dir):
        imgs = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))]
        if imgs:
            sample_path = os.path.join(class_dir, imgs[0])
            run_shap_explanation(sample_path)
        else:
            print("❌ No images found in sample class directory.")
    else:
        print(f"❌ Training path not found: {class_dir}")
