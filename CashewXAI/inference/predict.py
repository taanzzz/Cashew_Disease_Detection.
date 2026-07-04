# =============================================================================
# Inference — Real-World Prediction on New Images
# =============================================================================
# Multi-XAI Hybrid CNN-Transformer Framework for Cashew Disease Detection
#
# Loads the trained hybrid model and performs inference on external images.
# Outputs a structured prediction table with confidence scores.
# =============================================================================

import cv2
import torch
import pandas as pd
import os
from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs.config import (
    CHECKPOINT_DIR, TRAIN_PATH, DEVICE, CLASS_NAMES, IMAGE_SIZE,
)
from src.model import HybridCashewModel
from src.dataset import test_transform


# ─────────────────────────────────────────────────────────────────────────────
# Prediction Functions
# ─────────────────────────────────────────────────────────────────────────────

def predict_single_image(model, img_path, device=None):
    """
    Predict the disease class of a single cashew leaf image.

    Args:
        model:    Trained HybridCashewModel
        img_path: Path to the image file
        device:   Computation device

    Returns:
        dict: {"Image": filename, "Predicted": class_name, "Confidence": "XX.XX%"}
    """
    device = device or DEVICE
    model.eval()

    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    tensor = test_transform(image=img)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    return {
        "Image": os.path.basename(img_path),
        "Predicted": CLASS_NAMES[pred_idx.item()],
        "Confidence": f"{confidence.item() * 100:.2f}%",
    }


def predict_directory(model, image_dir, device=None, max_images=10):
    """
    Predict disease classes for all images in a directory.

    Args:
        model:      Trained HybridCashewModel
        image_dir:  Directory containing images
        device:     Computation device
        max_images: Maximum number of images to process

    Returns:
        pd.DataFrame: Prediction results table
    """
    device = device or DEVICE
    model.eval()

    img_paths = (
        list(Path(image_dir).glob("*.jpg")) +
        list(Path(image_dir).glob("*.png"))
    )

    if not img_paths:
        print(f"⚠ No images found in: {image_dir}")
        return None

    results = []
    for img_path in img_paths[:max_images]:
        result = predict_single_image(model, img_path, device)
        results.append(result)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Full Inference Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(image_dir=None, checkpoint_name="hybrid_cnn_transformer_best.pth"):
    """
    Complete inference pipeline: load model → predict → display results.

    Args:
        image_dir:       Directory with images (defaults to first class in train)
        checkpoint_name: Checkpoint filename
    """
    device = DEVICE
    model = HybridCashewModel().to(device)

    ckpt_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print("✅ Model loaded for inference.")

    # Default to first class directory
    if image_dir is None:
        classes = sorted([
            d for d in os.listdir(TRAIN_PATH)
            if os.path.isdir(os.path.join(TRAIN_PATH, d))
        ])
        if classes:
            image_dir = os.path.join(TRAIN_PATH, classes[0])
        else:
            print("❌ No data found.")
            return

    df_results = predict_directory(model, image_dir, device)

    if df_results is not None:
        print("\n" + "="*60)
        print("Prediction Results")
        print("="*60)
        print(df_results.to_string(index=False))

    return df_results


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    run_inference()
