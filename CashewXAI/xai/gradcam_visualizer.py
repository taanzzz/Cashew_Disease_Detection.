# =============================================================================
# XAI — Grad-CAM Heatmap Visualization
# =============================================================================
# Multi-XAI Hybrid CNN-Transformer Framework for Cashew Disease Detection
#
# Generates class-discriminative localization maps using gradient-weighted
# class activation mapping (Grad-CAM) on the CNN branch of the hybrid model.
# =============================================================================

import cv2
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs.config import (
    IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE,
    CHECKPOINT_DIR, TRAIN_PATH, DEVICE, XAI_OUTPUT_DIR,
)
from src.model import HybridCashewModel


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_gradcam(model, target_layer, img_path, class_names, device=None):
    """
    Generate Grad-CAM heatmap for a given image.

    Args:
        model:        Trained hybrid model
        target_layer: CNN layer to extract gradients from
        img_path:     Path to the input image
        class_names:  List of class label strings
        device:       Computation device

    Returns:
        tuple: (original_image, heatmap_overlay, predicted_label)
    """
    device = device or DEVICE

    # Load and preprocess image
    rgb_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (IMAGE_SIZE, IMAGE_SIZE))
    img_float = rgb_img.astype(np.float32) / 255.0
    input_tensor = preprocess_image(rgb_img, mean=IMAGENET_MEAN, std=IMAGENET_STD).to(device)

    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Get model prediction
    model.eval()
    output = model(input_tensor)
    pred_idx = torch.argmax(output).item()

    # Generate heatmap
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(pred_idx)],
    )[0, :]

    # Overlay heatmap on original image
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    return rgb_img, visualization, class_names[pred_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_gradcam(model, dataset_classes, sample_path=None, save_fig=True):
    """
    Load model, generate Grad-CAM, and display side-by-side visualization.

    Args:
        model:           Initialized HybridCashewModel
        dataset_classes: List of class names
        sample_path:     Path to a sample image (auto-selects if None)
        save_fig:        Whether to save the figure
    """
    device = DEVICE
    target_layer = model.cnn.blocks[-1][-1]

    # Auto-select sample image if not provided
    if sample_path is None:
        sample_class = dataset_classes[0]
        class_dir = os.path.join(TRAIN_PATH, sample_class)
        img_list = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))]
        sample_path = os.path.join(class_dir, img_list[0])

    try:
        orig, vis, pred_label = generate_gradcam(
            model, target_layer, sample_path, dataset_classes, device,
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(orig)
        axes[0].set_title(f"Original Image", fontsize=13)
        axes[0].axis('off')

        axes[1].imshow(vis)
        axes[1].set_title(f"Grad-CAM Heatmap (Pred: {pred_label})", fontsize=13)
        axes[1].axis('off')

        plt.suptitle('Grad-CAM Visualization — Hybrid Model', fontsize=15, fontweight='bold')
        plt.tight_layout()

        if save_fig:
            os.makedirs(XAI_OUTPUT_DIR, exist_ok=True)
            fig.savefig(
                os.path.join(XAI_OUTPUT_DIR, 'gradcam_heatmap.png'),
                dpi=150, bbox_inches='tight',
            )
            print(f"✅ Grad-CAM saved to: {XAI_OUTPUT_DIR}/gradcam_heatmap.png")

        plt.show()

    except Exception as e:
        print(f"❌ Error generating Grad-CAM: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from configs.config import CLASS_NAMES

    model = HybridCashewModel().to(DEVICE)
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'hybrid_cnn_transformer_best.pth')

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print("✅ Model loaded.")
        visualize_gradcam(model, CLASS_NAMES)
    else:
        print(f"❌ Checkpoint not found: {ckpt_path}")
