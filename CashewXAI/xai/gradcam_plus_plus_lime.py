# =============================================================================
# XAI — Grad-CAM++ & LIME Combined Explanations
# =============================================================================
# Multi-XAI Hybrid CNN-Transformer Framework for Cashew Disease Detection
#
# Grad-CAM++: Improved gradient-weighted activation mapping with pixel-wise
#             gradient weighting for better localization.
# LIME:       Local Interpretable Model-agnostic Explanations via superpixel
#             perturbation-based feature importance.
# =============================================================================

import cv2
import torch
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from lime import lime_image
from skimage.segmentation import mark_boundaries

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs.config import (
    IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE,
    CHECKPOINT_DIR, TRAIN_PATH, DEVICE, XAI_OUTPUT_DIR,
)
from src.model import HybridCashewModel
from src.dataset import test_transform


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM++ Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_gradcam_plus_plus(model, target_layer, img_path, class_names, device=None):
    """
    Generate Grad-CAM++ heatmap for a given image.

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

    rgb_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (IMAGE_SIZE, IMAGE_SIZE))
    img_float = rgb_img.astype(np.float32) / 255.0
    input_tensor = preprocess_image(rgb_img, mean=IMAGENET_MEAN, std=IMAGENET_STD).to(device)

    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])

    model.eval()
    output = model(input_tensor)
    pred_idx = torch.argmax(output).item()

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(pred_idx)],
    )[0, :]

    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    return rgb_img, visualization, class_names[pred_idx]


# ─────────────────────────────────────────────────────────────────────────────
# LIME Explanation
# ─────────────────────────────────────────────────────────────────────────────

def generate_lime_explanation(model, img_path, device=None):
    """
    Generate LIME explanation for a given image.

    Uses superpixel-based perturbation to identify the most important
    image regions for the model's prediction.

    Args:
        model:    Trained hybrid model
        img_path: Path to the input image
        device:   Computation device

    Returns:
        np.ndarray: LIME visualization with boundary overlay
    """
    device = device or DEVICE

    def batch_predict(images):
        """LIME batch prediction function."""
        model.eval()
        normalized_images = []
        for img in images:
            transformed = test_transform(image=img)["image"]
            normalized_images.append(transformed)
        batch = torch.stack(normalized_images, dim=0).to(device)
        logits = model(batch)
        return torch.softmax(logits, dim=1).detach().cpu().numpy()

    explainer = lime_image.LimeImageExplainer()

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    explanation = explainer.explain_instance(
        img, batch_predict,
        top_labels=1, num_samples=200,
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True, num_features=5, hide_rest=False,
    )

    return mark_boundaries(temp / 255.0, mask)


# ─────────────────────────────────────────────────────────────────────────────
# Combined Visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_gradcam_pp_and_lime(model=None, save_fig=True):
    """
    Generate and display side-by-side Grad-CAM++ and LIME visualizations.

    Automatically selects a random sample image from the training set.

    Args:
        model:    Trained HybridCashewModel (loads from checkpoint if None)
        save_fig: Whether to save the composite figure
    """
    device = DEVICE

    # Load model if not provided
    if model is None:
        model = HybridCashewModel().to(device)
        ckpt_path = os.path.join(CHECKPOINT_DIR, 'hybrid_cnn_transformer_best.pth')
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print("✅ Hybrid Model loaded.")
        else:
            print(f"❌ Checkpoint not found: {ckpt_path}")
            return

    # Discover class names from training directory
    class_names = sorted([
        d for d in os.listdir(TRAIN_PATH)
        if os.path.isdir(os.path.join(TRAIN_PATH, d))
    ])

    # Select a random sample image
    random_class = random.choice(class_names)
    class_dir = os.path.join(TRAIN_PATH, random_class)
    random_img = random.choice([
        f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))
    ])
    img_path = os.path.join(class_dir, random_img)

    target_layer = model.cnn.blocks[-1][-1]

    print(f"🔍 Generating XAI outputs for: {img_path}")

    # Generate explanations
    orig, gcpp_vis, pred_label = generate_gradcam_plus_plus(
        model, target_layer, img_path, class_names, device,
    )
    lime_vis = generate_lime_explanation(model, img_path, device)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(orig)
    axes[0].set_title(f"Original ({random_class})", fontsize=13)
    axes[0].axis('off')

    axes[1].imshow(gcpp_vis)
    axes[1].set_title(f"Grad-CAM++ (Pred: {pred_label})", fontsize=13)
    axes[1].axis('off')

    axes[2].imshow(lime_vis)
    axes[2].set_title("LIME Explanation", fontsize=13)
    axes[2].axis('off')

    plt.suptitle('Multi-XAI Explanations — Hybrid Model', fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_fig:
        os.makedirs(XAI_OUTPUT_DIR, exist_ok=True)
        fig.savefig(
            os.path.join(XAI_OUTPUT_DIR, 'gradcam_pp_lime.png'),
            dpi=150, bbox_inches='tight',
        )
        print(f"✅ Figure saved to: {XAI_OUTPUT_DIR}/gradcam_pp_lime.png")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    visualize_gradcam_pp_and_lime()
