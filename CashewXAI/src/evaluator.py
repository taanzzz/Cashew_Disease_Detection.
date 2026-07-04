# =============================================================================
# Evaluator — Detailed Model Evaluation & Confusion Matrix
# =============================================================================
# Multi-XAI Hybrid CNN-Transformer Framework for Cashew Disease Detection
# =============================================================================

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs.config import CHECKPOINT_DIR, DEVICE, XAI_OUTPUT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, data_loader, class_names, device=None, save_fig=True):
    """
    Evaluate a trained model and generate classification report + confusion matrix.

    Args:
        model:        Trained PyTorch model
        data_loader:  DataLoader for evaluation data
        class_names:  List of class label strings
        device:       Computation device (defaults to config)
        save_fig:     Whether to save confusion matrix as PNG

    Returns:
        tuple: (all_predictions, all_labels) as numpy arrays
    """
    device = device or DEVICE
    model.eval()
    all_preds = []
    all_labels = []

    print("\nGenerating predictions for evaluation...")
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # ── Classification Report ───────────────────────────────────────────
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # ── Confusion Matrix ────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix — Hybrid CNN-Transformer Model', fontsize=14)
    plt.tight_layout()

    if save_fig:
        os.makedirs(XAI_OUTPUT_DIR, exist_ok=True)
        fig_path = os.path.join(XAI_OUTPUT_DIR, 'confusion_matrix.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {fig_path}")

    plt.show()
    return np.array(all_preds), np.array(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Load & Evaluate Convenience Function
# ─────────────────────────────────────────────────────────────────────────────

def load_and_evaluate(model, data_loader, class_names, checkpoint_name="hybrid_cnn_transformer_best.pth"):
    """
    Load a saved checkpoint and run full evaluation.

    Args:
        model:           Uninitialized model with matching architecture
        data_loader:     DataLoader for evaluation
        class_names:     List of class label strings
        checkpoint_name: Filename of the checkpoint in CHECKPOINT_DIR
    """
    device = DEVICE
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    print(f"✅ Loaded checkpoint: {checkpoint_path}")

    evaluate_model(model, data_loader, class_names, device)
