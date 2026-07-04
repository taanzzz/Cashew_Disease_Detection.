# =============================================================================
# Baseline Training — 5 CNN Architectures
# =============================================================================
# Multi-XAI Hybrid CNN-Transformer Framework for Cashew Disease Detection
#
# CNN Baselines:
#   1. ResNet-50
#   2. DenseNet-121
#   3. EfficientNet-B0
#   4. VGG-16
#   5. MobileNetV3-Large
# =============================================================================

import pandas as pd
import timm
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs.config import (
    CNN_BASELINES, NUM_CLASSES, BASELINE_EPOCHS,
    BASELINE_LR, RESULTS_DIR,
)
from src.dataset import get_train_loader
from src.trainer import train_model, cleanup_memory


def train_cnn_baselines():
    """
    Train all 5 CNN baseline models sequentially and save results.

    Each model is trained for BASELINE_EPOCHS epochs with AdamW optimizer.
    Best checkpoints are saved automatically by the trainer.

    Returns:
        pd.DataFrame: Results with model names and best accuracies
    """
    train_loader, dataset = get_train_loader()
    if train_loader is None:
        print("❌ Cannot train: DataLoader is empty.")
        return None

    results = []

    for name in CNN_BASELINES:
        print(f"\n{'─'*60}")
        print(f"Initializing CNN Baseline: {name}")
        print(f"{'─'*60}")

        model = timm.create_model(name, pretrained=True, num_classes=NUM_CLASSES)

        best_acc = train_model(
            model, train_loader,
            epochs=BASELINE_EPOCHS,
            lr=BASELINE_LR,
            model_name=f"cnn_{name}",
        )
        results.append({"Model": name, "Best_Train_Acc": best_acc})

        # Free GPU memory before loading next model
        cleanup_memory(model)

    # Save results to CSV
    df_results = pd.DataFrame(results)
    result_dir = os.path.join(RESULTS_DIR, 'baseline_cnn')
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, 'cnn_results.csv')
    df_results.to_csv(csv_path, index=False)

    print("\n" + "="*60)
    print("CNN Baseline Results")
    print("="*60)
    print(df_results.to_string(index=False))
    print(f"\n✅ Results saved to: {csv_path}")

    return df_results


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train_cnn_baselines()
