# =============================================================================
# Trainer — Training Loop with Checkpoint Management
# =============================================================================
# Multi-XAI Hybrid CNN-Transformer Framework for Cashew Disease Detection
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import os
import gc

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs.config import CHECKPOINT_DIR, DEVICE


# ─────────────────────────────────────────────────────────────────────────────
# Training Function
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, train_loader, epochs=10, lr=1e-4, model_name="model"):
    """
    Train a model with AdamW optimizer and CrossEntropy loss.

    Saves the best checkpoint (by training accuracy) to the checkpoints
    directory. Prints epoch-wise loss and accuracy.

    Args:
        model:        PyTorch model to train
        train_loader: DataLoader for training data
        epochs:       Number of training epochs
        lr:           Learning rate
        model_name:   Name prefix for the saved checkpoint file

    Returns:
        float: Best training accuracy achieved
    """
    device = DEVICE
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_acc = 0.0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training {model_name} on {device}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        print(f"  Epoch [{epoch+1}/{epochs}]  "
              f"Loss: {epoch_loss:.4f}  "
              f"Acc: {epoch_acc:.4f}")

        # Save best checkpoint
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            save_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best checkpoint: {save_path}")

    print(f"\n✅ Training complete. Best accuracy: {best_acc:.4f}\n")
    return best_acc


# ─────────────────────────────────────────────────────────────────────────────
# Memory Cleanup Utility
# ─────────────────────────────────────────────────────────────────────────────

def cleanup_memory(model=None):
    """
    Release GPU memory after training a model.

    Args:
        model: Model reference to delete (optional)
    """
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
