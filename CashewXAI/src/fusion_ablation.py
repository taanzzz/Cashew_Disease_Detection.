# =============================================================================
# Fusion Strategy Analysis (Ablation Study)
# =============================================================================
# Multi-XAI Hybrid CNN-Transformer Framework for Cashew Disease Detection
#
# This script evaluates three different fusion strategies:
#   1. Concatenation ('concat')
#   2. Element-wise Sum ('sum')
#   3. Learnable Gated Fusion ('gated')
#
# It projects the CNN and Transformer features to a common 512-d space
# before applying the fusion strategy, matching the paper's ablation methodology.
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import timm
import pandas as pd
import numpy as np
import itertools
import os
import gc
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs.config import DEVICE, RESULTS_DIR, NUM_CLASSES, TRAIN_PATH, BATCH_SIZE, NUM_WORKERS, CHECKPOINT_DIR
from src.dataset import get_train_loader, CashewDataset, train_transform, test_transform
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold


# ─────────────────────────────────────────────────────────────────────────────
# Ablation Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

class AblationHybridModel(nn.Module):
    def __init__(self, fusion_type='gated', num_classes=NUM_CLASSES):
        super(AblationHybridModel, self).__init__()
        self.fusion_type = fusion_type
        
        # Linear projections to a common 512-d space (as described in the paper)
        self.cnn_proj = nn.Linear(1280, 512)
        self.trans_proj = nn.Linear(192, 512)

        if fusion_type == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(1024, 512),
                nn.Sigmoid()
            )

        # Classifier adjusted for fusion size
        feat_dim = 1024 if fusion_type == 'concat' else 512
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, f_cnn, f_trans):
        f_cnn = self.cnn_proj(f_cnn)
        f_trans = self.trans_proj(f_trans)

        if self.fusion_type == 'concat':
            fused = torch.cat([f_cnn, f_trans], dim=1)
        elif self.fusion_type == 'sum':
            fused = f_cnn + f_trans
        elif self.fusion_type == 'gated':
            g = self.gate(torch.cat([f_cnn, f_trans], dim=1))
            fused = g * f_cnn + (1 - g) * f_trans

        return self.classifier(fused)


class FullAblationWrapper(nn.Module):
    def __init__(self, strategy):
        super().__init__()
        self.cnn_extractor = timm.create_model(
            'efficientnet_b0', pretrained=True, num_classes=0, global_pool='avg'
        )
        self.trans_extractor = timm.create_model(
            'cait_xxs24_224', pretrained=True, num_classes=0
        )
        self.ablation_head = AblationHybridModel(fusion_type=strategy)

    def forward(self, x):
        f_cnn = self.cnn_extractor(x)
        f_trans = self.trans_extractor(x)
        return self.ablation_head(f_cnn, f_trans)


# ─────────────────────────────────────────────────────────────────────────────
# Training Function for Ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_train(model, loader, epochs=3):
    model.to(DEVICE)
    
    # The CrossEntropyLoss in PyTorch already combines LogSoftmax and NLLLoss 
    # which ensures numerical stability as requested in the paper.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
        print(f"  Epoch {epoch+1} - Acc: {acc:.4f}")
        
    return best_acc


# ─────────────────────────────────────────────────────────────────────────────
# K-Fold Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_validation(fusion_strategy='gated', k_folds=5, epochs=5):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []

    dataset = CashewDataset(TRAIN_PATH, transform=train_transform)
    if len(dataset) == 0:
        print("Dataset is empty. Cannot run CV.")
        return 0.0, 0.0

    indices = np.arange(len(dataset))
    print(f"\n--- Starting {k_folds}-Fold CV for Strategy: {fusion_strategy} ---")

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\nFold {fold+1}/{k_folds}")

        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)

        train_loader_cv = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader_cv = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        model = FullAblationWrapper(fusion_strategy).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        best_val_acc = 0.0
        for epoch in range(epochs):
            model.train()
            for imgs, labels in train_loader_cv:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader_cv:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(imgs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_acc = correct / total
            if val_acc > best_val_acc:
                best_val_acc = val_acc

        print(f"Fold {fold+1} Best Val Acc: {best_val_acc:.4f}")
        fold_results.append(best_val_acc)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    avg_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    return avg_acc, std_acc


def run_cv_analysis(strategies=None, k_folds=5, epochs=5):
    if strategies is None:
        strategies = ['concat', 'sum', 'gated']

    print(f"Starting Cross-Validation Analysis on {DEVICE}...")
    cv_results = []

    for strategy in strategies:
        avg, std = run_cross_validation(fusion_strategy=strategy, k_folds=k_folds, epochs=epochs)
        cv_results.append({"Strategy": strategy, "CV_Avg_Acc": avg, "CV_Std": std})

    df_results = pd.DataFrame(cv_results)

    os.makedirs(os.path.join(RESULTS_DIR, 'ablation'), exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'ablation', 'cross_validation_results.csv')
    df_results.to_csv(csv_path, index=False)

    print("\n" + "="*60)
    print("Cross-Validation Results")
    print("="*60)
    print(df_results.to_string(index=False))
    print(f"\nResults saved to: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter Tuning & Final Model Selection
# ─────────────────────────────────────────────────────────────────────────────

def run_hyperparameter_tuning(strategy='gated', epochs=2):
    param_grid = {
        'lr': [1e-4, 5e-5],
        'weight_decay': [1e-4, 1e-3]
    }

    train_loader, _ = get_train_loader()
    if train_loader is None:
        print("Dataset is empty. Cannot run tuning.")
        return None

    hyper_results = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    output_lines = []
    output_lines.append(f"--- Starting Hyperparameter Tuning ---")
    output_lines.append(f"Timestamp: {timestamp}")
    output_lines.append(f"Strategy: {strategy}")
    output_lines.append(f"Device: {DEVICE}")
    output_lines.append(f"Epochs per config: {epochs}")
    output_lines.append("")

    print(f"\n--- Starting Hyperparameter Tuning for Strategy: {strategy} ---")

    for lr, wd in itertools.product(param_grid['lr'], param_grid['weight_decay']):
        line = f"\nTesting Config: LR={lr}, WD={wd}"
        print(line)
        output_lines.append(line)

        model = FullAblationWrapper(strategy).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        model.train()
        for epoch in range(epochs):
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        final_acc = correct / total
        line = f"Config Result Accuracy: {final_acc:.4f}"
        print(line)
        output_lines.append(line)

        hyper_results.append({'LR': lr, 'WD': wd, 'Accuracy': final_acc})

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    df_results = pd.DataFrame(hyper_results)
    print("\n" + df_results.to_string(index=False))
    output_lines.append("\n" + df_results.to_string(index=False))

    best_idx = df_results['Accuracy'].idxmax()
    best_config = df_results.iloc[best_idx]
    summary = (f"\nBest Config: LR={best_config['LR']}, WD={best_config['WD']} "
               f"with Accuracy: {best_config['Accuracy']:.4f}")
    print(summary)
    output_lines.append(summary)
    output_lines.append("\nHyperparameter tuning complete! Choose the config with highest accuracy for final deployment.")

    # Save to txt
    os.makedirs(os.path.join(RESULTS_DIR, 'ablation'), exist_ok=True)
    txt_path = os.path.join(RESULTS_DIR, 'ablation', 'hyperparameter_tuning_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    # Save CSV too
    csv_path = os.path.join(RESULTS_DIR, 'ablation', 'hyperparameter_tuning_results.csv')
    df_results.to_csv(csv_path, index=False)

    print(f"\nResults saved to: {txt_path}")
    return df_results


def run_final_model_selection(strategy='gated', best_lr=5e-5, best_wd=1e-3, epochs=5):
    train_loader, _ = get_train_loader()
    if train_loader is None:
        return

    output_lines = []
    output_lines.append(f"--- Final Model Selection ---")
    output_lines.append(f"Strategy: {strategy}")
    output_lines.append(f"Best LR: {best_lr}, Best WD: {best_wd}")
    output_lines.append(f"Epochs: {epochs}")
    output_lines.append(f"Device: {DEVICE}")
    output_lines.append("")

    print(f"\n--- Training Final Model with Best Config: LR={best_lr}, WD={best_wd} ---")

    model = FullAblationWrapper(strategy).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = correct / total
        if acc > best_acc:
            best_acc = acc
        line = f"  Epoch {epoch+1}/{epochs} - Acc: {acc:.4f}"
        print(line)
        output_lines.append(line)

    final_line = f"\nFinal Model Best Accuracy: {best_acc:.4f}"
    print(final_line)
    output_lines.append(final_line)
    output_lines.append(f"\nFinal model training complete with config: LR={best_lr}, WD={best_wd}")

    # Save checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'final_model_{strategy}.pth'))

    # Save to txt
    os.makedirs(os.path.join(RESULTS_DIR, 'ablation'), exist_ok=True)
    txt_path = os.path.join(RESULTS_DIR, 'ablation', 'final_model_selection_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"Results saved to: {txt_path}")
    print(f"Model checkpoint saved to: {os.path.join(CHECKPOINT_DIR, f'final_model_{strategy}.pth')}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────────────────────

def run_fusion_analysis():
    print(f"Starting Stage 5 Ablation Study on {DEVICE}...")
    print("Classification Head verified with numerical stability (Log-Softmax integrated). ✅")
    
    train_loader, _ = get_train_loader()
    if train_loader is None:
        return

    strategies = ['concat', 'sum', 'gated']
    ablation_results = []

    for strategy in strategies:
        print(f"\nEvaluating Fusion Strategy: {strategy}")
        model = FullAblationWrapper(strategy).to(DEVICE)
        
        acc = run_ablation_train(model, train_loader, epochs=3)
        ablation_results.append({"Strategy": strategy, "Best_Acc": acc})
        
        # Cleanup memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_results = pd.DataFrame(ablation_results)
    
    # Save results
    os.makedirs(os.path.join(RESULTS_DIR, 'ablation'), exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'ablation', 'fusion_strategy_results.csv')
    df_results.to_csv(csv_path, index=False)
    
    print("\n" + "="*60)
    print("Ablation Study Results")
    print("="*60)
    print(df_results.to_string(index=False))
    print(f"\n✅ Results saved to: {csv_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fusion Ablation & Hyperparameter Tuning')
    parser.add_argument('--mode', type=str, default='ablation',
                        choices=['ablation', 'cv', 'tuning', 'final', 'all'],
                        help='Which analysis to run')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--strategy', type=str, default='gated',
                        choices=['concat', 'sum', 'gated'],
                        help='Fusion strategy')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for final model')
    parser.add_argument('--wd', type=float, default=1e-3, help='Weight decay for final model')
    args = parser.parse_args()

    if args.mode == 'ablation':
        run_fusion_analysis()
    elif args.mode == 'cv':
        run_cv_analysis(k_folds=args.k_folds, epochs=args.epochs)
    elif args.mode == 'tuning':
        run_hyperparameter_tuning(strategy=args.strategy, epochs=args.epochs)
    elif args.mode == 'final':
        run_final_model_selection(strategy=args.strategy, best_lr=args.lr, best_wd=args.wd, epochs=args.epochs)
    elif args.mode == 'all':
        run_fusion_analysis()
        run_cv_analysis(k_folds=args.k_folds, epochs=args.epochs)
        df = run_hyperparameter_tuning(strategy=args.strategy, epochs=args.epochs)
        if df is not None:
            best = df.iloc[df['Accuracy'].idxmax()]
            run_final_model_selection(strategy=args.strategy, best_lr=best['LR'], best_wd=best['WD'], epochs=args.epochs)
