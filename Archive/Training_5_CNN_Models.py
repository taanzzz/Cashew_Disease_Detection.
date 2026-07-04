
import torch.nn as nn
import torch.optim as optim
import time
import json
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import timm
import gc
import os

# প্রজেক্ট পাথ কনফিগারেশন নিশ্চিত করা
PROJECT_ROOT = '/content/drive/MyDrive/Cashew_XAI_Project'
CKPT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

# ডিরেক্টরি তৈরি নিশ্চিত করা
os.makedirs(CKPT_DIR, exist_ok=True)

def train_model(model, train_loader, epochs=10, lr=1e-4, model_name="model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_acc = 0.0
    print(f"Training {model_name} on {device}...")

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

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

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            # global CKPT_DIR variable is used here
            save_path = os.path.join(CKPT_DIR, f"{model_name}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best checkpoint to: {save_path}")

    return best_acc
cnn_names = ["resnet50", "densenet121", "efficientnet_b0", "vgg16", "mobilenetv3_large_100"]
cnn_results = []

for name in cnn_names:
    print(f"\nInitializing {name}...")
    model = timm.create_model(name, pretrained=True, num_classes=5)

    # দ্রুত টেস্ট করার জন্য আমরা ৫টি ইপোক ব্যবহার করছি
    best_acc = train_model(model, train_loader, epochs=5, model_name=f"cnn_{name}")
    cnn_results.append({"Model": name, "Best_Train_Acc": best_acc})

    # মেমোরি খালি করা
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

df_cnn_results = pd.DataFrame(cnn_results)
print("\n--- CNN Baseline Results ---")
print(df_cnn_results)