import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_hybrid_model(model, loader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []

    print("Generating predictions for evaluation...")
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 1. Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # 2. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Hybrid Model')
    plt.show()

# ডিভাইস সেট করা
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# সেরা চেকপয়েন্ট লোড করা
checkpoint_path = os.path.join(CKPT_DIR, "hybrid_cnn_transformer_best.pth")
hybrid_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
hybrid_model.to(device)

# ইভালুয়েশন চালানো
if train_loader:
    evaluate_hybrid_model(hybrid_model, train_loader, device, dataset.classes)
else:
    print("Error: DataLoader not found!")