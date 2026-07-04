# =============================================================================
# Data Utilities — Dataset Download, Sync & Preprocessing
# =============================================================================
# Multi-XAI Hybrid CNN-Transformer Framework for Cashew Disease Detection
# =============================================================================

import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs.config import (
    PROJECT_ROOT, DATASET_DIR, TRAIN_PATH,
    KAGGLE_DATASET_SLUG, is_colab,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Download (Kaggle)
# ─────────────────────────────────────────────────────────────────────────────

def download_dataset():
    """
    Download the cashew disease dataset from Kaggle using kagglehub.

    Returns:
        str: Path to the downloaded dataset directory
    """
    import kagglehub

    print("📥 Downloading dataset from Kaggle...")
    raw_path = kagglehub.dataset_download(KAGGLE_DATASET_SLUG)
    base_img_path = os.path.join(raw_path, 'Cashew')
    print(f"✅ Dataset downloaded to: {base_img_path}")
    return base_img_path


# ─────────────────────────────────────────────────────────────────────────────
# Data Sync (Kaggle Cache → Google Drive)
# ─────────────────────────────────────────────────────────────────────────────

def sync_data_to_drive(source, destination):
    """
    Copy dataset from Kaggle cache to Google Drive for persistence.

    Skips copy if destination already contains data.

    Args:
        source:      Path to source directory (Kaggle cache)
        destination: Path to destination directory (Google Drive)
    """
    if not os.path.exists(source):
        print(f"❌ Source path not found: {source}")
        return

    if not os.path.exists(destination) or len(os.listdir(destination)) == 0:
        print(f"📂 Copying data from {source} → {destination}...")
        shutil.copytree(source, destination, dirs_exist_ok=True)
        print("✅ Data sync complete!")
    else:
        print("✅ Data already exists in Drive. Skipping copy.")


def setup_dataset():
    """
    Full dataset setup pipeline: download from Kaggle and sync to Drive.

    Returns:
        str: Path to the training data directory
    """
    os.makedirs(DATASET_DIR, exist_ok=True)

    source_path = download_dataset()
    sync_data_to_drive(source_path, TRAIN_PATH)

    return TRAIN_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Summary & Visualization
# ─────────────────────────────────────────────────────────────────────────────

def summarize_dataset(data_path=None):
    """
    Print a summary table showing image counts per disease category.

    Args:
        data_path: Path to the dataset root (defaults to TRAIN_PATH)

    Returns:
        pd.DataFrame: Summary dataframe with Category and Count columns
    """
    data_path = data_path or TRAIN_PATH

    summary = []
    for category in sorted(os.listdir(data_path)):
        category_path = os.path.join(data_path, category)
        if os.path.isdir(category_path):
            num_images = len([
                f for f in os.listdir(category_path)
                if os.path.isfile(os.path.join(category_path, f))
            ])
            summary.append({'Category': category, 'Count': num_images})

    df = pd.DataFrame(summary).sort_values(by='Count', ascending=False)
    print("\n📊 Dataset Summary:")
    print(df.to_string(index=False))
    return df


def display_sample_images(data_path=None, num_images=5):
    """
    Display random sample images from each disease category.

    Args:
        data_path:  Path to the dataset root (defaults to TRAIN_PATH)
        num_images: Number of sample images to display
    """
    data_path = data_path or TRAIN_PATH

    categories = [
        d for d in sorted(os.listdir(data_path))
        if os.path.isdir(os.path.join(data_path, d))
    ]
    selected = random.sample(categories, min(num_images, len(categories)))

    fig, axes = plt.subplots(1, len(selected), figsize=(20, 10))
    if len(selected) == 1:
        axes = [axes]

    for i, category in enumerate(selected):
        category_path = os.path.join(data_path, category)
        files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
        if not files:
            continue

        img_name = random.choice(files)
        img_path = os.path.join(category_path, img_name)
        img = mpimg.imread(img_path)

        axes[i].imshow(img)
        axes[i].set_title(category, fontsize=14, fontweight='bold')
        axes[i].axis('off')

    plt.suptitle('Sample Images — Cashew Disease Categories', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Google Colab Utilities
# ─────────────────────────────────────────────────────────────────────────────

def mount_google_drive():
    """Mount Google Drive in Colab environment."""
    if is_colab():
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted.")
    else:
        print("ℹ Not running in Colab. Skipping Drive mount.")


def setup_kaggle_credentials():
    """Upload kaggle.json credentials in Colab."""
    if is_colab():
        from google.colab import files
        print("📄 Please upload your kaggle.json file:")
        files.upload()
        os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
        shutil.copy('kaggle.json', os.path.expanduser('~/.kaggle/kaggle.json'))
        os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
        print("✅ Kaggle credentials configured.")
    else:
        print("ℹ Not in Colab. Ensure ~/.kaggle/kaggle.json exists locally.")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    summarize_dataset()
    display_sample_images()
