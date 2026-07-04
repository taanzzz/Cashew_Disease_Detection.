# =============================================================================
# Project Configuration — Centralized Hyperparameters & Path Management
# =============================================================================
# Multi-XAI Hybrid CNN-Transformer Framework for Cashew Disease Detection
# =============================================================================

import os
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Environment Detection
# ─────────────────────────────────────────────────────────────────────────────

def is_colab():
    """Detect if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Path Configuration
# ─────────────────────────────────────────────────────────────────────────────

if is_colab():
    # Google Colab: data persisted on Google Drive
    PROJECT_ROOT = '/content/drive/MyDrive/Cashew_XAI_Project'
else:
    # Local machine: use relative project directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR   = os.path.join(PROJECT_ROOT, 'datasets')
TRAIN_PATH    = os.path.join(DATASET_DIR, 'train')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
RESULTS_DIR   = os.path.join(PROJECT_ROOT, 'results')
XAI_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'figures')
LOG_DIR       = os.path.join(PROJECT_ROOT, 'outputs', 'training_logs')


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Configuration
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ['anthracnose', 'gumosis', 'healthy', 'leaf miner', 'red rust']
NUM_CLASSES = len(CLASS_NAMES)

KAGGLE_DATASET_SLUG = 'olaidegabriel/cashew-disease-detection-dataset'


# ─────────────────────────────────────────────────────────────────────────────
# ImageNet Normalization Constants
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224


# ─────────────────────────────────────────────────────────────────────────────
# Training Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE        = 32
NUM_WORKERS       = 2
BASELINE_EPOCHS   = 5
HYBRID_EPOCHS     = 10
BASELINE_LR       = 1e-4
HYBRID_LR         = 5e-5
DROPOUT_RATE      = 0.3
FUSION_HIDDEN_DIM = 512


# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture Configuration
# ─────────────────────────────────────────────────────────────────────────────

# CNN Branch (best baseline)
CNN_BACKBONE = 'efficientnet_b0'

# Transformer Branch (best baseline)
TRANSFORMER_BACKBONE = 'cait_xxs24_224'

# CNN Baselines for comparative study
CNN_BASELINES = [
    'resnet50',
    'densenet121',
    'efficientnet_b0',
    'vgg16',
    'mobilenetv3_large_100',
]

# Transformer Baselines for comparative study
TRANSFORMER_BASELINES = [
    'vit_tiny_patch16_224',
    'swin_tiny_patch4_window7_224',
    'deit_tiny_patch16_224',
    'cait_xxs24_224',
    'crossvit_9_240',
]


# ─────────────────────────────────────────────────────────────────────────────
# Device Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Configuration
# ─────────────────────────────────────────────────────────────────────────────

SHAP_NUM_SAMPLES = 20  # Reduced to prevent OOM in Colab free tier


# ─────────────────────────────────────────────────────────────────────────────
# Directory Initialization
# ─────────────────────────────────────────────────────────────────────────────

def initialize_directories():
    """Create all required project directories if they do not exist."""
    directories = [
        DATASET_DIR,
        TRAIN_PATH,
        CHECKPOINT_DIR,
        RESULTS_DIR,
        XAI_OUTPUT_DIR,
        LOG_DIR,
        os.path.join(RESULTS_DIR, 'baseline_cnn'),
        os.path.join(RESULTS_DIR, 'baseline_transformer'),
        os.path.join(RESULTS_DIR, 'final'),
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)


if __name__ == '__main__':
    initialize_directories()
    print(f"Project Root  : {PROJECT_ROOT}")
    print(f"Device        : {DEVICE}")
    print(f"Classes       : {CLASS_NAMES}")
    print(f"Directories initialized successfully. ✅")
