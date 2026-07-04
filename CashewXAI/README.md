# 🔬 Multi-XAI Hybrid CNN–Transformer Framework for Cashew Disease Detection

**Researcher / Developer:** [Your Name/ID]  
**Supervisor:** [Professor's Name]

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

> **Sir, this repository contains the complete implementation of my research work: A novel explainable AI framework combining CNN and Vision Transformer architectures with multi-modal XAI interpretability for automated cashew leaf disease classification.**

---

## 📋 Table of Contents

- [Research Abstract](#research-abstract)
- [How I Structured the Project](#how-i-structured-the-project)
- [My Proposed Methodology](#my-proposed-methodology)
- [Dataset Preparation](#dataset-preparation)
- [Experimental Results](#experimental-results)
- [How to Run My Code](#how-to-run-my-code)

---

## 📄 Research Abstract

In this research, I have developed a **Multi-XAI Hybrid CNN–Transformer Framework** for cashew disease detection. I designed the architecture to fuse spatial features from **EfficientNet-B0** (which acts as my CNN branch) with global contextual representations from **CaiT-XXS24** (my Transformer branch) through a late-fusion classifier. 

Through my experiments, the framework achieved **99.02% accuracy** across five disease categories. To ensure the model isn't just a "black box," I implemented a multi-modal XAI pipeline that integrates **Grad-CAM**, **Grad-CAM++**, **LIME**, and **SHAP** explanations so we can visually verify what the model is learning.

### The Disease Categories I Analyzed:
| Class | Description |
|-------|-------------|
| Anthracnose | Fungal leaf spot disease |
| Gumosis | Gum exudation disease |
| Healthy | Disease-free leaves |
| Leaf Miner | Insect-induced leaf damage |
| Red Rust | Algal leaf disease |

---

## 📁 How I Structured the Project

To ensure my code is professional, modular, and easy for you to review, I organized my work into the following structure:

```
CashewXAI/
│
├── README.md                           # My project documentation
├── requirements.txt                    # All Python dependencies I used
│
├── configs/
│   └── config.py                       # I centralized all my hyperparameters and paths here
│
├── src/
│   ├── dataset.py                      # My dataset class and data augmentation pipeline
│   ├── model.py                        # My custom Hybrid CNN-Transformer architecture
│   ├── trainer.py                      # My training loop and checkpoint saving logic
│   ├── evaluator.py                    # My script to generate classification reports & confusion matrix
│   └── data_utils.py                   # My scripts for Kaggle downloads and Google Drive syncing
│
├── xai/
│   ├── gradcam_visualizer.py           # Script I wrote for Grad-CAM heatmaps
│   ├── gradcam_plus_plus_lime.py       # Script I wrote to compare Grad-CAM++ & LIME
│   └── shap_explainer.py               # Script I wrote for SHAP feature attribution
│
├── baselines/
│   ├── train_cnn_baselines.py          # Script I used to train the 5 CNN baseline models
│   └── train_transformer_baselines.py  # Script I used to train the 5 Transformer models
│
├── inference/
│   └── predict.py                      # Script I wrote to test the model on real-world unseen images
│
├── outputs/                            # Where I save all my generated results
│   ├── training_logs/                  # My text logs for loss and accuracy
│   └── figures/                        # All the heatmaps and matrices my code generates
│
├── docs/                               # My research methodology and manuscript drafts
├── notebooks/                          
│   └── colab_full_pipeline.md          # My step-by-step guide for executing in Colab
├── local_training/
│   └── local_train_guide.md            # My guide for executing on a local computer
│
├── checkpoints/                        # Where my saved model weights (.pth files) are stored
├── datasets/                           # Where the raw image data is stored
└── results/                            # CSV files containing my benchmark comparisons
```

---

## 🔬 My Proposed Methodology

### Architecture Overview

I designed the model to process images simultaneously through two distinct branches before fusing their outputs:

```text
Input Image (224×224×3)
        │
        ├──────────────────┐
        │                  │
   CNN Branch         Transformer Branch
  (EfficientNet-B0)    (CaiT-XXS24-224)
        │                  │
  AdaptiveAvgPool2d    Identity Head
   (1280-dim)          (192-dim)
        │                  │
        └────── Concat ────┘
                  │
           Fusion (1472-dim)
                  │
          FC → ReLU → Dropout
                  │
          FC → 5 Classes
```

### My Multi-XAI Pipeline
To prove the model's reliability, I integrated four explanation methods:
1. **Grad-CAM** — I used this to show class-discriminative localization via gradient-weighted activation maps.
2. **Grad-CAM++** — I added this improved variant for pixel-wise gradient weighting.
3. **LIME** — I implemented this to show local superpixel perturbation-based explanations.
4. **SHAP** — I utilized this for mathematically rigorous gradient-based feature attribution.

---

## 📊 Experimental Results

### Baseline Comparisons I Conducted

To prove my hybrid model is superior, I trained and tested 10 different architectures against it. Here are the best accuracies I recorded:

| Architecture | Model | Best Accuracy |
|-------------|-------|--------------|
| CNN | ResNet-50 | 96.15% |
| CNN | DenseNet-121 | 97.54% |
| CNN | **EfficientNet-B0** | **97.82%** |
| CNN | VGG-16 | 96.32% |
| CNN | MobileNetV3-Large | 97.48% |
| Transformer | ViT-Tiny | 97.42% |
| Transformer | Swin-Tiny | 97.36% |
| Transformer | DeiT-Tiny | 96.92% |
| Transformer | **CaiT-XXS24** | **97.54%** |
| Transformer | CrossViT-9 | 97.34% |
| **My Hybrid** | **EfficientNet-B0 + CaiT-XXS24** | **99.02%** |

### Classification Report (My Hybrid Model)

When I evaluated the final model, it produced the following metrics:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Anthracnose | 1.00 | 0.98 | 0.99 | 1,729 |
| Gumosis | 1.00 | 1.00 | 1.00 | 392 |
| Healthy | 0.99 | 1.00 | 1.00 | 1,368 |
| Leaf Miner | 0.99 | 1.00 | 0.99 | 1,378 |
| Red Rust | 1.00 | 1.00 | 1.00 | 1,682 |
| **Weighted Avg** | **0.99** | **0.99** | **0.99** | **6,549** |

---

## 🚀 How to Run My Code

Sir, I have prepared two detailed documents to show exactly how you or any reviewer can run my code from start to finish:

1. **For Google Colab:** Please review `notebooks/colab_full_pipeline.md`
2. **For Local Workstations:** Please review `local_training/local_train_guide.md`

All dependencies I used are listed in `requirements.txt`.
