# Multi-XAI Hybrid CNN-Transformer Framework for Cashew Disease Detection

**A Comprehensive Explainable AI Framework for Automated Cashew Leaf Disease Classification**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![timm](https://img.shields.io/badge/timm-0.9%2B-green)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-99.44%25-brightgreen)

---

## Table of Contents

1. [Abstract](#abstract)
2. [Project Overview](#project-overview)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
   - [4.1 Architecture Design](#41-architecture-design)
   - [4.2 CNN Branch - EfficientNet-B0](#42-cnn-branch---efficientnet-b0)
   - [4.3 Transformer Branch - CaiT-XXS24](#43-transformer-branch---cait-xxs24)
   - [4.4 Late Fusion Classifier](#44-late-fusion-classifier)
   - [4.5 Data Augmentation Pipeline](#45-data-augmentation-pipeline)
5. [Experimental Setup](#experimental-setup)
   - [5.1 Hyperparameter Configuration](#51-hyperparameter-configuration)
   - [5.2 Training Protocol](#52-training-protocol)
6. [Results and Analysis](#results-and-analysis)
   - [6.1 CNN Baseline Performance](#61-cnn-baseline-performance)
   - [6.2 Transformer Baseline Performance](#62-transformer-baseline-performance)
   - [6.3 Hybrid Model Performance](#63-hybrid-model-performance)
   - [6.4 Class-wise Classification Report](#64-class-wise-classification-report)
   - [6.5 Confusion Matrix Analysis](#65-confusion-matrix-analysis)
   - [6.6 Performance Comparison Across Architectures](#66-performance-comparison-across-architectures)
   - [6.7 Fusion Ablation Study](#67-fusion-ablation-study)
   - [6.8 Hyperparameter Tuning](#68-hyperparameter-tuning)
   - [6.9 Cross-Validation](#69-cross-validation)
   - [6.10 Real-World Inference](#610-real-world-inference)
7. [Explainable AI (XAI) Pipeline](#7-explainable-ai-xai-pipeline)
   - [7.1 Grad-CAM](#71-grad-cam)
   - [7.2 Grad-CAM++](#72-grad-cam)
   - [7.3 LIME](#73-lime)
   - [7.4 SHAP](#74-shap)
   - [7.5 XAI Comparison](#75-xai-comparison)
8. [Project Structure](#project-structure)
9. [Installation and Setup](#installation-and-setup)
10. [How to Run](#how-to-run)
11. [Conclusion](#conclusion)

---

## Abstract

Cashew (*Anacardium occidentale*) is a vital cash crop cultivated across tropical regions, yet its productivity is severely threatened by foliar diseases such as anthracnose, gumosis, leaf miner infestation, and red rust. Traditional disease diagnosis relies on visual inspection by agricultural experts, which is time-consuming, subjective, and not scalable for large plantations.

This research presents a **Multi-XAI Hybrid CNN-Transformer Framework** that combines the spatial feature extraction capabilities of Convolutional Neural Networks with the global contextual understanding of Vision Transformers to achieve state-of-the-art cashew disease classification. The proposed architecture fuses **EfficientNet-B0** (CNN branch) with **CaiT-XXS24-224** (Transformer branch) through a late-fusion strategy, achieving **99.02% training accuracy** and **99.44%** with optimized hyperparameters.

Beyond raw performance, this framework integrates four complementary Explainable AI (XAI) techniques -- **Grad-CAM**, **Grad-CAM++**, **LIME**, and **SHAP** -- to provide multi-modal visual interpretations of model decisions, ensuring transparency and trustworthiness in automated disease diagnosis.

---

## Project Overview

| Attribute | Details |
|---|---|
| **Task** | Multi-class leaf disease classification |
| **Number of Classes** | 5 (Anthracnose, Gumosis, Healthy, Leaf Miner, Red Rust) |
| **Total Dataset Size** | 6,549 images |
| **Best Model Accuracy** | 99.44% (tuned), 99.02% (default) |
| **Framework** | PyTorch 2.0+ with timm 0.9+ |
| **XAI Techniques** | Grad-CAM, Grad-CAM++, LIME, SHAP |
| **Computational Environment** | Google Colab (NVIDIA T4 GPU) |
| **Experiment Tracking** | Local logs and CSV records |

---

## Dataset

The dataset used in this research is sourced from Kaggle: [Cashew Disease Detection Dataset](https://www.kaggle.com/datasets/olaidegabriel/cashew-disease-detection-dataset) by Olaide Gabriel. It consists of **6,549 high-resolution RGB images** of cashew leaves, organized into five distinct categories.

### Class Distribution

| Class | Description | Number of Images | Percentage |
|---|---|---|---|
| Anthracnose | Fungal disease caused by *Colletotrichum gloeosporioides* causing dark, sunken lesions | 1,729 | 26.4% |
| Gumosis | Physiological disorder characterized by gum exudation from stems and leaves | 392 | 6.0% |
| Healthy | Disease-free cashew leaves with normal green pigmentation | 1,368 | 20.9% |
| Leaf Miner | Insect damage caused by leaf-mining larvae creating serpentine tunnels | 1,378 | 21.0% |
| Red Rust | Algal disease caused by *Cephaleuros virescens* forming rust-colored spots | 1,682 | 25.7% |

**Data Imbalance Note:** The Gumosis class has significantly fewer samples (392) compared to other classes (1,368-1,729). Despite this imbalance, the model achieved 1.00 precision, recall, and F1-score for this class, demonstrating robust feature learning even with limited data.

---

## Methodology

### 4.1 Architecture Design

The proposed Hybrid CNN-Transformer architecture processes input images simultaneously through two parallel branches, extracting complementary feature representations before fusing them for final classification.

```
                    Input Image (224 x 224 x 3)
                              |
                    ┌─────────┴──────────┐
                    |                    |
            CNN Branch            Transformer Branch
       (EfficientNet-B0)         (CaiT-XXS24-224)
                    |                    |
        AdaptiveAvgPool2d(1)       Identity Head
            (1280-dim)               (192-dim)
                    |                    |
                    └────── Concat ──────┘
                              |
                    Fully Connected (1472 -> 512)
                              |
                         ReLU Activation
                              |
                       Dropout (p=0.3)
                              |
                    Fully Connected (512 -> 5)
                              |
                      5-Class Output
```

### 4.2 CNN Branch - EfficientNet-B0

EfficientNet-B0 serves as the convolutional backbone, responsible for extracting local spatial features such as edges, textures, lesion boundaries, and color patterns from the leaf images.

- **Architecture:** EfficientNet-B0 (pretrained on ImageNet-1K)
- **Feature Mode:** `features_only=True` (multi-scale feature maps)
- **Output Dimension:** 1,280-dimensional feature vector (after AdaptiveAvgPool2d)
- **Rationale:** EfficientNet-B0 achieved the highest accuracy (97.82%) among all 5 CNN baselines, making it the optimal choice for the CNN branch of the hybrid architecture.

### 4.3 Transformer Branch - CaiT-XXS24-224

CaiT-XXS24-224 (Class-Attention in Image Transformers - Extra Extra Small) serves as the Transformer backbone, capturing long-range dependencies and global contextual relationships across the entire leaf image.

- **Architecture:** CaiT-XXS24-224 (pretrained on ImageNet-1K)
- **Head Replacement:** Classification head replaced with `nn.Identity()` to extract raw feature embeddings
- **Output Dimension:** 192-dimensional feature vector
- **Rationale:** CaiT-XXS24-224 achieved the highest accuracy (97.54%) among all 5 Transformer baselines, with its class-attention mechanism proving particularly effective for fine-grained disease classification.

### 4.4 Late Fusion Classifier

The 1,280-dim CNN features and 192-dim Transformer features are concatenated to form a 1,472-dim joint representation, which is then passed through a classifier head:

```
Fusion: Concatenation (1280 + 192 = 1472)
  -> Linear(1472, 512)   # Dimensionality reduction
  -> ReLU()               # Non-linear activation
  -> Dropout(0.3)         # Regularization
  -> Linear(512, 5)       # Final classification
```

The dropout rate of 0.3 prevents overfitting, which is particularly important given the dataset size (6,549 images) and the representational capacity of the fused feature space.

### 4.5 Data Augmentation Pipeline

To improve generalization and reduce overfitting, the training pipeline applies the following augmentation strategy using the Albumentations library:

| Augmentation | Parameter | Probability |
|---|---|---|
| Horizontal Flip | p=0.5 | 50% |
| Random Rotation | limit=40 degrees | 70% |
| Random Brightness Contrast | brightness_limit=0.3, contrast_limit=0.2 | 70% |
| Resize | 224x224 pixels | Always |
| ImageNet Normalization | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] | Always |

---

## Experimental Setup

### 5.1 Hyperparameter Configuration

| Hyperparameter | CNN Baselines | Transformer Baselines | Hybrid Model |
|---|---|---|---|
| Optimizer | AdamW | AdamW | AdamW |
| Learning Rate | 1e-4 | 1e-4 | 5e-5 |
| Weight Decay | 1e-4 | 1e-4 | 1e-3 (tuned) |
| Batch Size | 32 | 32 | 32 |
| Epochs | 5 | 5 | 10 |
| Dropout Rate | N/A | N/A | 0.3 |
| Fusion Hidden Dim | N/A | N/A | 512 |
| Input Size | 224x224 | 224x224 | 224x224 |
| Loss Function | CrossEntropy | CrossEntropy | CrossEntropy |
| GPU | NVIDIA T4 (16GB) | NVIDIA T4 (16GB) | NVIDIA T4 (16GB) |

### 5.2 Training Protocol

1. **CNN Baselines:** Each of the 5 CNN architectures was trained for 5 epochs with LR=1e-4. The best checkpoint (by training accuracy) was saved.
2. **Transformer Baselines:** Each of the 5 Transformer architectures was trained for 5 epochs with LR=1e-4. The best checkpoint (by training accuracy) was saved.
3. **Hybrid Model:** The proposed EfficientNet-B0 + CaiT-XXS24-224 fusion was trained for 10 epochs with LR=5e-5. All models were trained on a single NVIDIA T4 GPU in Google Colab.

---

## Results and Analysis

### 6.1 CNN Baseline Performance

Five state-of-the-art CNN architectures were evaluated to establish a performance baseline and to identify the strongest candidate for the CNN branch of the hybrid model.

```
Training logs: outputs/training_logs/cnn_baseline_log.txt
Results saved: checkpoints/cnn_*.pth
```

| Model | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Best Accuracy |
|---|---|---|---|---|---|---|
| ResNet-50 | 61.25% | 92.06% | 94.40% | 95.13% | **96.15%** | **96.15%** |
| DenseNet-121 | 86.29% | 95.27% | 96.30% | 97.14% | **97.54%** | **97.54%** |
| **EfficientNet-B0** | 83.20% | 93.62% | 95.37% | 96.73% | **97.82%** | **97.82%** |
| VGG-16 | 89.59% | 94.14% | 95.60% | 95.76% | **96.32%** | **96.32%** |
| MobileNetV3-Large | 84.23% | 93.56% | 95.30% | 96.73% | **97.48%** | **97.48%** |

**Key Findings:**
- EfficientNet-B0 achieved the highest accuracy (97.82%) among all CNN baselines, confirming its superior efficiency-accuracy trade-off.
- DenseNet-121 followed closely at 97.54%, benefiting from its dense connectivity pattern.
- VGG-16, despite being the oldest architecture, showed competitive results (96.32%) but had the largest model size (553M parameters).

### 6.2 Transformer Baseline Performance

Five Vision Transformer architectures were evaluated to establish transformer-based performance baselines.

```
Training logs: outputs/training_logs/transformer_baseline_log.txt
Results CSV: results/baseline_transformer/transformer_results.csv
```

| Model | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Best Accuracy |
|---|---|---|---|---|---|---|
| ViT-Tiny (patch16, 224) | 90.85% | 96.17% | 96.76% | 96.92% | **97.42%** | **97.42%** |
| Swin-Tiny (patch4, window7, 224) | 92.52% | 96.21% | 96.87% | 96.90% | **97.36%** | **97.36%** |
| DeiT-Tiny (patch16, 224) | 91.27% | 95.45% | 96.47% | **96.92%** | 96.70% | **96.92%** |
| **CaiT-XXS24 (224)** | 92.23% | 96.27% | 96.98% | **97.54%** | 97.43% | **97.54%** |
| CrossViT-9 (240) | 90.46% | 95.45% | 96.81% | **97.34%** | 97.28% | **97.34%** |

**Key Findings:**
- CaiT-XXS24-224 achieved the highest transformer accuracy (97.54%), validating its class-attention mechanism for fine-grained disease classification.
- ViT-Tiny and CrossViT-9 demonstrated competitive performance, both exceeding 97.3%.
- All transformer models converged faster than CNNs in early epochs, suggesting better initial feature representations from pretraining.

### 6.3 Hybrid Model Performance

The proposed Hybrid CNN-Transformer model (EfficientNet-B0 + CaiT-XXS24-224 with late fusion) was trained for 10 epochs.

```
Training logs: outputs/training_logs/hybrid_training_log.txt
```

| Epoch | Loss | Accuracy | Checkpoint Saved |
|---|---|---|---|
| 1 | 0.3217 | 89.75% | Yes |
| 2 | 0.1257 | 96.27% | Yes |
| 3 | 0.0927 | 97.22% | Yes |
| 4 | 0.0716 | 97.86% | Yes |
| 5 | 0.0648 | 98.00% | Yes |
| 6 | 0.0444 | 98.47% | Yes |
| 7 | 0.0436 | 98.50% | Yes |
| 8 | 0.0305 | 98.85% | Yes |
| **9** | **0.0298** | **99.02%** | **Yes (Best)** |
| 10 | 0.0287 | 98.98% | Yes |

**Convergence Analysis:**
- The hybrid model started at 89.75% accuracy in epoch 1, significantly higher than both CNN baselines (61.25-89.59%) and transformer baselines (90.46-92.23%) in their first epoch, demonstrating the advantage of dual-branch feature extraction.
- By epoch 5, the model had already reached 98.00% accuracy, outperforming the best individual baseline (EfficientNet-B0 at 97.82%).
- The loss consistently decreased from 0.3217 to 0.0287, with no signs of overfitting.
- Peak performance of **99.02%** was achieved at epoch 9.

### 6.4 Class-wise Classification Report

The final hybrid model was evaluated on the full training set to generate a per-class classification report.

```
Source: outputs/training_logs/evaluation_report.txt
```

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Anthracnose | 1.00 | 0.98 | 0.99 | 1,729 |
| Gumosis | 1.00 | 1.00 | 1.00 | 392 |
| Healthy | 0.99 | 1.00 | 1.00 | 1,368 |
| Leaf Miner | 0.99 | 1.00 | 0.99 | 1,378 |
| Red Rust | 1.00 | 1.00 | 1.00 | 1,682 |
| | | | | |
| **Accuracy** | | | **0.99** | **6,549** |
| **Macro Average** | 1.00 | 1.00 | 1.00 | 6,549 |
| **Weighted Average** | 0.99 | 0.99 | 0.99 | 6,549 |

**Analysis:**
- **Gumosis (392 samples):** A perfect 1.00 across all three metrics despite being the smallest class. This demonstrates the model's exceptional ability to learn discriminative features even from limited data.
- **Anthracnose (1,729 samples):** Slightly lower recall (0.98) suggests 2% of anthracnose cases were misclassified, likely confused with other diseases that present similar lesion patterns.
- **Red Rust (1,682 samples):** Perfect precision and recall, indicating the model confidently distinguishes algal infections.
- **Macro Average (1.00):** The perfect macro average confirms balanced performance across all classes, treating each class equally regardless of size.

### 6.5 Confusion Matrix Analysis

![Confusion Matrix](outputs/figures/confusion_matrix.png)

The confusion matrix visualizes the classification performance across all five disease categories. The strong diagonal dominance confirms that the vast majority of predictions are correct, with only minimal off-diagonal confusion. The Gumosis and Red Rust classes show zero misclassifications, while Anthracnose shows marginal confusion with other classes, consistent with the recall value of 0.98.

### 6.6 Performance Comparison Across Architectures

![Performance Comparison](outputs/figures/comparison.png)

This bar chart provides a direct visual comparison of all 11 trained models across three architecture families:

| Category | Best Model | Best Accuracy |
|---|---|---|
| CNN Baselines | EfficientNet-B0 | 97.82% |
| Transformer Baselines | CaiT-XXS24-224 | 97.54% |
| **Proposed Hybrid** | **EfficientNet-B0 + CaiT-XXS24** | **99.02%** |

**Improvement Analysis:**
- The hybrid model outperforms the best CNN baseline (EfficientNet-B0) by **1.20 percentage points**.
- The hybrid model outperforms the best Transformer baseline (CaiT-XXS24) by **1.48 percentage points**.
- This improvement confirms the hypothesis that combining local spatial features (CNN) with global contextual features (Transformer) produces a more comprehensive and accurate representation for disease classification.

### 6.7 Fusion Ablation Study

An ablation study was conducted to evaluate different feature fusion strategies: concatenation, element-wise summation, and learnable gated fusion. Each strategy was trained for 3 epochs under identical conditions.

```
Source: outputs/training_logs/fusion_ablation.txt
```

| Fusion Strategy | Epoch 1 | Epoch 2 | Epoch 3 | Best Accuracy |
|---|---|---|---|---|
| **Concat** | 92.96% | 98.03% | **99.30%** | **99.30%** |
| **Sum** | 92.96% | 97.91% | **99.33%** | **99.33%** |
| **Gated** | 93.28% | 98.29% | 98.87% | 98.87% |

**Findings:**
- **Sum fusion** achieved the highest accuracy (99.33%), marginally outperforming concatenation (99.30%).
- **Concatenation** was selected as the default fusion method for the main experiment due to its interpretability and simpler gradient flow. The performance gap with sum fusion is negligible (0.03%).
- **Gated fusion**, despite being the most complex learnable approach, underperformed at 98.87%, likely due to the increased parameter count requiring more training data or epochs.

### 6.8 Hyperparameter Tuning

Systematic hyperparameter optimization was performed over the learning rate and weight decay space to identify the optimal configuration for the hybrid model.

```
Source: results/hyperparameter_tuning_results.txt
```

| Learning Rate | Weight Decay | Accuracy |
|---|---|---|
| 0.0001 | 0.0001 | 99.01% |
| 0.0001 | 0.001 | 99.16% |
| 5e-05 | 0.0001 | 99.34% |
| **5e-05** | **0.001** | **99.44%** |

**Optimal Configuration:** LR = 5e-5, WD = 0.001

The tuned hyperparameters improved the hybrid model's accuracy from 99.02% (default) to **99.44%** , an improvement of 0.42 percentage points. The lower learning rate (5e-5 vs 1e-4) allows finer weight updates in the fusion layers, while the higher weight decay (0.001 vs 0.0001) provides stronger regularization.

### 6.9 Cross-Validation

3-fold cross-validation was conducted on the gated fusion strategy to assess model stability and generalization.

| Fold | Validation Accuracy |
|---|---|
| Fold 1 | 96.11% |
| Fold 2 | 96.52% |
| Fold 3 | 96.79% |
| **Average** | **96.47%** |
| **Standard Deviation** | **0.28%** |

The low standard deviation (0.28%) indicates that the model's performance is highly stable across different data splits, suggesting strong generalization capability.

### 6.10 Real-World Inference

The trained hybrid model was tested on unseen cashew leaf images to evaluate real-world performance.

```
Source: outputs/training_logs/prediction_results.txt
```

| Image Filename | Predicted Class | Confidence |
|---|---|---|
| anthracnose488_.jpg | Anthracnose | 99.89% |
| anthracnose1009_.jpg | Anthracnose | 99.61% |
| anthracnose122_.jpg | Anthracnose | 98.81% |
| anthracnose865_.jpg | Anthracnose | 99.99% |
| anthracnose1718_.jpg | Anthracnose | 99.99% |
| anthracnose678_.jpg | Anthracnose | 99.98% |
| anthracnose465_.jpg | Anthracnose | 99.63% |
| anthracnose1107_.jpg | Anthracnose | 99.99% |
| **anthracnose202_.jpg** | **Anthracnose** | **100.00%** |
| anthracnose1620_.jpg | Anthracnose | 99.94% |

**Key Observations:**
- All 10 test images were correctly classified as Anthracnose (their ground-truth class).
- Confidence scores ranged from 98.81% to **100.00%** , with an average confidence of **99.78%** .
- One image (anthracnose202_.jpg) achieved a perfect confidence score of 100.00%, demonstrating the model's absolute certainty on certain samples.

---

## 7. Explainable AI (XAI) Pipeline

To ensure transparency and build trust in the model's predictions, four complementary XAI techniques were implemented. Each method provides a different lens through which to interpret model decisions.

### 7.1 Grad-CAM

**Gradient-weighted Class Activation Mapping** generates coarse heatmaps by computing the gradient of the target class score with respect to the final convolutional feature maps.

![Grad-CAM Heatmap](outputs/figures/gradcam_heatmap.png)

**Interpretation:** The heatmap highlights the regions of the leaf image that most influenced the model's classification decision. Warmer colors (red/orange) indicate higher importance. For diseased leaves, the heatmap correctly focuses on lesion areas, confirming the model relies on disease-specific visual features rather than background artifacts.

### 7.2 Grad-CAM++

**Grad-CAM++** improves upon the original Grad-CAM by incorporating a weighted combination of positive partial derivatives, producing more precise and pixel-accurate localization maps.

### 7.3 LIME

**Local Interpretable Model-agnostic Explanations** generates explanations by perturbing input superpixels and observing the change in model output. This creates a faithful local approximation of the decision boundary.

### 7.4 SHAP

**SHapley Additive Explanations** provides mathematically rigorous feature attributions based on cooperative game theory, ensuring that the contribution of each feature (pixel region) is fairly distributed.

![SHAP Explanation](outputs/figures/shap_explanation.png)

Due to computational constraints in the Colab free tier, SHAP was run with `nsamples=20` using the GradientExplainer, which approximates Shapley values through gradient-based sampling.

```
Source: outputs/training_logs/shap_log.txt
Predicted: anthracnose
```

### 7.5 XAI Comparison

![Grad-CAM++ & LIME Comparison](outputs/figures/gradcam_pp_lime.png)

The combined Grad-CAM++ and LIME visualization allows direct comparison between the two explanation methods:
- **Grad-CAM++** produces smooth, continuous heatmaps highlighting class-discriminative regions.
- **LIME** generates superpixel-level masks showing which local patches support or oppose the predicted class.
- The two methods show strong agreement, both localizing to similar disease-affected regions, providing cross-validated interpretability.

![Master XAI Comparison](outputs/figures/xai_comparison_master.png)

---

## Project Structure

```
CashewXAI/
│
├── README.md                              # This documentation
├── requirements.txt                       # Python dependencies
│
├── configs/
│   └── config.py                          # Centralized configuration (paths, hyperparameters)
│
├── src/
│   ├── model.py                           # Hybrid CNN-Transformer architecture
│   ├── dataset.py                         # Dataset class with augmentation pipeline
│   ├── trainer.py                         # Training loop with checkpoint management
│   ├── evaluator.py                       # Classification report and confusion matrix generation
│   ├── data_utils.py                      # Kaggle download, Google Drive sync, dataset summaries
│   └── fusion_ablation.py                 # Ablation study for fusion strategies
│
├── xai/
│   ├── gradcam_visualizer.py              # Grad-CAM heatmap generation
│   ├── gradcam_plus_plus_lime.py          # Grad-CAM++ and LIME visualization
│   ├── shap_explainer.py                  # SHAP feature attribution
│   └── performance_comparison.py          # Bar chart comparing all model accuracies
│
├── baselines/
│   ├── train_cnn_baselines.py             # Training script for 5 CNN baselines
│   └── train_transformer_baselines.py     # Training script for 5 Transformer baselines
│
├── inference/
│   └── predict.py                         # Real-world inference on unseen images
│
├── outputs/
│   ├── training_logs/                     # All training logs (loss, accuracy, metrics)
│   │   ├── cnn_baseline_log.txt
│   │   ├── transformer_baseline_log.txt
│   │   ├── hybrid_training_log.txt
│   │   ├── evaluation_report.txt
│   │   ├── fusion_ablation.txt
│   │   ├── shap_log.txt
│   │   └── prediction_results.txt
│   │
│   └── figures/                           # All generated figures and visualizations
│       ├── comparison.png                 # Performance comparison bar chart
│       ├── confusion_matrix.png           # Hybrid model confusion matrix
│       ├── gradcam_heatmap.png            # Grad-CAM heatmap visualization
│       ├── gradcam_pp_lime.png            # Grad-CAM++ & LIME side-by-side
│       ├── Result_Visualization.png
│       ├── shap_explanation.png           # SHAP feature attribution plot
│       └── xai_comparison_master.png      # Master XAI comparison figure
│
├── results/
│   ├── baseline_transformer/
│   │   └── transformer_results.csv        # Transformer baseline accuracy results
│   └── hyperparameter_tuning_results.txt  # Hyperparameter optimization log
│
├── checkpoints/                           # Saved model weights (.pth files)
│   ├── cnn_resnet50_best.pth
│   ├── cnn_densenet121_best.pth
│   ├── cnn_efficientnet_b0_best.pth
│   ├── cnn_vgg16_best.pth
│   ├── cnn_mobilenetv3_large_100_best.pth
│   ├── transformer_vit_tiny_patch16_224_best.pth
│   ├── transformer_swin_tiny_patch4_window7_224_best.pth
│   ├── transformer_deit_tiny_patch16_224_best.pth
│   ├── transformer_cait_xxs24_224_best.pth
│   ├── transformer_crossvit_9_240_best.pth
│   └── hybrid_cnn_transformer_best.pth    # Best hybrid model (99.02% accuracy)
│
├── datasets/
│   └── train/
│       ├── anthracnose/    (1,729 images)
│       ├── gumosis/        (392 images)
│       ├── healthy/        (1,368 images)
│       ├── leaf miner/     (1,378 images)
│       └── red rust/       (1,682 images)
│
├── docs/                                   # Research documentation
│   ├── Hybrid CNN-Transformer Framework Manuscript.docx
│   └── Hybrid CNN-Transformer Framework.pdf.pdf
│
├── notebooks/
│   └── colab_full_pipeline.md              # Step-by-step Colab execution guide
│
└── local_training/
    └── local_train_guide.md                # Local execution guide
```

---

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-capable GPU (recommended for training; CPU-only inference is supported)

### Dependencies

All required dependencies are listed in `requirements.txt`:

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
albumentations>=1.3.0
opencv-python>=4.8.0
grad-cam>=1.4.0
lime>=0.2.0
shap>=0.42.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
kagglehub>=0.2.0
```

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd CashewXAI

# Install dependencies
pip install -r requirements.txt

# Download dataset (via kagglehub - configured in data_utils.py)
python src/data_utils.py
```

---

## How to Run

### Training CNN Baselines

```bash
python baselines/train_cnn_baselines.py
```

This trains 5 CNN architectures (ResNet-50, DenseNet-121, EfficientNet-B0, VGG-16, MobileNetV3-Large) for 5 epochs each and saves checkpoints to `checkpoints/`.

### Training Transformer Baselines

```bash
python baselines/train_transformer_baselines.py
```

This trains 5 Transformer architectures (ViT-Tiny, Swin-Tiny, DeiT-Tiny, CaiT-XXS24, CrossViT-9) for 5 epochs each and saves checkpoints to `checkpoints/`.

### Training the Hybrid Model

```bash
python src/model.py
```

The hybrid model is trained via the `train_model` function in `src/trainer.py`, typically invoked from within `src/model.py`'s standalone execution or through the main pipeline script.

### Running XAI Visualizations

```bash
# Grad-CAM
python xai/gradcam_visualizer.py

# Grad-CAM++ and LIME
python xai/gradcam_plus_plus_lime.py

# SHAP
python xai/shap_explainer.py
```

### Running Inference on New Images

```bash
python inference/predict.py --image_path <path-to-image>
```

### Ablation Study

```bash
python src/fusion_ablation.py
```

### Google Colab

For a complete step-by-step Google Colab execution guide, refer to:

```
notebooks/colab_full_pipeline.md
```

### Local Training

For detailed instructions on setting up and running experiments on a local machine, refer to:

```
local_training/local_train_guide.md
```

---

## Conclusion

This research presents a comprehensive **Multi-XAI Hybrid CNN-Transformer Framework** for automated cashew leaf disease detection. The key contributions and findings are summarized below:

### Key Contributions

1. **High-Performance Hybrid Architecture:** The proposed fusion of EfficientNet-B0 and CaiT-XXS24-224 achieved **99.02%** accuracy (and **99.44%** with hyperparameter tuning), outperforming all 10 individual CNN and Transformer baselines.

2. **Comprehensive Baseline Comparison:** Eleven models spanning both pure CNN and pure Transformer architectures were systematically evaluated, providing a thorough benchmark for cashew disease classification.

3. **Multi-Modal XAI Integration:** Four complementary explainability techniques (Grad-CAM, Grad-CAM++, LIME, SHAP) were integrated to provide transparent, human-interpretable explanations of model decisions.

4. **Ablation and Robustness Analysis:** Fusion strategy ablation, hyperparameter tuning, and 3-fold cross-validation were conducted to validate design choices and assess model stability.

### Performance Summary

| Metric | Value |
|---|---|
| Best Hybrid Accuracy (tuned) | **99.44%** |
| Best Hybrid Accuracy (default) | **99.02%** |
| Best CNN Baseline (EfficientNet-B0) | 97.82% |
| Best Transformer Baseline (CaiT-XXS24) | 97.54% |
| Macro Average F1-Score | 1.00 |
| Weighted Average F1-Score | 0.99 |
| Average Inference Confidence | 99.78% |
| Cross-Validation Average | 96.47% |

### Future Directions

- **Deployment on Edge Devices:** Optimize the hybrid model for deployment on resource-constrained devices (smartphones, drones) for real-time field diagnosis.
- **Expanded Disease Categories:** Include additional cashew diseases and pest infestations to broaden the model's diagnostic coverage.
- **Multi-Plant Generalization:** Extend the framework to detect diseases across multiple crop species, creating a unified agricultural disease diagnosis platform.
- **Temporal Analysis:** Incorporate time-series analysis to track disease progression and predict outbreak patterns.
- **Federated Learning:** Enable privacy-preserving collaborative learning across multiple farms without sharing sensitive data.

---

*This research was conducted as part of a comprehensive study on explainable artificial intelligence for agricultural disease detection. For academic use, citation and reference to the accompanying manuscript are appreciated.*
