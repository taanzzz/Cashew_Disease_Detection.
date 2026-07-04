# Project Workflow & Colab Execution Guide

**Prepared by:** [Your Name/ID]
**Presented to:** [Professor/Supervisor's Name]

Sir, this document outlines the complete workflow I followed to develop and execute the **Multi-XAI Hybrid CNN-Transformer Framework**. I have documented each step of my process in Google Colab, from dataset preparation to model training and generating the XAI visualizations, so you can easily review my methodology and reproduce the results.

---

## Step 1: Initial Environment Setup

For this research, I chose Google Colab as my primary computational environment because the Transformer models and SHAP explanations require significant GPU resources. 

To run my code, I set the hardware accelerator to **GPU (T4 or higher)** in the Colab runtime settings.

## Step 2: Google Drive Integration

To ensure that my datasets, model checkpoints, and generated figures are securely saved and not lost when the Colab session ends, I wrote a script to mount my Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 3: Project Initialization & Dependencies

I organized all my code into a modular structure. Here is how I clone the project into the Drive and install all the necessary libraries (like PyTorch, timm, SHAP, etc.) that I used for the implementation:

```python
import os

# I defined my primary project directory here
PROJECT_ROOT = '/content/drive/MyDrive/Cashew_XAI_Project'
os.makedirs(PROJECT_ROOT, exist_ok=True)

# Navigate to the project directory
%cd {PROJECT_ROOT}

# Install all the required packages I listed in requirements.txt
!pip install -q -r requirements.txt
```

## Step 4: Secure Kaggle Dataset Download

Instead of uploading the dataset manually, I integrated the Kaggle API to download the data directly. I set up the credentials like this:

```python
from google.colab import files
import os
import shutil

print("Uploading kaggle.json for authentication:")
files.upload()

# I configured the Kaggle directory securely
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
shutil.copy('kaggle.json', os.path.expanduser('~/.kaggle/kaggle.json'))
os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
```

## Step 5: Data Processing & Verification

After downloading the dataset, I wrote a utility script to organize it into proper train/validation structures and sync it with my Drive. I also added functions to summarize the class distribution to check for data imbalance.

```python
from src.data_utils import setup_dataset, summarize_dataset, display_sample_images

# My script to download and sync data
setup_dataset()

# Checking the dataset distribution across the 5 disease classes
summarize_dataset()

# Visualizing random samples to verify the data loading pipeline
display_sample_images()
```

## Step 6: Training the Baseline Models

Before finalizing my Hybrid model, I experimented with 10 different baseline architectures (5 CNNs and 5 Transformers) to establish a performance benchmark. Here is how I executed those baseline training loops:

```python
# Executing my CNN Baseline training script (ResNet, DenseNet, EfficientNet, etc.)
from baselines.train_cnn_baselines import train_cnn_baselines
train_cnn_baselines()

# Executing my Transformer Baseline training script (ViT, Swin, CaiT, etc.)
from baselines.train_transformer_baselines import train_transformer_baselines
train_transformer_baselines()
```

## Step 7: Training My Proposed Hybrid Model

This is the core contribution of my research. I designed a custom architecture that fuses features from EfficientNet-B0 and CaiT-XXS24. I trained it for 10 epochs using an AdamW optimizer.

```python
from src.model import HybridCashewModel
from src.dataset import get_train_loader
from src.trainer import train_model

# Loading the augmented dataset
train_loader, _ = get_train_loader()

# Initializing my custom Hybrid architecture
model = HybridCashewModel()

# Executing the training loop
best_acc = train_model(
    model, 
    train_loader, 
    epochs=10, 
    lr=5e-5, 
    model_name="hybrid_cnn_transformer"
)
```

## Step 8: Comprehensive Model Evaluation

To analyze how well my model learned, I wrote an evaluation script that generates a detailed classification report (Precision, Recall, F1-Score) and plots a Confusion Matrix.

```python
from src.model import HybridCashewModel
from src.dataset import get_test_loader
from src.evaluator import load_and_evaluate
from configs.config import CLASS_NAMES

test_loader, _ = get_train_loader() # Using train_loader here for demonstration

model = HybridCashewModel()
# This loads my best saved weights and generates the metrics
load_and_evaluate(model, test_loader, CLASS_NAMES)
```

## Step 9: Multi-Modal XAI Analysis

To make the "black-box" model interpretable, I integrated four different XAI techniques. I structured them into separate modules so I can analyze exactly which parts of the cashew leaf the model focuses on.

### 1. Grad-CAM (Targeting the CNN Branch)
```python
from src.model import HybridCashewModel
from xai.gradcam_visualizer import visualize_gradcam
from configs.config import CLASS_NAMES, DEVICE
import torch
import os

model = HybridCashewModel().to(DEVICE)
ckpt_path = '/content/drive/MyDrive/Cashew_XAI_Project/checkpoints/hybrid_cnn_transformer_best.pth'
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

# Generating Grad-CAM heatmap
visualize_gradcam(model, CLASS_NAMES)
```

### 2. Grad-CAM++ & LIME Comparison
I wrote this script to compare pixel-wise gradient weighting (Grad-CAM++) with superpixel perturbations (LIME) side-by-side.
```python
from xai.gradcam_plus_plus_lime import visualize_gradcam_pp_and_lime

visualize_gradcam_pp_and_lime(model)
```

### 3. SHAP (Feature Attribution)
*Note: Because SHAP is highly memory-intensive, I specifically configured this script to run on the CPU to prevent CUDA Out-Of-Memory errors during analysis.*
```python
from xai.shap_explainer import run_shap_explanation
import os
from configs.config import CLASS_NAMES, TRAIN_PATH

# Selecting a random image for my SHAP analysis
sample_class = CLASS_NAMES[0]
class_dir = os.path.join(TRAIN_PATH, sample_class)
sample_img = [f for f in os.listdir(class_dir) if f.endswith('.jpg')][0]
img_path = os.path.join(class_dir, sample_img)

run_shap_explanation(img_path, model)
```

## Step 10: Real-World Inference Testing

Finally, I wrote an inference script to test how my model performs on unseen real-world images. It outputs a structured DataFrame with prediction confidences.

```python
from inference.predict import run_inference

# This processes a directory of test images and prints my prediction results
df_predictions = run_inference()
```

## Step 11: Exporting Research Outputs

I added this final script to zip all my generated graphs, training logs, and `.pth` checkpoint files so I could download and include them in my research paper.

```python
import shutil
from google.colab import files

shutil.make_archive('/content/Cashew_Results', 'zip', '/content/drive/MyDrive/Cashew_XAI_Project')
files.download('/content/Cashew_Results.zip')
```

---
*Sir, please let me know if you would like me to explain any specific part of my code or methodology in more detail.*
