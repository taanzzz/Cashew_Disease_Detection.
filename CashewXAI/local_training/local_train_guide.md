# Local Environment Setup & Execution Report

**Prepared by:** [Your Name/ID]
**Presented to:** [Professor/Supervisor's Name]

Sir, in addition to the Google Colab workflow, I have also made sure that my entire **Multi-XAI Hybrid CNN-Transformer Framework** is fully compatible with local machines. This document outlines how I structured the project so that it can be executed natively on a Windows/Linux/Mac computer.

I designed the `config.py` file to automatically detect whether the code is running in Colab or locally, adjusting the file paths dynamically so no code changes are required.

---

## 1. Local Environment Configuration

To keep my project dependencies isolated and clean, I utilized a Python virtual environment. Here are the commands I used in my terminal to set it up:

```powershell
# Navigating to my project directory
cd E:\Cashew_Disease_Detection\CashewXAI

# Creating the virtual environment
python -m venv venv

# Activating the virtual environment
.\venv\Scripts\activate
```

## 2. Dependency Installation

Since I trained these deep learning models, I needed to install PyTorch with CUDA support to leverage GPU acceleration on my local machine. 

```powershell
# I installed PyTorch configured for my NVIDIA GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then I installed the rest of my project dependencies (like SHAP, LIME, timm):
pip install -r requirements.txt
```

## 3. Local Dataset Setup

For local execution, my code expects the dataset to be placed in the following structured format within the project folder:

```text
CashewXAI/
└── datasets/
    └── train/
        ├── anthracnose/
        ├── gumosis/
        ├── healthy/
        ├── leaf miner/
        └── red rust/
```

I wrote a utility script that can automatically download and structure this dataset directly from Kaggle:
```powershell
python src/data_utils.py
```

## 4. Executing My Training Pipelines

I made all my Python scripts executable directly from the command line. This modular approach allows me to run specific parts of the research without running a massive Jupyter Notebook.

### Training My Hybrid Model
I created a script to initialize my custom HybridCashewModel and run the training loop:
```powershell
python -c "
from src.model import HybridCashewModel
from src.dataset import get_train_loader
from src.trainer import train_model

train_loader, _ = get_train_loader()
model = HybridCashewModel()
train_model(model, train_loader, epochs=10, model_name='hybrid_cnn_transformer')
"
```

### Training the Benchmark Baselines
To gather the comparative data for my research, I ran these scripts to train the 10 baseline models:
```powershell
# Executing my CNN Baselines
python baselines/train_cnn_baselines.py

# Executing my Transformer Baselines
python baselines/train_transformer_baselines.py
```

## 5. Generating My XAI Explanations

After the model finished training and the best weights were saved in the `checkpoints/` folder, I ran my XAI scripts. I configured these scripts to automatically save the output visualizations as `.png` files in the `outputs/figures/` directory so I could easily import them into my research manuscript.

```powershell
# To generate my Grad-CAM Heatmaps:
python xai/gradcam_visualizer.py

# To generate my Grad-CAM++ and LIME comparisons:
python xai/gradcam_plus_plus_lime.py

# To generate my SHAP feature attributions:
python xai/shap_explainer.py
```

## 6. Running Inference on New Data

Finally, to demonstrate that my model works in practical scenarios, I use this script to feed it new images and output the prediction confidence scores:
```powershell
python inference/predict.py
```

---

### Challenges I Addressed During Local Implementation:

1. **GPU Memory Limitations:** During training, if I encountered Out of Memory (OOM) errors on my local GPU, I designed the `config.py` file so I could easily reduce the `BATCH_SIZE` from 32 to 16.
2. **SHAP Computational Cost:** SHAP calculations for high-resolution images are extremely resource-intensive. To prevent my local system from freezing, I specifically programmed the `shap_explainer.py` module to bypass the GPU and use the CPU, while limiting `nsamples=20`.

Sir, this setup allows the entire research methodology to be reproduced seamlessly on any standard local workstation.
