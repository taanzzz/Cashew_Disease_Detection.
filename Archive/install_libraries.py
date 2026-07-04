# লাইব্রেরিগুলো পুনরায় ইন্সটল করা হচ্ছে
!pip install -q grad-cam lime shap timm albumentations

import torch, torchvision, timm
import shap, lime
import albumentations
import pytorch_grad_cam

print(f'PyTorch version: {torch.__version__}')
print(f'Timm version: {timm.__version__}')
print('All libraries installed and imported successfully! ✅')