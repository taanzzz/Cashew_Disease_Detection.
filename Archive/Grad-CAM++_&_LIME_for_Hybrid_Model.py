import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import cv2
import os
import timm
import torch.nn as nn
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ১. হাইব্রিড মডেল ক্লাস
class HybridCashewModel(nn.Module):
    def __init__(self, cnn_name="efficientnet_b0", trans_name="cait_xxs24_224", num_classes=5):
        super(HybridCashewModel, self).__init__()
        self.cnn = timm.create_model(cnn_name, pretrained=True, features_only=True)
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)
        cnn_out_dim = self.cnn.feature_info[-1]['num_chs']
        self.transformer = timm.create_model(trans_name, pretrained=True)
        trans_out_dim = self.transformer.num_features
        self.transformer.head = nn.Identity()
        self.fusion_dim = cnn_out_dim + trans_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        cnn_feats = self.cnn(x)[-1]
        cnn_feats = self.cnn_pool(cnn_feats).view(cnn_feats.size(0), -1)
        trans_feats = self.transformer(x)
        combined = torch.cat((cnn_feats, trans_feats), dim=1)
        return self.classifier(combined)

# ২. XAI ফাংশনসমূহ
def run_gradcam_plus_plus(model, target_layer, img_path, class_names, device):
    rgb_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (224, 224))
    img_float = rgb_img.astype(np.float32) / 255
    input_tensor = preprocess_image(rgb_img, mean=MEAN, std=STD).to(device)

    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    model.eval()
    output = model(input_tensor)
    pred_idx = torch.argmax(output).item()

    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_idx)])[0, :]
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    return rgb_img, visualization, class_names[pred_idx]

def run_lime(model, img_path, device):
    def batch_predict(images):
        model.eval()
        normalized_images = []
        for img in images:
            temp = test_transform(image=img)["image"]
            normalized_images.append(temp)
        batch = torch.stack(normalized_images, dim=0).to(device)
        logits = model(batch)
        return torch.softmax(logits, dim=1).detach().cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    explanation = explainer.explain_instance(img, batch_predict, top_labels=1, num_samples=200)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    return mark_boundaries(temp / 255.0, mask)

# ৩. এক্সিকিউশন
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FIX: normalization constants (ImageNet defaults) ---
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# --- FIX: transform used by LIME's batch_predict ---
# NOTE: this should match whatever transform was used during training/validation.
# If your training pipeline used different resize/normalize settings, replace this
# with that exact transform for accurate LIME explanations.
test_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

hybrid_model = HybridCashewModel().to(device)
CKPT_PATH = '/content/drive/MyDrive/Cashew_XAI_Project/checkpoints/hybrid_cnn_transformer_best.pth'
TRAIN_PATH = '/content/drive/MyDrive/Cashew_XAI_Project/datasets/train'

if os.path.exists(CKPT_PATH):
    hybrid_model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    print("Hybrid Model loaded successfully! ✅")

    # --- FIX: class_names, derived from the training folder structure ---
    # NOTE: sorted() should match the class-to-index order used during training
    # (e.g. if you used ImageFolder, it also sorts folder names alphabetically).
    class_names = sorted([d for d in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, d))])

    # ডাইনামিকভাবে একটি স্যাম্পল ইমেজ খুঁজে বের করা
    available_classes = class_names
    random_class = random.choice(available_classes)
    class_dir = os.path.join(TRAIN_PATH, random_class)
    random_img = random.choice([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))])
    dynamic_img_path = os.path.join(class_dir, random_img)

    target_layer = hybrid_model.cnn.blocks[-1][-1]

    print(f"Generating XAI Outputs for: {dynamic_img_path}")
    orig, gcpp_vis, pred_l = run_gradcam_plus_plus(hybrid_model, target_layer, dynamic_img_path, class_names, device)
    lime_vis = run_lime(hybrid_model, dynamic_img_path, device)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(orig); plt.title(f"Original ({random_class})"); plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(gcpp_vis); plt.title(f"Grad-CAM++ (Pred: {pred_l})"); plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(lime_vis); plt.title("LIME Explanation"); plt.axis('off')
    plt.show()
else:
    print("Error: Checkpoint not found!")