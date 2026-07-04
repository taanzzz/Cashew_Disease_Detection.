# Grad-CAM লাইব্রেরি এবং লেয়ার ফিক্স
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def run_gradcam(model, target_layer, img_path, class_names, device):
    # ইমেজ লোড এবং প্রিপ্রসেসিং
    rgb_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (224, 224))
    img_float = rgb_img.astype(np.float32) / 255
    input_tensor = preprocess_image(rgb_img, mean=MEAN, std=STD).to(device)

    # Grad-CAM সেটআপ
    cam = GradCAM(model=model, target_layers=[target_layer])

    # মডেল প্রেডিকশন
    model.eval()
    # Grad-CAM requires gradients, so no torch.no_grad() here
    output = model(input_tensor)
    pred_idx = torch.argmax(output).item()

    # হিটম্যাপ জেনারেশন
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_idx)])
    grayscale_cam = grayscale_cam[0, :]

    # অরিজিনাল ইমেজের উপর হিটম্যাপ বসানো
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    return rgb_img, visualization, class_names[pred_idx]

# EfficientNet-B0 এর জন্য সঠিক target_layer সিলেক্ট করা
# timm EfficientNet এ 'blocks' এর শেষ এলিমেন্টটি নিতে হবে
target_layer = hybrid_model.cnn.blocks[-1][-1]

# স্যাম্পল ইমেজ পাথ
sample_class = dataset.classes[0]
sample_img_list = os.listdir(os.path.join(TRAIN_PATH, sample_class))
sample_path = os.path.join(TRAIN_PATH, sample_class, sample_img_list[0])

try:
    orig, vis, pred_label = run_gradcam(hybrid_model, target_layer, sample_path, dataset.classes, device)

    # প্লট করা
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(orig)
    plt.title(f"Original: {sample_class}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(vis)
    plt.title(f"Grad-CAM Heatmap (Pred: {pred_label})")
    plt.axis('off')
    plt.show()
except Exception as e:
    print(f"Error generating Grad-CAM: {e}")