import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# Adjusting path to the actual image directory
base_img_path = os.path.join(path_cashew_disease, 'Cashew')

# 1. Check sub-folder structure
print("Dataset categories found in Cashew folder:")
sub_categories = [d for d in os.listdir(base_img_path) if os.path.isdir(os.path.join(base_img_path, d))]
print(sub_categories)

# 2. Display sample images from sub-folders
def display_samples_fixed(base_path, categories, num_samples=4):
    plt.figure(figsize=(15, 8))
    selected_cats = random.sample(categories, min(num_samples, len(categories)))
    for i, category in enumerate(selected_cats):
        cat_path = os.path.join(base_path, category)
        # Filter to only pick files, avoiding directories
        files = [f for f in os.listdir(cat_path) if os.path.isfile(os.path.join(cat_path, f))]
        if not files: continue

        img_name = random.choice(files)
        img_path = os.path.join(cat_path, img_name)

        img = mpimg.imread(img_path)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(category)
        plt.axis('off')
    plt.show()

if sub_categories:
    display_samples_fixed(base_img_path, sub_categories)