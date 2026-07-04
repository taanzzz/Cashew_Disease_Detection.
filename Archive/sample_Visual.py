import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

def display_random_images(base_path, num_images=5):
    categories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    selected_categories = random.sample(categories, min(num_images, len(categories)))

    plt.figure(figsize=(20, 10))
    for i, category in enumerate(selected_categories):
        category_path = os.path.join(base_path, category)
        image_name = random.choice(os.listdir(category_path))
        img_path = os.path.join(category_path, image_name)

        img = mpimg.imread(img_path)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(category)
        plt.axis('off')
    plt.show()

# স্যাম্পল ছবি প্রদর্শন
display_random_images(path)