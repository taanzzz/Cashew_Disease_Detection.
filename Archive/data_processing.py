import shutil
import os
import kagglehub

# প্রজেক্ট পাথ ডিফাইন করা
PROJECT_ROOT = '/content/drive/MyDrive/Cashew_XAI_Project'
DATASET_DIR = os.path.join(PROJECT_ROOT, 'datasets')

# ডাইনামিকভাবে সঠিক পাথ ডাউনলোড/খুঁজে বের করা
print("Fetching dataset path...")
path_raw = kagglehub.dataset_download("olaidegabriel/cashew-disease-detection-dataset")
base_img_path = os.path.join(path_raw, 'Cashew')

def sync_data_to_drive(source, destination):
    if not os.path.exists(source):
        print(f"Error: Source {source} not found!")
        return
    if not os.path.exists(destination) or len(os.listdir(destination)) == 0:
        print(f"Copying data from {source} to {destination}...")
        shutil.copytree(source, destination, dirs_exist_ok=True)
        print("Copy complete! ✅")
    else:
        print("Data already exists in Drive. Skipping copy. ✅")

# ট্রেইন ফোল্ডারে ডাটা সিঙ্ক করা
sync_data_to_drive(base_img_path, os.path.join(DATASET_DIR, 'train'))