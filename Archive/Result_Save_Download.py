import shutil
import os
from google.colab import files

# পাথ পুনরায় ডিফাইন করা
PROJECT_ROOT = '/content/drive/MyDrive/Cashew_XAI_Project'

# প্রজেক্ট রেজাল্ট জিপ করা
zip_filename = "/content/Cashew_Disease_Detection_Results"
if os.path.exists(PROJECT_ROOT):
    shutil.make_archive(zip_filename, 'zip', PROJECT_ROOT)
    print(f"Results zipped successfully at: {zip_filename}.zip")
else:
    print(f"Error: PROJECT_ROOT path not found: {PROJECT_ROOT}")

# ডাউনলোড কমান্ড
# files.download(f"{zip_filename}.zip")