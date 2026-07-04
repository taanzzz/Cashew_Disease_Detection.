import kagglehub
import os

# ডেটাসেট ডাউনলোড
path = kagglehub.dataset_download("nirmalsankalana/crop-pest-and-disease-detection")

print("Path to dataset files:", path)

# ডেটাসেটের ফাইলগুলো দেখুন
print("\nFiles in dataset:")
for root, dirs, files in os.walk(path):
    level = root.replace(path, '').count(os.sep)
    indent = ' ' * 4 * (level)
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 1)
    for f in files[:5]: # প্রতিটি ফোল্ডারের প্রথম ৫টি ফাইল দেখাবে
        print(f"{subindent}{f}")