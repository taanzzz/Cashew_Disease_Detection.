import os

# প্রজেক্টের মূল ফোল্ডার পাথ (গুগল ড্রাইভ)
PROJECT_ROOT = '/content/drive/MyDrive/Cashew_XAI_Project'

# ফোল্ডারগুলোর তালিকা
folders = [
    '',                    # root
    'datasets',            # raw data
    'datasets/train',
    'datasets/val',
    'datasets/test',
    'checkpoints',         # model save (.pth files)
    'results',             # CSV, JSON results
    'results/baseline_cnn',
    'results/baseline_transformer',
    'results/9_combinations',
    'results/final',
    'xai_outputs',         # XAI heatmaps
    'logs',
]

for f in folders:
    path = os.path.join(PROJECT_ROOT, f)
    os.makedirs(path, exist_ok=True)
    print(f'Created/Verified: {path}')

# পাথ ভেরিয়েবল সেট করা (ভবিষ্যতে ব্যবহারের জন্য)
DATASET_DIR = os.path.join(PROJECT_ROOT, 'datasets')
CKPT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
RESULT_DIR = os.path.join(PROJECT_ROOT, 'results')
XAI_DIR = os.path.join(PROJECT_ROOT, 'xai_outputs')