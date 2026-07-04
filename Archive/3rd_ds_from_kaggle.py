import kagglehub

# Download latest version
path_cashew_disease = kagglehub.dataset_download("olaidegabriel/cashew-disease-detection-dataset")

print("Path to dataset files:", path_cashew_disease)