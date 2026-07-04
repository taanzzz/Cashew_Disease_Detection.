import zipfile
import os

zip_name = "cashew-image-dataset.zip"  # তোমার downloaded file

with zipfile.ZipFile(zip_name, 'r') as zip_ref:
    zip_ref.extractall("dataset")
    print("Extracted!")

# দেখো কী কী folder আছে
!ls dataset/
!ls dataset/*/  # subfolder দেখাও