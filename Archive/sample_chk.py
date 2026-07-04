import pandas as pd

# প্রতিটি ক্যাটাগরিতে কয়টি ছবি আছে তার একটি তালিকা তৈরি করছি
data_summary = []
for category in os.listdir(path):
    category_path = os.path.join(path, category)
    if os.path.isdir(category_path):
        num_images = len(os.listdir(category_path))
        data_summary.append({'Category': category, 'Count': num_images})

df_summary = pd.DataFrame(data_summary).sort_values(by='Count', ascending=False)
print("Dataset Summary:")
print(df_summary.to_string(index=False))