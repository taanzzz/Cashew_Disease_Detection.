import pandas as pd

# Summarize the new cashew dataset counts
cashew_summary = []
for category in sub_categories:
    category_path = os.path.join(base_img_path, category)
    num_images = len([f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))])
    cashew_summary.append({'Category': category, 'Count': num_images})

df_cashew = pd.DataFrame(cashew_summary).sort_values(by='Count', ascending=False)
print("New Cashew Dataset Summary:")
print(df_cashew.to_string(index=False))