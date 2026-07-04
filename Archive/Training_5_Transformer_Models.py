import pandas as pd
import os
import timm
import gc
import torch

transformer_names = [
    "vit_tiny_patch16_224",
    "swin_tiny_patch4_window7_224",
    "deit_tiny_patch16_224",
    "cait_xxs24_224",
    "crossvit_9_240" # সঠিক নাম সংশোধিত
]
transformer_results = []

for name in transformer_names:
    try:
        print(f"\nInitializing Transformer: {name}...")
        model = timm.create_model(name, pretrained=True, num_classes=5)

        # ৫টি ইপোকের জন্য ট্রেইনিং
        best_acc = train_model(model, train_loader, epochs=5, model_name=f"transformer_{name}")
        transformer_results.append({"Model": name, "Best_Train_Acc": best_acc})

        # মেমোরি পরিষ্কার করা
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error training {name}: {e}")
        print("Available CrossViT models: ", timm.list_models('*crossvit*'))

# রেজাল্ট প্রদর্শন এবং সেভ করা
df_transformer_results = pd.DataFrame(transformer_results)
TRANS_RESULT_PATH = os.path.join(PROJECT_ROOT, 'results/baseline_transformer/transformer_results.csv')
os.makedirs(os.path.dirname(TRANS_RESULT_PATH), exist_ok=True)
df_transformer_results.to_csv(TRANS_RESULT_PATH, index=False)

print("\n--- Transformer Baseline Results ---")
display(df_transformer_results)

