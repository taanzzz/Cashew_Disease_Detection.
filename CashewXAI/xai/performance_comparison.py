import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ১. ট্রেইনিং লজ থেকে ডাটা সংগ্রহ করে পুনরায় ডিফাইন করা
# CNN রেজাল্ট
cnn_data = {'Model': ['resnet50', 'densenet121', 'efficientnet_b0'], 'Best_Train_Acc': [0.9615, 0.9754, 0.9782]}
df_cnn_results = pd.DataFrame(cnn_data)

# Transformer রেজাল্ট
trans_data = {'Model': ['vit_tiny', 'swin_tiny', 'cait_xxs24'], 'Best_Train_Acc': [0.9742, 0.9736, 0.9754]}
df_transformer_results = pd.DataFrame(trans_data)

# Hybrid Model রেজাল্ট (আপনার রান করা সর্বশেষ রেজাল্ট)
best_hybrid_acc = 0.9902

# ২. কম্প্যারিশন ডাটা প্রস্তুত করা
best_cnn_acc = df_cnn_results['Best_Train_Acc'].max()
best_trans_acc = df_transformer_results['Best_Train_Acc'].max()

comparison_data = {
    'Model Category': ['Best CNN', 'Best Transformer', 'Proposed Hybrid'],
    'Accuracy': [best_cnn_acc, best_trans_acc, best_hybrid_acc]
}

df_compare = pd.DataFrame(comparison_data)

# ৩. ভিজ্যুয়ালাইজেশন
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
ax = sns.barplot(x='Model Category', y='Accuracy', data=df_compare, palette='viridis')

# বারের উপরে মান বসানো
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.4f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center',
                   xytext = (0, 9),
                   textcoords = 'offset points')

plt.title('Performance Comparison: Hybrid Model vs Baselines')
plt.ylim(0.95, 1.0) # পার্থক্য ভালোমতো বোঝার জন্য লিমিট সেট করা
plt.ylabel('Training Accuracy')
plt.show()