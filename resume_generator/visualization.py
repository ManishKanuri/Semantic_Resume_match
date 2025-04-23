import matplotlib.pyplot as plt
import numpy as np

# Suppose metrics_df is your DataFrame with columns: Method, Precision@5, Recall@5, F1@5
methods = metrics_df['Method']
precision = metrics_df['Precision@5']
recall = metrics_df['Recall@5']
f1 = metrics_df['F1@5']

bar_width = 0.2
indices = np.arange(len(methods))

plt.figure(figsize=(10,6))
plt.bar(indices - bar_width, precision, width=bar_width, label='Precision@5', color='#00bcd4')
plt.bar(indices, recall, width=bar_width, label='Recall@5', color='#ffc085')
plt.bar(indices + bar_width, f1, width=bar_width, label='F1@5', color='#b23a48')

plt.xticks(indices, methods, rotation=15)
plt.ylabel('Score')
plt.title('Evaluation Metrics Comparison for Different Methods')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()
