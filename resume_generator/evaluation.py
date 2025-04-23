import pandas as pd
import re

# Load data
gt = pd.read_excel("ground_truth.xlsx", sheet_name=0)
results = pd.read_excel("jd_resume_match_results_final.xlsx")

# Extract JD Name from the Job Description column
def extract_jd_name(text):
    if isinstance(text, str):
        m = re.match(r'\s*\d+\.\s*([^\s]+)', text)
        return m.group(1).strip() if m else text.strip()
    return ''

gt["JD Name"] = gt["Job Description"].apply(extract_jd_name)

# Parse Sanjana's resumes (handles commas, newlines, and numbered lists)
def parse_resumes(cell):
    if not isinstance(cell, str):
        return set()
    items = re.split(r'[\n,]+', cell)
    cleaned = []
    for item in items:
        # Remove leading numbering (e.g., '1. ', '2) ')
        item = re.sub(r'^\s*\d+[\.\)]?\s*', '', item).strip()
        if item and ('.json' in item or '.txt' in item):
            cleaned.append(item)
    return set(cleaned)

gt["Sanjana_Resumes"] = gt["Sanjana"].apply(parse_resumes)
gt_dict = dict(zip(gt["JD Name"], gt["Sanjana_Resumes"]))

# Calculate metrics for each JD/Method
def compute_metrics(group, k=5):
    jd_name = group.name[0] if isinstance(group.name, tuple) else group.name
    resumes_gt = gt_dict.get(jd_name, set())
    if not resumes_gt:
        return pd.Series({'Precision@5': None, 'Recall@5': None, 'F1@5': None})
    top_k = group.sort_values("Rank").head(k)["Resume"].tolist()
    tp = sum(1 for res in top_k if res in resumes_gt)
    precision = tp / k
    recall = tp / len(resumes_gt) if resumes_gt else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return pd.Series({'Precision@5': precision, 'Recall@5': recall, 'F1@5': f1})

metrics = results.groupby(["JD Name", "Method"]).apply(compute_metrics).reset_index()

print(metrics[["JD Name", "Method", "Precision@5", "Recall@5", "F1@5"]])


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the results Excel file
results = pd.read_excel("jd_resume_match_results_final.xlsx")

# Calculate average time taken per JD per method
avg_time = results.groupby(["JD Name", "Method"])['Time Taken (s)'].mean().reset_index()

# Pivot the data for plotting
pivot_df = avg_time.pivot(index='JD Name', columns='Method', values='Time Taken (s)')

# Plotting
plt.figure(figsize=(12, 6))

pivot_df.plot(kind='bar')

plt.title('Average Time Taken per JD per Method')
plt.ylabel('Average Time Taken (seconds)')
plt.xlabel('Job Description (JD)')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Method')
plt.tight_layout()
plt.show()
