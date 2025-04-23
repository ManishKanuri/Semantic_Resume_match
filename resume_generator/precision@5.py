import pandas as pd
import re

# Load data
gt = pd.read_excel("ground_truth.xlsx", sheet_name=0)
results = pd.read_excel("jd_resume_match_results_final.xlsx")

# Extract JD Name
def extract_jd_name(text):
    if isinstance(text, str):
        m = re.match(r'\s*\d+\.\s*([^\s]+)', text)
        return m.group(1).strip() if m else text.strip()
    return ''

gt["JD Name"] = gt["Job Description"].apply(extract_jd_name)

# Parse resumes from Sanjana's column
def parse_resumes(cell):
    if not isinstance(cell, str):
        return set()
    items = re.split(r'[\n,]+', cell)
    cleaned = []
    for item in items:
        item = re.sub(r'^\s*\d+[\.\)]?\s*', '', item).strip()
        if item and ('.json' in item or '.txt' in item):
            cleaned.append(item)
    return set(cleaned)

gt["Sanjana_Resumes"] = gt["Sanjana"].apply(parse_resumes)
gt_dict = dict(zip(gt["JD Name"], gt["Sanjana_Resumes"]))

# Metric calculation
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

# Apply and save
metrics = results.groupby(["JD Name", "Method"]).apply(compute_metrics).reset_index()
metrics.to_excel("resume_matching_metrics.xlsx", index=False)
