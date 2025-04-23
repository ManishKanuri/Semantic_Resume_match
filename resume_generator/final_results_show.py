import json
import re
import faiss
import time
import torch
import requests
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Ollama endpoints and models
OLLAMA_URL = "http://localhost:11434/api/generate"
LLAMA_MODEL = "llama3"
MISTRAL_MODEL = "mistral:instruct"

# Paths
RESUME_DIR = Path("resume_generator/resume_data")
JD_DIR = Path("jd_data/raw_jd_texts")
JD_PARSED_DIR = Path("jd_data/parsed")

# --- Extract JSON safely ---
def extract_json_block(text):
    try:
        return text[text.index("{"):text.rindex("}") + 1]
    except:
        return None

# --- Estimate experience ---
def estimate_experience_years(experience):
    total_years = 0
    for job in experience:
        duration = job.get("duration", "")
        match = re.search(r"(\d{4})\s*[-–—to]+\s*(\d{4}|present)", duration, re.I)
        if match:
            start = int(match.group(1))
            end = match.group(2)
            end_year = int(end) if end.isdigit() else 2025
            total_years += max(0, end_year - start)
        elif "summer" in duration.lower() and re.search(r"\d{4}", duration):
            total_years += 0.25
    return total_years

# --- Flatten resume text ---
def extract_text_from_resume(resume):
    summary = resume.get("summary", "")
    skills = ", ".join(resume.get("skills", []))
    exp_text = "\n".join(
        f"{e.get('title', '')} at {e.get('company', '')}: {'; '.join(e.get('details', []))}"
        for e in resume.get("experience", [])
    )
    education = f"{resume.get('education', {}).get('degree', '')} from {resume.get('education', {}).get('institution', '')}"
    certs = ", ".join(resume.get("certifications", []))
    return f"{summary}\nSkills: {skills}\nExperience:\n{exp_text}\nEducation: {education}\nCertifications: {certs}"

# --- Explanation using LLM ---
def get_explanation(model_name, jd_text, resume_text):
    prompt = f"""
You are an expert recruiter. Given the following job description and resume, explain in 2-3 bullet points why this resume is a good match for the job.

Job Description:
{jd_text}

Resume:
{resume_text}

Explain why this candidate is a good match:
"""
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": model_name,
            "prompt": prompt,
            "stream": False
        })
        return response.json().get('response', '⚠️ No response returned').strip()
    except Exception as e:
        return f"⚠️ Error: {e}"

# --- Main Matcher ---
def match_resumes_with_explanations(jd_txt_path):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    jd_path = Path(jd_txt_path)
    jd_name = jd_path.stem
    jd_text = jd_path.read_text()

    # Get experience limits
    parsed_path = JD_PARSED_DIR / f"{jd_name}.json"
    min_exp, max_exp = 0, 100
    if parsed_path.exists():
        jd_struct = json.load(open(parsed_path))
        min_exp = jd_struct.get("min_experience", 0)
        max_exp = jd_struct.get("max_experience", 100)
        print(f"🔍 Experience Filter: {min_exp} to {max_exp} years")
    else:
        print("⚠️ No parsed JD metadata found")

    jd_embed = model.encode(jd_text, convert_to_tensor=True)
    jd_embed_np = jd_embed.cpu().numpy().reshape(1, -1)

    embeddings, metadata, resume_texts = [], [], []

    for path in RESUME_DIR.glob("*.json"):
        try:
            content = path.read_text()
            json_text = extract_json_block(content)
            resume = json.loads(json_text)
            years = estimate_experience_years(resume.get("experience", []))

            if years < min_exp or years > max_exp:
                print(f"Skipping {path.name} ({years} yrs)")
                continue

            text = extract_text_from_resume(resume)
            emb = model.encode(text, normalize_embeddings=False)
            embeddings.append(emb)
            metadata.append(path.name)
            resume_texts.append(text)
        except Exception as e:
            print(f"Error reading {path.name}: {e}")
            continue

    if not embeddings:
        print("No resumes matched.")
        return

    embeddings_np = np.vstack(embeddings)
    faiss_l2 = faiss.IndexFlatL2(embeddings_np.shape[1])
    faiss_l2.add(embeddings_np)
    D_l2, I_l2 = faiss_l2.search(np.array(jd_embed_np, dtype=np.float32), 5)

    print(f"\nTop 5 L2 Matches for JD: {jd_name}")
    for i, idx in enumerate(I_l2[0]):
        name = metadata[idx]
        distance = D_l2[0][i]
        resume_text = resume_texts[idx]
        print(f"\n#{i+1}: {name} — Distance: {distance:.4f}")

        llama_expl = get_explanation(LLAMA_MODEL, jd_text, resume_text)
        mistral_expl = get_explanation(MISTRAL_MODEL, jd_text, resume_text)

        print(f"\nLLaMA:\n{llama_expl}")
        print(f"\nMistral:\n{mistral_expl}")
        print("-" * 80)


# ▶️ Run
if __name__ == "__main__":
    match_resumes_with_explanations("jd_data/raw_jd_texts/genai_llm.txt")
