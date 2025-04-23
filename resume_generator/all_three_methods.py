import json
import re
import time
import faiss
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# --- Extract structured JSON safely ---
def extract_json_block(text):
    try:
        first = text.index("{")
        last = text.rindex("}")
        return text[first:last+1]
    except:
        return None

# --- Estimate total experience ---
def estimate_experience_years(experience):
    total_years = 0
    for job in experience:
        duration = job.get("duration", "")
        match = re.search(r"(\d{4})\s*[-–—to]+\s*(\d{4}|present)", duration, re.I)
        if match:
            start_year = int(match.group(1))
            end = match.group(2).lower()
            end_year = int(end) if end.isdigit() else 2025
            total_years += max(0, end_year - start_year)
        elif "summer" in duration.lower() and re.search(r"\d{4}", duration):
            total_years += 0.25
    return total_years

# --- Flatten resume into a single text block ---
def extract_text_from_resume(resume):
    summary = resume.get("summary", "")
    skills = " ".join(resume.get("skills", []))
    exp_text = " ".join(
        f"{e.get('title', '')} at {e.get('company', '')}. {' '.join(e.get('details', []))}"
        for e in resume.get("experience", [])
    )
    education = f"{resume.get('education', {}).get('degree', '')} from {resume.get('education', {}).get('institution', '')}"
    certs = " ".join(resume.get("certifications", []))
    return f"{summary} {skills} {exp_text} {education} {certs}"

# --- Main Matching Function ---
def match_resumes_to_jd(jd_txt_path):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    resume_dir = Path(__file__).parent / "resume_data"

    jd_path = Path(jd_txt_path)
    jd_name = jd_path.stem
    jd_text = jd_path.read_text()

    # Updated: Parse structured JD info
    parsed_path = Path("jd_data/parsed") / f"{jd_name}.json"
    min_exp = 0
    max_exp = 100
    if parsed_path.exists():
        jd_struct = json.load(open(parsed_path))
        min_exp = jd_struct.get("min_experience", 0)
        max_exp = jd_struct.get("max_experience", 100)
        print(f"Required: Min Exp: {min_exp}, Max Exp: {max_exp}")
    else:
        print(f" No parsed JD metadata found at {parsed_path}")

    jd_embed = model.encode(jd_text, convert_to_tensor=True)
    jd_embed_np = jd_embed.cpu().numpy().reshape(1, -1)

    resume_paths = list(resume_dir.glob("*.json"))
    embeddings = []
    metadata = []

    print(f"\nMatching resumes for JD: {jd_name}")

    for path in resume_paths:
        try:
            content = path.read_text()
            json_text = extract_json_block(content)
            resume = json.loads(json_text)
            exp_years = estimate_experience_years(resume.get("experience", []))
            if exp_years < min_exp or exp_years > max_exp:
                print(f"⛔ Skipping {path.name} — {exp_years} years (out of bounds)")
                continue
            text = extract_text_from_resume(resume)
            emb = model.encode(text, normalize_embeddings=False)
            embeddings.append(emb)
            metadata.append(path.name)
        except Exception as e:
            print(f"Skipping {path.name} — {e}")

    if not embeddings:
        print("No resumes matched the experience criteria.")
        return

    embeddings_np = np.vstack(embeddings)

    # Vanilla Cosine
    start1 = time.time()
    cosine_scores = [(metadata[i], float(util.cos_sim(jd_embed, torch.tensor(embeddings_np[i:i+1]))[0])) for i in range(len(embeddings_np))]
    cosine_scores.sort(key=lambda x: x[1], reverse=True)
    end1 = time.time()

    # FAISS Cosine (Normalized)
    start2 = time.time()
    norm_embeds = embeddings_np.copy().astype(np.float32)
    faiss.normalize_L2(norm_embeds)
    faiss_cos = faiss.IndexFlatIP(norm_embeds.shape[1])
    faiss_cos.add(norm_embeds)

    jd_norm = np.array(jd_embed_np, dtype=np.float32)  
    faiss.normalize_L2(jd_norm)

    D_cos, I_cos = faiss_cos.search(jd_norm, 5)
    faiss_cosine = [(metadata[idx], D_cos[0][i]) for i, idx in enumerate(I_cos[0])]
    end2 = time.time()


    # FAISS L2
    start3 = time.time()
    faiss_l2 = faiss.IndexFlatL2(embeddings_np.shape[1])
    faiss_l2.add(embeddings_np)
    D_l2, I_l2 = faiss_l2.search(np.array(jd_embed_np, dtype=np.float32), 5)

    faiss_l2_out = [(metadata[idx], D_l2[0][i]) for i, idx in enumerate(I_l2[0])]
    end3 = time.time()

    print("\nVanilla Cosine Similarity:")
    for i, (name, score) in enumerate(cosine_scores[:5], 1):
        print(f"{i}. {name} — Score: {score:.4f}")

    print("\nFAISS Cosine Similarity (Normalized IP):")
    for i, (name, score) in enumerate(faiss_cosine, 1):
        print(f"{i}. {name} — Score: {score:.4f}")

    print("\nFAISS L2 Distance (Lower is Better):")
    for i, (name, score) in enumerate(faiss_l2_out, 1):
        print(f"{i}. {name} — Distance: {score:.4f}")

    print("\nTime Taken:")
    print(f"Vanilla Cosine:   {end1 - start1:.4f}s")
    print(f"FAISS Cosine:     {end2 - start2:.4f}s")
    print(f"FAISS L2:         {end3 - start3:.4f}s")

    

# ▶️ Run
if __name__ == "__main__":
    match_resumes_to_jd("jd_data/raw_jd_texts/genai_llm.txt")
