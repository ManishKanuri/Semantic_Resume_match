import os
import re
import json
from pathlib import Path

# Directories
jd_dir = Path("jd_data/raw_jd_texts")
output_dir = Path("jd_data/parsed")
output_dir.mkdir(parents=True, exist_ok=True)

def extract_metadata(jd_text):
    structured = {
        "jd_text": jd_text,
        "min_experience": None,
        "max_experience": None,
        "education_required": None
    }

    # Pattern: e.g., "2–5 years", "0-2 years", "1 to 3 years"
    range_match = re.search(r"(\d+)\s*[-–—to]+\s*(\d+)\s+years?", jd_text, re.I)
    if range_match:
        structured["min_experience"] = int(range_match.group(1))
        structured["max_experience"] = int(range_match.group(2))
    else:
        # Pattern: "3+ years of experience"
        min_only = re.search(r"(\d+)\+?\s+years? of.*?experience", jd_text, re.I)
        if min_only:
            structured["min_experience"] = int(min_only.group(1))
            structured["max_experience"] = None  # open-ended

    # Extract education requirement
    edu_match = re.search(r"(bachelor[’']?s.*?or master[’']?s.*?(in.*?)?)", jd_text, re.I)
    if edu_match:
        structured["education_required"] = edu_match.group(0).strip()

    return structured

# Loop through all .txt JDs and save parsed metadata
for jd_file in jd_dir.glob("*.txt"):
    try:
        jd_text = jd_file.read_text()
        parsed = extract_metadata(jd_text)
        parsed["jd_filename"] = jd_file.name

        out_path = output_dir / jd_file.with_suffix(".json").name
        with open(out_path, "w") as f:
            json.dump(parsed, f, indent=2)

        print(f"Parsed: {jd_file.name} ➜ {out_path.name}")
    except Exception as e:
        print(f"Failed parsing {jd_file.name}: {e}")
