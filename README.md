# Semantic Resume Matcher 🔍📄

A semantic search pipeline that intelligently matches job descriptions with the most relevant resumes using embeddings, FAISS indexing, and LLM-powered explanations.

## Project Overview

This project helps recruiters identify the top-k resumes for a given job description based on semantic similarity, going beyond keyword matching. It uses sentence embeddings, FAISS similarity search, and LLM-generated explanations to rank resumes effectively.

### 🔧 Features

- Resume–Job Description matching using `all-MiniLM-L6-v2` embeddings
- Fast similarity search via FAISS (Cosine & L2)
- Top-5 resume retrieval with ranking scores
- Explanations generated via RAG using Ollama + LLaMA/Mistral
- Experience-based filtering (e.g., only resumes with 0–2 years for GenAI roles)
- Synthetic resume generation across 20 domains and 3 experience levels
- Output results in Excel format with scores and explanations

## 🛠 Tech Stack

- Python 3.10+
- [SentenceTransformers](https://www.sbert.net/)
- FAISS
- Torch
- OpenPyXL
- Ollama (for LLM inference)




