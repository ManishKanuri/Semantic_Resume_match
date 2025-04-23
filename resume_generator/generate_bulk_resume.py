from jinja2 import Template
import subprocess
import json
import re
from pathlib import Path

# All target domains
domains = {
    "genai": "GenAI, LLM fine-tuning, prompt engineering, LangChain, vector databases, transformers like GPT and LLaMA.",
    "recsys": "Recommendation systems using collaborative filtering, matrix factorization, ranking models, and PyTorch or TensorFlow.",
    "computer_vision": "Real-time object detection, CNNs, OpenCV, YOLOv8, image classification and segmentation pipelines.",
    "nlp": "Text classification, sentiment analysis, topic modeling, NER, SpaCy, HuggingFace Transformers, and BERT-based models.",
    "time_series": "Time series forecasting using ARIMA, LSTM, Prophet, trend analysis, anomaly detection, and seasonal decomposition.",
    "reinforcement_learning": "Q-learning, policy gradients, multi-armed bandits, reward modeling, and OpenAI Gym environments.",
    "ml_research": "Research publications, arXiv submissions, experimental modeling, academic ML projects, and benchmarking studies.",
    "data_analytics": "Exploratory data analysis, pandas, SQL, KPI dashboards, hypothesis testing, and stakeholder reporting.",
    "product_analytics": "User funnel analysis, A/B testing, retention metrics, Mixpanel, Amplitude, customer behavior modeling.",
    "marketing_analytics": "Campaign ROI, user segmentation, attribution models, churn prediction, and marketing funnel optimization.",
    "financial_analytics": "Credit risk modeling, fraud detection, financial forecasting, loan scoring, and investment optimization.",
    "healthcare_analytics": "Electronic health records, patient risk scoring, treatment outcome modeling, and FHIR/HL7 data handling.",
    "data_visualization": "Interactive dashboards using Tableau, Power BI, D3.js, Plotly, and storytelling with visual analytics.",
    "data_engineering": "ETL development, Airflow, Apache Beam, dbt, data lake pipelines, and distributed system optimization.",
    "big_data": "Spark, Hadoop, Hive, Presto, batch/streaming data processing, and high-volume data transformation at scale.",
    "streaming_data": "Kafka, Spark Streaming, Flink, real-time ingestion, window functions, and event-driven architecture.",
    "cloud_data_engineering": "Data solutions on AWS (Glue, Redshift), GCP (BigQuery, Dataflow), Azure Synapse, cloud-native design.",
    "data_governance": "Data quality monitoring, lineage, metadata management, Great Expectations, and compliance frameworks.",
    "ml_ops": "Model deployment, versioning, CI/CD for ML, Docker, MLflow, SageMaker, and production model monitoring.",
    "analytics_engineering": "dbt, dimensional modeling, SQL optimization, data transformation pipelines for analytics teams.",
    "data_product_management": "Technical product planning, data use cases, cross-functional coordination, and product lifecycle in data."
}

# Experience levels
levels = {
    "fresher": "0–1 years experience, strong academic background and projects.",
    "mid": "2–5 years hands-on experience, end-to-end system building.",
    "senior": "10+ years of leadership, architecture design, and strategic delivery."
}

# Config
n_per_combo = 2
model = "llama3"
template_path = Path("prompts/prompt_template.txt")
prompt_template = Template(template_path.read_text())

Path("resume_generator/resume_data").mkdir(parents=True, exist_ok=True)

# Regex JSON extractor
def extract_json_block(text):
    match = re.search(r"\{(?:[^{}]|(?R))*\}", text, re.DOTALL)
    return match.group(0) if match else None

# Resume generation loop
for domain, domain_desc in domains.items():
    for level, level_desc in levels.items():
        for i in range(1, n_per_combo + 1):
            job_desc = f"{domain_desc} {level_desc}"
            if level == "fresher":
                job_desc += " Include internship or research work. Highlight academic projects, assistantships, and certifications."
            elif level == "senior":
                job_desc += " Include leadership, architecture contributions, mentoring, and cross-team collaboration."

            prompt = prompt_template.render(
                domain=domain.replace("_", " "),
                level=level,
                job_description=job_desc,
                experience_years=level_desc
            )

            temp_prompt = Path("resume_generator/temp_prompt.txt")
            temp_prompt.write_text(prompt)

            json_path = Path(f"resume_generator/resume_data/{domain}_{level}_{i}.json")
            print(f"\nGenerating: {json_path.name}...")

            # Run Ollama
            with open(json_path, "w") as f:
                subprocess.run(["ollama", "run", model], stdin=temp_prompt.open(), stdout=f)

            # Post-process
            try:
                raw_text = json_path.read_text().strip()
                if not raw_text:
                    raise ValueError("Empty LLM response.")

                print("LLM response preview:\n", raw_text[:300], "\n")

                clean_json = extract_json_block(raw_text)
                if not clean_json:
                    raise ValueError("No JSON object found.")

                json_data = json.loads(clean_json)
                json_path.write_text(json.dumps(json_data, indent=2))

                print(f"Clean JSON Saved: {json_path}")

            except Exception as e:
                print(f"Failed to generate {json_path.name} | {e}")
                print("Raw Output:\n", raw_text)
