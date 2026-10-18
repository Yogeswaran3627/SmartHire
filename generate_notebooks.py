"""
Generates the 5 project Jupyter notebooks specified in Section 7 of the SmartHire brief.
Valid JSON nbformat v4 notebooks.
"""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.13.7"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [s + "\n" for s in source.split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in source.split("\n")]
    }

# 1. 01_eda.ipynb
nb1_cells = [
    md_cell("# SmartHire — Exploratory Data Analysis (EDA)\n## Exploring Job Corpus and Candidate Resumes\n\nThis notebook analyzes the distribution of candidate resumes and merged job postings across Naukri, LinkedIn, and standard job listings."),
    code_cell("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import CLEAN_RESUMES_PATH, CLEAN_JOBS_PATH
from src.features.match_features import extract_skills

sns.set_theme(style='whitegrid')"""),
    md_cell("### 1. Load Processed Datasets"),
    code_cell("""resumes = pd.read_csv(CLEAN_RESUMES_PATH)
jobs = pd.read_csv(CLEAN_JOBS_PATH)

print(f"Resumes shape: {resumes.shape}")
print(f"Jobs corpus shape: {jobs.shape}")
resumes.head(3)"""),
    md_cell("### 2. Resume Category Distribution"),
    code_cell("""plt.figure(figsize=(10, 6))
cat_counts = resumes['category'].value_counts()
sns.barplot(x=cat_counts.values, y=cat_counts.index, palette='viridis')
plt.title("Distribution of Resumes Across 24 Professional Categories")
plt.xlabel("Count")
plt.ylabel("Category")
plt.tight_layout()
plt.show()"""),
    md_cell("### 3. Job Sources Breakdown"),
    code_cell("""plt.figure(figsize=(6, 4))
jobs['source'].value_counts().plot(kind='pie', autopct='%1.1f%%', colormap='Set2')
plt.title("Job Corpus Distribution by Source Dataset")
plt.ylabel("")
plt.show()"""),
    md_cell("### 4. Top Extracted Skills in Resumes vs Jobs"),
    code_cell("""resume_sample_skills = []
for text in resumes['resume_text'].sample(300, random_state=42):
    resume_sample_skills.extend(extract_skills(text))

job_sample_skills = []
for text in jobs['job_text'].sample(300, random_state=42):
    job_sample_skills.extend(extract_skills(text))

top_resume_skills = pd.Series(resume_sample_skills).value_counts().head(12)
top_job_skills = pd.Series(job_sample_skills).value_counts().head(12)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(x=top_resume_skills.values, y=top_resume_skills.index, ax=axes[0], palette='Blues_r')
axes[0].set_title("Top In-Demand Skills in Resumes")

sns.barplot(x=top_job_skills.values, y=top_job_skills.index, ax=axes[1], palette='Oranges_r')
axes[1].set_title("Top In-Demand Skills in Job Postings")
plt.tight_layout()
plt.show()""")
]

# 2. 02_resume_classifier.ipynb
nb2_cells = [
    md_cell("# SmartHire — Supervised Resume Category Classifier (Model A)\n\nTrains and evaluates a multi-class text classification model predicting candidate role domains using TF-IDF and Logistic Regression / Random Forest."),
    code_cell("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import CLEAN_RESUMES_PATH, CLASSIFIER_PATH
from src.models.classifier import train_classifier, predict_category, load_classifier

resumes = pd.read_csv(CLEAN_RESUMES_PATH)
print(f"Loaded {len(resumes)} labeled resumes.")"""),
    md_cell("### Train & Evaluate Classifier"),
    code_cell("""model, metrics = train_classifier(resumes, save=True)
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
print(f"Macro F1-Score: {metrics['f1_macro']:.4f}")"""),
    md_cell("### Sample Inference Test"),
    code_cell("""sample_cv = "Senior Software Engineer with 6 years experience in Python, Django, React, PostgreSQL, Docker, AWS and microservices architecture."
result = predict_category(model, sample_cv)
print("Predicted Domain:", result['predicted_category'])
print(f"Confidence: {result['confidence_pct']}%")
print("Top 3 Predictions:", result['top_predictions'][:3])""")
]

# 3. 03_recommender.ipynb
nb3_cells = [
    md_cell("# SmartHire — Unsupervised Job Recommender (Core Engine)\n\nComputes content-based recommendations using TF-IDF and Cosine Similarity, with Precision@K evaluation."),
    code_cell("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
from src.config import CLEAN_RESUMES_PATH, CLEAN_JOBS_PATH
from src.models.recommender import recommend_jobs, evaluate_precision_at_k
from src.features.text_features import load_tfidf_vectorizer

resumes = pd.read_csv(CLEAN_RESUMES_PATH)
jobs = pd.read_csv(CLEAN_JOBS_PATH)
vectorizer = load_tfidf_vectorizer()

print(f"Jobs database size: {len(jobs)} postings.")"""),
    md_cell("### Run Job Matching for Sample Candidate"),
    code_cell("""sample_resume = resumes[resumes['category'] == 'INFORMATION-TECHNOLOGY'].iloc[0]['resume_text']
recommendations = recommend_jobs(sample_resume, jobs, top_n=10, vectorizer=vectorizer)
recommendations[['rank', 'title', 'company', 'location', 'match_percent']].head(10)"""),
    md_cell("### Quantitative Evaluation: Precision@5"),
    code_cell("""p_at_5 = evaluate_precision_at_k(recommendations, 'information technology', k=5)
print(f"Recommender Precision@5: {p_at_5 * 100:.1f}%")""")
]

# 4. 04_clustering_topics.ipynb
nb4_cells = [
    md_cell("# SmartHire — Unsupervised Job Clustering & Topic Discovery\n\nDiscovers natural job families using K-Means, evaluates optimal k with Elbow and Silhouette methods, and visualizes clusters with 2D PCA."),
    code_cell("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
import matplotlib.pyplot as plt
from src.config import CLEAN_JOBS_PATH
from src.models.clustering import cluster_jobs, find_optimal_k, compute_pca_projection, extract_job_topics

jobs = pd.read_csv(CLEAN_JOBS_PATH)"""),
    md_cell("### 1. Elbow Method & Silhouette Score"),
    code_cell("""elbow = find_optimal_k(jobs.head(1500), k_range=range(2, 7))
fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()
ax1.plot(elbow['k_values'], elbow['inertias'], 'b-o')
ax2.plot(elbow['k_values'], elbow['silhouette_scores'], 'g-s')
ax1.set_xlabel('k (Number of Clusters)')
ax1.set_ylabel('Inertia', color='b')
ax2.set_ylabel('Silhouette Score', color='g')
plt.title('Optimal k Selection for Job Clusters')
plt.show()"""),
    md_cell("### 2. K-Means Clustering & 2D PCA Visualization"),
    code_cell("""clustered, km, vec, info = cluster_jobs(jobs, max_rows=3000, save=True)
texts = clustered['job_text'].fillna('').astype(str)
mat = vec.transform(texts)
coords = compute_pca_projection(mat, n_components=2)

plt.figure(figsize=(9, 6))
plt.scatter(coords[:, 0], coords[:, 1], c=clustered['cluster'], cmap='tab10', alpha=0.5, s=15)
plt.title("2D PCA Projection of Job Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label='Cluster ID')
plt.show()"""),
    md_cell("### 3. Latent Dirichlet Allocation (LDA) Topic Extraction"),
    code_cell("""topics = extract_job_topics(jobs.head(2000), n_topics=5)
for t in topics:
    print(f"Topic {t['topic_id']}: {t['theme']} -> Keywords: {', '.join(t['keywords'])}")""")
]

# 5. 05_fit_predictor.ipynb
nb5_cells = [
    md_cell("# SmartHire — Supervised Shortlisting / Fit Predictor (Model B)\n\nPredicts candidate shortlisting probability based on engineered features: skill overlap, experience match, education match, and text similarity. Compares Logistic Regression vs Random Forest vs XGBoost."),
    code_cell("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
import matplotlib.pyplot as plt
from src.config import CLEAN_RESUMES_PATH, CLEAN_JOBS_PATH
from src.models.fit_predictor import train_fit_predictor, predict_fit_score
from src.features.text_features import load_tfidf_vectorizer

resumes = pd.read_csv(CLEAN_RESUMES_PATH)
jobs = pd.read_csv(CLEAN_JOBS_PATH)
vectorizer = load_tfidf_vectorizer()"""),
    md_cell("### 1. Train Models & Compare Metrics"),
    code_cell("""pipe, results = train_fit_predictor(resumes_df=resumes, jobs_df=jobs, vectorizer=vectorizer, save=True)
comp_table = pd.DataFrame(results['comparison_table']).T
comp_table[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]"""),
    md_cell("### 2. Feature Importances"),
    code_cell("""feat_imp = pd.Series(results['feature_importances']).sort_values(ascending=False)
plt.figure(figsize=(8, 4))
feat_imp.plot(kind='barh', color='teal')
plt.title("Fit Predictor Feature Importances")
plt.xlabel("Weight / Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()"""),
    md_cell("### 3. Inference on Sample Candidate-Job Pair"),
    code_cell("""cand_cv = "Data Scientist with expertise in Python, Machine Learning, Scikit-learn, Pandas, SQL and Tableau with 3 years experience."
job_req = "Senior Data Analyst required. Must have 3+ years experience in Python, SQL, Tableau, data visualization and statistical modeling."

fit_out = predict_fit_score(cand_cv, job_req, model=pipe, vectorizer=vectorizer)
print("Fit Probability:", fit_out['fit_percent'], "%")
print("Shortlist Status:", fit_out['status'])
print("Feature Breakdown:", fit_out['features'])""")
]

# Write out all notebooks
notebooks_to_create = [
    ("01_eda.ipynb", nb1_cells),
    ("02_resume_classifier.ipynb", nb2_cells),
    ("03_recommender.ipynb", nb3_cells),
    ("04_clustering_topics.ipynb", nb4_cells),
    ("05_fit_predictor.ipynb", nb5_cells)
]

for name, cells in notebooks_to_create:
    nb_path = NOTEBOOKS_DIR / name
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"Created notebook: {nb_path}")

print("All 5 project notebooks generated successfully.")
