# SmartHire — Resume-to-Job Matching & Career Guidance Engine

An end-to-end Machine Learning industrial project combining supervised and unsupervised learning techniques to recommend jobs, predict candidate domain classification, assess candidate shortlisting fit probability, and generate actionable skill-gap reports.

> **Project Principle:** Built entirely using classical machine learning (Scikit-Learn, TF-IDF, Logistic Regression, Random Forest, XGBoost, K-Means, LDA). No costly black-box generative AI or LLMs.

---

## 🚀 Key Features

1. **Resume Category Classification (Supervised Model A)**
   - Predicts candidate job domain across 24 professional fields (Information Technology, Data Science, HR, Finance, Engineering, Healthcare, etc.).
   - Multiclass TF-IDF pipeline with probability distribution and confidence score.
2. **Content-Based Job Recommendation (Unsupervised Core Engine)**
   - High-dimensional TF-IDF vectorization and Cosine Similarity ranking over 25,000+ real-world job postings (Naukri, LinkedIn, and standard job corpora).
   - Filtering by target role, title, and location. Precision@K evaluation.
3. **Candidate Shortlisting / Fit Predictor (Supervised Model B)**
   - 8-Dimensional feature engineering: Skill overlap (Jaccard), coverage ratio, matched skills count, missing skills count, TF-IDF cosine similarity, experience level match, education level match, and text length ratio.
   - Evaluates Logistic Regression, Random Forest, and XGBoost with **0.9891 ROC-AUC**.
4. **Skill-Gap Analytics & Career Guidance**
   - Extracts candidate skills against employer requirements.
   - Categorizes skills into matched and missing competencies with actionable preparation tips.
5. **Job-Family Clustering & LDA Topic Discovery (Unsupervised)**
   - K-Means clustering with Elbow method (Inertia) and Silhouette score analysis.
   - 2D PCA cluster projection visualization.
   - Latent Dirichlet Allocation (LDA) topic modeling to surface in-demand industry skill themes.
6. **Rule-Based AI Career Mentor (Stretch Goal)**
   - Interactive career doubt-clearing assistant answering questions regarding required skills, resume optimization, and qualification gaps without external LLM dependencies.
7. **Comprehensive Technical Report & Visualizations**
   - Publication-quality figures (confusion matrix, ROC curves, PCA plots, elbow curves).
   - Automated PDF technical report generation (`reports/final_report.pdf`).

---

## 📁 Repository Structure

```text
smarthire/
├── README.md                      # Project documentation and quick start
├── requirements.txt               # Pinned dependencies
├── .gitignore                     # Ignored files (venv, raw dumps)
│
├── data/
│   ├── raw/                       # Original downloaded datasets (never modified)
│   ├── interim/                   # Partially cleaned / merged interim files
│   └── processed/                 # Model-ready clean datasets (clean_resumes.csv, clean_jobs.csv)
│
├── notebooks/
│   ├── 01_eda.ipynb               # Explore jobs and resumes
│   ├── 02_resume_classifier.ipynb # Supervised: category classification model
│   ├── 03_recommender.ipynb       # Unsupervised: cosine similarity ranking
│   ├── 04_clustering_topics.ipynb # Unsupervised: KMeans clusters + LDA topics
│   └── 05_fit_predictor.ipynb     # Supervised: candidate shortlisting model
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Project paths, hyperparameters, constants
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py           # Robust multi-format dataset loaders
│   │   └── preprocess.py          # Text cleaning and unified corpus builder
│   ├── features/
│   │   ├── __init__.py
│   │   ├── text_features.py       # TF-IDF vectorization & similarity
│   │   └── match_features.py      # Skill extraction, experience, education, fit features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py          # Supervised resume category classifier
│   │   ├── recommender.py         # Content-based cosine similarity engine
│   │   ├── clustering.py          # KMeans clustering, PCA & LDA topic modeling
│   │   └── fit_predictor.py       # Candidate shortlisting / fit predictor
│   ├── parsing/
│   │   ├── __init__.py
│   │   └── resume_parser.py       # PDF, DOCX, TXT multi-format text extraction
│   └── evaluate.py                # Evaluation metrics across all models
│
├── models/                        # Serialized model artifacts (.pkl)
│   ├── classifier.pkl             # Trained resume category classifier
│   ├── tfidf_vectorizer.pkl       # Fitted global TF-IDF vectorizer
│   ├── fit_predictor.pkl          # Winning candidate fit predictor model
│   └── clustering_model.pkl       # Fitted KMeans clustering bundle
│
├── app/
│   └── streamlit_app.py           # Streamlit web portal application
│
├── reports/
│   ├── figures/                   # Confusion matrix, ROC curves, PCA plots
│   └── final_report.pdf           # Formatted written technical report
│
├── tests/
│   └── test_features.py           # Comprehensive unit tests
│
├── train_pipeline.py              # End-to-end pipeline training & figure generation
├── generate_notebooks.py          # Script to generate project notebooks
└── generate_report_pdf.py         # Script to compile final PDF technical report
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows, macOS, or Linux

### 2. Environment Setup
```powershell
# Create and activate virtual environment (optional)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## ⚡ Running the Application

Launch the Streamlit web demo with:
```powershell
streamlit run app/streamlit_app.py
```
Or with Python module syntax:
```powershell
python -m streamlit run app/streamlit_app.py
```
The portal will open automatically at `http://localhost:8501`.

---

## 🧪 Running Unit Tests

Run the full test suite with:
```powershell
python -m unittest discover tests
```

---

## 🔄 Re-Running the Training Pipeline

To rebuild the preprocessed corpus, retrain all supervised and unsupervised models, and regenerate figures:
```powershell
python train_pipeline.py
python generate_report_pdf.py
```

---

## 📊 Model Evaluation Summary

| Model Component | Algorithm | Primary Metrics | Status |
| :--- | :--- | :--- | :--- |
| **Resume Classifier** | TF-IDF + Logistic Regression | Accuracy: 64.6% (24 classes), Weighted F1: 0.626 | Saved (`classifier.pkl`) |
| **Job Recommender** | TF-IDF + Cosine Similarity | Precision@5: 40-60%, Rank Score | Active Engine |
| **Job Clustering** | K-Means + PCA + LDA | Inertia Elbow & Silhouette score | Saved (`clustering_model.pkl`) |
| **Fit Predictor** | 8-D Features + Logistic Regression / XGBoost | **ROC-AUC: 0.9891**, Accuracy: 97.8%, F1: 0.974 | Saved (`fit_predictor.pkl`) |
