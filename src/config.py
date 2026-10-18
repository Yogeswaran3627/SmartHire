from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".json", ".jsonl", ".ldjson"}
MAX_JOB_ROWS = 50000
RANDOM_STATE = 42
DEFAULT_TOP_N = 15
TFIDF_MAX_FEATURES = 30000

# Saved model paths according to section 7 specification
CLASSIFIER_PATH = MODEL_DIR / "classifier.pkl"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
FIT_PREDICTOR_PATH = MODEL_DIR / "fit_predictor.pkl"
CLUSTERING_PATH = MODEL_DIR / "clustering_model.pkl"

# Processed data paths
CLEAN_RESUMES_PATH = PROCESSED_DATA_DIR / "clean_resumes.csv"
CLEAN_JOBS_PATH = PROCESSED_DATA_DIR / "clean_jobs.csv"

for folder in (INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, REPORTS_DIR, FIGURES_DIR):
    folder.mkdir(parents=True, exist_ok=True)

